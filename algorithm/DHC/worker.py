import os
import random
import threading
import time
from copy import deepcopy
from typing import Tuple

import numpy as np
import ray
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from . import configs
from .buffer import LocalBuffer, SumTree
from .dhc_env import DHCAVGEnv
from .model import Network


@ray.remote(num_cpus=1)
class GlobalBuffer:
    def __init__(
        self,
        episode_capacity=configs.episode_capacity,
        local_buffer_capacity=configs.max_episode_length,
        init_env_settings=configs.init_env_settings,
        alpha=configs.prioritized_replay_alpha,
        beta=configs.prioritized_replay_beta,
    ):
        self.capacity = episode_capacity
        self.local_buffer_capacity = local_buffer_capacity
        self.size = 0
        self.ptr = 0

        self.priority_tree = SumTree(episode_capacity * local_buffer_capacity)
        self.alpha = alpha
        self.beta = beta

        self.counter = 0
        self.batched_data = []
        self.stat_dict = {init_env_settings: []}
        self.lock = threading.Lock()
        self.env_settings_set = ray.put([init_env_settings])

        self.obs_buf = np.zeros(
            ((local_buffer_capacity + 1) * episode_capacity, configs.max_num_agents, *configs.obs_shape),
            dtype=bool,
        )
        self.act_buf = np.zeros(
            (local_buffer_capacity * episode_capacity, configs.max_num_agents),
            dtype=np.uint8,
        )
        self.rew_buf = np.zeros(
            (local_buffer_capacity * episode_capacity, configs.max_num_agents),
            dtype=np.float16,
        )
        self.hid_buf = np.zeros(
            (local_buffer_capacity * episode_capacity, configs.max_num_agents, configs.hidden_dim),
            dtype=np.float16,
        )
        self.done_buf = np.zeros(episode_capacity, dtype=bool)
        self.size_buf = np.zeros(episode_capacity, dtype=np.uint32)
        self.comm_mask_buf = np.zeros(
            ((local_buffer_capacity + 1) * episode_capacity, configs.max_num_agents, configs.max_num_agents),
            dtype=bool,
        )

    def __len__(self):
        return self.size

    def run(self):
        self.background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        self.background_thread.start()

    def prepare_data(self):
        while True:
            if len(self.batched_data) <= 4:
                data = self.sample_batch(configs.batch_size)
                self.batched_data.append(ray.put(data))
            else:
                time.sleep(0.1)

    def get_data(self):
        if not self.batched_data:
            return ray.put(self.sample_batch(configs.batch_size))
        return self.batched_data.pop(0)

    def add(self, data: Tuple):
        """
        data:
        actor_id 0, num_agents 1, map_len 2, obs_buf 3, act_buf 4,
        rew_buf 5, hid_buf 6, td_errors 7, done 8, size 9, comm_mask 10
        """
        stat_key = (data[1], data[2])
        self.stat_dict.setdefault(stat_key, []).append(data[8])
        if len(self.stat_dict[stat_key]) > 200:
            self.stat_dict[stat_key].pop(0)

        with self.lock:
            step_start_idx = self.ptr * self.local_buffer_capacity
            obs_start_idx = self.ptr * (self.local_buffer_capacity + 1)
            idxes = np.arange(step_start_idx, step_start_idx + self.local_buffer_capacity)

            self.size -= int(self.size_buf[self.ptr])
            self.size += data[9]
            self.counter += data[9]

            self.priority_tree.batch_update(idxes, data[7] ** self.alpha)

            self.obs_buf[obs_start_idx : obs_start_idx + data[9] + 1, : data[1]] = data[3]
            self.act_buf[step_start_idx : step_start_idx + data[9], : data[1]] = data[4]
            self.rew_buf[step_start_idx : step_start_idx + data[9], : data[1]] = data[5]
            self.hid_buf[step_start_idx : step_start_idx + data[9], : data[1]] = data[6]
            self.done_buf[self.ptr] = data[8]
            self.size_buf[self.ptr] = data[9]
            self.comm_mask_buf[obs_start_idx : obs_start_idx + data[9] + 1] = 0
            self.comm_mask_buf[obs_start_idx : obs_start_idx + data[9] + 1, : data[1], : data[1]] = data[10]

            self.ptr = (self.ptr + 1) % self.capacity

    def sample_batch(self, batch_size: int) -> Tuple:
        b_obs, b_action, b_reward, b_done, b_steps, b_seq_len, b_comm_mask = [], [], [], [], [], [], []
        b_hidden = []

        with self.lock:
            idxes, priorities = self.priority_tree.batch_sample(batch_size)
            global_idxes = idxes // self.local_buffer_capacity
            local_idxes = idxes % self.local_buffer_capacity

            for idx, global_idx, local_idx in zip(idxes.tolist(), global_idxes.tolist(), local_idxes.tolist()):
                episode_size = int(self.size_buf[global_idx])
                assert local_idx < episode_size, f"index is {local_idx} but size is {episode_size}"

                steps = min(configs.forward_steps, episode_size - local_idx)
                seq_len = min(local_idx + 1, configs.seq_len)

                episode_obs_start = global_idx * (self.local_buffer_capacity + 1)
                step_obs_idx = episode_obs_start + local_idx

                if local_idx < configs.seq_len - 1:
                    obs = self.obs_buf[episode_obs_start : step_obs_idx + 1 + steps]
                    comm_mask = self.comm_mask_buf[episode_obs_start : step_obs_idx + 1 + steps]
                    hidden = np.zeros((configs.max_num_agents, configs.hidden_dim), dtype=np.float16)
                else:
                    seq_start = step_obs_idx + 1 - configs.seq_len
                    obs = self.obs_buf[seq_start : step_obs_idx + 1 + steps]
                    comm_mask = self.comm_mask_buf[seq_start : step_obs_idx + 1 + steps]
                    if local_idx == configs.seq_len - 1:
                        hidden = np.zeros((configs.max_num_agents, configs.hidden_dim), dtype=np.float16)
                    else:
                        hidden = self.hid_buf[idx - configs.seq_len]

                if obs.shape[0] < configs.seq_len + configs.forward_steps:
                    pad_len = configs.seq_len + configs.forward_steps - obs.shape[0]
                    obs = np.pad(obs, ((0, pad_len), (0, 0), (0, 0), (0, 0), (0, 0)))
                    comm_mask = np.pad(comm_mask, ((0, pad_len), (0, 0), (0, 0)))

                action = self.act_buf[idx]
                reward = np.zeros(configs.max_num_agents, dtype=np.float32)
                for i in range(steps):
                    reward += self.rew_buf[idx + i].astype(np.float32) * (configs.gamma ** i)

                done = bool(self.done_buf[global_idx] and local_idx >= episode_size - configs.forward_steps)

                b_obs.append(obs)
                b_action.append(action)
                b_reward.append(reward)
                b_done.append(done)
                b_steps.append(steps)
                b_seq_len.append(seq_len)
                b_hidden.append(hidden)
                b_comm_mask.append(comm_mask)

            min_p = max(float(np.min(priorities)), 1e-6)
            weights = np.power(priorities / min_p, -self.beta)

            data = (
                torch.from_numpy(np.stack(b_obs).astype(np.float32)),
                torch.from_numpy(np.stack(b_action).astype(np.int64)),
                torch.from_numpy(np.stack(b_reward).astype(np.float32)),
                torch.FloatTensor(b_done).unsqueeze(1),
                torch.FloatTensor(b_steps).unsqueeze(1),
                torch.LongTensor(b_seq_len),
                torch.from_numpy(np.concatenate(b_hidden).astype(np.float32)),
                torch.from_numpy(np.stack(b_comm_mask)),
                idxes,
                torch.from_numpy(weights.astype(np.float32)).unsqueeze(1),
                self.ptr,
            )
            return data

    def update_priorities(self, idxes: np.ndarray, priorities: np.ndarray, old_ptr: int):
        with self.lock:
            if self.ptr > old_ptr:
                mask = (idxes < old_ptr * self.local_buffer_capacity) | (
                    idxes >= self.ptr * self.local_buffer_capacity
                )
                idxes = idxes[mask]
                priorities = priorities[mask]
            elif self.ptr < old_ptr:
                mask = (idxes < old_ptr * self.local_buffer_capacity) & (
                    idxes >= self.ptr * self.local_buffer_capacity
                )
                idxes = idxes[mask]
                priorities = priorities[mask]

            self.priority_tree.batch_update(np.copy(idxes), np.copy(priorities) ** self.alpha)

    def stats(self, interval: int):
        print(f"buffer update speed: {self.counter / interval}/s")
        print(f"buffer size: {self.size}")
        available = ", ".join(str(k) for k in sorted(self.stat_dict))
        print(f"observed env settings: {available}")
        self.env_settings_set = ray.put(sorted(self.stat_dict.keys()))
        self.counter = 0

    def ready(self):
        return len(self) >= configs.learning_starts

    def get_env_settings(self):
        return self.env_settings_set

    def check_done(self):
        # This project trains on a fixed warehouse environment and stops by
        # `training_times`, so we do not rely on the original curriculum stop.
        return False


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer: GlobalBuffer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Network().to(self.device)
        self.tar_model = deepcopy(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=configs.learning_rate)
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=list(configs.lr_milestones),
            gamma=configs.lr_gamma,
        )
        self.buffer = buffer
        self.counter = 0
        self.last_counter = 0
        self.done = False
        self.loss = 0.0

        if configs.load_model is not None and os.path.exists(configs.load_model):
            print(f"\n加载模型权重: {configs.load_model}")
            state_dict = torch.load(configs.load_model, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.tar_model.load_state_dict(state_dict)

            filename = os.path.basename(configs.load_model)
            import re

            match = re.search(r"(\d+)\.pth", filename)
            if match:
                resume_step = int(match.group(1))
                self.counter = resume_step
                self.last_counter = resume_step
                print(f"恢复训练步数: {resume_step}")
        else:
            print("未找到模型，从头训练")

        os.makedirs(configs.save_path, exist_ok=True)
        self.store_weights()

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights_id = ray.put(state_dict)

    def run(self):
        self.learning_thread = threading.Thread(target=self.train, daemon=True)
        self.learning_thread.start()

    def train(self):
        scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

        while self.counter < configs.training_times:
            data_id = ray.get(self.buffer.get_data.remote())
            data = ray.get(data_id)

            (
                b_obs,
                b_action,
                b_reward,
                b_done,
                b_steps,
                b_seq_len,
                b_hidden,
                b_comm_mask,
                idxes,
                weights,
                old_ptr,
            ) = data

            b_obs = b_obs.to(self.device)
            b_action = b_action.to(self.device)
            b_reward = b_reward.to(self.device)
            b_done = b_done.to(self.device)
            b_steps = b_steps.to(self.device)
            b_seq_len = b_seq_len.to(self.device)
            b_hidden = b_hidden.to(self.device)
            b_comm_mask = b_comm_mask.to(self.device)
            weights = weights.to(self.device)

            b_next_seq_len = torch.LongTensor(
                [(seq_len + step_count).item() for seq_len, step_count in zip(b_seq_len, b_steps)]
            ).to(self.device)

            with torch.no_grad():
                b_q_next = self.tar_model(b_obs, b_next_seq_len, b_hidden, b_comm_mask).max(dim=2)[0]
                b_q_next = (1 - b_done) * b_q_next

            b_q = self.model(
                b_obs[:, :-configs.forward_steps],
                b_seq_len,
                b_hidden,
                b_comm_mask[:, :-configs.forward_steps],
            ).gather(2, b_action.unsqueeze(-1)).squeeze(-1)

            td_target = b_reward + (configs.gamma ** b_steps) * b_q_next
            td_error = b_q - td_target

            priorities = td_error.detach().abs().mean(dim=1).clamp(1e-4).cpu().numpy()
            loss = (weights * self.huber_loss(td_error).mean(dim=1, keepdim=True)).mean()
            self.loss += float(loss.item())

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), configs.grad_norm_dqn)
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()

            self.buffer.update_priorities.remote(idxes, priorities, old_ptr)
            self.counter += 1

            if self.counter % 5 == 0:
                self.store_weights()

            if self.counter % configs.target_network_update_freq == 0:
                self.tar_model.load_state_dict(self.model.state_dict())

            if self.counter % configs.save_interval == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(configs.save_path, f"{self.counter}.pth"),
                )

        self.done = True

    def huber_loss(self, td_error, kappa=1.0):
        abs_td_error = td_error.abs()
        flag = (abs_td_error < kappa).float()
        return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)

    def stats(self, interval: int):
        print(f"number of updates: {self.counter}")
        print(f"update speed: {(self.counter - self.last_counter) / interval}/s")
        if self.counter != self.last_counter:
            print(f"loss: {self.loss / (self.counter - self.last_counter):.4f}")

        self.last_counter = self.counter
        self.loss = 0.0
        return self.done


@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, worker_id: int, epsilon: float, learner: Learner, buffer: GlobalBuffer):
        self.id = worker_id
        self.model = Network()
        self.model.eval()
        self.env = DHCAVGEnv(curriculum=True, seed_offset=worker_id * 100000)
        self.epsilon = epsilon
        self.learner = learner
        self.global_buffer = buffer
        self.max_episode_length = configs.max_episode_length
        self.counter = 0

    def run(self):
        self.update_weights()
        obs, pos, local_buffer = self.reset()

        while True:
            actions, q_val, hidden, comm_mask = self.model.step(
                torch.from_numpy(obs.astype(np.float32)),
                torch.from_numpy(pos.astype(np.float32)),
            )

            for agv_id in range(self.env.num_agents):
                if random.random() < self.epsilon:
                    actions[agv_id] = np.random.randint(0, configs.action_dim)

            (next_obs, next_pos), rewards, done, _ = self.env.step(actions)
            local_buffer.add(
                q_val,
                np.asarray(actions, dtype=np.uint8),
                np.asarray(rewards, dtype=np.float32),
                next_obs,
                hidden,
                comm_mask,
            )

            if not done and self.env.steps < self.max_episode_length:
                obs, pos = next_obs, next_pos
            else:
                if done:
                    data = local_buffer.finish()
                else:
                    _, q_val, _, comm_mask = self.model.step(
                        torch.from_numpy(next_obs.astype(np.float32)),
                        torch.from_numpy(next_pos.astype(np.float32)),
                    )
                    data = local_buffer.finish(q_val, comm_mask)

                self.global_buffer.add.remote(data)
                obs, pos, local_buffer = self.reset()

            self.counter += 1
            if self.counter >= configs.actor_update_steps:
                self.update_weights()
                self.counter = 0

    def update_weights(self):
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)

        new_env_settings_set = ray.get(self.global_buffer.get_env_settings.remote())
        self.env.update_env_settings_set(ray.get(new_env_settings_set))

    def reset(self):
        self.model.reset()
        obs, pos = self.env.reset()
        local_buffer = LocalBuffer(self.id, self.env.num_agents, self.env.map_size[1], obs)
        return obs, pos, local_buffer
