import numpy as np

from . import configs


class SumTree:
    """Prioritized replay tree."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        leaf_count = 1
        while leaf_count < capacity:
            leaf_count *= 2

        self.leaf_count = leaf_count
        self.layer = int(np.log2(leaf_count)) + 1
        self.leaf_offset = leaf_count - 1
        self.tree = np.zeros(2 * leaf_count - 1, dtype=np.float64)
        self.capacity = capacity
        self.size = 0

    def sum(self):
        leaf_sum = np.sum(self.tree[self.leaf_offset : self.leaf_offset + self.capacity])
        assert leaf_sum - self.tree[0] < 0.1, (
            f"sum is {leaf_sum} but root is {self.tree[0]}"
        )
        return self.tree[0]

    def __getitem__(self, idx: int):
        assert 0 <= idx < self.capacity
        return self.tree[self.leaf_offset + idx]

    def batch_sample(self, batch_size: int):
        p_sum = self.tree[0]
        interval = p_sum / batch_size
        prefixsums = (
            np.arange(0, p_sum, interval, dtype=np.float64)
            + np.random.uniform(0, interval, batch_size)
        )

        idxes = np.zeros(batch_size, dtype=int)
        for _ in range(self.layer - 1):
            left_nodes = self.tree[idxes * 2 + 1]
            idxes = np.where(prefixsums < left_nodes, idxes * 2 + 1, idxes * 2 + 2)
            prefixsums = np.where(idxes % 2 == 0, prefixsums - self.tree[idxes - 1], prefixsums)

        priorities = self.tree[idxes]
        idxes -= self.leaf_offset

        assert np.all(priorities > 0), f"idx: {idxes}, priority: {priorities}"
        assert np.all(idxes >= 0) and np.all(idxes < self.capacity)

        return idxes, priorities

    def batch_update(self, idxes: np.ndarray, priorities: np.ndarray):
        idxes += self.leaf_offset
        self.tree[idxes] = priorities

        for _ in range(self.layer - 1):
            idxes = (idxes - 1) // 2
            idxes = np.unique(idxes)
            self.tree[idxes] = self.tree[2 * idxes + 1] + self.tree[2 * idxes + 2]

        leaf_sum = np.sum(self.tree[self.leaf_offset : self.leaf_offset + self.capacity])
        assert leaf_sum - self.tree[0] < 0.1, (
            f"sum is {leaf_sum} but root is {self.tree[0]}"
        )


class LocalBuffer:
    __slots__ = (
        "actor_id",
        "map_len",
        "num_agents",
        "obs_buf",
        "act_buf",
        "rew_buf",
        "hid_buf",
        "comm_mask_buf",
        "q_buf",
        "capacity",
        "size",
        "done",
    )

    def __init__(
        self,
        actor_id: int,
        num_agents: int,
        map_len: int,
        init_obs: np.ndarray,
        capacity: int = configs.max_episode_length,
        obs_shape=configs.obs_shape,
        hidden_dim=configs.hidden_dim,
        action_dim=configs.action_dim,
    ):
        self.actor_id = actor_id
        self.num_agents = num_agents
        self.map_len = map_len

        self.obs_buf = np.zeros((capacity + 1, num_agents, *obs_shape), dtype=bool)
        self.act_buf = np.zeros((capacity, num_agents), dtype=np.uint8)
        self.rew_buf = np.zeros((capacity, num_agents), dtype=np.float16)
        self.hid_buf = np.zeros((capacity, num_agents, hidden_dim), dtype=np.float16)
        self.comm_mask_buf = np.zeros((capacity + 1, num_agents, num_agents), dtype=bool)
        self.q_buf = np.zeros((capacity + 1, num_agents, action_dim), dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.obs_buf[0] = init_obs

    def __len__(self):
        return self.size

    def add(
        self,
        q_val: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        hidden: np.ndarray,
        comm_mask: np.ndarray,
    ):
        assert self.size < self.capacity

        self.act_buf[self.size] = action
        self.rew_buf[self.size] = reward
        self.obs_buf[self.size + 1] = next_obs
        self.q_buf[self.size] = q_val
        self.hid_buf[self.size] = hidden
        self.comm_mask_buf[self.size] = comm_mask
        self.size += 1

    def finish(self, last_q_val=None, last_comm_mask=None):
        if last_q_val is None:
            done = True
        else:
            done = False
            self.q_buf[self.size] = last_q_val
            self.comm_mask_buf[self.size] = last_comm_mask

        self.obs_buf = self.obs_buf[: self.size + 1]
        self.act_buf = self.act_buf[: self.size]
        self.rew_buf = self.rew_buf[: self.size]
        self.hid_buf = self.hid_buf[: self.size]
        self.q_buf = self.q_buf[: self.size + 1]
        self.comm_mask_buf = self.comm_mask_buf[: self.size + 1]

        td_errors = np.zeros(self.capacity, dtype=np.float32)
        agent_idx = np.arange(self.num_agents)

        for t in range(self.size):
            steps = min(configs.forward_steps, self.size - t)
            discounted_reward = np.zeros(self.num_agents, dtype=np.float32)
            for i in range(steps):
                discounted_reward += self.rew_buf[t + i].astype(np.float32) * (configs.gamma ** i)

            if done and t >= self.size - configs.forward_steps:
                bootstrap = np.zeros(self.num_agents, dtype=np.float32)
            else:
                bootstrap = np.max(self.q_buf[t + steps], axis=1)

            target = discounted_reward + (configs.gamma ** steps) * bootstrap
            q_val = self.q_buf[t, agent_idx, self.act_buf[t]]
            td_errors[t] = max(float(np.mean(np.abs(target - q_val))), 1e-4)

        return (
            self.actor_id,
            self.num_agents,
            self.map_len,
            self.obs_buf,
            self.act_buf,
            self.rew_buf,
            self.hid_buf,
            td_errors,
            done,
            self.size,
            self.comm_mask_buf,
        )
