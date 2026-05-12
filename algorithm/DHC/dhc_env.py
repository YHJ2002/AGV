from typing import Dict, List, Tuple

from config.settings import SimConfig
from core.agv import StepInfo
from core.agvmanager import AGVManager
from core.env import Env
from core.fault_manager import FaultManager
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from utils.algorithm_factory import build_scheduler

from . import configs
from .dhc_converter import DHCCompatibleConverter


ACTION_DELTA = {
    0: (0, 0),
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}

DHC_REWARD = dict(configs.reward_fn)
DIST_REWARD_SCALE = configs.distance_reward_scale


class DHCAVGEnv:
    """Project-backed environment adapter for DHC training."""

    def __init__(self, curriculum, seed_offset: int = 0):
        self.curriculum = curriculum
        self.seed_offset = seed_offset
        self.episode_idx = 0
        self.env_settings_set = [configs.init_env_settings]

        if curriculum:
            # Training needs a feasible horizon; the project defaults are tuned
            # for evaluation and can be too harsh for early learning.
            SimConfig.total_orders_limit = configs.train_total_orders_limit
            SimConfig.order_processing_timeout = configs.train_order_processing_timeout
            SimConfig.max_steps = configs.train_max_steps

        grid_map = GridMap()
        self.ordermanager = OrderManager(grid_map)
        self.agv_manager = AGVManager(grid_map, self.ordermanager)
        self.real_env = Env(self.agv_manager, grid_map, self.ordermanager)
        self.fault_manager = FaultManager(self.agv_manager, self.real_env, grid_map)

        self.scheduler = build_scheduler(
            self.real_env,
            self.agv_manager,
            self.ordermanager,
            grid_map,
            self.fault_manager,
        )
        # The task scheduler does not depend on a path planner during DHC
        # training, so keep this unset instead of recursively loading a planner.
        self.planner = None

        self.obs_radius = configs.obs_radius
        self.converter = DHCCompatibleConverter(
            num_agvs=self.agv_manager.num_agvs,
            gridmap=grid_map,
            agvmanager=self.agv_manager,
        )
        self.steps = 0
        self.num_agents = self.agv_manager.num_agvs
        self.map_size = (self.real_env.map.height, self.real_env.map.width)
        self.prev_goal_distances: Dict[int, int] = {}
        self.all_agv_ids = sorted(self.agv_manager.all_agv_ids)

        SimConfig.force_replan_every_step = True

    def reset(self):
        self.steps = 0
        if self.curriculum and configs.train_randomize_order_seed:
            SimConfig.order_seed = configs.train_order_seed_base + self.seed_offset + self.episode_idx
            self.episode_idx += 1
        self.real_env.reset()
        self.fault_manager.reset()
        if self.ordermanager.can_generate_more_orders():
            self.ordermanager.step()
        self.scheduler.reset()
        self.prev_goal_distances.clear()

        idle_agv_set = self.agv_manager.get_idle_agv_ids()
        if idle_agv_set:
            agv_tasks = self.scheduler.assign_tasks(idle_agv_set, self.planner)
            if agv_tasks:
                self.agv_manager.assign_tasks(agv_tasks)

        replanning_targets = self.agv_manager.get_replan_targets()
        for agv_id, (curr_pos, goal_pos) in replanning_targets.items():
            dist = abs(curr_pos[0] - goal_pos[0]) + abs(curr_pos[1] - goal_pos[1])
            self.prev_goal_distances[agv_id] = dist

        return self.observe()

    def step(self, actions: List[int]) -> Tuple:
        replanning_targets = self.agv_manager.get_replan_targets()
        next_pos_dict: Dict[int, List[Tuple[int, int]]] = {}
        for agv_id, (current_pos, _) in replanning_targets.items():
            if agv_id >= len(actions):
                raise IndexError(f"AGV {agv_id} action index exceeds action list length {len(actions)}")

            action = actions[agv_id]
            if action not in ACTION_DELTA:
                raise ValueError(f"AGV {agv_id} action {action} is invalid")

            dx, dy = ACTION_DELTA[action]
            next_pos_dict[agv_id] = [(current_pos[0] + dx, current_pos[1] + dy)]

        self.agv_manager.replan_paths(next_pos_dict)
        step_info_dict = self.real_env.step()

        idle_agv_set = self.agv_manager.get_idle_agv_ids()
        if idle_agv_set:
            agv_tasks = self.scheduler.assign_tasks(idle_agv_set, self.planner)
            if agv_tasks:
                self.agv_manager.assign_tasks(agv_tasks)

        agvs_needing_rest = self.agv_manager.get_need_rest_agv_ids()
        if agvs_needing_rest:
            rest_assignments = self.scheduler.assign_rest_areas(agvs_needing_rest)
            self.agv_manager.assign_rest_zones(rest_assignments)

        replanning_targets = self.agv_manager.get_replan_targets()
        obs, pos = self.observe()

        self.steps += 1
        if self.ordermanager.can_generate_more_orders():
            self.ordermanager.step()
        timeout_events = self.ordermanager.consume_timeout_events()
        timeout_counts: Dict[int, int] = {}
        for _, agv_id in timeout_events:
            timeout_counts[agv_id] = timeout_counts.get(agv_id, 0) + 1

        rewards = []
        for agv_id in self.all_agv_ids:
            info = step_info_dict[agv_id]

            if info == StepInfo.ORDER_COMPLETE:
                reward = DHC_REWARD["order_complete"]
            elif info == StepInfo.FINISH:
                reward = DHC_REWARD["finish"]
            elif info == StepInfo.MOVE:
                reward = DHC_REWARD["move"]
            elif info == StepInfo.COLLISION:
                reward = DHC_REWARD["collision"]
            elif info == StepInfo.STAY_OFF_GOAL:
                reward = DHC_REWARD["stay_off_goal"]
            elif info == StepInfo.STAY_ON_GOAL:
                reward = DHC_REWARD["stay_on_goal"]
            elif info == StepInfo.OTHER:
                reward = DHC_REWARD["other"]
            else:
                raise ValueError(f"Unknown StepInfo: {info}")

            if agv_id in timeout_counts:
                reward += DHC_REWARD["timeout"] * timeout_counts[agv_id]

            if agv_id in replanning_targets:
                curr_pos, goal_pos = replanning_targets[agv_id]
                curr_dist = abs(curr_pos[0] - goal_pos[0]) + abs(curr_pos[1] - goal_pos[1])
                if info in (StepInfo.FINISH, StepInfo.ORDER_COMPLETE):
                    self.prev_goal_distances[agv_id] = curr_dist
                else:
                    prev_dist = self.prev_goal_distances.get(agv_id, curr_dist)
                    reward += DIST_REWARD_SCALE * (prev_dist - curr_dist)
                    self.prev_goal_distances[agv_id] = curr_dist
            else:
                self.prev_goal_distances.pop(agv_id, None)

            rewards.append(reward)

        overall_done = self.ordermanager.is_all_orders_completed() or self.steps >= SimConfig.max_steps
        return (obs, pos), rewards, overall_done, step_info_dict

    def render(self):
        self.real_env.render()

    def close(self):
        self.real_env.close()

    def observe(self):
        env_info = self.real_env.get_env_info()
        return self.converter.convert(
            static_grid=env_info["static_grid"],
            agv_positions_xy=env_info["current_grid_pos"],
            targets=self.agv_manager.get_replan_targets(),
        )

    def update_env_settings_set(self, env_settings):
        self.env_settings_set = env_settings
