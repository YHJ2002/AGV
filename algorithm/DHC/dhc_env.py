import numpy as np
from typing import List, Dict, Tuple
from config.settings import SimConfig
from core.agv import StepInfo
from core.agvmanager import AGVManager
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from core.env import Env
from core.fault_manager import FaultManager
from utils.algorithm_factory import build_scheduler, build_planner
from .dhc_converter import DHCCompatibleConverter
from . import configs


# 动作到坐标偏移的映射，基于仓储地图常用的 (x, y) 坐标系。
ACTION_DELTA = {
    0: (0,  0),   # stay
    1: (0, -1),   # up
    2: (0,  1),   # down
    3: (-1, 0),   # left
    4: (1,  0)    # right
}

# 将底层环境的步进反馈映射到 DHC 风格奖励。
DHC_REWARD = {
    'move':          -0.075,
    'collision':     -0.5,
    'stay_off_goal': -0.075,
    'stay_on_goal':   0.0,
    'finish':        +5.0,
    'other':         -0.075,
}

# 额外距离塑形奖励的缩放系数。
DIST_REWARD_SCALE = 0.1


class DHCAVGEnv:
    """
    对外提供与 DHC Environment 一致的接口，
    底层实际运行的是仓储 AGV 仿真环境。
    """
    def __init__(self, curriculum):
        grid_map = GridMap()
        self.ordermanager = OrderManager(grid_map)
        self.agv_manager = AGVManager(grid_map, self.ordermanager)
        self.real_env = Env(self.agv_manager, grid_map, self.ordermanager)
        self.fault_manager = FaultManager(self.agv_manager, self.real_env, grid_map)

        # 复用项目中的调度器 / 规划器工厂，保证接口与主仿真保持一致。
        self.scheduler = build_scheduler(
            self.real_env,
            self.agv_manager,
            self.ordermanager,
            grid_map,
            self.fault_manager
        )
        self.planner = build_planner(
            self.real_env,
            self.agv_manager,
            self.ordermanager,
            grid_map,
            self.fault_manager
        )

        self.obs_radius = configs.obs_radius
        self.converter = DHCCompatibleConverter(
            num_agvs=self.agv_manager.num_agvs,
            gridmap=grid_map,
            agvmanager=self.agv_manager
        )
        self.steps = 0
        self.num_agents = self.agv_manager.num_agvs
        self.map_size = (self.real_env.map.height, self.real_env.map.width)

        # 记录每个 AGV 上一步到目标点的曼哈顿距离，用于 shaping reward。
        self.prev_goal_distances = {}
        SimConfig.force_replan_every_step = True

    def reset(self):
        self.steps = 0
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
        """
        actions: 当前参与决策的 AGV 动作列表，动作值范围为 0~4
        返回: ((obs, pos), rewards, done, info)
        """
        replanning_targets = self.agv_manager.get_replan_targets()
        next_pos_dict: Dict[int, List[Tuple[int, int]]] = {}
        for agv_id, (current_pos, goal_pos) in replanning_targets.items():
            if agv_id >= len(actions):
                raise IndexError(f"AGV {agv_id} 的动作索引超出 actions 列表长度 {len(actions)}")

            action = actions[agv_id]
            if action not in ACTION_DELTA:
                raise ValueError(f"AGV {agv_id} 的动作值 {action} 非法")

            dx, dy = ACTION_DELTA[action]
            next_x = current_pos[0] + dx
            next_y = current_pos[1] + dy
            next_pos_dict[agv_id] = [(next_x, next_y)]

        # DHC 直接给出下一步位置，因此这里把这些位置当作新的规划结果写回 AGV。
        self.agv_manager.replan_paths(next_pos_dict)
        step_info = self.real_env.step()

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

        env_info = self.real_env.get_env_info()
        static_grid = env_info['static_grid']
        agv_positions = env_info['current_grid_pos']

        obs, pos = self.converter.convert(
            static_grid=static_grid,
            agv_positions_xy=agv_positions,
            targets=replanning_targets
        )

        self.steps += 1
        if self.ordermanager.can_generate_more_orders():
            self.ordermanager.step()

        all_agv_ids = self.agv_manager.all_agv_ids

        rewards = []
        for agv_id in all_agv_ids:
            info = step_info[agv_id]

            if info == StepInfo.FINISH:
                r = DHC_REWARD['finish']
            elif info == StepInfo.MOVE:
                r = DHC_REWARD['move']
            elif info == StepInfo.COLLISION:
                r = DHC_REWARD['collision']
            elif info == StepInfo.STAY_OFF_GOAL:
                r = DHC_REWARD['stay_off_goal']
            elif info == StepInfo.STAY_ON_GOAL:
                r = DHC_REWARD['stay_on_goal']
            elif info == StepInfo.OTHER:
                r = DHC_REWARD['other']
            else:
                raise ValueError(f"Unknown StepInfo: {info}")

            if agv_id in replanning_targets:
                curr_pos, goal_pos = replanning_targets[agv_id]
                curr_dist = abs(curr_pos[0] - goal_pos[0]) + abs(curr_pos[1] - goal_pos[1])
                if info == StepInfo.FINISH:
                    self.prev_goal_distances[agv_id] = curr_dist
                else:
                    prev_dist = self.prev_goal_distances.get(agv_id, curr_dist)
                    delta_dist = prev_dist - curr_dist
                    r += DIST_REWARD_SCALE * delta_dist
                    self.prev_goal_distances[agv_id] = curr_dist
            else:
                self.prev_goal_distances.pop(agv_id, None)

            rewards.append(r)

        overall_done = self.ordermanager.is_all_orders_completed() or self.steps >= SimConfig.max_steps
        return (obs, pos), rewards, overall_done, info

    def render(self):
        self.real_env.render()

    def close(self):
        self.real_env.close()

    def observe(self):
        env_info = self.real_env.get_env_info()
        static_grid = env_info['static_grid']
        agv_positions = env_info['current_grid_pos']
        replanning_targets = self.agv_manager.get_replan_targets()
        obs, pos = self.converter.convert(
            static_grid=static_grid,
            agv_positions_xy=agv_positions,
            targets=replanning_targets
        )

        return obs, pos

    def update_env_settings_set(self, test):
        pass
