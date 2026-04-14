from typing import Deque, List, Tuple, Optional, Dict, Set, Generator
from core.agv import AGV, AGVAction, StepInfo
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from config.settings import SimConfig
import json
from utils.logger import global_logger


class AGVManager:
    def __init__(self, map_inst: GridMap, order_manager: OrderManager):
        # 从地图配置文件中读取 AGV 和等待区初始信息
        with open(SimConfig.map_file, "r") as f:
            data = json.load(f)

        # 读取 AGV 配置列表
        agv_data = data.get("agvs", [])
        # 读取等待区配置，格式为 {wait_zone_id: position}
        wait_zones = {w["wait_zone_id"]: tuple(w["position"]) for w in data.get("wait_zones", [])}

        agv_list = []
        for agv_entry in agv_data:
            agv_id = agv_entry["agv_id"]
            # 这里默认 AGV 的等待区编号与 AGV 编号一致
            wait_id = agv_id
            agv_size = agv_entry.get("size", 1)
            # AGV 初始位置设置为对应等待区位置
            init_grid = wait_zones[wait_id]

            # 创建 AGV 实例
            agv = AGV(
                agv_id=agv_id,
                size=agv_size,
                init_grid_pos=init_grid,
                map_inst=map_inst,
                order_manager=order_manager
            )
            agv_list.append(agv)

        # 用字典统一管理所有 AGV，键为 agv_id
        self._agvs: Dict[int, AGV] = {agv.id: agv for agv in agv_list}

        # 当前空闲 AGV 集合
        self.idle_agvs: Set[int] = {agv.id for agv in agv_list}
        # 当前需要分配休息区的 AGV 集合
        self.need_rest_agvs: Set[int] = set(self.idle_agvs)
        # 当前需要重新规划路径的 AGV 集合
        self.need_replan_agvs: Set[int] = set(self.idle_agvs)

        # 记录每个 AGV 连续被阻塞的次数
        self.block_counts: Dict[int, int] = {agv.id: 0 for agv in agv_list}

        # 按尺寸分类保存 AGV，便于异构车队管理
        self.agvs_by_size: Dict[int, Set[int]] = {}
        for agv in agv_list:
            self.agvs_by_size.setdefault(agv.size, set()).add(agv.id)

        # 保存每个 AGV 的尺寸
        self.agv_sizes: Dict[int, int] = {agv.id: agv.size for agv in agv_list}

        # AGV 总数
        self.num_agvs = len(agv_list)
        # 所有 AGV 的编号集合
        self.all_agv_ids = set(self._agvs.keys())

    def get_agv(self, agv_id: int) -> AGV:
        """根据 AGV 编号获取 AGV 对象"""
        return self._agvs[agv_id]

    def get_agv_speed(self, agv_id: int) -> float:
        """获取指定 AGV 的最大速度"""
        agv = self._agvs.get(agv_id)
        return agv.max_speed

    def get_grid_position(self, agv_id: int) -> Tuple[int, int]:
        """获取指定 AGV 的栅格坐标"""
        agv = self._agvs.get(agv_id)
        return agv.grid_pos

    def get_real_position(self, agv_id: int) -> Tuple[float, float]:
        """获取指定 AGV 的连续空间坐标"""
        agv = self._agvs.get(agv_id)
        return agv.real_pos

    def get_agv_size(self, agv_id: int) -> int:
        """获取指定 AGV 的尺寸"""
        return self.agv_sizes.get(agv_id, 1)

    def get_agv_ids_by_size(self, size: int) -> Set[int]:
        """返回指定尺寸的 AGV 编号集合"""
        return self.agvs_by_size.get(size, set())

    def all_agvs(self) -> Generator[AGV, None, None]:
        """遍历所有 AGV 对象"""
        yield from self._agvs.values()

    def get_idle_agv_ids(self) -> List[int]:
        """返回当前空闲 AGV 列表"""
        return list(self.idle_agvs)

    def get_need_rest_agv_ids(self) -> List[int]:
        """返回当前需要去休息区的 AGV 列表"""
        return list(self.need_rest_agvs)

    def get_need_replan_agv_ids(self) -> List[int]:
        """返回当前需要重新规划路径的 AGV 列表"""
        return list(self.need_replan_agvs)

    def get_carrying_status(self) -> Dict[int, bool]:
        """返回每个 AGV 是否正在载货"""
        return {
            agv_id: agv.carried_box_id is not None
            for agv_id, agv in self._agvs.items()
        }

    def get_carried_box_ids(self) -> Dict[int, Optional[int]]:
        """返回每个 AGV 当前携带的货箱编号"""
        return {
            agv_id: agv.carried_box_id
            for agv_id, agv in self._agvs.items()
        }

    def get_all_current_pos(self) -> Dict[int, Tuple[int, int]]:
        """返回所有 AGV 的当前栅格坐标"""
        return {agv_id: agv.grid_pos for agv_id, agv in self._agvs.items()}

    def get_all_next_pos(self) -> Dict[int, Tuple[int, int]]:
        """返回所有 AGV 下一步计划到达的位置"""
        return {agv_id: agv.get_next_pos() for agv_id, agv in self._agvs.items()}

    def get_all_real_positions(self) -> Dict[int, Tuple[float, float]]:
        """返回所有 AGV 的连续空间坐标"""
        return {agv_id: agv.real_pos for agv_id, agv in self._agvs.items()}

    def get_all_speeds(self) -> Dict[int, float]:
        """返回所有 AGV 的最大速度"""
        return {agv_id: agv.max_speed for agv_id, agv in self._agvs.items()}

    def get_all_action_queues(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        返回所有 AGV 当前动作队列。
        若 AGV 正在前往休息区，则用休息目标位置重复填充一个短队列，
        方便规划器和环境模块统一处理。
        """
        result = {}
        for agv_id, agv in self._agvs.items():
            if agv.is_resting:
                result[agv_id] = [agv.rest_target] * 10  # type: ignore
            else:
                result[agv_id] = list(agv.action_queue)
        return result

    def get_aligned_agv_ids(self) -> Set[int]:
        """返回 real_pos 与 grid_pos 对齐的 AGV 编号集合"""
        return {agv_id for agv_id, agv in self._agvs.items() if agv.is_aligned()}

    def increment_block_count(self, agv_id: int):
        """
        增加 AGV 的阻塞计数。
        若 AGV 不是已经停在休息目标点上，则记录一次碰撞/阻塞事件。
        """
        agv = self._agvs[agv_id]
        if agv.rest_target is None or agv.grid_pos != agv.rest_target:
            self.block_counts[agv_id] += 1
            global_logger.record_agv_collision(agv_id)

    def reset_block_count(self, agv_id: int):
        """重置指定 AGV 的阻塞计数"""
        self.block_counts[agv_id] = 0

    def step_all(self, next_positions: Dict[int, Tuple[int, int]]) -> Dict[int, StepInfo]:
        """
        推进所有 AGV 执行一步。
        next_positions 为每个 AGV 本步允许前往的目标位置。
        返回每个 AGV 的步进结果信息。
        """
        step_info_dict: Dict[int, StepInfo] = {}

        for agv_id, next_pos in next_positions.items():
            agv = self._agvs[agv_id]
            # 单个 AGV 执行一步，返回是否需要重规划以及步进信息
            need_replan, step_info = agv.step(next_pos)
            step_info_dict[agv_id] = step_info

            # 若 AGV 变为空闲，则加入空闲集合
            if agv.is_idle:
                self.idle_agvs.add(agv_id)
                # 若还没有分配休息区，则加入待休息集合
                if agv.rest_target is None:
                    self.need_rest_agvs.add(agv_id)

            # 若 AGV 自身请求重规划，或连续阻塞次数过多，则加入待重规划集合
            if need_replan or self.block_counts[agv_id] >= 3:
                self.need_replan_agvs.add(agv_id)
                self.reset_block_count(agv_id)

        return step_info_dict

    def assign_tasks(self, task_dict: Dict[int, List[Tuple[Tuple[int, int], AGVAction, int]]]):
        """
        将调度器分配的任务写入对应 AGV。
        task_dict: {agv_id: [(position, action, extra), ...]}
        """
        for agv_id, task_list in task_dict.items():
            agv = self._agvs[agv_id]
            agv.assign_task(task_list)
            # 已分配任务后，不再视为空闲或待休息
            self.idle_agvs.discard(agv_id)
            self.need_rest_agvs.discard(agv_id)

    def assign_rest_zones(self, rest_dict: Dict[int, Tuple[int, int]]):
        """
        为空闲 AGV 分配休息区。
        分配完成后，加入待重规划集合，后续由规划器生成去休息区的路径。
        """
        for agv_id, rest_pos in rest_dict.items():
            agv = self._agvs[agv_id]
            agv.assign_rest_zone(rest_pos)
            self.need_rest_agvs.discard(agv_id)
            self.need_replan_agvs.add(agv_id)

    def get_replan_targets(self) -> Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        获取需要重规划的 AGV 的起点和目标点。
        返回格式: {agv_id: (current_pos, target_pos)}
        """
        result = {}

        # 若配置要求每步都重规划，则所有 AGV 都参与；否则只处理待重规划 AGV
        if SimConfig.force_replan_every_step:
            replan_agvs = self.all_agv_ids
        else:
            replan_agvs = set(self.need_replan_agvs)

        for agv_id in replan_agvs:
            agv = self._agvs[agv_id]
            current = agv.grid_pos
            # 若还有任务，则目标为当前任务队列的第一个位置；否则目标为休息区
            if agv.task_queue:
                target = agv.task_queue[0][0]
            else:
                target = agv.rest_target
            result[agv_id] = (current, target)

        return result

    def replan_paths(self, path_dict: Dict[int, List[Tuple[int, int]]]):
        """
        将规划器生成的新路径写回对应 AGV。
        写回后，将 AGV 从待重规划集合中移除。
        """
        for agv_id, path in path_dict.items():
            agv = self._agvs[agv_id]
            agv.set_new_plan(path)
            self.need_replan_agvs.discard(agv_id)

    def set_agv_status(self, agv_id, is_working):
        """设置指定 AGV 的工作状态"""
        agv = self._agvs.get(agv_id)
        if agv:
            agv.is_working = is_working

    def reset_agvs(self):
        """
        重置所有 AGV 到初始状态。
        包括位置、任务、空闲集合、待休息集合、待重规划集合和阻塞计数。
        """
        for agv in self._agvs.values():
            agv.reset()

        all_agv_ids = set(self._agvs.keys())
        self.idle_agvs = all_agv_ids.copy()
        self.need_rest_agvs = all_agv_ids.copy()
        self.need_replan_agvs = all_agv_ids.copy()
        self.block_counts = {agv_id: 0 for agv_id in all_agv_ids}

        global_logger.add_runtime_log("[AGVManager] All AGVs have been reset to initial states.")