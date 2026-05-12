import itertools
from copy import deepcopy
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set
from core.gridmap import GridMap
from core.order import Order
from core.ordermanager import OrderManager
from core.agv import AGVAction
from core.agvmanager import AGVManager
from core.env import Env
from core.fault_manager import FaultManager
from scheduler.base_scheduler import BaseScheduler
from scipy.optimize import linear_sum_assignment
from utils.base_utils import orders_to_tasks


class TAScheduler(BaseScheduler):
    def __init__(
        self, 
        env: Env,
        agv_manager: AGVManager,
        order_manager: OrderManager, 
        map: GridMap,
        fault_manager: FaultManager
    ):
        # 初始化父类，注入环境、AGV 管理器、订单管理器、地图和故障管理器
        super().__init__(env, agv_manager, order_manager, map, fault_manager)

    def compute_manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """计算两个网格坐标之间的曼哈顿距离。"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _select_best_box_for_group(
        self,
        agv_pos: Tuple[int, int],
        order_group: List[Order],
    ) -> Tuple[int, Optional[int]]:
        """
        为某组同 goods_id 的订单选择一个最合适的箱子。

        这里显式枚举所有候选箱子，选总代价最小的那个。
        这样“匈牙利分配用的代价”和“真正执行时的任务路径”就一致了，
        不会再被两次随机选箱子的噪声带偏。
        """
        if not order_group:
            return float("inf"), None

        first_order = order_group[0]
        box_ids = self.map.get_boxes_by_goods(first_order.goods_id)
        if not box_ids:
            return float("inf"), None

        best_cost = float("inf")
        best_box_id: Optional[int] = None

        for box_id in sorted(box_ids):
            box_pos = self.map.get_box_position(box_id)

            total_cost = self.compute_manhattan_distance(agv_pos, box_pos)

            receiver_pos = self.map.get_receiver_position(first_order.receiver_id)
            total_cost += self.compute_manhattan_distance(box_pos, receiver_pos)
            prev_receiver_pos = receiver_pos

            for order in order_group[1:]:
                next_receiver_pos = self.map.get_receiver_position(order.receiver_id)
                total_cost += self.compute_manhattan_distance(prev_receiver_pos, next_receiver_pos)
                prev_receiver_pos = next_receiver_pos

            total_cost += self.compute_manhattan_distance(prev_receiver_pos, box_pos)

            if total_cost < best_cost or (total_cost == best_cost and (best_box_id is None or box_id < best_box_id)):
                best_cost = total_cost
                best_box_id = box_id

        return best_cost, best_box_id

    def compute_task_cost(self, agv_pos: Tuple[int, int], order_group: List[Order]) -> int:
        """
        计算一个 AGV 执行一组连续订单的总代价。
        路径逻辑：
        AGV当前位置 -> 货箱 -> 第一个收货点 -> 后续收货点 ... -> 返回货箱
        """
        cost, _ = self._select_best_box_for_group(agv_pos, order_group)
        return cost

    def build_cost_matrix(
        self,
        idle_agv_ids: List[int],
        grouped_orders: List[List['Order']]
    ) -> Tuple[List[List[int]], List[List[Optional[int]]]]:
        """
        构建代价矩阵：
        cost_matrix[agv_idx][order_group_idx] 表示某个 AGV 执行某组订单的总代价。
        """
        cost_matrix = []
        box_choice_matrix: List[List[Optional[int]]] = []

        # 遍历所有空闲 AGV
        for agv_id in idle_agv_ids:
            agv_pos = self.agv_manager.get_grid_position(agv_id)
            agv_costs = []
            agv_boxes: List[Optional[int]] = []

            # 计算该 AGV 对每个订单组的执行成本
            for order_group in grouped_orders:
                cost, box_id = self._select_best_box_for_group(agv_pos, order_group)
                agv_costs.append(cost)
                agv_boxes.append(box_id)

            cost_matrix.append(agv_costs)
            box_choice_matrix.append(agv_boxes)

        return cost_matrix, box_choice_matrix

    def task_assignment(self, cost_matrix: List[List[int]]) -> Dict[int, int]:
        """
        使用匈牙利算法进行任务分配。

        说明：
        - A = AGV 数量
        - M = 任务组数量

        情况1：M >= A
            直接做最优匹配，给每个 AGV 分配一个任务组。
        情况2：M < A
            用高代价虚拟任务补齐矩阵，避免某些 AGV 被错误分配到真实任务；
            最后只保留分配到真实任务的结果。

        返回：
            {agv_idx: task_idx}
            其中 agv_idx 是 AGV 在 cost_matrix 中的行索引，
            task_idx 是 grouped_orders 中的任务组索引。
        """
        # 如果代价矩阵为空，直接返回空分配
        if not cost_matrix:
            return {}

        A = len(cost_matrix)                     # AGV 数量
        M = len(cost_matrix[0]) if cost_matrix[0] else 0   # 任务数量

        # 如果没有任务，直接返回空分配
        if M == 0:
            return {}

        # 校验矩阵每一行长度是否一致
        for row in cost_matrix:
            if len(row) != M:
                raise ValueError("All rows in cost_matrix must have same number of columns")

        # 如果任务数量不少于 AGV 数量，直接使用匈牙利算法
        if M >= A:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return {int(r): int(c) for r, c in zip(row_ind, col_ind)}

        # 如果任务数量少于 AGV 数量，需要补齐虚拟列
        flat_max = max(max(row) for row in cost_matrix)

        # 构造一个足够大的代价值，确保虚拟任务不会优先于真实任务被选择
        if flat_max <= 0:
            big_cost = 10**9
        else:
            big_cost = int(flat_max * (A + 10)) + 1

        # 给每一行补充 (A - M) 个虚拟任务
        padded = [row + [big_cost] * (A - M) for row in cost_matrix]

        # 在补齐后的方阵上做匈牙利匹配
        row_ind, col_ind = linear_sum_assignment(padded)

        assignment: Dict[int, int] = {}

        # 只保留分配到真实任务列（c < M）的 AGV
        for r, c in zip(row_ind, col_ind):
            if c < M:
                assignment[int(r)] = int(c)

        return assignment

    def assign_tasks(
        self, idle_agv_ids: Set[int], planner
    ) -> Dict[int, List[Tuple[Tuple[int, int], AGVAction, int]]]:
        """
        为所有空闲 AGV 分配任务。

        分配策略：
        1. 先按 AGV 尺寸与订单所需尺寸匹配
        2. 再按 goods_id 对订单分组
        3. 基于代价矩阵使用匈牙利算法完成 AGV 与订单组的最优分配
        4. 将订单转换为实际任务序列
        """
        # 如果全部订单已经完成，或者没有空闲 AGV，则无需分配
        if self.order_manager.is_all_orders_completed() or not idle_agv_ids:
            return {}

        # 按订单所需尺寸分类
        size_to_orders = defaultdict(list)
        for order in self.order_manager.get_unprocessed_orders():
            size_to_orders[order.required_size].append(order)

        # 按 AGV 尺寸分类
        size_to_agvs = defaultdict(list)
        for agv_id in idle_agv_ids:
            agv_size = self.agv_manager.get_agv_size(agv_id)
            size_to_agvs[agv_size].append(agv_id)

        # 最终返回结果：agv_id -> 任务列表
        agv_task_map = {}

        # 遍历每一种尺寸的 AGV
        for size, agv_ids in size_to_agvs.items():
            # 取出该尺寸 AGV 能处理的订单
            valid_orders = size_to_orders.get(size, [])
            if not valid_orders:
                continue

            # 将订单按 goods_id 分组，同类货物组成一个任务组
            goods_to_orders = defaultdict(list)
            for order in valid_orders:
                goods_to_orders[order.goods_id].append(order)

            grouped_orders = list(goods_to_orders.values())

            # 构建 AGV 与订单组之间的代价矩阵
            cost_matrix, box_choice_matrix = self.build_cost_matrix(agv_ids, grouped_orders)

            # 求解最优分配
            assignment = self.task_assignment(cost_matrix)

            # 将“索引级别的分配”转换为“真实 AGV -> 订单组”
            agv_to_orders = defaultdict(list)
            for agv_idx, task_idx in assignment.items():
                agv_id = agv_ids[agv_idx]
                agv_to_orders[agv_id] = grouped_orders[task_idx]

            # 为每台 AGV 生成具体任务
            for agv_id, orders in agv_to_orders.items():
                agv_idx = agv_ids.index(agv_id)
                task_idx = assignment[agv_idx]
                selected_box_id = box_choice_matrix[agv_idx][task_idx]
                if selected_box_id is None:
                    raise ValueError(f"No available box found for goods_id={orders[0].goods_id}")

                # 深拷贝，避免直接修改原始订单对象
                copied_orders = deepcopy(orders)

                # 估价阶段和真正执行阶段使用同一个箱子，减少分配噪声。
                for order in copied_orders:
                    order.box_id = selected_box_id

                # 将订单列表转换成 AGV 可执行的任务序列
                agv_task_map[agv_id] = orders_to_tasks(copied_orders, self.map)

                # 将原始订单标记为“处理中”
                for original_order in orders:
                    self.order_manager.mark_order_as_processing(original_order.order_id, agv_id)

        return agv_task_map
