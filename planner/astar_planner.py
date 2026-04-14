import heapq
from typing import Dict, Tuple, List, Set
from collections import defaultdict
from core.env import Env
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from core.fault_manager import FaultManager
from core.agvmanager import AGVManager
from planner.base_planner import BasePlanner

# A* 搜索中允许扩展的最大节点数，用于限制搜索开销
MAX_ASTAR_NODES = 800


class AStarPlanner(BasePlanner):
    def __init__(
        self,
        env: Env,
        agv_manager: AGVManager,
        order_manager: OrderManager,
        map: GridMap,
        fault_manager: FaultManager
    ):
        # 调用父类构造函数，保存环境、AGV 管理器、订单管理器、地图和故障管理器
        super().__init__(env, agv_manager, order_manager, map, fault_manager)

        # 路径规划的最大时间深度，超过后停止搜索
        self.max_time = 100

        # 从环境中获取 AGV 尺寸信息，支持异构 AGV 占用判断
        env_info = self.env.get_env_info()
        self.agv_sizes = env_info['agv_sizes']

    def plan(
        self,
        targets: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
        scheduler
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        基于预约表的集中式 A* 路径规划。
        输入：
            targets = {agv_id: (start_pos, target_pos)}
        输出：
            {agv_id: path}
        返回路径不包含起点，path[0] 即下一步位置。
        """
        if not targets:
            return {}

        # 从环境中读取当前已有路径和载货状态
        env_info = self.env.get_env_info()
        current_paths = env_info['action_queues']
        carrying_status = env_info['carrying_status']

        # 根据当前已有路径构建预约表
        reservation_table = self._build_reservation_table(current_paths)

        paths = {}

        # 逐个 AGV 规划路径
        for agv_id, (start, goal) in targets.items():
            carrying = carrying_status.get(agv_id, False)

            path = self._a_star_with_reservation(
                agv_id, start, goal, carrying, reservation_table
            )

            if path:
                # 返回结果中去掉起点，只保留后续动作路径
                paths[agv_id] = path[1:] if len(path) > 1 else []

                # 将新路径写入预约表，供后续 AGV 避让
                self._add_to_reservation_table(agv_id, reservation_table, path)
            else:
                # 若搜索失败，则退化为原地等待
                paths[agv_id] = [start]

        return paths

    def _build_reservation_table(
        self,
        current_paths: Dict[int, List[Tuple[int, int]]],
    ) -> Dict[int, Set[Tuple[int, int]]]:
        """
        构建预约表：
        time_step -> occupied_cells
        用于记录每个时间步哪些栅格已被其他 AGV 占用。
        """
        table = defaultdict(set)

        for agv_id, path in current_paths.items():
            for t, pos in enumerate(path):
                occupied = self._get_occupied_cells(agv_id, pos)
                for cell in occupied:
                    table[t].add(cell)

        return table

    def _add_to_reservation_table(
        self,
        agv_id: int,
        table: Dict[int, Set[Tuple[int, int]]],
        path: List[Tuple[int, int]]
    ):
        """
        将新生成的路径加入预约表。
        这样后续 AGV 规划时会自动避开该路径。
        """
        for t, pos in enumerate(path):
            occupied = self._get_occupied_cells(agv_id, pos)
            for cell in occupied:
                table[t].add(cell)

    def _a_star_with_reservation(
        self,
        agv_id: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        carrying: bool,
        reservation_table: Dict[int, Set[Tuple[int, int]]]
    ) -> List[Tuple[int, int]]:
        """
        带预约表约束的 A* 搜索。
        在搜索过程中同时避免：
        1. 顶点冲突（同一时刻占用同一栅格）
        2. 边冲突（两车交换位置）
        """
        open_set = []

        # 优先队列元素：(f, g, current_pos, path)
        heapq.heappush(open_set, (0 + self._heuristic(start, goal), 0, start, [start]))

        # closed_set 中保存 (位置, 时间步)
        closed_set = set()

        expanded_nodes = 0

        while open_set:
            expanded_nodes += 1

            # 若扩展节点过多则停止搜索
            if expanded_nodes >= MAX_ASTAR_NODES:
                break

            f, g, current, path = heapq.heappop(open_set)

            if (current, g) in closed_set:
                continue
            closed_set.add((current, g))

            # 若到达目标点，且未来两个时间步仍安全，则返回路径
            if (
                current == goal
                and self._is_free(agv_id, current, g + 1, reservation_table)
                and self._is_free(agv_id, current, g + 2, reservation_table)
            ):
                return path + [goal] * 2

            # 扩展所有可通行邻居
            for neighbor in self.env.get_walkable_neighbors(agv_id, current, carrying):
                # 检查该时刻目标位置是否已被预约
                if not self._is_free(agv_id, neighbor, g + 1, reservation_table):
                    continue

                # 检查是否产生交换位置冲突
                if self._is_edge_conflict(agv_id, current, neighbor, g + 1, reservation_table):
                    continue

                new_path = path + [neighbor]

                heapq.heappush(
                    open_set,
                    (g + 1 + self._heuristic(neighbor, goal), g + 1, neighbor, new_path)
                )

            # 若规划时间步过深，则停止搜索
            if g > self.max_time:
                break

        # 搜索失败时返回 None
        return None

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        A* 启发函数：曼哈顿距离
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_free(
        self,
        agv_id: int,
        pos: Tuple[int, int],
        t: int,
        reservation_table: Dict[int, Set[Tuple[int, int]]]
    ) -> bool:
        """
        判断在时间步 t 时，
        指定 AGV 若位于 top-left = pos，其占用的所有栅格是否都空闲。
        """
        for cell in self._get_occupied_cells(agv_id, pos):
            if cell in reservation_table.get(t, set()):
                return False
        return True

    def _is_edge_conflict(
        self,
        agv_id: int,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        t: int,
        reservation_table: Dict[int, Set[Tuple[int, int]]]
    ) -> bool:
        """
        检查边冲突/交换位置冲突。
        若本车从 from_pos -> to_pos，
        同时有其他车在 t-1 -> t 走了相反方向，则视为边冲突。
        """
        occ_from_t = self._get_occupied_cells(agv_id, from_pos)
        occ_to_tminus1 = self._get_occupied_cells(agv_id, to_pos)

        conflict_now = any(cell in reservation_table.get(t, set()) for cell in occ_from_t)
        conflict_prev = any(cell in reservation_table.get(t - 1, set()) for cell in occ_to_tminus1)

        return conflict_now and conflict_prev

    def _get_occupied_cells(self, agv_id: int, top_left: Tuple[int, int]) -> set[Tuple[int, int]]:
        """
        根据 AGV 左上角位置和尺寸，返回该 AGV 实际占用的所有栅格。
        用于支持异构 AGV 的冲突检测。
        """
        size = self.agv_sizes.get(agv_id, 1)
        x, y = top_left
        return {(x + dx, y + dy) for dx in range(size) for dy in range(size)}