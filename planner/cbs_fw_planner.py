from abc import ABC
from typing import Dict, Tuple, List
import heapq
from collections import defaultdict
from planner.base_planner import BasePlanner
from core.env import Env
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from core.fault_manager import FaultManager
from core.agvmanager import AGVManager

# CBS 搜索允许扩展的最大节点数，用于控制搜索开销
MAX_CBS_NODES = 800


class FixedWindowCBSPlanner(BasePlanner):
    def __init__(
        self,
        env: Env,
        agv_manager: AGVManager,
        order_manager: OrderManager,
        map: GridMap,
        fault_manager: FaultManager
    ):
        # 调用父类初始化
        super().__init__(env, agv_manager, order_manager, map, fault_manager)

        # 固定窗口大小，只在有限时间窗口内做 CBS 规划
        self.window_size = 10

    def plan(
        self,
        targets: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
        scheduler
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        固定窗口 CBS 集中式路径规划入口。
        输入：
            targets = {agv_id: (start_pos, target_pos)}
        输出：
            planned_paths = {agv_id: path}
        返回路径不包含起点，path[0] 表示下一步位置。
        """
        if not targets:
            return {}

        # 从环境中读取当前 AGV 载货状态、动作队列、当前位置和尺寸
        env_info = self.env.get_env_info()
        carrying_status = env_info["carrying_status"]
        action_queues = env_info["action_queues"]
        current_pos = env_info["current_grid_pos"]
        self.agv_sizes = env_info["agv_sizes"]

        # 把“当前位置 + 后续动作队列”拼成完整路径
        full_paths = {
            agv_id: [current_pos[agv_id]] + path
            for agv_id, path in action_queues.items()
        }

        # 不参与本轮重规划的 AGV 视为固定路径智能体
        fixed_agents = {
            agv_id: path for agv_id, path in full_paths.items() if agv_id not in targets
        }

        # 在固定窗口内执行 CBS
        window_paths = self._cbs_window(targets, carrying_status, fixed_agents)

        # 返回给主流程的路径不包含起点
        planned_paths = {}
        for agv_id, path in window_paths.items():
            if agv_id in targets:
                planned_paths[agv_id] = path[1:] if len(path) > 1 else []

        return planned_paths

    def _cbs_window(
        self,
        targets: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
        carrying_status: Dict[int, bool],
        fixed_agents: Dict[int, List[Tuple[int, int]]],
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        固定窗口 CBS 主过程。
        只对本轮需要规划的 AGV 执行约束搜索；
        未参与重规划的 AGV 作为固定路径约束参与冲突检测。
        """
        planning_agents = set(targets.keys())

        # 根节点：无约束、空路径、总代价为 0
        root = {'constraints': [], 'paths': {}, 'cost': 0}

        # 先为所有待规划 AGV 生成无约束的初始路径
        for agv_id, (start, goal) in targets.items():
            path = self._a_star_with_constraints(
                agv_id, start, goal, carrying_status[agv_id], [], goal
            )

            # 若规划失败，则退化为原地等待
            if path is None:
                path = [start]

            # 若起点就是终点，则尝试补一个驻留动作
            if len(path) == 1 and start == goal:
                if self._is_vertex_free(agv_id, start, 1, root['constraints']):
                    path = [start, start]

            root['paths'][agv_id] = path
            root['cost'] += len(path) - 1

        # 固定 AGV 只保留窗口内路径
        for agv_id, path in fixed_agents.items():
            root['paths'][agv_id] = path[: self.window_size + 1]

        # CBS 开放列表：按总代价排序
        open_list = []
        heapq.heappush(open_list, (root['cost'], 0, root))

        node_id = 1
        expanded_nodes = 0

        while open_list:
            expanded_nodes += 1

            # 超过节点上限则停止 CBS 搜索
            if expanded_nodes >= MAX_CBS_NODES:
                break

            cost, _, node = heapq.heappop(open_list)

            # 检测当前路径集合中是否存在冲突
            conflict = self._detect_conflict(
                node['paths'],
                planning_agents,
                set(fixed_agents.keys())
            )

            # 无冲突则直接返回
            if conflict is None:
                return node['paths']

            # 冲突格式：(a1, a2, time, loc)
            a1, a2, time, loc = conflict

            # 分别为两个冲突智能体构造约束，生成子节点
            for agent, constraint in zip((a1, a2), self._build_constraints(a1, a2, time, loc)):
                # 固定路径智能体不重规划
                if agent not in planning_agents:
                    continue

                child = {
                    'constraints': node['constraints'] + [constraint],
                    'paths': dict(node['paths']),
                    'cost': 0
                }

                start, goal = targets[agent]

                # 对该 AGV 在新约束下重新规划
                new_path = self._a_star_with_constraints(
                    agent, start, goal, carrying_status[agent], child['constraints'], goal
                )

                if new_path is None:
                    continue

                # 起点即终点时尝试补驻留动作
                if len(new_path) == 1 and start == goal:
                    if self._is_vertex_free(agent, start, 1, child['constraints']):
                        new_path = [start, start]

                child['paths'][agent] = new_path

                # 更新总代价
                child['cost'] = sum(len(p) - 1 for p in child['paths'].values() if p)

                heapq.heappush(open_list, (child['cost'], node_id, child))
                node_id += 1

        # 若 CBS 失败，则退化为逐车独立 A* 结果
        fallback = {}
        for agv_id, (start, goal) in targets.items():
            path = self._a_star_with_constraints(
                agv_id, start, goal, carrying_status[agv_id], [], goal
            )

            if path is None:
                path = [start]

            if len(path) == 1 and start == goal:
                if self._is_vertex_free(agv_id, start, 1, []):
                    path = [start, start]

            fallback[agv_id] = path

        return fallback

    def _a_star_with_constraints(
        self,
        agv_id: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        carrying: bool,
        constraints: List[Dict],
        true_goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        带约束的单车 A* 搜索。
        约束分为：
        1. 顶点约束：某时刻不能占用某栅格
        2. 边约束：某时刻不能走某条边
        """
        vertex_cons = defaultdict(set)
        edge_cons = defaultdict(set)

        # 预处理当前 AGV 相关约束
        for c in constraints:
            if c['agent'] != agv_id:
                continue
            t = c['time']
            if len(c['loc']) == 1:
                vertex_cons[t].add(tuple(c['loc'][0]))
            else:
                edge_cons[t].add((tuple(c['loc'][0]), tuple(c['loc'][1])))

        # 起点若一开始就违反顶点约束，则无解
        if self._occupied_cells(agv_id, start) & vertex_cons.get(0, set()):
            return None

        start_state = (start, 0)

        # 开放表元素：(f, g, (pos, t), parent)
        open_heap = []
        gscore = {start_state: 0}
        heapq.heappush(open_heap, (self._h(start, goal), 0, start_state, None))

        parents = {}
        closed = set()
        INF = 10**9

        while open_heap:
            f, g, (pos, t), parent = heapq.heappop(open_heap)

            if g > gscore.get((pos, t), INF):
                continue

            # 当前状态违反顶点约束则跳过
            if self._occupied_cells(agv_id, pos) & vertex_cons.get(t, set()):
                continue

            parents[(pos, t)] = parent
            closed.add((pos, t))

            # 到达目标或达到窗口上限，则返回路径
            if pos == goal or t >= self.window_size:
                path = self._reconstruct_path((pos, t), parents)
                return path

            # 允许等待动作：原地停留一格时间
            if not (self._occupied_cells(agv_id, pos) & vertex_cons.get(t + 1, set())):
                succ = (pos, t + 1)
                ng = g + 1
                if gscore.get(succ, INF) > ng:
                    gscore[succ] = ng
                    heapq.heappush(open_heap, (ng + self._h(pos, goal), ng, succ, (pos, t)))

            # 扩展邻居动作
            for nb in self.env.get_walkable_neighbors(agv_id, pos, carrying):
                # 下一个时刻不能违反顶点约束
                if self._occupied_cells(agv_id, nb) & vertex_cons.get(t + 1, set()):
                    continue

                # 当前时刻不能违反边约束
                if (pos, nb) in edge_cons.get(t, set()):
                    continue

                succ = (nb, t + 1)
                ng = g + 1
                if gscore.get(succ, INF) > ng:
                    gscore[succ] = ng
                    heapq.heappush(open_heap, (ng + self._h(nb, goal), ng, succ, (pos, t)))

        return None

    def _h(self, a, b):
        """曼哈顿距离启发函数"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(self, state, parents):
        """根据 parent 指针回溯重建路径"""
        path = []
        cur = state
        while cur is not None:
            pos, t = cur
            path.append(pos)
            cur = parents.get(cur)
        path.reverse()
        return path

    def _detect_conflict(self, paths: Dict[int, List[Tuple[int, int]]], planning_agents: set, fixed_agents: set):
        """
        检测路径集合中的冲突。
        返回：
            (a1, a2, time, loc)
        其中 loc 为：
            - [cell]      表示顶点冲突
            - [u, v]      表示边冲突（交换位置）
        """
        agents = set(paths.keys())
        max_len = max((len(paths[aid]) for aid in agents), default=0)

        for t in range(max_len):
            occupied_by = defaultdict(list)

            # 检查顶点冲突：同一时刻占用同一栅格
            for agv_id, path in paths.items():
                if t >= len(path):
                    continue
                pos = path[t]
                for cell in self._occupied_cells(agv_id, pos):
                    occupied_by[cell].append(agv_id)

            for cell, agvs in occupied_by.items():
                if len(agvs) > 1:
                    conflicting_agvs = [a for a in agvs if a in planning_agents]
                    fixed_conflicting = [a for a in agvs if a in fixed_agents]

                    if len(conflicting_agvs) >= 2:
                        a1, a2 = conflicting_agvs[0], conflicting_agvs[1]
                    elif len(conflicting_agvs) == 1 and fixed_conflicting:
                        a1 = conflicting_agvs[0]
                        a2 = fixed_conflicting[0]
                    else:
                        continue

                    return a1, a2, t, [cell]

            # 检查边冲突：两车交换位置
            if t > 0:
                ids = list(agents)
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        ai, aj = ids[i], ids[j]
                        if t >= len(paths[ai]) or t >= len(paths[aj]):
                            continue

                        prev_i, cur_i = paths[ai][t - 1], paths[ai][t]
                        prev_j, cur_j = paths[aj][t - 1], paths[aj][t]

                        if prev_i == cur_j and prev_j == cur_i and cur_i != cur_j:
                            if ai in fixed_agents and aj in fixed_agents:
                                continue
                            return ai, aj, t - 1, [prev_i, cur_i]

        return None

    def _build_constraints(self, a1, a2, time, loc):
        """
        根据冲突信息为两个智能体分别构造约束。
        顶点冲突：双方都不能在该时刻占用该点
        边冲突：双方都不能在该时刻走冲突边
        """
        if len(loc) == 1:
            c1 = {'agent': a1, 'loc': [loc[0]], 'time': time}
            c2 = {'agent': a2, 'loc': [loc[0]], 'time': time}
        else:
            u, v = loc
            c1 = {'agent': a1, 'loc': [u, v], 'time': time}
            c2 = {'agent': a2, 'loc': [v, u], 'time': time}
        return c1, c2

    def _is_vertex_free(self, agv_id: int, pos: Tuple[int, int], time: int, constraints: List[Dict]) -> bool:
        """
        判断某 AGV 在指定时刻能否占用 pos。
        实际判断的是该 AGV 占用的所有栅格是否都不在顶点约束中。
        """
        forbidden_cells = set()
        for c in constraints:
            if c.get('agent') != agv_id:
                continue
            if c.get('time') == time and len(c.get('loc', [])) == 1:
                forbidden_cells.add(tuple(c['loc'][0]))
        return self._occupied_cells(agv_id, pos) & forbidden_cells == set()

    def _occupied_cells(self, agv_id, pos: Tuple[int, int]) -> set:
        """
        根据 AGV 左上角位置和尺寸，返回该 AGV 占用的所有栅格。
        用于支持异构 AGV 冲突检测。
        """
        size = self.agv_sizes.get(agv_id, 1)
        x, y = pos
        return {(x + dx, y + dy) for dx in range(size) for dy in range(size)}