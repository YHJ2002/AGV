from typing import Dict, Tuple, Set, List
from core.gridmap import GridMap
from core.agvmanager import AGVManager
from core.agv import StepInfo
from core.ordermanager import OrderManager

# 浮点数比较误差，用于判断AGV是否正好位于网格中心
epsilon = 1e-4


class Env:
    def __init__(self, agv_manager: AGVManager, map_inst: GridMap, order_manager: OrderManager):
        # 管理所有AGV状态的对象
        self.agv_manager = agv_manager
        # 地图对象，负责可通行性判断等
        self.map = map_inst
        # 订单管理器，reset时会一起重置
        self.order_manager = order_manager

    def get_env_info(self):
        # 获取静态地图
        static_grid = self.map.static_grid
        # 获取每台AGV是否载货
        carrying_status = self.agv_manager.get_carrying_status()
        # 获取每台AGV的动作队列
        action_queues = self.agv_manager.get_all_action_queues()
        # 获取每台AGV当前所在网格坐标
        current_grid_pos = self.agv_manager.get_all_current_pos()
        # 获取每台AGV尺寸
        agv_sizes = self.agv_manager.agv_sizes

        # 将环境信息打包返回，供规划器等模块使用
        return {
            'static_grid': static_grid,
            'carrying_status': carrying_status,
            'action_queues': action_queues,
            'current_grid_pos': current_grid_pos,
            'agv_sizes': agv_sizes
        }
    
    def get_walkable_neighbors(self, agv_id: int, pos: Tuple[int, int], carrying_goods: bool) -> List[Tuple[int, int]]:
        # 返回指定AGV在当前位置可到达的相邻网格
        return self.map.get_walkable_neighbors(self.agv_manager.get_agv_size(agv_id), pos, carrying_goods)
    
    def is_walkable(self, agv_id: int, to_pos: Tuple[int, int], from_pos: Tuple[int, int], carrying_goods: bool) -> bool:
        # 判断指定AGV从from_pos移动到to_pos是否合法、可通行
        return self.map.is_walkable(self.agv_manager.get_agv_size(agv_id), to_pos, from_pos, carrying_goods)

    def step(self) -> Dict[int, StepInfo]:
        # 先做冲突消解，得到最终下一步位置和被阻塞的AGV
        next_positions, block_agvs = self.resolve_conflicts()
        # 让所有AGV按最终位置执行一步移动
        step_info_dict = self.agv_manager.step_all(next_positions)
        # 被阻塞的AGV标记为碰撞/冲突状态
        for agv_id in block_agvs:
            step_info_dict[agv_id] = StepInfo.COLLISION
        return step_info_dict

    def resolve_conflicts(self) -> Tuple[Dict[int, Tuple[int, int]], Set[int]]:
        # 当前所有AGV所在网格
        current_pos = self.agv_manager.get_all_current_pos()
        # 当前所有AGV计划前往的下一网格
        next_pos = self.agv_manager.get_all_next_pos()
        # 当前所有AGV真实坐标（可能处于格子中间）
        real_pos = self.agv_manager.get_all_real_positions()
        # 当前所有AGV是否载货
        carrying_status = self.agv_manager.get_carrying_status()

        # 最终执行的下一位置，初始先拷贝计划位置
        final_next_pos: Dict[int, Tuple[int, int]] = dict(next_pos)
        # 被阻塞的AGV集合
        block_agvs: Set[int] = set()

        # 检查非法移动：一次只能走一个曼哈顿距离单位
        for agv_id, tgt in next_pos.items():
            cur = current_pos[agv_id]
            dx = abs(tgt[0] - cur[0])
            dy = abs(tgt[1] - cur[1])
            if dx + dy > 1:
                print(f"[Warning] AGV {agv_id} invalid move {cur} -> {tgt}, forced to stay.")
                # 非法移动则强制原地不动
                next_pos[agv_id] = cur

        # 根据真实位置把AGV分成“在格子中心”和“不在格子中心”两类
        in_center, not_in_center = self.classify_by_grid_center(real_pos)

        # 顶点冲突表：某个格子被哪些AGV占用
        vertex_conflict_dict: Dict[Tuple[int, int], Set[int]] = dict()

        # 先处理不在格子中心的AGV
        # 这些AGV正在移动过程中，占用情况是确定的
        for agv_id in not_in_center:
            cur = current_pos[agv_id]
            tgt = final_next_pos[agv_id]
            # 计算该AGV本时间步将占用的所有网格
            occ = self._get_next_occupied_positions(agv_id, cur, tgt)

            for pos in occ:
                if pos not in vertex_conflict_dict:
                    vertex_conflict_dict[pos] = set()
                # 如果已有其他AGV占用，说明出现静态阶段冲突，直接报错
                if vertex_conflict_dict[pos]:
                    print("current_pos:", current_pos)
                    print("next_pos:", next_pos)
                    print("real_pos:", real_pos)
                    print("conflict at:", pos, "by agv:", agv_id, "and agv(s):", vertex_conflict_dict[pos])
                    raise ValueError(f"Conflict in static phase for AGV {agv_id} at {pos}")
                vertex_conflict_dict[pos].add(agv_id)

        # 处于格子中心的AGV，先默认保持原地不动
        for agv_id in in_center:
            final_next_pos[agv_id] = current_pos[agv_id]

        # 迭代地尝试放行更多AGV，直到结果不再变化
        while True:
            changed = False

            # 当前顶点占用表副本
            cur_vertex_dict: Dict[Tuple[int, int], Set[int]] = {
                k: set(v) for k, v in vertex_conflict_dict.items()
            }
            # 边冲突集合：用于检测两个AGV交换位置
            edge_conflict_set: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

            # 先把已经确定的最终位置写入占用表
            for agv_id in in_center:
                cur = current_pos[agv_id]
                tgt = final_next_pos[agv_id]
                occ = self._get_next_occupied_positions(agv_id, cur, tgt)
                for pos in occ:
                    if pos not in cur_vertex_dict:
                        cur_vertex_dict[pos] = set()
                    cur_vertex_dict[pos].add(agv_id)
                # 如果AGV真的移动了，则记录边
                if cur != tgt:
                    edge_conflict_set.add((cur, tgt))

            # 逐个尝试让处于中心的AGV移动
            for agv_id in in_center:
                cur = current_pos[agv_id]
                tgt = next_pos[agv_id]
                carrying = carrying_status.get(agv_id, False)

                # 如果目标位置就是当前位置，跳过
                if tgt == cur:
                    continue

                # 判断地图上是否可走
                walkable = self.map.is_walkable(self.agv_manager.get_agv_size(agv_id), tgt, cur, carrying)
                # 计算移动过程中占用的格子
                occ = self._get_next_occupied_positions(agv_id, cur, tgt)

                # 是否与其他AGV发生顶点冲突
                has_vertex_conflict = any(
                    (cell in cur_vertex_dict and len(cur_vertex_dict[cell] - {agv_id}) > 0)
                    for cell in occ
                )
                # 是否发生边冲突（交换位置）
                has_edge_conflict = (tgt, cur) in edge_conflict_set

                # 可走、无顶点冲突、无边冲突，才允许移动
                if walkable and not has_vertex_conflict and not has_edge_conflict:
                    if final_next_pos[agv_id] != tgt:
                        final_next_pos[agv_id] = tgt
                        changed = True
                    # 更新顶点占用表
                    for pos in occ:
                        if pos not in cur_vertex_dict:
                            cur_vertex_dict[pos] = set()
                        cur_vertex_dict[pos].add(agv_id)
                    # 记录边占用
                    edge_conflict_set.add((cur, tgt))
                else:
                    # 否则该AGV保持原地
                    final_next_pos[agv_id] = cur
                    edge_conflict_set.add((cur, cur))

            # 如果这一轮没有变化，说明冲突消解结果稳定，结束迭代
            if not changed:
                for agv_id in in_center:
                    # 最终位置和原计划位置不同，说明被阻塞
                    if final_next_pos[agv_id] != next_pos[agv_id]:
                        self.agv_manager.increment_block_count(agv_id)
                        block_agvs.add(agv_id)
                break

        return final_next_pos, block_agvs

    def _get_next_occupied_positions(
        self, agv_id: int, cur: Tuple[int, int], tgt: Tuple[int, int]
    ) -> Set[Tuple[int, int]]:
        """返回AGV从cur移动到tgt过程中占用的所有网格（考虑尺寸）"""
        size = self.agv_manager.get_agv_size(agv_id)

        # 计算AGV在某个网格位置时占用的所有格子
        def footprint(pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
            x, y = pos
            return {(x + dx, y + dy) for dx in range(size) for dy in range(size)}

        # 如果没有移动，直接返回当前位置占用区域
        if cur == tgt:
            return footprint(cur)

        # 获取真实坐标和速度，用于判断本时间步会占用哪些区域
        real_pos = self.agv_manager.get_real_position(agv_id)
        speed = self.agv_manager.get_agv_speed(agv_id)
        time_step = 1
        offset = speed * time_step
        x, y = real_pos
        dx = tgt[0] - cur[0]
        dy = tgt[1] - cur[1]

        cur_fp = footprint(cur)   # 当前格占用区域
        tgt_fp = footprint(tgt)   # 目标格占用区域

        occupied: Set[Tuple[int, int]] = set()

        # 横向移动
        if dx != 0:
            target_x = tgt[0] + 0.5
            # 如果本步已经足够接近目标中心，则只占用目标区域
            if abs(target_x - x) <= offset + epsilon:
                occupied |= tgt_fp
            else:
                # 否则认为当前区域和目标区域都可能占用
                occupied |= (cur_fp | tgt_fp)

        # 纵向移动
        elif dy != 0:
            target_y = tgt[1] + 0.5
            if abs(target_y - y) <= offset + epsilon:
                occupied |= tgt_fp
            else:
                occupied |= (cur_fp | tgt_fp)

        # 不移动
        else:
            occupied |= cur_fp

        return occupied

    def classify_by_grid_center(self, real_positions: Dict[int, Tuple[float, float]]) -> Tuple[Set[int], Set[int]]:
        """将AGV分为两类：位于格子中心 / 不位于格子中心"""
        in_center = set()
        not_in_center = set()

        for agv_id, (x, y) in real_positions.items():
            # 若x和y的小数部分都接近0.5，则认为AGV位于格子中心
            if abs(x % 1 - 0.5) < epsilon and abs(y % 1 - 0.5) < epsilon:
                in_center.add(agv_id)
            else:
                not_in_center.add(agv_id)

        return in_center, not_in_center

    def reset(self):
        # 重置AGV状态
        self.agv_manager.reset_agvs()
        # 重置地图状态
        self.map.reset_map()
        # 重置订单状态
        self.order_manager.reset_order()