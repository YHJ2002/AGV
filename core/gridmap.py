from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from config.settings import SimConfig
import json
from utils.logger import global_logger


class GridMap:
    def __init__(self):
        # 读取地图配置文件
        with open(SimConfig.map_file, "r") as f:
            map_data = json.load(f)

        # 地图尺寸
        self.width = map_data["map"]["width"]
        self.height = map_data["map"]["height"]

        # static_grid 含义：
        # -2 = 障碍物
        # -1 = 空闲可通行区域
        # >=0 = 对应 box_id 的货箱区域
        self.static_grid = np.full((self.height, self.width), -1, dtype=int)

        # 货箱相关数据
        self.box_positions: Dict[int, Tuple[int, int]] = {}   # 货箱左上角位置
        self.box_sizes: Dict[int, int] = {}                   # 货箱尺寸
        self.box_to_goods: Dict[int, List[int]] = {}          # 货箱 -> 货物列表
        self.box_status: Dict[int, bool] = {}                 # 货箱是否在原位
        self.box_id_set: Set[int] = set()                     # 所有货箱编号
        self.goods_to_boxes: Dict[int, List[int]] = {}        # 货物 -> 所属货箱
        self.goods_id_set: Set[int] = set()                   # 所有货物编号

        # 解析货箱信息，并写入静态地图
        for box in map_data.get("boxes", []):
            box_id = box["box_id"]
            x, y = box["position"]
            goods_ids = box.get("goods_ids", [])
            size = box.get("size", 1)

            self.box_positions[box_id] = (x, y)
            self.box_sizes[box_id] = size
            self.box_to_goods[box_id] = goods_ids
            self.box_status[box_id] = True
            self.box_id_set.add(box_id)

            # 建立 goods_id -> box_id 的映射
            for gid in goods_ids:
                self.goods_to_boxes.setdefault(gid, []).append(box_id)
                self.goods_id_set.add(gid)

            # 将货箱区域写入静态地图
            for dx in range(size):
                for dy in range(size):
                    self.static_grid[y + dy][x + dx] = box_id

        # 障碍物集合
        self.obstacles: Set[Tuple[int, int]] = set()
        for x, y in map_data.get("obstacles", []):
            self.static_grid[y][x] = -2
            self.obstacles.add((x, y))

        # 接收区相关信息
        self.receiver_zones: Dict[int, Tuple[int, int]] = {}
        self.receiver_id_set: Set[int] = set()
        self.receiver_zones_size: Dict[int, int] = {}
        for r in map_data.get("receivers", []):
            rid = r["receiver_id"]
            pos = tuple(r["position"])
            size = r.get("size", 1)
            self.receiver_zones[rid] = pos
            self.receiver_id_set.add(rid)
            self.receiver_zones_size[rid] = size

        # 等待区相关信息
        self.wait_zones: Dict[int, Tuple[int, int]] = {}
        self.wait_zones_size: Dict[int, int] = {}
        for zone in map_data.get("wait_zones", []):
            zid = zone["wait_zone_id"]
            pos = tuple(zone["position"])
            size = zone.get("size", 1)
            self.wait_zones[zid] = pos
            self.wait_zones_size[zid] = size

        # 动态占用区域
        # 用于记录临时不可用栅格，例如故障撤离路径等
        self.dynamic_occupied: Dict[str, list[Tuple[int, int]]] = {}

    def add_dynamic_occupancy(self, key: str, cells: List[Tuple[int, int]]):
        """注册一组临时占用栅格，例如故障维修路径"""
        self.dynamic_occupied[key] = cells

    def remove_dynamic_occupancy(self, key: str):
        """移除一组临时占用栅格"""
        if key in self.dynamic_occupied:
            del self.dynamic_occupied[key]

    def is_occupied(self, x: int, y: int) -> bool:
        """判断某个栅格是否被动态占用（不包含静态障碍物）"""
        for cells in self.dynamic_occupied.values():
            if (x, y) in cells:
                return True
        return False

    def is_walkable(
        self,
        agv_size: int,
        to_pos: Tuple[int, int],
        from_pos: Tuple[int, int],
        carrying_goods: bool
    ) -> bool:
        """
        判断 AGV 是否可以从 from_pos 移动到 to_pos
        这里考虑了：
        1. 单步移动约束
        2. 地图边界约束
        3. 障碍物约束
        4. 动态占用约束
        5. 货箱区域与载货状态约束
        """
        x_from, y_from = from_pos
        x_to, y_to = to_pos
        dx, dy = x_to - x_from, y_to - y_from

        # 只能上下左右移动一步
        if abs(dx) + abs(dy) != 1:
            return False

        # 检查目标位置是否越界
        if not (0 <= x_to and x_to + agv_size - 1 < self.width and
                0 <= y_to and y_to + agv_size - 1 < self.height):
            return False

        # 检查当前起始位置是否越界
        if not (0 <= x_from and x_from + agv_size - 1 < self.width and
                0 <= y_from and y_from + agv_size - 1 < self.height):
            return False

        # 计算 AGV 前进方向上的“头部边缘”
        if dx == 1:
            head_positions = [(x_from + agv_size - 1, y_from + i) for i in range(agv_size)]
        elif dx == -1:
            head_positions = [(x_from, y_from + i) for i in range(agv_size)]
        elif dy == 1:
            head_positions = [(x_from + i, y_from + agv_size - 1) for i in range(agv_size)]
        else:
            head_positions = [(x_from + i, y_from) for i in range(agv_size)]

        # 下一步前进后将要占用的“新头部边缘”
        next_positions = [(hx + dx, hy + dy) for (hx, hy) in head_positions]

        # 检查头部与目标边缘是否越界
        for (hx, hy) in head_positions + next_positions:
            if not (0 <= hx < self.width and 0 <= hy < self.height):
                return False

        # 读取当前头部边缘和目标边缘对应的地图值
        head_vals = [self.static_grid[hy][hx] for (hx, hy) in head_positions]
        next_vals = [self.static_grid[ny][nx] for (nx, ny) in next_positions]

        # 只要有障碍物，直接不可通行
        if any(v == -2 for v in head_vals + next_vals):
            return False

        # 若存在动态占用，也不能通行
        if self.dynamic_occupied:
            for (hx, hy) in head_positions + next_positions:
                if self.is_occupied(hx, hy):
                    return False

        def classify_group(vals):
            """
            将一组栅格分类为：
            - empty: 全部为空地
            - shelf: 全部为同一个货箱区域
            - mixed: 混合状态，不允许通行
            """
            if all(v == -1 for v in vals):
                return ("empty", None)
            if all(v >= 0 for v in vals):
                first = vals[0]
                if all(v == first for v in vals):
                    return ("shelf", first)
                else:
                    return ("mixed", None)
            return ("mixed", None)

        head_type, head_id = classify_group(head_vals)
        next_type, next_id = classify_group(next_vals)

        # 混合区域不允许移动
        if head_type == "mixed" or next_type == "mixed":
            return False

        # 携带货物时的通行规则更严格
        if carrying_goods:
            if head_type == "empty" and next_type == "empty":
                return True
            if head_type == "empty" and next_type == "shelf":
                # 允许进入已空出的货箱位置
                return not self.box_status.get(next_id, True)
            if head_type == "shelf" and next_type == "empty":
                return True
            if head_type == "shelf" and next_type == "shelf":
                # 只能在同一个货箱区域内移动
                return head_id == next_id
            return False

        # 未载货时通行规则更宽松
        else:
            if head_type == "empty" and next_type == "empty":
                return True
            if head_type == "empty" and next_type == "shelf":
                return True
            if head_type == "shelf" and next_type == "empty":
                return True
            if head_type == "shelf" and next_type == "shelf":
                return True
            return False

    def get_walkable_neighbors(
        self,
        agv_size: int,
        pos: Tuple[int, int],
        carrying_goods: bool
    ) -> List[Tuple[int, int]]:
        """
        获取 AGV 从当前位置 pos 出发，所有可行的相邻栅格
        """
        x, y = pos
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not (0 <= nx and nx + agv_size - 1 < self.width and
                    0 <= ny and ny + agv_size - 1 < self.height):
                continue
            if self.is_walkable(agv_size, (nx, ny), (x, y), carrying_goods):
                neighbors.append((nx, ny))

        return neighbors

    def pick_box_at(self, pos: Tuple[int, int]) -> Optional[int]:
        """
        在指定位置取货箱。
        若该位置确实存在货箱且当前仍在原位，则将其状态改为“已取走”并返回 box_id
        """
        x, y = pos
        box_id = self.static_grid[y][x]
        expected_pos = self.box_positions.get(box_id)

        # 必须与货箱定义位置一致
        if expected_pos != pos:
            return None

        # 若货箱还在原位，则取走
        if self.box_status[box_id]:
            self.box_status[box_id] = False
            return box_id

        return None

    def place_box_at(self, pos: Tuple[int, int], box_id: int) -> bool:
        """
        在指定位置放回货箱。
        只有目标位置与该货箱定义位置一致，且当前该货箱不在原位时，才允许放回
        """
        x, y = pos
        expected_pos = self.box_positions.get(box_id)

        if expected_pos != pos:
            return False

        if not self.box_status.get(box_id, True):
            self.box_status[box_id] = True
            return True

        return False

    def get_all_box_status(self) -> Dict[int, bool]:
        """返回所有货箱当前是否在原位"""
        return dict(self.box_status)

    def get_box_position(self, box_id: int) -> Optional[Tuple[int, int]]:
        """返回指定货箱的位置"""
        return self.box_positions.get(box_id)

    def get_goods_by_box(self, box_id: int) -> List[int]:
        """根据货箱编号获取其中包含的货物编号列表"""
        return self.box_to_goods.get(box_id, [])

    def get_boxes_by_goods(self, goods_id: int) -> List[int]:
        """根据货物编号获取可能所在的货箱编号列表"""
        return self.goods_to_boxes.get(goods_id, [])

    def get_all_goods_ids(self) -> Set[int]:
        """返回所有货物编号集合"""
        return self.goods_id_set

    def get_receiver_position(self, receiver_id: int) -> Optional[Tuple[int, int]]:
        """根据接收区编号获取位置"""
        return self.receiver_zones.get(receiver_id)

    def get_all_receiver_zone_ids(self) -> Set[int]:
        """返回所有接收区编号集合"""
        return self.receiver_id_set

    def get_wait_zone_position(self, zone_id: int) -> Optional[Tuple[int, int]]:
        """根据等待区编号获取位置"""
        return self.wait_zones.get(zone_id)

    def reset_map(self):
        """
        重置地图动态状态：
        1. 所有货箱恢复到“在原位”
        2. 清空动态占用信息
        """
        for box_id in self.box_status:
            self.box_status[box_id] = True
        self.dynamic_occupied.clear()
        global_logger.add_runtime_log("[GridMap] Map has been reset.")