import json
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from config.settings import SimConfig
from core.gridmap import GridMap
from core.order import Order
from config.settings import SimConfig


class OrderGenerationStrategy(ABC):
    """订单生成策略抽象基类。"""

    def __init__(self):
        # 使用固定随机种子初始化随机数生成器，保证实验可复现
        self.rng = random.Random(SimConfig.order_seed)

        # 按尺寸预处理地图中的所有货箱
        # 格式：{size: [box_dict, ...]}
        self._all_boxes_by_size: Dict[int, List[dict]] = self._prepare_boxes_by_size()

        # 按尺寸预处理地图中的所有接收区
        # 格式：{size: [receiver_dict, ...]}
        self._all_receivers_by_size: Dict[int, List[dict]] = self._prepare_receivers_by_size()

    @abstractmethod
    def update(self, current_step: int) -> List[Order]:
        """
        每个仿真步调用一次。
        子类需要实现该方法，用于生成当前步新增的订单列表。
        """
        pass

    def _prepare_boxes_by_size(self) -> Dict[int, List[dict]]:
        """
        从地图 JSON 文件中读取所有货箱，并按尺寸分组。
        返回格式：{size: [box_dict, ...]}
        """
        map_path = SimConfig.map_file
        with open(map_path, "r", encoding="utf-8") as f:
            map_data = json.load(f)

        boxes = map_data.get("boxes", [])
        boxes_by_size: Dict[int, List[dict]] = {}

        for box in boxes:
            size = box.get("size", 1)
            boxes_by_size.setdefault(size, []).append(box)

        return boxes_by_size

    def _prepare_receivers_by_size(self) -> Dict[int, List[dict]]:
        """
        从地图 JSON 文件中读取所有接收区，并按尺寸分组。
        返回格式：{size: [receiver_dict, ...]}
        """
        map_path = SimConfig.map_file
        with open(map_path, "r", encoding="utf-8") as f:
            map_data = json.load(f)

        receivers = map_data.get("receivers", [])
        receivers_by_size: Dict[int, List[dict]] = {}

        for recv in receivers:
            size = recv.get("size", 1)
            receivers_by_size.setdefault(size, []).append(recv)

        return receivers_by_size

    def _generate_single_order(self, size: int) -> Optional[Order]:
        """
        生成一个指定尺寸的订单。
        若当前尺寸下没有可用货箱或接收区，则返回 None。
        """
        boxes = self._all_boxes_by_size.get(size, [])
        receivers = self._all_receivers_by_size.get(size, [])

        if not boxes or not receivers:
            return None

        # 随机选择一个该尺寸的货箱
        box = self.rng.choice(boxes)
        goods_ids = box.get("goods_ids", [])

        # 如果该货箱没有货物，则无法生成订单
        if not goods_ids:
            return None

        # 从货箱中随机选一个货物
        goods_id = self.rng.choice(goods_ids)

        # 随机选一个同尺寸接收区
        receiver = self.rng.choice(receivers)

        # 构造订单对象
        return Order(
            order_id=-1,  # 真正编号在 OrderManager 中赋值
            goods_id=goods_id,
            receiver_id=receiver["receiver_id"],
            required_size=size
        )

    def _generate_batch_orders(self, batch_size: int) -> tuple[List[Order], int]:
        """
        按 batch_size 批量生成订单，并根据 size2_ratio 划分尺寸比例。
        返回生成后的订单列表。
        """
        # 按比例划分 size=2 和 size=1 的订单数量
        num_size2 = int(batch_size * SimConfig.size2_ratio)
        num_size1 = batch_size - num_size2

        new_orders: List[Order] = []

        # 分别生成两种尺寸的订单
        for size, count in [(1, num_size1), (2, num_size2)]:
            for _ in range(count):
                order = self._generate_single_order(size)
                if order:
                    new_orders.append(order)

        return new_orders