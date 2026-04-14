from typing import List, Dict

from core.order import Order
from config.settings import SimConfig, ContinuousParetoConfig
from order_strategies.order_generation_strategy import OrderGenerationStrategy


class ContinuousParetoStrategy(OrderGenerationStrategy):
    """
    Pareto 分布订单生成策略。
    在固定时间间隔生成一批订单，批量大小服从 Pareto 分布；
    同时在同尺寸货物中引入热点 SKU 偏置，使部分货物被更高概率选中。
    """

    def __init__(self):
        # 调用父类构造函数，初始化基础数据与随机数生成器
        super().__init__()

        # 读取 Pareto 策略配置
        self.config = ContinuousParetoConfig()

        # 下一次生成订单的仿真步
        self.next_generation_step = 0

        # 按尺寸收集所有可用 goods_id
        self.all_goods_ids_by_size = self._prepare_all_goods_ids_by_size()

        # 按尺寸预先选出热点 goods_id
        self.hot_goods_ids_by_size = self._prepare_hot_goods_by_size()

        # 若地图中没有可用货物，则无法生成订单
        if not any(self.all_goods_ids_by_size.values()):
            raise ValueError("Map has no goods; cannot generate orders.")

    def _prepare_all_goods_ids_by_size(self) -> Dict[int, List[int]]:
        """
        按尺寸收集所有 goods_id。
        例如：
        size=1 的所有货箱中有哪些 goods_id
        size=2 的所有货箱中有哪些 goods_id
        """
        goods_by_size = {}
        for size, boxes in self._all_boxes_by_size.items():
            ids = []
            for box in boxes:
                ids.extend(box.get("goods_ids", []))
            goods_by_size[size] = list(set(ids))
        return goods_by_size

    def _prepare_hot_goods_by_size(self) -> Dict[int, List[int]]:
        """
        为每种尺寸预先生成热点货物集合。
        热点数量由 hot_sku_percentage 决定，至少保留 1 个热点 SKU。
        """
        hot = {}
        for size, goods_ids in self.all_goods_ids_by_size.items():
            if not goods_ids:
                hot[size] = []
                continue

            num_hot = max(1, int(len(goods_ids) * self.config.hot_sku_percentage))
            hot[size] = self.rng.sample(goods_ids, num_hot)
        return hot

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------

    def _choose_goods_id(self, size: int) -> int:
        """
        对指定尺寸选择一个 goods_id。
        若存在热点货物，则以较高概率（这里是 0.8）从热点集合中选取；
        否则从该尺寸所有货物中随机选取。
        """
        all_goods = self.all_goods_ids_by_size.get(size, [])
        hot_goods = self.hot_goods_ids_by_size.get(size, [])

        if not all_goods:
            raise RuntimeError(f"No available goods for size={size}")

        if hot_goods and self.rng.random() < 0.8:
            return self.rng.choice(hot_goods)
        return self.rng.choice(all_goods)

    # ------------------------------------------------------------------
    # main update
    # ------------------------------------------------------------------

    def update(self, current_step: int) -> List[Order]:
        """
        每个仿真步调用一次。
        功能包括：
        1. 判断是否到达订单生成时间
        2. 用 Pareto 分布生成本批订单数量
        3. 按尺寸比例划分订单数量
        4. 按热点偏置选择 goods_id，并随机匹配可用货箱和接收区
        """
        new_orders: List[Order] = []

        # 若还没到下一个生成时间点，则本步不生成订单
        if current_step < self.next_generation_step:
            return new_orders

        # 根据 Pareto 分布生成订单批量大小
        raw_count = self.rng.paretovariate(self.config.alpha)
        batch_size = max(1, int(raw_count * self.config.scale))

        # 按配置比例拆分为 size=1 和 size=2 的订单数
        num_size2 = int(batch_size * SimConfig.size2_ratio)
        num_size1 = batch_size - num_size2

        for size, count in [(1, num_size1), (2, num_size2)]:
            # 获取该尺寸下可用货箱和接收区
            boxes = self._all_boxes_by_size.get(size, [])
            receivers = self._all_receivers_by_size.get(size, [])

            if not boxes or not receivers or count <= 0:
                continue

            for _ in range(count):
                # 先按热点偏置选 goods_id
                goods_id = self._choose_goods_id(size)

                # 再找出包含该 goods_id 的候选货箱
                candidate_boxes = [
                    box for box in boxes
                    if goods_id in box.get("goods_ids", [])
                ]

                if not candidate_boxes:
                    continue

                # 随机选一个候选货箱和接收区
                box = self.rng.choice(candidate_boxes)
                receiver = self.rng.choice(receivers)

                # 构造订单对象
                order = Order(
                    order_id=-1,  # 真正编号在 OrderManager 中赋值
                    goods_id=goods_id,
                    receiver_id=receiver["receiver_id"],
                    required_size=size
                )
                new_orders.append(order)

        # 更新下一次订单生成时间
        self.next_generation_step = (
            current_step + self.config.generation_interval_steps
        )

        return new_orders