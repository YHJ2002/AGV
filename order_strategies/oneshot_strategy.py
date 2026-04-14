from order_strategies.order_generation_strategy import OrderGenerationStrategy
from config.settings import SimConfig, OneShotConfig
from core.gridmap import GridMap
from core.order import Order
from typing import List


class OneShotStrategy(OrderGenerationStrategy):
    def __init__(self):
        # 调用父类构造函数，初始化基础数据与随机数生成器
        super().__init__()

        # 一次性模式下需要生成的订单总数
        self.orders_to_generate = SimConfig.total_orders_limit

        # 预留的订单编号计数器
        # 说明：实际系统中真正的 order_id 会在 OrderManager 中重新赋值
        self.next_order_id = 0

        # 标记是否已经在第一次调用 update 时完成订单生成
        self.generated_in_first_step = False

    def update(self, current_step: int) -> List[Order]:
        """
        每个仿真步调用一次。
        一次性订单模式下，仅在第一次调用时批量生成全部订单；
        后续调用不再生成新订单。
        """
        # 如果已经在第一步生成过订单，则后续不再生成
        if self.generated_in_first_step:
            return []

        new_orders = []

        # 剩余需要生成的订单数
        remaining = self.orders_to_generate

        # 按配置比例拆分为 size=2 和 size=1 两类订单
        num_size2 = int(remaining * SimConfig.size2_ratio)
        num_size1 = remaining - num_size2

        # 分别生成两种尺寸的订单
        for size, count in [(1, num_size1), (2, num_size2)]:
            for _ in range(count):
                order = self._generate_single_order(size)
                if order:
                    new_orders.append(order)

        # 标记已经完成首步订单生成
        self.generated_in_first_step = True

        return new_orders