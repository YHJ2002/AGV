from typing import List, Dict, Optional, Tuple
import random
from core.gridmap import GridMap
from core.order import Order
from config.settings import SimConfig, ContinuousConstantConfig
from order_strategies.order_generation_strategy import OrderGenerationStrategy


class ContinuousConstantStrategy(OrderGenerationStrategy):
    def __init__(self):
        # 调用父类构造函数，初始化订单生成策略的基础能力
        super().__init__()

        # 下一次生成订单的仿真步
        self.next_generation_step = 0

    def update(self, current_step: int) -> List[Order]:
        """
        每个仿真步调用一次。
        当当前步达到 next_generation_step 时，按固定批量生成一组订单，
        然后更新下一次生成时间。
        """
        new_orders = []

        # 只有达到设定生成时间点时才生成订单
        if current_step >= self.next_generation_step:
            # 按配置比例划分 size=2 和 size=1 的订单数量
            num_size2 = int(ContinuousConstantConfig.batch_size * SimConfig.size2_ratio)
            num_size1 = ContinuousConstantConfig.batch_size - num_size2

            # 分别生成两种尺寸的订单
            for size, count in [(1, num_size1), (2, num_size2)]:
                for _ in range(count):
                    order = self._generate_single_order(size)
                    if order:
                        new_orders.append(order)

            # 更新下一次订单生成时间
            self.next_generation_step = current_step + ContinuousConstantConfig.generation_interval_steps

        return new_orders