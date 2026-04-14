from typing import List

from core.order import Order
from config.settings import SimConfig, ContinuousBurstConfig
from order_strategies.order_generation_strategy import OrderGenerationStrategy
from utils.logger import global_logger


class ContinuousBurstStrategy(OrderGenerationStrategy):
    """
    爆单模式订单生成策略。
    平时按较低频率、小批量生成订单；
    在满足触发条件时，进入短时间高频率、大批量的“爆单阶段”。
    """

    def __init__(self):
        # 调用父类构造函数，通常用于初始化随机数生成器等基础能力
        super().__init__()

        # 读取爆单模式配置
        self.config = ContinuousBurstConfig()

        # 平时状态下的订单生成参数
        self.base_batch_size = self.config.base_batch_size                  # 平时每批订单数
        self.base_interval = self.config.generation_interval_steps         # 平时间隔多少步生成一次

        # 爆单状态下的订单生成参数
        self.burst_batch_size = self.config.burst_peak_batch_size          # 爆单时每批订单数
        self.burst_interval = self.config.burst_interval_steps             # 爆单时间隔多少步生成一次

        # 当前是否处于爆单阶段
        self.in_burst = False

        # 爆单状态还要持续多少步
        self.steps_remaining_in_burst = 0

        # 下一次允许生成订单的仿真步
        self.next_generation_step = 0

    def _try_trigger_burst(self, current_step: int) -> bool:
        """
        判断当前步是否触发一次爆单。
        触发方式基于概率：
        burst_probability_per_1000_steps / 1000 表示每一步的触发概率
        """
        prob_per_step = self.config.burst_probability_per_1000_steps / 1000.0
        return self.rng.random() < prob_per_step

    def update(self, current_step: int) -> List[Order]:
        """
        每个仿真步调用一次。
        功能包括：
        1. 判断是否触发爆单
        2. 更新爆单状态持续时间
        3. 在到达生成时间点时批量生成订单
        """
        new_orders = []

        # 若当前不在爆单状态，则尝试按概率触发爆单
        if not self.in_burst and self._try_trigger_burst(current_step):
            self.in_burst = True
            self.steps_remaining_in_burst = self.config.burst_duration_steps

            global_logger.add_runtime_log(
                f"[OrderManager] Burst promotion triggered at step {current_step} "
                f"for {self.config.burst_duration_steps} steps!"
            )

        # 若当前处于爆单状态，则每步减少剩余持续时间
        if self.in_burst:
            self.steps_remaining_in_burst -= 1

            # 爆单持续时间结束后恢复为普通状态
            if self.steps_remaining_in_burst <= 0:
                self.in_burst = False
                global_logger.add_runtime_log(
                    f"[OrderManager] Burst promotion ended at step {current_step}"
                )

        # 到达订单生成时刻后，按当前模式（普通/爆单）生成订单
        if current_step >= self.next_generation_step:
            if self.in_burst:
                # 爆单模式：大批量、短间隔
                current_batch_size = self.burst_batch_size
                next_interval = self.burst_interval
            else:
                # 普通模式：小批量、长间隔
                current_batch_size = self.base_batch_size
                next_interval = self.base_interval

            # 按配置的尺寸比例划分 1 尺寸和 2 尺寸订单数量
            num_size2 = int(current_batch_size * SimConfig.size2_ratio)
            num_size1 = current_batch_size - num_size2

            # 分别生成 size=1 和 size=2 的订单
            for size, count in [(1, num_size1), (2, num_size2)]:
                for _ in range(count):
                    order = self._generate_single_order(size)
                    if order:
                        new_orders.append(order)

            # 更新下一次生成订单的时间
            self.next_generation_step = current_step + next_interval

        return new_orders