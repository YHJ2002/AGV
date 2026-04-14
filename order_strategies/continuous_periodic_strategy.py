import math
from order_strategies.order_generation_strategy import OrderGenerationStrategy
from config.settings import SimConfig, ContinuousPeriodicConfig
from core.gridmap import GridMap
from core.order import Order
from typing import List


class ContinuousPeriodicStrategy(OrderGenerationStrategy):
    def __init__(self):
        # 调用父类构造函数，初始化基础数据与随机数生成器
        super().__init__()

        # 下一次生成订单的仿真步
        self.next_generation_step = 0

    def _current_multiplier(self, current_step: int) -> float:
        """
        根据当前仿真步计算当前周期下的订单生成倍率。
        支持不同波形：
        - sine   : 正弦波，订单量平滑变化
        - square : 方波，订单量在高低两档之间切换
        - 其他   : 默认恒定倍率
        """
        # 计算当前步在一个完整周期中的进度（0~1）
        progress = (
            (current_step % ContinuousPeriodicConfig.cycle_duration_steps)
            / ContinuousPeriodicConfig.cycle_duration_steps
        )

        # 根据波形类型计算归一化强度 normalized（0~1）
        if ContinuousPeriodicConfig.wave_type == "sine":
            angle = progress * 2 * math.pi
            normalized = (math.sin(angle) + 1) / 2
        elif ContinuousPeriodicConfig.wave_type == "square":
            normalized = 1.0 if progress < 0.5 else 0.0
        else:
            normalized = 1.0

        # 将归一化强度映射到 [valley_multiplier, peak_multiplier] 区间
        return (
            ContinuousPeriodicConfig.valley_multiplier
            + normalized * (
                ContinuousPeriodicConfig.peak_multiplier
                - ContinuousPeriodicConfig.valley_multiplier
            )
        )

    def update(self, current_step: int) -> List[Order]:
        """
        每个仿真步调用一次。
        当达到订单生成时间点时：
        1. 先根据当前周期位置计算批量倍率
        2. 再按倍率调整当前批量大小
        3. 按尺寸比例生成订单
        """
        new_orders = []

        # 只有达到设定的生成时刻才会生成订单
        if current_step >= self.next_generation_step:
            # 根据当前周期位置计算本次订单批量倍率
            multiplier = self._current_multiplier(current_step)

            # 基础批量乘以倍率，得到当前批次订单数
            current_batch_size = max(
                1,
                int(ContinuousPeriodicConfig.base_batch_size * multiplier)
            )

            # 按配置比例拆分为 size=1 和 size=2 两类订单
            num_size2 = int(current_batch_size * SimConfig.size2_ratio)
            num_size1 = current_batch_size - num_size2

            # 分别生成两种尺寸的订单
            for size, count in [(1, num_size1), (2, num_size2)]:
                for _ in range(count):
                    order = self._generate_single_order(size)
                    if order:
                        new_orders.append(order)

            # 更新下一次生成订单的时间
            self.next_generation_step = (
                current_step + ContinuousPeriodicConfig.generation_interval_steps
            )

        return new_orders