from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    """
    订单数据结构。
    用于描述系统中的单个订单任务，包括订单标识、目标货物、
    目标接收区以及订单在仿真过程中的关键时间信息。
    """

    # 订单编号，唯一标识一个订单
    order_id: int

    # 订单对应的货物编号
    goods_id: int

    # 订单目标接收区编号
    receiver_id: int

    # 订单要求的货物尺寸，用于与货箱/AGV尺寸匹配
    required_size: int

    # 订单创建时的仿真步数
    created_step: Optional[int] = None

    # 订单开始进入处理状态时的仿真步数
    start_processing_step: Optional[int] = None

    # 订单完成时的仿真步数
    finished_step: Optional[int] = None