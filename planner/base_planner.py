from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Set
from collections import defaultdict
from core.env import Env
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from core.fault_manager import FaultManager
from core.agvmanager import AGVManager
from typing import TYPE_CHECKING

# 仅用于类型检查，避免运行时循环导入
if TYPE_CHECKING:
    from scheduler.base_scheduler import BaseScheduler


class BasePlanner(ABC):
    """
    路径规划器抽象基类。
    所有具体规划器（如 A*、CBS、DHC）都应继承该类，
    并实现统一的 plan(...) 接口。
    """

    def __init__(
        self,
        env: Env,
        agv_manager: AGVManager,
        order_manager: OrderManager,
        map: GridMap,
        fault_manager: FaultManager
    ):
        # 保存环境对象，用于访问当前地图状态、可通行邻居、AGV 动态信息等
        self.env = env

        # AGV 管理器，用于访问所有 AGV 的状态
        self.agv_manager = agv_manager

        # 订单管理器，用于在规划过程中获取订单相关信息（如有需要）
        self.order_manager = order_manager

        # 地图对象，用于访问静态地图和动态占用信息
        self.map = map

        # 故障管理器，用于处理路径规划中与 AGV 故障相关的信息（如有需要）
        self.fault_manager = fault_manager

        # 统一定义的最大规划时间深度，可由子类继续调整
        self.max_time = 100

    @abstractmethod
    def plan(
        self,
        targets: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
        scheduler: BaseScheduler
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        多 AGV 集中式路径规划统一接口。

        参数：
            targets:
                {agv_id: (start_pos, target_pos)}
                表示每个需要规划的 AGV 的起点和终点。
            scheduler:
                当前调度器实例，可为规划器提供额外上下文信息。

        返回：
            {agv_id: path}
            其中 path 为栅格位置序列。

        说明：
            - 返回路径中不应包含起点
            - path[0] 表示 AGV 下一步应前往的栅格
            - 可通过 self.env.get_env_info() 获取：
                * action_queues   : 当前 AGV 的动作队列
                * current_grid_pos: 当前 AGV 的栅格位置
        """
        pass