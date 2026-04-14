from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Generator
import random
import json
from config.settings import SimConfig, OrderMode, OneShotConfig, ContinuousConstantConfig, ContinuousPeriodicConfig
from core.gridmap import GridMap
from order_strategies import (
    OrderGenerationStrategy,
    OneShotStrategy,
    ContinuousConstantStrategy,
    ContinuousPeriodicStrategy,
    ContinuousParetoStrategy,
    ContinuousBurstStrategy,
)
from utils.logger import global_logger
from core.order import Order
from utils.simulation_clock import clock


class OrderManager:
    def __init__(self, map_inst: GridMap):
        # 地图对象，用于后续根据 box_id 查询货物、根据 receiver_id 查询接收区位置
        self.map = map_inst

        # 订单总数上限
        self.total_orders_limit = SimConfig.total_orders_limit

        # 所有订单列表（完整历史）
        self.all_orders: List[Order] = []

        # 未处理订单：等待调度器分配
        self.unprocessed_orders: Dict[int, Order] = {}

        # 处理中订单：已分配给 AGV，正在执行
        self.processing_orders: Dict[int, Order] = {}

        # 已完成订单
        self.finished_orders: Dict[int, Order] = {}

        # 下一个订单编号
        self.next_order_id = 0

        # 根据配置创建订单生成策略
        self.strategy = self._create_strategy()

    def _create_strategy(self) -> OrderGenerationStrategy:
        """
        根据当前配置中的订单模式创建对应订单生成策略
        """
        mode = SimConfig.order_mode

        if mode == OrderMode.ONESHOT:
            return OneShotStrategy()
        elif mode == OrderMode.CONTINUOUS_CONSTANT:
            return ContinuousConstantStrategy()
        elif mode == OrderMode.CONTINUOUS_PERIODIC:
            return ContinuousPeriodicStrategy()
        elif mode == OrderMode.CONTINUOUS_PARETO:
            return ContinuousParetoStrategy()
        elif mode == OrderMode.CONTINUOUS_BURST:
            return ContinuousBurstStrategy()
        else:
            raise ValueError(f"Unknown order_mode: {mode}")

    def can_generate_more_orders(self) -> bool:
        """
        判断当前是否还能继续生成订单
        """
        return len(self.all_orders) < self.total_orders_limit

    def step(self):
        """
        每个仿真步调用一次。
        功能包括：
        1. 按当前订单生成策略生成新订单
        2. 将新订单加入未处理队列
        3. 检查处理中订单是否超时
        """
        if self.can_generate_more_orders():
            current_step = clock.now()

            # 调用当前策略生成新订单
            new_orders = self.strategy.update(current_step)

            accepted_count = 0

            # 将新订单加入系统，直到达到订单总数上限
            for order in new_orders:
                if self.can_generate_more_orders():
                    order.order_id = self.next_order_id
                    order.created_step = current_step
                    self.unprocessed_orders[self.next_order_id] = order
                    self.all_orders.append(order)
                    self.next_order_id += 1
                    accepted_count += 1
                else:
                    break

            # 若本步确实新增了订单，则记录日志
            if accepted_count:
                for order in new_orders[:accepted_count]:
                    global_logger.add_order_generation_log(
                        order_id=order.order_id,
                        receiver_id=order.receiver_id,
                        goods_id=order.goods_id,
                        box_id=getattr(order, 'box_id', None),
                    )

                global_logger.add_runtime_log(
                    f"[OrderManager] Step {current_step}: Accepted {accepted_count} new orders. Total orders: {len(self.all_orders)}"
                )

        # 每个仿真步都检查一次处理中订单是否超时
        self.check_processing_timeouts()

    def get_all_orders(self) -> List[Order]:
        """返回所有订单"""
        return self.all_orders

    def get_unprocessed_orders(self) -> List[Order]:
        """返回当前未处理订单列表"""
        return list(self.unprocessed_orders.values())

    def mark_order_as_processing(self, order_id: int, agv_id: int) -> bool:
        """
        将订单从未处理状态转为处理中状态。
        一般在调度器成功把订单分配给某个 AGV 后调用。
        """
        if order_id not in self.unprocessed_orders:
            return False

        order = self.unprocessed_orders.pop(order_id)

        # 记录订单开始处理的仿真步
        order.start_processing_step = clock.now()

        # 加入处理中订单字典
        self.processing_orders[order_id] = order

        # 记录订单分配日志
        box_id = getattr(order, 'box_id', None)
        global_logger.add_order_assignment_log(order_id=order_id, agv_id=agv_id, box_id=box_id)

        return True

    def complete_order(self, order_id: int, agv_id: int, box_id: Optional[int], agv_pos: Tuple[int, int]) -> bool:
        """
        完成订单。
        将订单从 processing_orders 或 unprocessed_orders 移入 finished_orders。
        同时校验：
        1. AGV 携带的货箱是否包含该订单所需 goods_id
        2. AGV 当前位置是否等于目标接收区位置
        """
        # 先判断订单来自哪里
        if order_id in self.processing_orders:
            order_source = self.processing_orders
        elif order_id in self.unprocessed_orders:
            order_source = self.unprocessed_orders
        else:
            global_logger.add_runtime_log(
                f"[ERROR] Order {order_id} not found in processing or unprocessed orders."
            )
            return False

        order = order_source[order_id]

        # 通过地图查询当前货箱中的货物列表
        goods_list = self.map.get_goods_by_box(box_id) if box_id is not None else []

        # 查询订单对应的接收区位置
        receiver_pos = self.map.get_receiver_position(order.receiver_id)

        # 校验是否真的完成了订单
        if order.goods_id in goods_list and agv_pos == receiver_pos:
            order.finished_step = clock.now()

            # 从原状态移除，加入已完成订单
            self.finished_orders[order_id] = order_source.pop(order_id)

            global_logger.add_runtime_log(
                f"[OrderManager] Order {order_id} completed by AGV {agv_id} at step {clock.now()}."
            )
            global_logger.add_order_completion_log(order_id=order_id, agv_id=agv_id)
            global_logger.record_order_completed(self.finished_orders[order_id])

            return True
        else:
            # 校验失败，说明货不对或位置不对
            global_logger.add_runtime_log(
                f"[FAIL] Order {order_id} not fulfilled by AGV {agv_id}. "
                f"Expected goods {order.goods_id} at receiver {receiver_pos}, "
                f"but got goods {goods_list} at {agv_pos} with box_id={box_id}."
            )
            return False

    def is_all_orders_completed(self) -> bool:
        """
        判断系统中的订单是否全部完成。
        条件：
        1. 没有未处理订单
        2. 没有处理中订单
        3. 不能再生成新订单
        """
        return (
            len(self.unprocessed_orders) == 0
            and len(self.processing_orders) == 0
            and not self.can_generate_more_orders()
        )

    def check_processing_timeouts(self):
        """
        检查处理中订单是否超时。
        若订单从 start_processing_step 开始，超过配置的超时时间，
        则将其退回未处理队列重新等待分配。
        """
        timeout_orders = []
        current_step = clock.now()

        for order_id, order in self.processing_orders.items():
            if order.start_processing_step is None:
                continue

            if current_step - order.start_processing_step > SimConfig.order_processing_timeout:
                timeout_orders.append(order_id)

        # 将超时订单退回未处理队列
        for order_id in timeout_orders:
            order = self.processing_orders.pop(order_id)
            order.start_processing_step = None
            self.unprocessed_orders[order_id] = order

            global_logger.add_runtime_log(
                f"[OrderManager] Order {order_id} timeout, returned to unprocessed queue."
            )

    def reset_order(self):
        """
        重置订单管理器状态。
        包括：
        1. 清空所有订单集合
        2. 重置订单编号计数器
        3. 重新创建订单生成策略
        """
        self.all_orders.clear()
        self.unprocessed_orders.clear()
        self.processing_orders.clear()
        self.finished_orders.clear()
        self.next_order_id = 0
        self.strategy = self._create_strategy()

        global_logger.add_runtime_log("[OrderManager] Orders have been reset.")