# core/simulator.py

from config.settings import SimConfig
from core.gridmap import GridMap
from core.agvmanager import AGVManager
from core.env import Env
from core.ordermanager import OrderManager
from scheduler.base_scheduler import BaseScheduler
from planner.base_planner import BasePlanner
from utils.simulation_clock import clock
from utils.logger import global_logger

class Simulator:
    def __init__(self, map_inst: GridMap,
                 agv_manager: AGVManager, order_manager: OrderManager, env: Env,
                 scheduler: BaseScheduler, planner: BasePlanner):
        # 保存地图、AGV管理器、订单管理器、环境、调度器和路径规划器
        self.map = map_inst
        self.agv_manager = agv_manager
        self.order_manager = order_manager
        self.env = env
        self.scheduler = scheduler
        self.planner = planner

    def step(self):
        """
        执行一次仿真步：
        1. 更新订单
        2. 给空闲AGV分配任务
        3. 给需要等待的AGV分配休息区
        4. 对需要重规划的AGV重新规划路径
        5. 执行环境一步（冲突检测与移动）
        """
        # 每30步打印一次日志
        if SimConfig.log_to_console and clock.now() % 30 == 0:
            print(f"\n--- Simulator Step {clock.now()} ---")
            global_logger.add_runtime_log(f"Simulator Step {clock.now()}")

        # 更新订单状态（生成新订单、检查超时等）
        self.order_manager.step()

        # 获取当前空闲的AGV
        idle_agv_set = self.agv_manager.get_idle_agv_ids()

        # 调用调度器为闲置AGV分配任务，并统计调度耗时
        with global_logger.computation_timer("scheduler"):
            agv_tasks = self.scheduler.assign_tasks(idle_agv_set, self.planner)

        # 如果有新任务，则分配给对应AGV
        if agv_tasks:
            self.agv_manager.assign_tasks(agv_tasks)

        # 获取需要去休息区的AGV
        agvs_needing_rest = self.agv_manager.get_need_rest_agv_ids()
        if agvs_needing_rest:
            # 为这些AGV分配等待区/休息区
            rest_assignments = self.scheduler.assign_rest_areas(agvs_needing_rest)
            self.agv_manager.assign_rest_zones(rest_assignments)

        # 获取需要重新规划路径的AGV目标
        replanning_targets = self.agv_manager.get_replan_targets()

        # 调用路径规划器生成新路径，并统计规划耗时
        with global_logger.computation_timer("planner"):
            new_paths = self.planner.plan(replanning_targets, self.scheduler)

        # 将新路径写回AGV管理器
        self.agv_manager.replan_paths(new_paths)

        # 环境执行一步：处理冲突并更新AGV位置
        self.env.step()

        # 仿真时钟前进一步
        clock.tick()

        # 检查订单是否全部完成
        if self.order_all_finished():
            print("All orders have been completed.")

    def order_all_finished(self) -> bool:
        """检查订单是否全部完成（当前仍是占位实现）"""
        return False  # TODO: 后续应接入 OrderManager 的真实完成状态判断