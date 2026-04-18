# auto_experiment_runner.py
# 全量实验运行脚本：
# 2种调度器 x 3种规划器 x 5种订单模式 x 3种场景
# 每种组合重复运行 NUM_RUNS 次
# 每个场景输出一个 CSV 文件：
#   行 = 算法组合
#   列 = 该组合在多次运行下的平均指标

import os
import csv
import itertools
from typing import Dict, List

from config.settings import (
    SimConfig,
    FaultConfig,
    SchedulerType,
    PlannerType,
    OrderMode,
)

# 从单次实验模块中导入：
# run_experiments: 执行多次实验
# summarize: 对多次实验结果做汇总统计（如求平均值）
from test.single_run import run_experiments, summarize


# 每种算法组合重复运行次数
NUM_RUNS = 100

# 随机种子基准值，用于保证实验可复现
BASE_SEED = 42

# 输出目录：所有批量实验结果的 CSV 都保存在这里
OUT_DIR = "batch_results"


# 定义三类实验场景
SCENES = {
    "homogeneous": {
        # 同构场景：地图固定，所有 AGV 规格一致，无故障
        "map_file": "config/maps/map_20_15_32.json",
        "size2_ratio": 0.0,
        "enable_faults": False,
    },
    "heterogeneous": {
        # 异构场景：包含不同规格 AGV，size2 占比 30%，无故障
        "map_file": "config/maps/map_20_15_hetero.json",
        "size2_ratio": 0.3,
        "enable_faults": False,
    },
    "fault": {
        # 故障场景：地图与同构场景一致，但开启故障注入
        "map_file": "config/maps/map_20_15_32.json",
        "size2_ratio": 0.0,
        "enable_faults": True,
    },
}


# 待测试的调度器集合
SCHEDULERS = [SchedulerType.RANDOM, SchedulerType.TA]

# 待测试的路径规划器集合
PLANNERS = [PlannerType.ASTAR, PlannerType.CBS_FW, PlannerType.DHC]

# 待测试的订单模式集合
ORDER_MODES = list(OrderMode)


def apply_scene(scene_cfg: Dict):
    """
    将场景配置写入全局仿真配置。
    
    参数：
        scene_cfg: 某个场景对应的配置字典，
                   包括地图文件、AGV 类型比例、是否启用故障等。
    """
    SimConfig.map_file = scene_cfg["map_file"]
    SimConfig.size2_ratio = scene_cfg["size2_ratio"]
    FaultConfig.enable_faults = scene_cfg["enable_faults"]


def apply_algorithm(scheduler, planner, order_mode):
    """
    将当前实验使用的算法组合写入全局仿真配置。
    
    参数：
        scheduler: 调度器类型
        planner: 路径规划器类型
        order_mode: 订单生成/处理模式
    """
    SimConfig.scheduler_type = scheduler
    SimConfig.planner_type = planner
    SimConfig.order_mode = order_mode


def combo_name():
    """
    生成当前算法组合的名称，便于日志打印。
    
    返回：
        形如 "random+astar+oneshot" 的字符串
    """
    return f"{SimConfig.scheduler_type.value}+{SimConfig.planner_type.value}+{SimConfig.order_mode.value}"


def run_all():
    """
    执行所有场景下的全部算法组合实验，并将结果分别保存为 CSV 文件。
    
    整体流程：
    1. 创建输出目录
    2. 遍历每个场景
    3. 对每个场景，遍历所有 调度器-规划器-订单模式 组合
    4. 每种组合运行 NUM_RUNS 次实验
    5. 对结果求平均
    6. 将当前场景下所有组合结果写入一个 CSV 文件
    """
    # 如果输出目录不存在，则自动创建
    os.makedirs(OUT_DIR, exist_ok=True)

    # 遍历每一个实验场景
    for scene_name, scene_cfg in SCENES.items():
        print(f"\n===== Scene: {scene_name} =====")

        # 应用当前场景配置到全局设置
        apply_scene(scene_cfg)

        # 用于保存当前场景下所有算法组合的统计结果
        scene_rows: List[Dict] = []

        # 穷举当前场景下所有算法组合
        for scheduler, planner, order_mode in itertools.product(
            SCHEDULERS, PLANNERS, ORDER_MODES
        ):
            # 应用当前算法组合配置
            apply_algorithm(scheduler, planner, order_mode)

            # 打印当前运行的组合名称，便于观察实验进度
            print(f"Running {combo_name()} ...")

            # 执行多次实验
            results = run_experiments(
                num_runs=NUM_RUNS,
                base_seed=BASE_SEED,
            )

            # 对多次实验结果进行汇总（例如计算平均值）
            avg = summarize(results)

            # 构建当前组合的一行输出结果
            row = {
                "scheduler": scheduler.value,
                "planner": planner.value,
                "order_mode": order_mode.value,
            }

            # 将统计指标加入结果行
            row.update(avg)

            # 添加到当前场景结果列表中
            scene_rows.append(row)

        # 当前场景的结果输出文件路径
        out_path = os.path.join(OUT_DIR, f"{scene_name}.csv")

        # 将当前场景下所有组合的统计结果写入 CSV
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=scene_rows[0].keys())
            writer.writeheader()
            writer.writerows(scene_rows)

        # 打印保存完成提示
        print(f"Saved scene results to {out_path}")


if __name__ == "__main__":
    # 程序入口：运行所有批量实验
    run_all()