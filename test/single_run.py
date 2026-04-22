"""
批量算法测试脚本（无可视化界面）。

功能说明：
1. 在固定算法组合下，重复运行多次完整仿真；
2. 每次运行记录一组实验指标；
3. 最终将每次结果和平均结果写入 CSV 文件。
"""

import os
import argparse
import csv
import random
import sys
import numpy as np
from typing import List, Dict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 系统配置项：包含仿真配置和故障配置
from config.settings import SimConfig, FaultConfig

# 仓储环境与仿真核心模块
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from core.agvmanager import AGVManager
from core.env import Env
from core.simulator import Simulator

# 算法工厂：根据配置动态创建调度器和规划器
from utils.algorithm_factory import build_scheduler, build_planner

# 故障管理、日志、全局时钟
from core.fault_manager import FaultManager
from utils.logger import global_logger
from utils.simulation_clock import clock

# 进度条显示
from tqdm import trange


def run_single_episode(seed: int) -> Dict:
    """
    运行一次完整仿真，并返回最终实验指标。

    参数:
        seed: 当前实验轮次使用的随机种子

    返回:
        metrics: 一个字典，保存本轮仿真的统计指标
    """
    # 固定 Python 和 NumPy 的随机种子，保证实验可重复
    random.seed(seed)
    np.random.seed(seed)

    # 同步设置订单生成与故障注入的随机种子
    SimConfig.order_seed = seed
    FaultConfig.fault_seed = seed

    # 重置全局时钟和日志器，避免上一轮实验影响当前实验
    clock.reset()
    global_logger.reset()

    # 初始化地图、订单管理器、AGV 管理器、环境和故障管理器
    grid_map = GridMap()
    ordermanager = OrderManager(grid_map)
    agv_manager = AGVManager(grid_map, ordermanager)
    env = Env(agv_manager, grid_map, ordermanager)
    fault_manager = FaultManager(agv_manager, env, grid_map)

    # 根据当前配置动态创建调度器和路径规划器
    scheduler = build_scheduler(env, agv_manager, ordermanager, grid_map, fault_manager)
    planner = build_planner(env, agv_manager, ordermanager, grid_map, fault_manager)

    # 创建仿真器对象，统一驱动整个仿真流程
    simulator = Simulator(
        grid_map,
        agv_manager,
        ordermanager,
        env,
        scheduler,
        planner,
    )

    # 主循环：
    # 条件1：订单还没有全部完成
    # 条件2：当前仿真步数未超过最大步数限制
    while (
        not ordermanager.is_all_orders_completed()
        and clock.now() < SimConfig.max_steps
    ):
        simulator.step()      # 执行一轮仿真步进（订单更新、调度、规划、移动等）
        fault_manager.step()  # 执行一轮故障注入/恢复逻辑

    # 从日志器中提取最终实验指标
    metrics = global_logger.get_final_metrics(clock.now())

    # 补充一些额外信息，便于后续实验分析
    metrics["seed"] = seed
    metrics["finished"] = ordermanager.is_all_orders_completed()
    metrics["sim_steps"] = clock.now()

    return metrics


def run_experiments(
    num_runs: int,
    base_seed: int,
) -> List[Dict]:
    """
    连续运行多轮实验。

    参数:
        num_runs: 总实验轮数
        base_seed: 初始随机种子

    返回:
        results: 列表，其中每个元素都是一轮实验对应的指标字典
    """
    results: List[Dict] = []

    # 使用 base_seed, base_seed+1, ... 作为每轮实验的随机种子
    for i in trange(num_runs, desc="Running episodes"):
        global_logger.add_runtime_log(
            f"=== Starting Run {i} with seed {base_seed + i} ==="
        )
        seed = base_seed + i
        print(f"[Run {i}] seed={seed}")

        # 运行单轮实验并保存结果
        metrics = run_single_episode(seed)
        results.append(metrics)

    return results


def summarize(results: List[Dict]) -> Dict:
    """
    对多轮实验结果求平均值。
    参数:
        results: 所有实验轮次的指标结果列表

    返回:
        summary: 平均指标字典
    """
    summary = {}

    # 自动筛选出数值型字段，并排除 seed 字段
    numeric_keys = [
        k for k in results[0].keys()
        if isinstance(results[0][k], (int, float))
        and k not in ("seed",)
    ]

    # 对每个数值型指标求平均
    for k in numeric_keys:
        summary[k] = sum(r[k] for r in results) / len(results)

    return summary


def save_csv(results: List[Dict], path: str):
    """
    将实验结果写入 CSV 文件。

    参数:
        results: 实验结果列表
        path: 输出文件路径
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def build_output_filename(
    num_runs: int,
    base_seed: int,
    ext: str = "csv",
) -> str:
    """
    根据当前实验配置自动生成输出文件名。

    文件名格式：
    调度器_规划器_订单模式_runs实验次数_seed初始种子.csv
    """
    scheduler = SimConfig.scheduler_type.value
    planner = SimConfig.planner_type.value
    order_mode = SimConfig.order_mode.value

    return (
        f"{scheduler}_{planner}_{order_mode}"
        f"_runs{num_runs}_seed{base_seed}.{ext}"
    )


def append_summary_row(results: List[Dict], summary: Dict) -> List[Dict]:
    """
    在实验结果列表最后追加一行“平均值”结果。

    参数:
        results: 每轮实验结果
        summary: 平均结果字典

    返回:
        results + [summary_row]
    """
    summary_row = {}

    for k in results[0].keys():
        if k in summary:
            summary_row[k] = summary[k]
        else:
            # 对于非平均字段，seed 用 avg 标记，其余留空
            if k == "seed":
                summary_row[k] = "avg"
            else:
                summary_row[k] = ""

    return results + [summary_row]


def main():
    """
    主函数：
    1. 解析命令行参数
    2. 打印当前实验配置
    3. 执行多轮实验
    4. 计算平均指标
    5. 保存结果到 CSV
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)     # 实验轮数
    parser.add_argument("--seed", type=int, default=42)      # 初始随机种子
    parser.add_argument("--out_dir", type=str, default="test/single")  # 输出目录
    args = parser.parse_args()

    # 若输出目录不存在则自动创建
    os.makedirs(args.out_dir, exist_ok=True)

    # 打印当前实验配置，便于检查
    print("==== Experiment Config ====")
    print(f"Scheduler: {SimConfig.scheduler_type}")
    print(f"Planner:   {SimConfig.planner_type}")
    print(f"OrderMode: {SimConfig.order_mode}")
    print("===========================")

    # 执行多轮实验
    results = run_experiments(
        num_runs=args.runs,
        base_seed=args.seed,
    )

    # 统计平均指标
    summary = summarize(results)

    # 将平均值拼接到结果末尾
    results_with_avg = append_summary_row(results, summary)

    # 在控制台输出平均指标
    print("\n==== Average Metrics ====")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    # 自动生成输出文件名
    filename = build_output_filename(
        num_runs=args.runs,
        base_seed=args.seed,
    )
    out_path = os.path.join(args.out_dir, filename)

    # 保存结果到 CSV 文件
    save_csv(results_with_avg, out_path)
    print(f"\nSaved detailed results to {out_path}")

    # 关闭日志器
    global_logger.close()


if __name__ == "__main__":
    main()
