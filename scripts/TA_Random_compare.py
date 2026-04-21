import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. 项目根目录
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# 2. 输入文件路径
# 请按你的实际文件名修改
# =========================
ta_file = os.path.join(PROJECT_ROOT, "test", "single", "ta_astar_oneshot_runs100_seed42.csv")
random_file = os.path.join(PROJECT_ROOT, "test", "single", "random_astar_oneshot_runs100_seed42.csv")

# =========================
# 3. 输出目录
# =========================
output_dir = os.path.join(PROJECT_ROOT, "figures", "TA_Random")
os.makedirs(output_dir, exist_ok=True)

print("TA file:", ta_file)
print("Random file:", random_file)
print("Output dir:", output_dir)

# =========================
# 4. 读取 CSV
# =========================
ta_df = pd.read_csv(ta_file)
random_df = pd.read_csv(random_file)

# 取最后一行平均值
ta_avg = ta_df.iloc[-1]
random_avg = random_df.iloc[-1]

# =========================
# 5. 整理核心指标
# =========================
compare_df = pd.DataFrame([
    {
        "Strategy": "TA",
        "Tasks Completed": float(ta_avg["Tasks Completed"]),
        "Throughput": float(ta_avg["Throughput"]),
        "Avg Task Time": float(ta_avg["Avg Task Time"]),
        "Total AGV Collisions": float(ta_avg["Total AGV Collisions"]),
    },
    {
        "Strategy": "Random",
        "Tasks Completed": float(random_avg["Tasks Completed"]),
        "Throughput": float(random_avg["Throughput"]),
        "Avg Task Time": float(random_avg["Avg Task Time"]),
        "Total AGV Collisions": float(random_avg["Total AGV Collisions"]),
    }
])

# 保存原始汇总表
summary_path = os.path.join(output_dir, "core_metrics_summary.csv")
compare_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
print("Saved summary:", summary_path)
print(compare_df)

# =========================
# 6. 单个指标双柱状图
# 纵轴尽量拉长：给上下留出 15%~20% 空间
# =========================
def draw_two_bar(metric_name, ylabel, title, save_name):
    values = compare_df[metric_name].values
    labels = compare_df["Strategy"].values

    min_val = float(np.min(values))
    max_val = float(np.max(values))

    # 防止两值太接近或完全一样导致图太扁
    if max_val == min_val:
        lower = 0
        upper = max_val * 1.2 if max_val != 0 else 1
    else:
        span = max_val - min_val
        lower = max(0, min_val - span * 0.2)
        upper = max_val + span * 0.2

    plt.figure(figsize=(7, 6))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.xlabel("Scheduling Strategy")
    plt.title(title)
    plt.ylim(lower, upper)

    # 在柱顶标数值
    for i, v in enumerate(values):
        plt.text(i, v + (upper - lower) * 0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved figure:", save_path)

# 生成四张核心指标图
draw_two_bar(
    metric_name="Tasks Completed",
    ylabel="Tasks Completed",
    title="Tasks Completed Comparison",
    save_name="tasks_completed_comparison.png"
)

draw_two_bar(
    metric_name="Throughput",
    ylabel="Throughput",
    title="Throughput Comparison",
    save_name="throughput_comparison.png"
)

draw_two_bar(
    metric_name="Avg Task Time",
    ylabel="Average Task Time",
    title="Average Task Time Comparison",
    save_name="avg_task_time_comparison.png"
)

draw_two_bar(
    metric_name="Total AGV Collisions",
    ylabel="Total AGV Collisions",
    title="Collision Comparison",
    save_name="collisions_comparison.png"
)

# =========================
# 7. 归一化综合对比图
# 越大越好：Tasks Completed, Throughput
# 越小越好：Avg Task Time, Total AGV Collisions
# =========================
benefit_metrics = ["Tasks Completed", "Throughput"]
cost_metrics = ["Avg Task Time", "Total AGV Collisions"]

normalized_df = compare_df.copy()

for metric in benefit_metrics:
    min_val = compare_df[metric].min()
    max_val = compare_df[metric].max()
    if max_val == min_val:
        normalized_df[metric] = 1.0
    else:
        normalized_df[metric] = (compare_df[metric] - min_val) / (max_val - min_val)

for metric in cost_metrics:
    min_val = compare_df[metric].min()
    max_val = compare_df[metric].max()
    if max_val == min_val:
        normalized_df[metric] = 1.0
    else:
        normalized_df[metric] = (max_val - compare_df[metric]) / (max_val - min_val)

# 保存归一化结果
normalized_path = os.path.join(output_dir, "normalized_core_metrics_summary.csv")
normalized_df.to_csv(normalized_path, index=False, encoding="utf-8-sig")
print("Saved normalized summary:", normalized_path)
print(normalized_df)

# 分组双柱状图
metrics = ["Tasks Completed", "Throughput", "Avg Task Time", "Total AGV Collisions"]
x = np.arange(len(metrics))
width = 0.35

ta_values = [normalized_df[normalized_df["Strategy"] == "TA"][m].values[0] for m in metrics]
random_values = [normalized_df[normalized_df["Strategy"] == "Random"][m].values[0] for m in metrics]

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, ta_values, width, label="TA")
plt.bar(x + width / 2, random_values, width, label="Random")

plt.xticks(x, metrics, rotation=15)
plt.ylabel("Normalized Score")
plt.xlabel("Metrics")
plt.title("Normalized Comparison of Core Metrics")
plt.ylim(0, 1.15)
plt.legend()

# 柱顶标数值
for i, v in enumerate(ta_values):
    plt.text(x[i] - width / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

for i, v in enumerate(random_values):
    plt.text(x[i] + width / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)


plt.tight_layout()
normalized_fig_path = os.path.join(output_dir, "normalized_core_metrics_comparison.png")
plt.savefig(normalized_fig_path, dpi=300)
plt.close()

print("Saved normalized figure:", normalized_fig_path)
print("All figures generated successfully.")