from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

astar_file = PROJECT_ROOT / "test" / "single3" / "ta_astar_oneshot_runs100_seed42.csv"
cbs_file = PROJECT_ROOT / "test" / "single3" / "ta_cbs_fw_oneshot_runs100_seed42.csv"
dhc_file = PROJECT_ROOT / "test" / "single3" / "ta_dhc_oneshot_run100_seed42.csv"

output_dir = PROJECT_ROOT / "figures" / "planners_compare"
output_dir.mkdir(parents=True, exist_ok=True)

print("A* file:", astar_file)
print("CBS-FW file:", cbs_file)
print("DHC file:", dhc_file)
print("Output dir:", output_dir)

astar_df = pd.read_csv(astar_file)
cbs_df = pd.read_csv(cbs_file)
dhc_df = pd.read_csv(dhc_file)

astar_avg = astar_df.iloc[-1]
cbs_avg = cbs_df.iloc[-1]
dhc_avg = dhc_df.iloc[-1]

core_metrics = [
    "Tasks Completed",
    "Task Success Rate",
    "Throughput",
    "Avg Task Time",
    "Total AGV Collisions",
    "Planner Avg Time",
]

compare_df = pd.DataFrame(
    [
        {
            "Planner": "A*",
            "Tasks Completed": float(astar_avg["Tasks Completed"]),
            "Task Success Rate": float(astar_avg["Task Success Rate"]),
            "Throughput": float(astar_avg["Throughput"]),
            "Avg Task Time": float(astar_avg["Avg Task Time"]),
            "Total AGV Collisions": float(astar_avg["Total AGV Collisions"]),
            "Planner Avg Time": float(astar_avg["Planner Avg Time"]),
        },
        {
            "Planner": "CBS-FW",
            "Tasks Completed": float(cbs_avg["Tasks Completed"]),
            "Task Success Rate": float(cbs_avg["Task Success Rate"]),
            "Throughput": float(cbs_avg["Throughput"]),
            "Avg Task Time": float(cbs_avg["Avg Task Time"]),
            "Total AGV Collisions": float(cbs_avg["Total AGV Collisions"]),
            "Planner Avg Time": float(cbs_avg["Planner Avg Time"]),
        },
        {
            "Planner": "DHC",
            "Tasks Completed": float(dhc_avg["Tasks Completed"]),
            "Task Success Rate": float(dhc_avg["Task Success Rate"]),
            "Throughput": float(dhc_avg["Throughput"]),
            "Avg Task Time": float(dhc_avg["Avg Task Time"]),
            "Total AGV Collisions": float(dhc_avg["Total AGV Collisions"]),
            "Planner Avg Time": float(dhc_avg["Planner Avg Time"]),
        },
    ]
)

summary_path = output_dir / "core_metrics_summary.csv"
compare_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
print("Saved summary:", summary_path)
print(compare_df)


def save_table_figure(df: pd.DataFrame, title: str, save_name: str) -> None:
    display_df = df.copy()
    for column in display_df.columns:
        if column != "Planner":
            display_df[column] = display_df[column].map(lambda x: f"{x:.4f}")

    fig_height = 1.6 + 0.55 * (len(display_df) + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=14)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9ead3")
        else:
            cell.set_facecolor("#ffffff")

    save_path = output_dir / save_name
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved table figure:", save_path)


def save_bar_figure(
    df: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    title: str,
    save_name: str,
) -> None:
    labels = df["Planner"].tolist()
    values = df[metric_name].tolist()
    max_val = max(values)
    upper = max_val * 1.2 if max_val != 0 else 1

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=["#6aa84f", "#3d85c6", "#cc4125"])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Path Planner")
    ax.set_title(title)
    ax.set_ylim(0, upper)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + upper * 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save_path = output_dir / save_name
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved bar figure:", save_path)


benefit_metrics = ["Tasks Completed", "Task Success Rate", "Throughput"]
cost_metrics = ["Avg Task Time", "Total AGV Collisions", "Planner Avg Time"]

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

normalized_path = output_dir / "normalized_core_metrics_summary.csv"
normalized_df.to_csv(normalized_path, index=False, encoding="utf-8-sig")
print("Saved normalized summary:", normalized_path)
print(normalized_df)

save_table_figure(
    compare_df,
    title="TA with Different Path Planners (single3)",
    save_name="core_metrics_table.png",
)

save_bar_figure(
    compare_df,
    metric_name="Tasks Completed",
    ylabel="Tasks Completed",
    title="Tasks Completed Comparison",
    save_name="tasks_completed_comparison.png",
)

save_bar_figure(
    compare_df,
    metric_name="Task Success Rate",
    ylabel="Task Success Rate",
    title="Task Success Rate Comparison",
    save_name="task_success_rate_comparison.png",
)

save_bar_figure(
    compare_df,
    metric_name="Throughput",
    ylabel="Throughput",
    title="Throughput Comparison",
    save_name="throughput_comparison.png",
)

save_bar_figure(
    compare_df,
    metric_name="Avg Task Time",
    ylabel="Average Task Time",
    title="Average Task Time Comparison",
    save_name="avg_task_time_comparison.png",
)

save_bar_figure(
    compare_df,
    metric_name="Total AGV Collisions",
    ylabel="Total AGV Collisions",
    title="Collision Comparison",
    save_name="collisions_comparison.png",
)

save_bar_figure(
    compare_df,
    metric_name="Planner Avg Time",
    ylabel="Planner Avg Time",
    title="Planner Avg Time Comparison",
    save_name="planner_avg_time_comparison.png",
)

print("Tables and bar charts generated successfully.")
