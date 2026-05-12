from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

normal_file = PROJECT_ROOT / "test" / "single" / "ta_astar_oneshot_runs100_seed42.csv"
fault_file = PROJECT_ROOT / "test" / "single2" / "fault_ta_astar_oneshot_runs100_seed42.csv"

output_dir = PROJECT_ROOT / "figures" / "TA_ASTAR_Fault"
output_dir.mkdir(parents=True, exist_ok=True)

print("Normal file:", normal_file)
print("Fault file:", fault_file)
print("Output dir:", output_dir)

normal_df = pd.read_csv(normal_file)
fault_df = pd.read_csv(fault_file)

normal_avg = normal_df.iloc[-1]
fault_avg = fault_df.iloc[-1]

core_metrics = [
    "Tasks Completed",
    "Task Success Rate",
    "Throughput",
    "Avg Task Time",
    "Total AGV Collisions",
]

compare_df = pd.DataFrame(
    [
        {
            "Scenario": "Normal A*",
            "Tasks Completed": float(normal_avg["Tasks Completed"]),
            "Task Success Rate": float(normal_avg["Task Success Rate"]),
            "Throughput": float(normal_avg["Throughput"]),
            "Avg Task Time": float(normal_avg["Avg Task Time"]),
            "Total AGV Collisions": float(normal_avg["Total AGV Collisions"]),
        },
        {
            "Scenario": "Fault A*",
            "Tasks Completed": float(fault_avg["Tasks Completed"]),
            "Task Success Rate": float(fault_avg["Task Success Rate"]),
            "Throughput": float(fault_avg["Throughput"]),
            "Avg Task Time": float(fault_avg["Avg Task Time"]),
            "Total AGV Collisions": float(fault_avg["Total AGV Collisions"]),
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
        if column != "Scenario":
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


def save_two_bar_figure(
    df: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    title: str,
    save_name: str,
) -> None:
    labels = df["Scenario"].tolist()
    values = df[metric_name].tolist()
    max_val = max(values)
    upper = max_val * 1.2 if max_val != 0 else 1

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(labels, values, color=["#6aa84f", "#cc4125"])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
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

normalized_path = output_dir / "normalized_core_metrics_summary.csv"
normalized_df.to_csv(normalized_path, index=False, encoding="utf-8-sig")
print("Saved normalized summary:", normalized_path)
print(normalized_df)

delta_row = {"Scenario": "Fault - Normal"}
for metric in core_metrics:
    delta_row[metric] = float(fault_avg[metric]) - float(normal_avg[metric])

delta_df = pd.DataFrame([delta_row])
delta_path = output_dir / "delta_summary.csv"
delta_df.to_csv(delta_path, index=False, encoding="utf-8-sig")
print("Saved delta summary:", delta_path)
print(delta_df)

plot_compare_df = compare_df.copy()
plot_compare_df["Scenario"] = ["Normal", "Fault"]

plot_delta_df = delta_df.copy()
plot_delta_df["Scenario"] = ["Difference"]

save_table_figure(
    plot_compare_df,
    title="TA Normal vs Fault Disturbance",
    save_name="core_metrics_table.png",
)

save_table_figure(
    plot_delta_df,
    title="TA Fault Disturbance Delta Table",
    save_name="delta_metrics_table.png",
)

save_two_bar_figure(
    plot_compare_df,
    metric_name="Tasks Completed",
    ylabel="Tasks Completed",
    title="Tasks Completed Comparison",
    save_name="tasks_completed_comparison.png",
)

save_two_bar_figure(
    plot_compare_df,
    metric_name="Task Success Rate",
    ylabel="Task Success Rate",
    title="Task Success Rate Comparison",
    save_name="task_success_rate_comparison.png",
)

save_two_bar_figure(
    plot_compare_df,
    metric_name="Throughput",
    ylabel="Throughput",
    title="Throughput Comparison",
    save_name="throughput_comparison.png",
)

save_two_bar_figure(
    plot_compare_df,
    metric_name="Avg Task Time",
    ylabel="Average Task Time",
    title="Average Task Time Comparison",
    save_name="avg_task_time_comparison.png",
)

save_two_bar_figure(
    plot_compare_df,
    metric_name="Total AGV Collisions",
    ylabel="Total AGV Collisions",
    title="Collision Comparison",
    save_name="collisions_comparison.png",
)

print("Tables and bar charts generated successfully.")
