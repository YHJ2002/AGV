import os

import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 1. Project root
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# 2. Input files
# =========================
ta_file = os.path.join(
    PROJECT_ROOT, "test", "single", "ta_astar_oneshot_runs100_seed42.csv"
)
random_file = os.path.join(
    PROJECT_ROOT, "test", "single", "random_astar_oneshot_runs100_seed42.csv"
)

# =========================
# 3. Output dir
# =========================
output_dir = os.path.join(PROJECT_ROOT, "figures", "TA_Random")
os.makedirs(output_dir, exist_ok=True)

print("TA file:", ta_file)
print("Random file:", random_file)
print("Output dir:", output_dir)

# =========================
# 4. Read CSV
# =========================
ta_df = pd.read_csv(ta_file)
random_df = pd.read_csv(random_file)

ta_avg = ta_df.iloc[-1]
random_avg = random_df.iloc[-1]

# =========================
# 5. Core metrics table
# =========================
compare_df = pd.DataFrame(
    [
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
        },
    ]
)

summary_path = os.path.join(output_dir, "core_metrics_summary.csv")
compare_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
print("Saved summary:", summary_path)
print(compare_df)


def save_table_figure(df: pd.DataFrame, title: str, save_name: str) -> None:
    display_df = df.copy()
    for column in display_df.columns:
        if column != "Strategy":
            display_df[column] = display_df[column].map(lambda x: f"{x:.4f}")

    fig_height = 1.6 + 0.55 * (len(display_df) + 1)
    fig, ax = plt.subplots(figsize=(11, fig_height))
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

    save_path = os.path.join(output_dir, save_name)
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
    labels = df["Strategy"].tolist()
    values = df[metric_name].tolist()
    max_val = max(values)
    upper = max_val * 1.2 if max_val != 0 else 1

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(labels, values, color=["#6aa84f", "#3d85c6"])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Scheduling Strategy")
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

    save_path = os.path.join(output_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved bar figure:", save_path)


# =========================
# 6. Normalized table
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

normalized_path = os.path.join(output_dir, "normalized_core_metrics_summary.csv")
normalized_df.to_csv(normalized_path, index=False, encoding="utf-8-sig")
print("Saved normalized summary:", normalized_path)
print(normalized_df)

# =========================
# 7. Save table figures
# =========================
save_table_figure(
    compare_df,
    title="Core Metrics Comparison Table",
    save_name="core_metrics_table.png",
)

save_two_bar_figure(
    compare_df,
    metric_name="Tasks Completed",
    ylabel="Tasks Completed",
    title="Tasks Completed Comparison",
    save_name="tasks_completed_comparison.png",
)

save_two_bar_figure(
    compare_df,
    metric_name="Throughput",
    ylabel="Throughput",
    title="Throughput Comparison",
    save_name="throughput_comparison.png",
)

save_two_bar_figure(
    compare_df,
    metric_name="Avg Task Time",
    ylabel="Average Task Time",
    title="Average Task Time Comparison",
    save_name="avg_task_time_comparison.png",
)

save_two_bar_figure(
    compare_df,
    metric_name="Total AGV Collisions",
    ylabel="Total AGV Collisions",
    title="Collision Comparison",
    save_name="collisions_comparison.png",
)

print("Tables and bar charts generated successfully.")
