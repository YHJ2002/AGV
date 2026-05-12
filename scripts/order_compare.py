import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "test", "single")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "figures", "order")

os.makedirs(OUTPUT_DIR, exist_ok=True)


INPUT_FILES = [
    ("OneShot", "ta_astar_oneshot_runs100_seed42.csv"),
    ("Constant", "ta_astar_continuous_constant_runs100_seed42.csv"),
    ("Periodic", "ta_astar_continuous_periodic_runs100_seed42.csv"),
    ("Pareto", "ta_astar_continuous_pareto_runs100_seed42.csv"),
    ("Burst", "ta_astar_continuous_burst_runs100_seed42.csv"),
]


METRICS = [
    ("Tasks Completed", "Tasks Completed", "tasks_completed_comparison.png"),
    ("Task Success Rate", "Task Success Rate", "task_success_rate_comparison.png"),
    ("Throughput", "Throughput", "throughput_comparison.png"),
    ("Avg Task Time", "Average Task Time", "avg_task_time_comparison.png"),
    (
        "Total AGV Collisions",
        "Total AGV Collisions",
        "total_agv_collisions_comparison.png",
    ),
]


def build_compare_table() -> pd.DataFrame:
    rows = []

    for mode_name, file_name in INPUT_FILES:
        file_path = os.path.join(INPUT_DIR, file_name)
        df = pd.read_csv(file_path)
        avg_row = df.iloc[-1]

        rows.append(
            {
                "Order Mode": mode_name,
                "Tasks Completed": float(avg_row["Tasks Completed"]),
                "Task Success Rate": float(avg_row["Task Success Rate"]),
                "Throughput": float(avg_row["Throughput"]),
                "Avg Task Time": float(avg_row["Avg Task Time"]),
                "Total AGV Collisions": float(avg_row["Total AGV Collisions"]),
            }
        )

    return pd.DataFrame(rows)


def save_table_figure(df: pd.DataFrame, title: str, save_name: str) -> None:
    display_df = df.copy()
    for column in display_df.columns:
        if column != "Order Mode":
            display_df[column] = display_df[column].map(lambda x: f"{x:.4f}")

    fig_height = 1.8 + 0.55 * (len(display_df) + 1)
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

    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved table figure:", save_path)


def save_bar_figure(
    df: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    save_name: str,
) -> None:
    labels = df["Order Mode"].tolist()
    values = df[metric_name].tolist()
    max_val = max(values)
    upper = max_val * 1.2 if max_val != 0 else 1
    colors = ["#6aa84f", "#3d85c6", "#f6b26b", "#cc79a7", "#76a5af"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=colors[: len(values)])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Order Mode")
    ax.set_title(f"{metric_name} Across Order Modes")
    ax.set_ylim(0, upper)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + upper * 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=10)
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved bar figure:", save_path)


def main() -> None:
    compare_df = build_compare_table()

    summary_path = os.path.join(OUTPUT_DIR, "order_metrics_summary.csv")
    compare_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("Saved summary:", summary_path)
    print(compare_df)

    save_table_figure(
        compare_df,
        title="Order Mode Comparison Table",
        save_name="order_metrics_table.png",
    )

    for metric_name, ylabel, save_name in METRICS:
        save_bar_figure(compare_df, metric_name, ylabel, save_name)

    print("All order comparison outputs generated successfully.")


if __name__ == "__main__":
    main()
