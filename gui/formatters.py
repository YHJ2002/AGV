"""Formatting helpers for the Chinese desktop UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from config.settings import OrderMode, PlannerType, SchedulerType


SCHEDULER_LABELS = {
    SchedulerType.TA.value: "TA 调度",
    SchedulerType.RANDOM.value: "随机调度",
}

PLANNER_LABELS = {
    PlannerType.ASTAR.value: "A* 规划",
    PlannerType.CBS_FW.value: "CBS-FW 规划",
    PlannerType.DHC.value: "DHC 规划",
}

ORDER_MODE_LABELS = {
    OrderMode.ONESHOT.value: "一次性订单",
    OrderMode.CONTINUOUS_CONSTANT.value: "匀速连续订单",
    OrderMode.CONTINUOUS_PERIODIC.value: "周期连续订单",
    OrderMode.CONTINUOUS_PARETO.value: "帕累托订单",
    OrderMode.CONTINUOUS_BURST.value: "爆发订单",
}

TASK_TYPE_LABELS = {
    "pick": "取货中",
    "handover": "交付中",
    "place": "返航中",
    None: "空闲",
}


def scheduler_options() -> List[tuple[str, str]]:
    return list(SCHEDULER_LABELS.items())


def planner_options() -> List[tuple[str, str]]:
    return list(PLANNER_LABELS.items())


def order_mode_options() -> List[tuple[str, str]]:
    return list(ORDER_MODE_LABELS.items())


def format_algorithm_summary(config: Dict[str, str]) -> str:
    return " / ".join(
        [
            SCHEDULER_LABELS.get(config.get("scheduler"), config.get("scheduler", "-")),
            PLANNER_LABELS.get(config.get("planner"), config.get("planner", "-")),
            ORDER_MODE_LABELS.get(config.get("order_mode"), config.get("order_mode", "-")),
        ]
    )


def format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_runtime_value(key: str, value: Any) -> str:
    if value in (None, "-"):
        return "-"

    if key == "step":
        return f"{value} step"

    if key == "sim_time":
        return f"{value} step"

    return str(value)


def format_named_metric_value(key: str, value: Any) -> str:
    if value in (None, "-"):
        return "-"

    if key in {"scheduler_avg_time", "planner_avg_time"}:
        try:
            return f"{float(value) * 1000:.3f} ms"
        except (TypeError, ValueError):
            return str(value)

    if key == "avg_task_time":
        try:
            return f"{float(value):.3f} step"
        except (TypeError, ValueError):
            return str(value)

    return format_metric_value(value)


def format_generation_log(entry: Dict[str, Any]) -> str:
    goods = f"货物 {entry['goods_id']} -> " if entry.get("goods_id") is not None else ""
    return f"订单 {entry.get('order_id', '-')}: {goods}收货区 {entry.get('receiver_id', '-')}"


def format_assignment_log(entry: Dict[str, Any]) -> str:
    box_text = f" 货箱 {entry['box_id']}" if entry.get("box_id") is not None else ""
    return f"订单 {entry.get('order_id', '-')} -> AGV {entry.get('agv_id', '-')}{box_text}"


def format_completion_log(entry: Dict[str, Any]) -> str:
    return f"AGV {entry.get('agv_id', '-')} 完成订单 {entry.get('order_id', '-')}"


def format_task_type(task_type: str | None, order_id: int | None) -> str:
    base = TASK_TYPE_LABELS.get(task_type, "执行中")
    if order_id is None or task_type is None:
        return base
    return f"{base}（订单 {order_id}）"


def list_map_files(map_dir: Path) -> List[Path]:
    return sorted(map_dir.glob("*.json"))


def recent_lines(items: Iterable[str], limit: int = 30) -> List[str]:
    values = list(items)
    return values[-limit:]
