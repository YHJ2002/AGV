"""Right-side data monitoring panel."""

from __future__ import annotations

from gui.formatters import (
    format_algorithm_summary,
    format_assignment_log,
    format_completion_log,
    format_generation_log,
    format_metric_value,
    format_named_metric_value,
    format_runtime_value,
    format_task_type,
    recent_lines,
)
from gui.qt import (
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QListWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class DataPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.summary_labels = {}
        self.order_labels = {}
        self.metric_labels = {}
        self._frozen = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.algorithm_summary = QLabel("当前算法：-")
        self.algorithm_summary.setObjectName("algorithmSummary")
        self.algorithm_summary.setWordWrap(True)
        layout.addWidget(self.algorithm_summary)

        layout.addWidget(self._build_summary_box())
        layout.addWidget(self._build_order_box())
        layout.addWidget(self._build_progress_box())
        layout.addWidget(self._build_metrics_box())
        layout.addWidget(self._build_export_box())
        layout.addWidget(self._build_log_box())

    def _build_summary_box(self):
        box = QGroupBox("运行概览")
        form = QFormLayout(box)
        for key, label in (
            ("status", "运行状态"),
            ("step", "当前步数"),
            ("sim_time", "仿真时间"),
            ("agv_total", "AGV 总数"),
            ("faults", "故障数量"),
        ):
            widget = QLabel("-")
            self.summary_labels[key] = widget
            form.addRow(label, widget)
        return box

    def _build_order_box(self):
        box = QGroupBox("订单监控")
        form = QFormLayout(box)
        for key, label in (
            ("unprocessed", "未处理"),
            ("processing", "处理中"),
            ("completed", "已完成"),
        ):
            widget = QLabel("0")
            self.order_labels[key] = widget
            form.addRow(label, widget)
        return box

    def _build_progress_box(self):
        box = QGroupBox("AGV 任务进度")
        layout = QVBoxLayout(box)
        self.agv_table = QTableWidget(0, 4)
        self.agv_table.setHorizontalHeaderLabels(["AGV", "状态", "订单", "进度"])
        self.agv_table.verticalHeader().setVisible(False)
        self.agv_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.agv_table.setAlternatingRowColors(True)
        self.agv_table.setMinimumHeight(220)
        layout.addWidget(self.agv_table)
        return box

    def _build_metrics_box(self):
        box = QGroupBox("运行指标")
        form = QFormLayout(box)
        for key, label in (
            ("completed_orders", "完成订单"),
            ("success_rate", "成功率"),
            ("throughput", "吞吐量"),
            ("avg_task_time", "平均任务时间"),
            ("total_agv_collisions", "碰撞次数"),
            ("scheduler_avg_time", "调度平均耗时"),
            ("planner_avg_time", "规划平均耗时"),
        ):
            widget = QLabel("-")
            self.metric_labels[key] = widget
            form.addRow(label, widget)
        return box

    def _build_export_box(self):
        box = QGroupBox("输出状态")
        form = QFormLayout(box)
        self.export_status = QLabel("未生成")
        self.export_file = QLabel("-")
        self.export_file.setWordWrap(True)
        form.addRow("路径图生成", self.export_status)
        form.addRow("最近输出", self.export_file)
        return box

    def _build_log_box(self):
        box = QGroupBox("实时日志")
        layout = QVBoxLayout(box)
        self.order_logs = QListWidget()
        self.runtime_logs = QListWidget()
        self.order_logs.setMinimumHeight(110)
        self.runtime_logs.setMinimumHeight(140)
        layout.addWidget(QLabel("订单事件"))
        layout.addWidget(self.order_logs)
        layout.addWidget(QLabel("系统日志"))
        layout.addWidget(self.runtime_logs)
        return box

    def set_algorithm_summary(self, config: dict):
        self.algorithm_summary.setText(f"当前算法：{format_algorithm_summary(config)}")

    def update_runtime_summary(self, data: dict):
        if self._frozen and data.get("status_text") in {"Ready", "Reset complete"}:
            return
        if data.get("status_text") == "Completed":
            self._frozen = True
        self.summary_labels["status"].setText(data.get("status_text", "-"))
        self.summary_labels["step"].setText(format_runtime_value("step", data.get("step", "-")))
        self.summary_labels["sim_time"].setText(format_runtime_value("sim_time", data.get("sim_time", "-")))
        self.summary_labels["agv_total"].setText(str(data.get("agv_total", "-")))
        self.summary_labels["faults"].setText(str(data.get("fault_count", 0)))

    def update_orders(self, orders: dict):
        if self._frozen and not orders.get("logs") and not orders.get("agv_progress") and not orders.get("counts", {}).get("completed", 0):
            return
        counts = orders.get("counts", {})
        self.order_labels["unprocessed"].setText(str(counts.get("unprocessed", 0)))
        self.order_labels["processing"].setText(str(counts.get("processing", 0)))
        self.order_labels["completed"].setText(str(counts.get("completed", 0)))

        items = []
        for entry in orders.get("logs", {}).get("generation", [])[-5:]:
            items.append("生成: " + format_generation_log(entry))
        for entry in orders.get("logs", {}).get("assignment", [])[-5:]:
            items.append("分配: " + format_assignment_log(entry))
        for entry in orders.get("logs", {}).get("completion", [])[-5:]:
            items.append("完成: " + format_completion_log(entry))

        self.order_logs.clear()
        self.order_logs.addItems(items[-15:])

        progress_items = orders.get("agv_progress", [])
        self.agv_table.setRowCount(len(progress_items))
        for row, item in enumerate(progress_items):
            self.agv_table.setItem(row, 0, QTableWidgetItem(str(item.get("agv_id", "-"))))
            self.agv_table.setItem(row, 1, QTableWidgetItem(format_task_type(item.get("task_type"), item.get("order_id"))))
            self.agv_table.setItem(row, 2, QTableWidgetItem(str(item.get("order_id", "-"))))
            self.agv_table.setItem(row, 3, QTableWidgetItem(f"{round((item.get('progress', 0.0) or 0.0) * 100)}%"))

    def update_metrics(self, metrics: dict):
        if self._frozen and all(metrics.get(key) in (None, "-", 0, 0.0) for key in self.metric_labels):
            return
        for key, label in self.metric_labels.items():
            label.setText(format_named_metric_value(key, metrics.get(key, "-")))

    def update_logs(self, lines: list[str]):
        if self._frozen:
            joined = " ".join(lines).lower()
            if (not lines) or ("reset" in joined) or ("ready" in joined):
                return
        self.runtime_logs.clear()
        self.runtime_logs.addItems(recent_lines(lines, limit=25))
        if self.runtime_logs.count():
            self.runtime_logs.scrollToBottom()

    def update_export_status(self, status_text: str, output_path: str = "-"):
        self.export_status.setText(status_text)
        self.export_file.setText(output_path)

    def clear_freeze(self):
        self._frozen = False
