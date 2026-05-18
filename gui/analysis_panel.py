"""Analysis page for runtime summary and exported route charts."""

from __future__ import annotations

from pathlib import Path

from gui.formatters import (
    format_algorithm_summary,
    format_assignment_log,
    format_completion_log,
    format_generation_log,
    format_named_metric_value,
    format_runtime_value,
    format_task_type,
)
from gui.qt import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPixmap,
    QScrollArea,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class AnalysisPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.summary_labels = {}
        self.order_labels = {}
        self.metric_labels = {}
        self.overview_chart_path: str | None = None
        self.route_dir_path: str | None = None
        self._agv_chart_paths: dict[int, str] = {}
        self._order_chart_paths: dict[int, str] = {}
        self._order_to_agv: dict[int, int] = {}
        self._current_chart_path: str | None = None
        self._current_chart_title = "总览图"
        self._has_final_result = False
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(10)

        hero = QFrame()
        hero.setObjectName("analysisHero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(16, 10, 16, 10)
        hero_layout.setSpacing(12)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)

        title = QLabel("结果分析")
        title.setObjectName("analysisHeroTitle")
        subtitle = QLabel("路径图与关键结果会在这里集中展示。")
        subtitle.setObjectName("sectionHint")
        subtitle.setWordWrap(True)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        self.algorithm_label = QLabel("最近一次结果：尚未运行")
        self.algorithm_label.setObjectName("algorithmSummary")
        self.algorithm_label.setWordWrap(True)

        hero_layout.addLayout(title_block, 3)
        hero_layout.addWidget(self.algorithm_label, 4)
        root.addWidget(hero)

        content = QHBoxLayout()
        content.setSpacing(10)
        content.addWidget(self._build_chart_stage(), 16)
        content.addWidget(self._build_sidebar_scroll(), 6)
        root.addLayout(content, 1)

        self._set_initial_state()

    def _build_chart_stage(self):
        box = QGroupBox("路径图")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(10)
        self.chart_title = QLabel("总览图")
        self.chart_title.setObjectName("sectionTitle")
        self.chart_hint = QLabel("运行结束并导出后，路径图会完整适配到这里。")
        self.chart_hint.setObjectName("sectionHint")
        self.chart_hint.setWordWrap(True)
        header.addWidget(self.chart_title, 0, Qt.AlignTop)
        header.addWidget(self.chart_hint, 1)
        layout.addLayout(header)

        self.chart_viewport = QFrame()
        self.chart_viewport.setObjectName("chartViewport")
        viewport_layout = QVBoxLayout(self.chart_viewport)
        viewport_layout.setContentsMargins(10, 10, 10, 10)

        self.chart_label = QLabel("运行结束并导出后，这里会显示完整路径图。")
        self.chart_label.setObjectName("chartPlaceholder")
        self.chart_label.setMinimumHeight(0)
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setWordWrap(True)
        viewport_layout.addWidget(self.chart_label, 1)

        layout.addWidget(self.chart_viewport, 1)
        return box

    def _build_sidebar_scroll(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(460)

        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_control_box())
        layout.addWidget(self._build_runtime_box())
        layout.addWidget(self._build_metrics_box())
        layout.addWidget(self._build_progress_box())
        layout.addWidget(self._build_event_box())
        layout.addStretch(1)

        scroll.setWidget(wrapper)
        return scroll

    def _build_control_box(self):
        box = QGroupBox("查看控制")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)

        self.chart_mode = QComboBox()
        self.chart_mode.setMinimumHeight(38)
        self.chart_mode.addItem("总览图", ("overview", None))
        self.chart_mode.addItem("按 AGV 查看", ("agv", None))
        self.chart_mode.addItem("按订单查看", ("order", None))
        self.chart_mode.currentIndexChanged.connect(self._refresh_target_selector)

        self.target_selector = QComboBox()
        self.target_selector.setMinimumHeight(38)
        self.target_selector.addItem("总览图", ("overview", None))
        self.target_selector.currentIndexChanged.connect(self._handle_target_changed)

        layout.addWidget(self._field_label("查看方式"))
        layout.addWidget(self.chart_mode)
        layout.addWidget(self._field_label("目标"))
        layout.addWidget(self.target_selector)

        self.export_status = QLabel("尚未生成分析图")
        self.export_status.setObjectName("analysisStatus")
        self.export_status.setWordWrap(True)
        layout.addWidget(self.export_status)

        self.overview_path_label = QLabel("总览图：-")
        self.overview_path_label.setObjectName("metaLine")
        self.overview_path_label.setWordWrap(True)
        self.routes_path_label = QLabel("路径目录：-")
        self.routes_path_label.setObjectName("metaLine")
        self.routes_path_label.setWordWrap(True)
        layout.addWidget(self.overview_path_label)
        layout.addWidget(self.routes_path_label)
        return box

    def _build_runtime_box(self):
        box = QGroupBox("运行摘要")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(10)

        runtime_title = QLabel("运行概览")
        runtime_title.setObjectName("compactTitle")
        layout.addWidget(runtime_title)

        runtime_form = QFormLayout()
        runtime_form.setSpacing(6)
        runtime_form.setLabelAlignment(Qt.AlignLeft)
        runtime_form.setFormAlignment(Qt.AlignTop)
        for key, label in (
            ("status", "运行状态"),
            ("step", "当前步数"),
            ("sim_time", "仿真时间"),
            ("agv_total", "AGV 总数"),
            ("faults", "故障数量"),
        ):
            value = QLabel("-")
            self.summary_labels[key] = value
            runtime_form.addRow(label, value)
        layout.addLayout(runtime_form)

        order_title = QLabel("订单监控")
        order_title.setObjectName("compactTitle")
        layout.addWidget(order_title)

        order_form = QFormLayout()
        order_form.setSpacing(6)
        order_form.setLabelAlignment(Qt.AlignLeft)
        order_form.setFormAlignment(Qt.AlignTop)
        for key, label in (
            ("unprocessed", "未处理"),
            ("processing", "处理中"),
            ("completed", "已完成"),
        ):
            value = QLabel("0")
            self.order_labels[key] = value
            order_form.addRow(label, value)
        layout.addLayout(order_form)
        return box

    def _build_metrics_box(self):
        box = QGroupBox("运行指标")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        items = (
            ("completed_orders", "完成订单"),
            ("success_rate", "成功率"),
            ("throughput", "吞吐量"),
            ("avg_task_time", "平均任务时间"),
            ("total_agv_collisions", "碰撞次数"),
            ("scheduler_avg_time", "调度平均耗时"),
            ("planner_avg_time", "规划平均耗时"),
            ("final_step", "结束步数"),
        )
        for index, (key, label_text) in enumerate(items):
            card = QFrame()
            card.setObjectName("metricCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 8, 10, 8)
            card_layout.setSpacing(2)

            label = QLabel(label_text)
            label.setObjectName("metricLabel")
            value = QLabel("-")
            value.setObjectName("metricValue")
            value.setWordWrap(True)
            self.metric_labels[key] = value

            card_layout.addWidget(label)
            card_layout.addWidget(value)
            grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(grid)
        return box

    def _build_progress_box(self):
        box = QGroupBox("AGV 任务进度")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)

        self.progress_empty = QLabel("运行结束后，这里会显示各 AGV 的任务进度。")
        self.progress_empty.setObjectName("sectionHint")
        self.progress_empty.setWordWrap(True)
        layout.addWidget(self.progress_empty)

        self.agv_table = QTableWidget(0, 4)
        self.agv_table.setHorizontalHeaderLabels(["AGV", "状态", "订单", "进度"])
        self.agv_table.verticalHeader().setVisible(False)
        self.agv_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.agv_table.setAlternatingRowColors(True)
        self.agv_table.setMinimumHeight(150)
        self.agv_table.setMaximumHeight(220)
        layout.addWidget(self.agv_table)
        return box

    def _build_event_box(self):
        box = QGroupBox("最近事件")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)

        self.events_empty = QLabel("运行结束后，这里会显示订单生成、分配与完成事件。")
        self.events_empty.setObjectName("sectionHint")
        self.events_empty.setWordWrap(True)
        layout.addWidget(self.events_empty)

        self.event_list = QListWidget()
        self.event_list.setMinimumHeight(150)
        self.event_list.setMaximumHeight(240)
        layout.addWidget(self.event_list)
        return box

    def _field_label(self, text: str):
        label = QLabel(text)
        label.setObjectName("sideLabel")
        return label

    def _set_initial_state(self):
        self.progress_empty.show()
        self.agv_table.hide()
        self.events_empty.show()
        self.event_list.hide()

    def update_results(self, payload: dict):
        runtime = payload.get("runtime", {})
        orders = payload.get("orders", {})
        metrics = payload.get("metrics", {})

        if self._has_final_result and runtime.get("status_text") in {"Ready", "Reset complete"}:
            return

        self._has_final_result = runtime.get("status_text") == "Completed"

        algo = {
            "scheduler": runtime.get("scheduler"),
            "planner": runtime.get("planner"),
            "order_mode": runtime.get("order_mode"),
        }
        self.algorithm_label.setText(f"最近一次结果：{format_algorithm_summary(algo)}")

        self.summary_labels["status"].setText(runtime.get("status_text", "-"))
        self.summary_labels["step"].setText(format_runtime_value("step", runtime.get("step", "-")))
        self.summary_labels["sim_time"].setText(format_runtime_value("sim_time", runtime.get("sim_time", "-")))
        self.summary_labels["agv_total"].setText(str(runtime.get("agv_total", "-")))
        self.summary_labels["faults"].setText(str(runtime.get("fault_count", 0)))

        counts = orders.get("counts", {})
        self.order_labels["unprocessed"].setText(str(counts.get("unprocessed", 0)))
        self.order_labels["processing"].setText(str(counts.get("processing", 0)))
        self.order_labels["completed"].setText(str(counts.get("completed", 0)))

        for key, label in self.metric_labels.items():
            if key == "final_step":
                label.setText(format_runtime_value("step", runtime.get("step", "-")))
            else:
                label.setText(format_named_metric_value(key, metrics.get(key, "-")))

        progress_items = orders.get("agv_progress", [])
        self.agv_table.setRowCount(len(progress_items))
        for row, item in enumerate(progress_items):
            self.agv_table.setItem(row, 0, QTableWidgetItem(str(item.get("agv_id", "-"))))
            self.agv_table.setItem(row, 1, QTableWidgetItem(format_task_type(item.get("task_type"), item.get("order_id"))))
            self.agv_table.setItem(row, 2, QTableWidgetItem(str(item.get("order_id", "-"))))
            self.agv_table.setItem(row, 3, QTableWidgetItem(f"{round((item.get('progress', 0.0) or 0.0) * 100)}%"))
        if progress_items:
            self.progress_empty.hide()
            self.agv_table.show()

        events = []
        for entry in orders.get("logs", {}).get("generation", [])[-4:]:
            events.append("生成: " + format_generation_log(entry))
        for entry in orders.get("logs", {}).get("assignment", [])[-5:]:
            events.append("分配: " + format_assignment_log(entry))
        for entry in orders.get("logs", {}).get("completion", [])[-5:]:
            events.append("完成: " + format_completion_log(entry))
        self.event_list.clear()
        if events:
            self.events_empty.hide()
            self.event_list.show()
            self.event_list.addItems(events[-14:])
            self.event_list.scrollToBottom()

        self._rebuild_order_mapping(orders)
        self._refresh_target_selector()

    def update_export_artifacts(self, payload: dict):
        paths = payload.get("paths", {})
        self.overview_chart_path = str(paths.get("overview_png")) if paths.get("overview_png") else None
        self.route_dir_path = str(paths.get("routes_dir")) if paths.get("routes_dir") else None
        self._agv_chart_paths = self._collect_agv_chart_paths(self.route_dir_path)
        self._order_chart_paths = self._collect_order_chart_paths(self.route_dir_path)

        self.export_status.setText(payload.get("status_text", "已完成"))
        self.overview_path_label.setText(f"总览图：{self.overview_chart_path or '-'}")
        self.routes_path_label.setText(f"路径目录：{self.route_dir_path or '-'}")

        self._refresh_target_selector()
        self._show_selected_chart()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_chart()

    def _collect_agv_chart_paths(self, route_dir_path: str | None) -> dict[int, str]:
        if not route_dir_path:
            return {}
        route_dir = Path(route_dir_path)
        if not route_dir.exists():
            return {}
        result = {}
        for file_path in route_dir.glob("agv_*.png"):
            try:
                agv_id = int(file_path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            result[agv_id] = str(file_path)
        return dict(sorted(result.items()))

    def _collect_order_chart_paths(self, route_dir_path: str | None) -> dict[int, str]:
        if not route_dir_path:
            return {}
        route_dir = Path(route_dir_path)
        if not route_dir.exists():
            return {}
        result = {}
        for file_path in route_dir.glob("order_*.png"):
            try:
                order_id = int(file_path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            result[order_id] = str(file_path)
        return dict(sorted(result.items()))

    def _rebuild_order_mapping(self, orders: dict):
        mapping = {}
        for entry in orders.get("logs", {}).get("assignment", []):
            order_id = entry.get("order_id")
            agv_id = entry.get("agv_id")
            if order_id is not None and agv_id is not None:
                mapping[int(order_id)] = int(agv_id)
        for item in orders.get("agv_progress", []):
            order_id = item.get("order_id")
            agv_id = item.get("agv_id")
            if order_id is not None and agv_id is not None:
                mapping[int(order_id)] = int(agv_id)
        for entry in orders.get("logs", {}).get("completion", []):
            order_id = entry.get("order_id")
            agv_id = entry.get("agv_id")
            if order_id is not None and agv_id is not None:
                mapping[int(order_id)] = int(agv_id)
        self._order_to_agv = dict(sorted(mapping.items()))

    def _refresh_target_selector(self):
        mode = self.chart_mode.currentData()
        mode_key = mode[0] if isinstance(mode, tuple) else "overview"
        current_data = self.target_selector.currentData()
        self.target_selector.blockSignals(True)
        self.target_selector.clear()

        if mode_key == "agv":
            if self._agv_chart_paths:
                for agv_id in self._agv_chart_paths:
                    self.target_selector.addItem(f"AGV {agv_id}", ("agv", agv_id))
            else:
                self.target_selector.addItem("暂无 AGV 路径图", ("none", None))
        elif mode_key == "order":
            if self._order_chart_paths:
                for order_id, chart_path in self._order_chart_paths.items():
                    agv_id = self._order_to_agv.get(order_id)
                    if agv_id is None:
                        self.target_selector.addItem(f"订单 {order_id}", ("order", order_id))
                        continue
                    self.target_selector.addItem(f"订单 {order_id} -> AGV {agv_id}", ("order", order_id))
            else:
                self.target_selector.addItem("暂无订单映射", ("none", None))
        else:
            self.target_selector.addItem("总览图", ("overview", None))

        if current_data is not None:
            for index in range(self.target_selector.count()):
                if self.target_selector.itemData(index) == current_data:
                    self.target_selector.setCurrentIndex(index)
                    break
        self.target_selector.blockSignals(False)
        self._show_selected_chart()

    def _handle_target_changed(self):
        self._show_selected_chart()

    def _show_selected_chart(self):
        data = self.target_selector.currentData()
        if not isinstance(data, tuple):
            self._current_chart_title = "总览图"
            self._current_chart_path = self.overview_chart_path
            self.chart_hint.setText("运行结束并导出后，路径图会完整适配到这里。")
            self._render_chart()
            return

        kind, value = data
        if kind == "agv" and value in self._agv_chart_paths:
            self._current_chart_title = f"AGV {value} 路径图"
            self._current_chart_path = self._agv_chart_paths[value]
            self.chart_hint.setText("当前显示单台 AGV 的完整路径图。")
        elif kind == "order" and value in self._order_to_agv:
            agv_id = self._order_to_agv[value]
            self._current_chart_title = f"订单 {value} 专属路径图"
            self._current_chart_path = self._order_chart_paths.get(value, self.overview_chart_path)
            self.chart_hint.setText(f"当前只显示订单 {value} 在 AGV {agv_id} 上执行时的那一段真实路径。")
        elif kind == "order" and value in self._order_chart_paths:
            self._current_chart_title = f"订单 {value} 专属路径图"
            self._current_chart_path = self._order_chart_paths.get(value, self.overview_chart_path)
            self.chart_hint.setText("当前只显示该订单自己的那一段真实路径。")
        else:
            self._current_chart_title = "总览图"
            self._current_chart_path = self.overview_chart_path
            self.chart_hint.setText("当前显示整轮仿真的总览路径分析图。")
        self._render_chart()

    def _render_chart(self):
        self.chart_title.setText(self._current_chart_title)
        if not self._current_chart_path:
            self.chart_label.setPixmap(QPixmap())
            self.chart_label.setText("运行结束并导出后，这里会显示完整路径图。")
            return

        chart_path = Path(self._current_chart_path)
        if not chart_path.exists():
            self.chart_label.setPixmap(QPixmap())
            self.chart_label.setText("图表已导出，但当前无法加载预览。")
            return

        pixmap = QPixmap(str(chart_path))
        if pixmap.isNull():
            self.chart_label.setPixmap(QPixmap())
            self.chart_label.setText("图表已导出，但当前无法加载预览。")
            return

        label_size = self.chart_label.contentsRect().size()
        if label_size.width() > 1 and label_size.height() > 1:
            target_width = label_size.width()
            target_height = label_size.height()
        else:
            viewport = self.chart_viewport.contentsRect().size()
            target_width = max(viewport.width() - 24, 1)
            target_height = max(viewport.height() - 24, 1)
        scaled = pixmap.scaled(target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.chart_label.setPixmap(scaled)
        self.chart_label.setText("")
