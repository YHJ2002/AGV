"""Main desktop window for WareRover."""

from __future__ import annotations

import os

from config.settings import PROJECT_ROOT
from config.settings import OrderMode, PlannerType, SchedulerType, SimConfig
from core.agvmanager import AGVManager
from core.data_generator import generate_send_data
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from gui.analysis_panel import AnalysisPanel
from gui.control_panel import ControlPanel
from gui.formatters import format_algorithm_summary, list_map_files
from gui.map_view import MapView
from gui.qt import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QStackedWidget,
    Qt,
    QVBoxLayout,
    QWidget,
)
from gui.right_panel import DataPanel
from gui.sim_worker import SimulationThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能AGV仿真系统")
        self.resize(1680, 960)
        self.worker: SimulationThread | None = None

        map_files = list_map_files(PROJECT_ROOT / "config" / "maps")
        self.control_panel = ControlPanel(map_files)
        self.map_view = MapView()
        self.analysis_panel = AnalysisPanel()
        self.data_panel = DataPanel()
        self.map_view.on_entity_selected = self.handle_entity_selected

        self.status_badge = QLabel("未启动")
        self.step_badge = QLabel("步数 0")
        self.algorithm_badge = QLabel("当前算法")

        self._build_ui()
        self._apply_theme()
        self._wire_signals()
        self._apply_initial_sidebar_state()

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 6, 8, 8)
        root_layout.setSpacing(6)

        root_layout.addWidget(self._build_header())
        root_layout.addWidget(self._build_mode_bar())

        self.page_stack = QStackedWidget()
        self.page_stack.addWidget(self._build_run_page())
        self.page_stack.addWidget(self.analysis_panel)
        root_layout.addWidget(self.page_stack, 1)

        self.setCentralWidget(root)
        status = QStatusBar()
        status.showMessage("系统就绪")
        self.setStatusBar(status)

    def _build_header(self):
        frame = QFrame()
        frame.setObjectName("headerCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(8)

        title = QLabel("智能AGV仿真系统")
        title.setObjectName("heroTitle")
        title.setAlignment(Qt.AlignCenter)

        self.status_badge.setObjectName("statusBadge")
        self.step_badge.setObjectName("stepBadge")
        self.algorithm_badge.setObjectName("algorithmBadge")

        badge_row = QHBoxLayout()
        badge_row.setSpacing(12)
        badge_row.addStretch(1)
        badge_row.addWidget(self.algorithm_badge)
        badge_row.addWidget(self.status_badge)
        badge_row.addWidget(self.step_badge)
        badge_row.addStretch(1)

        layout.addWidget(title)
        layout.addLayout(badge_row)
        return frame

    def _build_mode_bar(self):
        frame = QFrame()
        frame.setObjectName("modeBar")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(12)

        self.run_mode_btn = QPushButton("运行界面")
        self.analysis_mode_btn = QPushButton("分析界面")
        for btn in (self.run_mode_btn, self.analysis_mode_btn):
            btn.setCheckable(True)
            btn.setObjectName("modeSwitchButton")
            btn.setMinimumSize(132, 44)

        self.run_mode_btn.setChecked(True)
        layout.addWidget(self.run_mode_btn)
        layout.addWidget(self.analysis_mode_btn)
        layout.addStretch(1)

        self.run_mode_btn.clicked.connect(lambda: self._switch_page(0))
        self.analysis_mode_btn.clicked.connect(lambda: self._switch_page(1))
        return frame

    def _switch_page(self, index: int):
        self.page_stack.setCurrentIndex(index)
        self.run_mode_btn.setChecked(index == 0)
        self.analysis_mode_btn.setChecked(index == 1)

    def _build_run_page(self):
        frame = QFrame()
        layout = QHBoxLayout(frame)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_scroll())
        splitter.addWidget(self._build_map_panel())
        splitter.addWidget(self._build_right_scroll())
        splitter.setSizes([360, 980, 420])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        layout.addWidget(splitter, 1)
        return frame

    def _build_left_scroll(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.control_panel)
        scroll.setMinimumWidth(360)
        return scroll

    def _build_map_panel(self):
        frame = QFrame()
        frame.setObjectName("mapCard")
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        title = QLabel("仓库实时运行视图")
        title.setObjectName("sectionTitle")
        hint = QLabel("支持拖拽查看。")
        hint.setObjectName("sectionHint")
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(self.map_view, 1)
        return frame

    def _build_right_scroll(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.addWidget(self.data_panel)
        layout.addStretch(1)
        scroll.setWidget(wrapper)
        return scroll

    def _wire_signals(self):
        self.control_panel.start_requested.connect(self.start_simulation)
        self.control_panel.pause_requested.connect(lambda: self._with_worker(lambda worker: worker.pause(), "已暂停"))
        self.control_panel.resume_requested.connect(lambda: self._with_worker(lambda worker: worker.resume(), "继续运行"))
        self.control_panel.step_requested.connect(lambda: self._with_worker(lambda worker: worker.request_step(), "单步执行"))
        self.control_panel.reset_requested.connect(lambda: self._with_worker(lambda worker: worker.request_reset(), "正在重置"))
        self.control_panel.stop_requested.connect(self.stop_simulation)
        self.control_panel.apply_config_requested.connect(self.apply_config)
        self.control_panel.damage_requested.connect(lambda agv_id: self._with_worker(lambda worker: worker.request_damage(agv_id), f"已注入 AGV {agv_id} 故障"))
        self.control_panel.repair_requested.connect(lambda agv_id: self._with_worker(lambda worker: worker.request_repair(agv_id), f"已请求修复 AGV {agv_id}"))
        self.control_panel.export_current_requested.connect(lambda: self._with_worker(lambda worker: worker.request_export_current(), "正在生成当前路径图"))
        self.control_panel.output_options_changed.connect(self.handle_output_options)
        self.control_panel.display_options_changed.connect(self.handle_display_options)

    def _apply_initial_sidebar_state(self):
        self.data_panel.clear_freeze()
        self.handle_display_options(self.control_panel.current_display_options())
        self.data_panel.set_algorithm_summary(self.control_panel.current_algorithm_config())
        self._load_preview_map()

    def _load_preview_map(self):
        try:
            grid_map = GridMap()
            order_manager = OrderManager(grid_map)
            agv_manager = AGVManager(grid_map, order_manager)
            payload = generate_send_data(grid_map, agv_manager, order_manager, "init")
            payload["runtime"] = {
                "status_text": "未启动",
                "step": 0,
                "sim_time": 0,
                "agv_total": agv_manager.num_agvs,
                "fault_count": 0,
                "faulty_agvs": [],
                "scheduler": self.control_panel.current_algorithm_config().get("scheduler"),
                "planner": self.control_panel.current_algorithm_config().get("planner"),
                "order_mode": self.control_panel.current_algorithm_config().get("order_mode"),
                "logs": [],
            }
            self.handle_snapshot(payload)
        except Exception as exc:
            self.statusBar().showMessage(f"地图预览加载失败: {exc}")

    def start_simulation(self):
        if self.worker and self.worker.isRunning():
            self.data_panel.clear_freeze()
            self.worker.resume()
            self.statusBar().showMessage("继续运行")
            return

        self.data_panel.clear_freeze()
        self.worker = SimulationThread(self)
        self.worker.snapshot_ready.connect(self.handle_snapshot)
        self.worker.analysis_ready.connect(self.handle_analysis_result)
        self.worker.status_changed.connect(self.handle_worker_status)
        self.worker.export_finished.connect(self.handle_export_status)
        self.worker.error_raised.connect(self.handle_worker_error)
        self.worker.configure(self.control_panel.current_algorithm_config())
        self.worker.update_output_options(self.control_panel.current_output_options())
        self.worker.start()
        self.statusBar().showMessage("已启动仿真线程")

    def stop_simulation(self):
        if not self.worker:
            return
        self.worker.request_stop()
        self.worker.wait(3000)
        self.statusBar().showMessage("已停止运行")

    def apply_config(self, config: dict):
        self.data_panel.set_algorithm_summary(config)
        if self.worker and self.worker.isRunning():
            self.worker.configure(config)
            self.statusBar().showMessage("已保存设置，重置后生效")
        else:
            self._sync_config_to_simconfig(config)
            self._load_preview_map()
            self.statusBar().showMessage("设置已更新，启动时生效")

    def handle_output_options(self, options: dict):
        if options.get("open_dir"):
            os.startfile(options.get("output_dir", "."))  # type: ignore[attr-defined]
            return
        if self.worker and self.worker.isRunning():
            self.worker.update_output_options(options)
        self.statusBar().showMessage(f"输出目录：{options.get('output_dir', '-')}")

    def handle_display_options(self, options: dict):
        self.map_view.set_display_options(options)
        self.statusBar().showMessage("已更新显示选项")

    def handle_entity_selected(self, kind: str, payload: dict):
        if kind == "agv":
            detail = {
                "type": "AGV",
                "id": payload.get("id", "-"),
                "task": "运行中",
                "cargo": "未知",
                "target": "未知",
                "conflict": "无",
                "step": self.data_panel.summary_labels.get("step").text() if "step" in self.data_panel.summary_labels else "-",
            }
            self.data_panel.update_entity_detail(detail)

    def handle_snapshot(self, payload: dict):
        if payload.get("type") == "init":
            self.map_view.load_snapshot(payload)
        else:
            self.map_view.update_snapshot(payload)

        runtime = payload.get("runtime", {})
        orders = payload.get("orders", {})
        metrics = payload.get("metrics", {})

        self.map_view.mark_faults(set(runtime.get("faulty_agvs", [])))
        self.data_panel.update_runtime_summary(runtime)
        self.data_panel.update_orders(orders)
        self.data_panel.update_metrics(metrics)
        self.data_panel.update_logs(runtime.get("logs", []))

        algo = {
            "scheduler": runtime.get("scheduler"),
            "planner": runtime.get("planner"),
            "order_mode": runtime.get("order_mode"),
        }
        self.algorithm_badge.setText(f"当前算法：{format_algorithm_summary(algo)}")
        self.status_badge.setText(runtime.get("status_text", "未启动"))
        self.step_badge.setText(f"步数 {runtime.get('step', 0)}")

    def handle_worker_status(self, payload: dict):
        self.status_badge.setText(payload.get("status_text", ""))
        self.step_badge.setText(f"步数 {payload.get('step', 0)}")
        self.statusBar().showMessage(payload.get("status_text", ""))

    def handle_analysis_result(self, payload: dict):
        runtime = payload.get("runtime", {})
        orders = payload.get("orders", {})
        metrics = payload.get("metrics", {})
        self.data_panel.update_runtime_summary(runtime)
        self.data_panel.update_orders(orders)
        self.data_panel.update_metrics(metrics)
        self.data_panel.update_logs(runtime.get("logs", []))
        self.analysis_panel.update_results(payload)

    def handle_export_status(self, payload: dict):
        paths = payload.get("paths", {})
        output_path = paths.get("overview_png") or paths.get("html") or "-"
        self.analysis_panel.update_export_artifacts(payload)
        self.data_panel.update_export_status(payload.get("status_text", "已完成"), output_path)
        self.statusBar().showMessage(payload.get("status_text", "已导出"))

    def handle_worker_error(self, message: str):
        QMessageBox.critical(self, "运行错误", message)
        self.statusBar().showMessage("运行失败")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(3000)
        super().closeEvent(event)

    def _with_worker(self, action, status_text: str):
        if self.worker and self.worker.isRunning():
            action(self.worker)
            self.statusBar().showMessage(status_text)
        else:
            self.statusBar().showMessage("请先启动仿真")

    def _sync_config_to_simconfig(self, config: dict):
        if config.get("map_file"):
            SimConfig.map_file = str(config["map_file"])
        if config.get("scheduler"):
            SimConfig.scheduler_type = SchedulerType(config["scheduler"])
        if config.get("planner"):
            SimConfig.planner_type = PlannerType(config["planner"])
        if config.get("order_mode"):
            SimConfig.order_mode = OrderMode(config["order_mode"])

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #edf5ff;
                color: #16314f;
                font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QFrame#headerCard, QFrame#mapCard, QScrollArea, QGroupBox {
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid #dbe8f8;
                border-radius: 14px;
            }
            QFrame#modeBar {
                background: transparent;
                border: none;
            }
            QFrame#headerCard, QFrame#mapCard {
                padding: 8px;
            }
            QLabel#heroTitle {
                font-size: 30px;
                font-weight: 700;
                color: #16395e;
                padding: 10px 26px;
                background: #eaf2ff;
                border-radius: 16px;
                min-width: 520px;
            }
            QLabel#algorithmBadge, QLabel#statusBadge, QLabel#stepBadge {
                background: #eff6ff;
                border: 1px solid #d5e4f8;
                border-radius: 10px;
                padding: 6px 12px;
                font-weight: 600;
                color: #28527a;
            }
            QLabel#sectionTitle {
                font-size: 16px;
                font-weight: 700;
                color: #193b60;
            }
            QLabel#sectionHint {
                color: #6b87a8;
            }
            QLabel#algorithmSummary {
                background: #f2f8ff;
                border: 1px solid #dbe8f7;
                border-radius: 12px;
                padding: 11px 14px;
                font-size: 14px;
                font-weight: 600;
                color: #21466d;
            }
            QFrame#analysisHero {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f8fbff, stop:1 #eef5ff);
                border: 1px solid #d9e7f7;
                border-radius: 16px;
            }
            QLabel#analysisHeroTitle {
                font-size: 22px;
                font-weight: 700;
                color: #183b61;
            }
            QFrame#compactCard {
                background: #f8fbff;
                border: 1px solid #d9e6f6;
                border-radius: 14px;
            }
            QLabel#compactTitle {
                font-size: 14px;
                font-weight: 700;
                color: #1b4169;
            }
            QLabel#sideLabel {
                color: #5f7b99;
                font-size: 12px;
                font-weight: 600;
            }
            QFrame#metricCard {
                background: #f6faff;
                border: 1px solid #d8e6f6;
                border-radius: 12px;
            }
            QLabel#metricLabel {
                color: #6b87a8;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#metricValue {
                color: #12385d;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#analysisStatus {
                background: #eef6ff;
                border: 1px solid #d9e7f7;
                border-radius: 12px;
                padding: 10px 12px;
                color: #1d466f;
                font-weight: 600;
            }
            QLabel#metaLine {
                color: #5e7897;
                padding: 2px 0;
            }
            QLabel#chartPlaceholder {
                background: #f7fbff;
                border: 1px dashed #c4d8ee;
                border-radius: 16px;
                color: #6f89a7;
                padding: 10px;
                font-size: 15px;
            }
            QFrame#chartViewport {
                background: #fbfdff;
                border: 1px solid #d7e4f5;
                border-radius: 18px;
            }
            QGroupBox {
                margin-top: 10px;
                padding: 12px 10px 10px 10px;
                font-weight: 700;
                color: #193b60;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
            }
            QPushButton {
                background: #e5f0ff;
                border: 1px solid #c8dbf4;
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 600;
                color: #16314f;
            }
            QPushButton#modeSwitchButton {
                background: #edf5ff;
                border: 1px solid #bed4f1;
                border-radius: 14px;
                padding: 10px 18px;
                font-size: 16px;
                font-weight: 700;
                min-width: 132px;
            }
            QPushButton#modeSwitchButton:checked {
                background: #dcecff;
                border: 1px solid #8fb3df;
                color: #123b63;
            }
            QPushButton:hover {
                background: #d8eafe;
            }
            QPushButton:pressed {
                background: #caddf6;
            }
            QLineEdit, QComboBox, QListWidget, QTableWidget {
                background: #f8fbff;
                border: 1px solid #d8e4f5;
                border-radius: 10px;
                padding: 6px 8px;
            }
            QComboBox {
                min-height: 36px;
                padding: 6px 12px;
            }
            QCheckBox {
                spacing: 8px;
            }
            QHeaderView::section {
                background: #ecf5ff;
                color: #1d446d;
                border: none;
                padding: 6px;
                font-weight: 700;
            }
            QTableWidget {
                gridline-color: #e1ecfa;
                alternate-background-color: #f2f8ff;
                min-height: 0px;
            }
            QListWidget {
                padding: 8px;
            }
            QStatusBar {
                background: #edf5ff;
                color: #33587d;
                border-top: 1px solid #d7e4f8;
            }
            QScrollArea {
                background: transparent;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #bfd5f1;
                border-radius: 5px;
                min-height: 24px;
            }
            """
        )


def launch():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
    return exec_fn()
