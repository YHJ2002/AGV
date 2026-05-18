"""Left-side control panel."""

from __future__ import annotations

from pathlib import Path

from config.settings import SimConfig
from gui.formatters import order_mode_options, planner_options, scheduler_options
from gui.qt import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    Signal,
)


class ControlPanel(QWidget):
    start_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    step_requested = Signal()
    reset_requested = Signal()
    stop_requested = Signal()
    apply_config_requested = Signal(dict)
    damage_requested = Signal(int)
    repair_requested = Signal(int)
    export_current_requested = Signal()
    output_options_changed = Signal(dict)
    display_options_changed = Signal(dict)

    def __init__(self, map_files: list[Path], parent: QWidget | None = None):
        super().__init__(parent)
        self.map_files = map_files
        self._build_ui()
        self._wire_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self.setMinimumWidth(340)

        layout.addWidget(self._build_run_group())
        layout.addWidget(self._build_algorithm_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_fault_group())
        layout.addWidget(self._build_output_group())
        layout.addStretch(1)

    def _build_run_group(self):
        box = QGroupBox("仿真控制")
        layout = QVBoxLayout(box)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.start_button = QPushButton("开始仿真")
        self.pause_button = QPushButton("暂停")
        self.resume_button = QPushButton("继续")
        self.step_button = QPushButton("单步执行")
        self.reset_button = QPushButton("重置仿真")
        self.stop_button = QPushButton("停止运行")

        buttons = (
            self.start_button,
            self.pause_button,
            self.resume_button,
            self.step_button,
            self.reset_button,
            self.stop_button,
        )
        for button in buttons:
            button.setMinimumHeight(38)

        grid.addWidget(self.start_button, 0, 0)
        grid.addWidget(self.pause_button, 0, 1)
        grid.addWidget(self.resume_button, 1, 0)
        grid.addWidget(self.step_button, 1, 1)
        grid.addWidget(self.reset_button, 2, 0)
        grid.addWidget(self.stop_button, 2, 1)

        self.apply_hint = QLabel("运行前可选择地图与算法；运行中修改后需要重置生效。")
        self.apply_hint.setWordWrap(True)

        layout.addLayout(grid)
        layout.addWidget(self.apply_hint)
        return box

    def _build_algorithm_group(self):
        box = QGroupBox("算法与场景")
        form = QFormLayout(box)

        self.map_combo = QComboBox()
        for map_file in self.map_files:
            self.map_combo.addItem(map_file.name, str(map_file))
        current_map = str(Path(SimConfig.map_file))
        if current_map:
            index = self.map_combo.findData(current_map)
            if index >= 0:
                self.map_combo.setCurrentIndex(index)

        self.scheduler_combo = QComboBox()
        for value, label in scheduler_options():
            self.scheduler_combo.addItem(label, value)
        self.scheduler_combo.setCurrentIndex(self.scheduler_combo.findData(SimConfig.scheduler_type.value))

        self.planner_combo = QComboBox()
        for value, label in planner_options():
            self.planner_combo.addItem(label, value)
        self.planner_combo.setCurrentIndex(self.planner_combo.findData(SimConfig.planner_type.value))

        self.order_mode_combo = QComboBox()
        for value, label in order_mode_options():
            self.order_mode_combo.addItem(label, value)
        self.order_mode_combo.setCurrentIndex(self.order_mode_combo.findData(SimConfig.order_mode.value))

        self.apply_button = QPushButton("应用设置")

        form.addRow("地图文件", self.map_combo)
        form.addRow("调度算法", self.scheduler_combo)
        form.addRow("路径规划", self.planner_combo)
        form.addRow("订单模式", self.order_mode_combo)
        form.addRow("", self.apply_button)
        return box

    def _build_display_group(self):
        box = QGroupBox("显示设置")
        layout = QVBoxLayout(box)
        self.show_agv_id = QCheckBox("显示 AGV 编号")
        self.show_box_id = QCheckBox("显示货箱编号")
        self.show_receiver_id = QCheckBox("显示收货区编号")
        self.show_paths = QCheckBox("显示安全路径")
        self.show_grid = QCheckBox("显示网格")
        self.highlight_faults = QCheckBox("高亮故障车辆")

        self.show_agv_id.setChecked(True)
        self.show_grid.setChecked(True)
        self.show_paths.setChecked(True)
        self.highlight_faults.setChecked(True)

        for widget in (
            self.show_agv_id,
            self.show_box_id,
            self.show_receiver_id,
            self.show_paths,
            self.show_grid,
            self.highlight_faults,
        ):
            layout.addWidget(widget)
        return box

    def _build_fault_group(self):
        box = QGroupBox("故障模拟")
        layout = QVBoxLayout(box)
        self.fault_input = QLineEdit()
        self.fault_input.setPlaceholderText("输入 AGV 编号")
        row = QHBoxLayout()
        self.damage_button = QPushButton("注入故障")
        self.repair_button = QPushButton("修复车辆")
        self.damage_button.setMinimumHeight(36)
        self.repair_button.setMinimumHeight(36)
        row.addWidget(self.damage_button)
        row.addWidget(self.repair_button)
        layout.addWidget(self.fault_input)
        layout.addLayout(row)
        return box

    def _build_output_group(self):
        box = QGroupBox("结果输出")
        layout = QVBoxLayout(box)

        self.export_paths = QCheckBox("生成路径图")
        self.export_overview = QCheckBox("生成总览图")
        self.export_single = QCheckBox("生成单车轨迹图")
        self.export_logs = QCheckBox("导出运行日志")
        self.export_paths.setChecked(True)
        self.export_overview.setChecked(True)
        self.export_single.setChecked(True)
        self.export_logs.setChecked(True)

        self.output_dir = QLineEdit(SimConfig.log_dir)
        self.output_dir.setPlaceholderText("选择路径图生成目录")
        self.output_dir.setPlaceholderText("输出目录")

        path_row = QHBoxLayout()
        browse_button = QPushButton("浏览...")
        open_row = QHBoxLayout()
        self.export_current_button = QPushButton("生成当前路径图")
        self.open_dir_button = QPushButton("打开输出目录")
        self.export_current_button.setMinimumHeight(36)
        self.open_dir_button.setMinimumHeight(36)
        path_row.addWidget(self.output_dir)
        path_row.addWidget(browse_button)
        open_row.addWidget(self.export_current_button)
        open_row.addWidget(self.open_dir_button)

        output_label = QLabel("路径图生成地址")

        for widget in (
            self.export_paths,
            self.export_overview,
            self.export_single,
            self.export_logs,
        ):
            layout.addWidget(widget)
        layout.addWidget(output_label)
        layout.addLayout(path_row)
        layout.addLayout(open_row)

        self.browse_button = browse_button
        return box

    def _wire_signals(self):
        self.start_button.clicked.connect(self.start_requested.emit)
        self.pause_button.clicked.connect(self.pause_requested.emit)
        self.resume_button.clicked.connect(self.resume_requested.emit)
        self.step_button.clicked.connect(self.step_requested.emit)
        self.reset_button.clicked.connect(self.reset_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.apply_button.clicked.connect(self._emit_config)
        self.damage_button.clicked.connect(lambda: self._emit_agv_command(self.damage_requested))
        self.repair_button.clicked.connect(lambda: self._emit_agv_command(self.repair_requested))
        self.export_current_button.clicked.connect(self.export_current_requested.emit)
        self.browse_button.clicked.connect(self._select_output_dir)
        self.open_dir_button.clicked.connect(lambda: self.output_options_changed.emit(self.current_output_options() | {"open_dir": True}))

        for widget in (
            self.export_paths,
            self.export_overview,
            self.export_single,
            self.export_logs,
        ):
            widget.stateChanged.connect(self._emit_output_options)
        self.output_dir.editingFinished.connect(self._emit_output_options)

        for widget in (
            self.show_agv_id,
            self.show_box_id,
            self.show_receiver_id,
            self.show_paths,
            self.show_grid,
            self.highlight_faults,
        ):
            widget.stateChanged.connect(self._emit_display_options)

    def _emit_agv_command(self, signal):
        text = self.fault_input.text().strip()
        if text.isdigit():
            signal.emit(int(text))

    def _emit_config(self):
        self.apply_config_requested.emit(self.current_algorithm_config())

    def _emit_output_options(self):
        self.output_options_changed.emit(self.current_output_options())

    def _emit_display_options(self):
        self.display_options_changed.emit(self.current_display_options())

    def _select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir.text().strip() or ".")
        if path:
            self.output_dir.setText(path)
            self._emit_output_options()

    def current_algorithm_config(self) -> dict:
        return {
            "map_file": self.map_combo.currentData(),
            "scheduler": self.scheduler_combo.currentData(),
            "planner": self.planner_combo.currentData(),
            "order_mode": self.order_mode_combo.currentData(),
        }

    def current_output_options(self) -> dict:
        return {
            "generate_paths": self.export_paths.isChecked(),
            "generate_overview": self.export_overview.isChecked(),
            "generate_single_agv": self.export_single.isChecked(),
            "export_logs": self.export_logs.isChecked(),
            "output_dir": self.output_dir.text().strip() or SimConfig.log_dir,
        }

    def current_display_options(self) -> dict:
        return {
            "show_agv_id": self.show_agv_id.isChecked(),
            "show_box_id": self.show_box_id.isChecked(),
            "show_receiver_id": self.show_receiver_id.isChecked(),
            "show_paths": self.show_paths.isChecked(),
            "show_grid": self.show_grid.isChecked(),
            "highlight_faults": self.highlight_faults.isChecked(),
        }
