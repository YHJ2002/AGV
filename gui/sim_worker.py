"""Simulation worker thread for the PyQt desktop app."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

from config.settings import OrderMode, PlannerType, SchedulerType, SimConfig
from core.agvmanager import AGVManager
from core.data_generator import generate_send_data
from core.env import Env
from core.fault_manager import FaultManager
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from core.simulator import Simulator
from gui.formatters import format_algorithm_summary
from gui.qt import QThread, Signal
from utils.algorithm_factory import build_planner, build_scheduler
from utils.logger import global_logger
from utils.simulation_clock import clock


class SimulationThread(QThread):
    snapshot_ready = Signal(dict)
    analysis_ready = Signal(dict)
    status_changed = Signal(dict)
    export_finished = Signal(dict)
    error_raised = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = Lock()
        self._paused = False
        self._step_once = False
        self._reset_requested = False
        self._stop_requested = False
        self._fault_queue: list[dict] = []
        self._config_pending: dict | None = None
        self._output_options = {
            "generate_paths": True,
            "generate_overview": True,
            "generate_single_agv": True,
            "export_logs": True,
            "output_dir": SimConfig.log_dir,
        }
        self._manual_export_requested = False
        self._auto_export_done = False

    def configure(self, config: dict):
        with self._lock:
            self._config_pending = dict(config)

    def update_output_options(self, options: dict):
        with self._lock:
            self._output_options.update(options)

    def pause(self):
        with self._lock:
            self._paused = True

    def resume(self):
        with self._lock:
            self._paused = False

    def request_step(self):
        with self._lock:
            self._paused = True
            self._step_once = True

    def request_reset(self):
        with self._lock:
            self._reset_requested = True

    def request_stop(self):
        with self._lock:
            self._stop_requested = True

    def request_damage(self, agv_id: int):
        with self._lock:
            self._fault_queue.append({"cmd": "damage", "agv_id": agv_id})

    def request_repair(self, agv_id: int):
        with self._lock:
            self._fault_queue.append({"cmd": "repair", "agv_id": agv_id})

    def request_export_current(self):
        with self._lock:
            self._manual_export_requested = True

    def run(self):
        try:
            self._apply_pending_config()
            self._setup_simulation()
            while True:
                with self._lock:
                    stop_requested = self._stop_requested
                    reset_requested = self._reset_requested
                    paused = self._paused
                    step_once = self._step_once
                    fault_queue = list(self._fault_queue)
                    manual_export = self._manual_export_requested
                    self._fault_queue.clear()
                    self._manual_export_requested = False

                if stop_requested:
                    break

                if reset_requested:
                    self._perform_reset()
                    continue

                self._apply_pending_config()
                self._apply_faults(fault_queue)

                if manual_export:
                    self._emit_export(self._export_paths("Current path figures exported"))

                if self.ordermanager.is_all_orders_completed() or clock.now() >= SimConfig.max_steps:
                    self._emit_final_analysis()
                    self._auto_export_if_needed()
                    self._perform_reset(emit_status=False)
                    with self._lock:
                        self._paused = True
                    self._emit_status("Completed and reset", paused=True)
                    time.sleep(0.15)
                    continue

                if paused and not step_once:
                    self._emit_status("Paused", paused=True)
                    time.sleep(0.1)
                    continue

                self.fault_manager.step()
                self.simulator.step()
                self._emit_update("Running")

                with self._lock:
                    self._step_once = False
                time.sleep(0.1)
        except Exception as exc:  # pragma: no cover - UI thread reports error
            self.error_raised.emit(str(exc))
        finally:
            global_logger.close()

    def _perform_reset(self, emit_status: bool = True):
        self._apply_pending_config()
        self._setup_simulation()
        with self._lock:
            self._reset_requested = False
            self._step_once = False
            self._paused = False
        if emit_status:
            self._emit_status("Reset complete", paused=False)

    def _setup_simulation(self):
        Path(SimConfig.log_dir).mkdir(parents=True, exist_ok=True)
        global_logger.reset()
        clock.reset()
        self._auto_export_done = False

        self.grid_map = GridMap()
        self.ordermanager = OrderManager(self.grid_map)
        self.agv_manager = AGVManager(self.grid_map, self.ordermanager)
        self.env = Env(self.agv_manager, self.grid_map, self.ordermanager)
        self.fault_manager = FaultManager(self.agv_manager, self.env, self.grid_map)
        self.scheduler = build_scheduler(self.env, self.agv_manager, self.ordermanager, self.grid_map, self.fault_manager)
        self.planner = build_planner(self.env, self.agv_manager, self.ordermanager, self.grid_map, self.fault_manager)
        self.simulator = Simulator(self.grid_map, self.agv_manager, self.ordermanager, self.env, self.scheduler, self.planner)
        global_logger.record_agv_positions(clock.now(), self.agv_manager)

        payload = generate_send_data(self.grid_map, self.agv_manager, self.ordermanager, data_type="init")
        payload["runtime"] = self._runtime_payload("Ready")
        self.snapshot_ready.emit(payload)

    def _apply_pending_config(self):
        with self._lock:
            config = self._config_pending
            self._config_pending = None
        if not config:
            return

        if config.get("map_file"):
            SimConfig.map_file = str(config["map_file"])
        if config.get("scheduler"):
            SimConfig.scheduler_type = SchedulerType(config["scheduler"])
        if config.get("planner"):
            SimConfig.planner_type = PlannerType(config["planner"])
        if config.get("order_mode"):
            SimConfig.order_mode = OrderMode(config["order_mode"])

        summary = format_algorithm_summary(
            {
                "scheduler": SimConfig.scheduler_type.value,
                "planner": SimConfig.planner_type.value,
                "order_mode": SimConfig.order_mode.value,
            }
        )
        global_logger.add_runtime_log(f"[DesktopUI] Applied settings: {summary}")

    def _apply_faults(self, queue: list[dict]):
        for item in queue:
            self.fault_manager.handle_message(item)

    def _emit_update(self, status_text: str):
        payload = generate_send_data(self.grid_map, self.agv_manager, self.ordermanager, data_type="update")
        payload["runtime"] = self._runtime_payload(status_text)
        payload["runtime"]["faulty_agvs"] = sorted(self.fault_manager.active_faults.keys())
        payload["runtime"]["logs"] = global_logger.get_runtime_logs(30)
        self.snapshot_ready.emit(payload)
        self._emit_status(status_text, paused=False)

    def _emit_status(self, status_text: str, paused: bool):
        self.status_changed.emit(
            {
                "status_text": status_text,
                "paused": paused,
                "step": clock.now(),
                "algorithm_text": format_algorithm_summary(
                    {
                        "scheduler": SimConfig.scheduler_type.value,
                        "planner": SimConfig.planner_type.value,
                        "order_mode": SimConfig.order_mode.value,
                    }
                ),
            }
        )

    def _runtime_payload(self, status_text: str) -> dict:
        return {
            "status_text": status_text,
            "step": clock.now(),
            "sim_time": clock.now() * SimConfig.time_step,
            "agv_total": self.agv_manager.num_agvs,
            "fault_count": len(self.fault_manager.active_faults),
            "faulty_agvs": sorted(self.fault_manager.active_faults.keys()),
            "scheduler": SimConfig.scheduler_type.value,
            "planner": SimConfig.planner_type.value,
            "order_mode": SimConfig.order_mode.value,
            "logs": global_logger.get_runtime_logs(30),
        }

    def _auto_export_if_needed(self):
        if self._auto_export_done or not self._output_options.get("generate_paths", True):
            return
        export_payload = self._export_paths("Path figures auto-exported after completion")
        self._auto_export_done = True
        self._emit_export(export_payload)

    def _export_paths(self, status_text: str) -> dict:
        output_dir = self._output_options.get("output_dir") or SimConfig.log_dir
        SimConfig.log_dir = output_dir
        exported = global_logger.export_agv_paths(
            self.grid_map.width,
            self.grid_map.height,
            self.ordermanager.get_all_orders(),
        )
        return {
            "status_text": status_text if exported else "No path figures available to export",
            "paths": exported or {},
        }

    def _emit_export(self, payload: dict):
        self.export_finished.emit(payload)

    def _emit_final_analysis(self):
        payload = generate_send_data(self.grid_map, self.agv_manager, self.ordermanager, data_type="update")
        payload["runtime"] = self._runtime_payload("Completed")
        payload["runtime"]["faulty_agvs"] = sorted(self.fault_manager.active_faults.keys())
        payload["runtime"]["logs"] = global_logger.get_runtime_logs(30)
        self.analysis_ready.emit(payload)
