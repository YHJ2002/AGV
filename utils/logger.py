from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import time
from contextlib import contextmanager
from config.settings import SimConfig
from core.order import Order
import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from core.agvmanager import AGVManager

class GlobalLogger:
    """Global logger singleton for single-threaded simulation."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.reset()

    # ================= Reset =================
    def reset(self):
        # ---------- Runtime Logs ----------
        self._runtime_logs: List[str] = []
        self._max_runtime_logs = 200
        self._log_to_console = SimConfig.log_to_console
        self._log_to_file = SimConfig.log_to_file

        self.total_agv_collisions = 0
        self._agv_path_history: Dict[int, List[Dict[str, Any]]] = {}
        self._agv_path_export_count = 0
        self._conflict_events: List[Dict[str, Any]] = []

        # ---------- Order Panel Logs (separate from runtime logs) ----------
        self._order_generation_logs: List[Dict[str, Any]] = []
        self._order_assignment_logs: List[Dict[str, Any]] = []
        self._order_completion_logs: List[Dict[str, Any]] = []
        self._max_order_logs = 50

        # ---------- Order Statistics ----------
        self.total_orders = SimConfig.total_orders_limit
        self.completed_orders = 0
        self.completed_task_time = 0.0  # sum(finished - created)
        self._first_completed_order_step_by_agv: Dict[int, int] = {}
        self._first_pick_position_by_agv: Dict[int, Tuple[int, int]] = {}

        # ---------- Computation Statistics ----------
        self._computation_stats = {
            "scheduler": {"total_time": 0.0, "calls": 0},
            "planner": {"total_time": 0.0, "calls": 0},
        }
        if self._log_to_console:
            print("[GlobalLogger] Logger has been reset.")

        # ---------- File Logger ----------
        self._log_file = None
        if self._log_to_file:
            os.makedirs(SimConfig.log_dir, exist_ok=True)
            self._log_file_path = os.path.join(
                SimConfig.log_dir,
                SimConfig.log_file_name
            )

            mode = "w" if SimConfig.log_overwrite else "a"
            self._log_file = open(self._log_file_path, mode, encoding="utf-8")


    # ================= Runtime Logs =================
    def add_runtime_log(self, msg: str):
        timestamp = time.strftime("[%H:%M:%S]")
        line = f"{timestamp} {msg}"

        self._runtime_logs.append(line)
        if len(self._runtime_logs) > self._max_runtime_logs:
            self._runtime_logs.pop(0)
        if self._log_to_console:
            print(line)
        if self._log_to_file and self._log_file:
            self._log_file.write(line + "\n")
            self._log_file.flush()

    def get_runtime_logs(self, n: int = 10) -> List[str]:
        return self._runtime_logs[-n:]

    # ================= Order Panel Logs (structured, for frontend) =================
    def add_order_generation_log(self, order_id: int, receiver_id: int, goods_id: Optional[int] = None, box_id: Optional[int] = None):
        """Record when an order is generated. box_id may be None until assignment."""
        entry = {"order_id": order_id, "receiver_id": receiver_id, "goods_id": goods_id, "box_id": box_id}
        self._order_generation_logs.append(entry)
        if len(self._order_generation_logs) > self._max_order_logs:
            self._order_generation_logs.pop(0)

    def add_order_assignment_log(self, order_id: int, agv_id: int, box_id: Optional[int] = None):
        """Record when an order is assigned to an AGV."""
        entry = {"order_id": order_id, "agv_id": agv_id, "box_id": box_id}
        self._order_assignment_logs.append(entry)
        if len(self._order_assignment_logs) > self._max_order_logs:
            self._order_assignment_logs.pop(0)

    def add_order_completion_log(self, order_id: int, agv_id: int):
        """Record when an order is completed by an AGV."""
        entry = {"order_id": order_id, "agv_id": agv_id}
        self._order_completion_logs.append(entry)
        if len(self._order_completion_logs) > self._max_order_logs:
            self._order_completion_logs.pop(0)

    def get_order_logs_for_panel(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get recent order logs for the frontend panel."""
        return {
            "generation": self._order_generation_logs[-self._max_order_logs:],
            "assignment": self._order_assignment_logs[-self._max_order_logs:],
            "completion": self._order_completion_logs[-self._max_order_logs:],
        }

    def get_agv_order_progress(self, agv_manager: "AGVManager") -> List[Dict[str, Any]]:
        """
        Get each AGV's current task progress for the order panel.
        Returns list of {agv_id, task_type, order_id, progress} where progress is 0..1.
        For PICK: order_id from first HANDOVER in queue; for HANDOVER: order_id; for PLACE: None.
        """
        result = []
        for agv in agv_manager.all_agvs():
            if not agv.task_queue:
                result.append({
                    "agv_id": agv.id,
                    "task_type": None,
                    "order_id": None,
                    "progress": 0.0,
                })
                continue
            task_pos, action, extra = agv.task_queue[0]
            target_pos = task_pos
            last_pos = agv.last_completed_task_pos
            # Use grid_pos for Manhattan distance (simpler, consistent)
            def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
                return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
            total_dist = manhattan(last_pos, target_pos)
            if total_dist <= 0:
                progress = 1.0
            else:
                remain = manhattan(agv.grid_pos, target_pos)
                progress = 1.0 - (remain / total_dist)
                progress = max(0.0, min(1.0, progress))
            order_id = None
            from core.agv import AGVAction
            if action == AGVAction.HANDOVER:
                order_id = extra
            elif action == AGVAction.PICK:
                # Look for first HANDOVER in queue to get order_id
                for t in agv.task_queue:
                    if t[1] == AGVAction.HANDOVER:
                        order_id = t[2]
                        break
            result.append({
                "agv_id": agv.id,
                "task_type": action.value if action else None,
                "order_id": order_id,
                "progress": round(progress, 3),
            })
        return result
    
    def record_agv_collision(self, agv_id: int):
        """
        Record an AGV collision event.
        """
        self.total_agv_collisions += 1

    def record_conflict_event(self, step: int, cell: Tuple[int, int], agv_ids: List[int], kind: str):
        """Record a conflict at a concrete grid cell for later export/plotting."""
        event = {
            "step": int(step),
            "grid_x": int(cell[0]),
            "grid_y": int(cell[1]),
            "real_x": cell[0] + 0.5,
            "real_y": cell[1] + 0.5,
            "agv_ids": sorted({int(agv_id) for agv_id in agv_ids}),
            "kind": kind,
        }
        if self._conflict_events and self._conflict_events[-1] == event:
            return
        self._conflict_events.append(event)

    def record_agv_positions(self, current_step: int, agv_manager: "AGVManager"):
        """Append one path sample per AGV for the given simulation step."""
        for agv in agv_manager.all_agvs():
            history = self._agv_path_history.setdefault(agv.id, [])
            sample = {
                "step": current_step,
                "grid_x": agv.grid_pos[0],
                "grid_y": agv.grid_pos[1],
                "real_x": round(agv.real_pos[0], 4),
                "real_y": round(agv.real_pos[1], 4),
            }
            if history and history[-1] == sample:
                continue
            history.append(sample)

    def record_pick_position(self, agv_id: int, grid_pos: Tuple[int, int]):
        """Record the first pick location for an AGV."""
        if agv_id not in self._first_pick_position_by_agv:
            self._first_pick_position_by_agv[agv_id] = grid_pos

    def export_agv_paths(self, map_width: int, map_height: int, orders: Optional[List[Order]] = None) -> Optional[Dict[str, str]]:
        """Export AGV path history plus paper-style route figures under logs/."""
        if not self._agv_path_history:
            return None

        export_root = Path(SimConfig.log_dir)
        export_root.mkdir(parents=True, exist_ok=True)
        self._agv_path_export_count += 1
        suffix = self._next_export_suffix(export_root)

        json_path = str(export_root / f"agv_paths_{suffix}.json")
        csv_path = str(export_root / f"agv_paths_{suffix}.csv")
        html_path = str(export_root / f"agv_paths_{suffix}.html")
        routes_dir = export_root / "routes" / suffix
        routes_dir.mkdir(parents=True, exist_ok=True)

        map_payload = self._load_map_payload()
        export_paths, export_scope = self._full_path_history_scope()
        if not export_paths:
            return None

        payload = {
            "map": {"width": map_width, "height": map_height},
            "map_file": str(SimConfig.map_file),
            "map_layout": map_payload,
            "export_scope": export_scope,
            "conflicts": list(self._conflict_events),
            "pick_positions": {
                str(agv_id): {"grid_x": pos[0], "grid_y": pos[1], "real_x": pos[0] + 0.5, "real_y": pos[1] + 0.5}
                for agv_id, pos in self._first_pick_position_by_agv.items()
                if agv_id in export_paths
            },
            "drop_positions": {
                str(agv_id): {
                    "grid_x": samples[-1]["grid_x"],
                    "grid_y": samples[-1]["grid_y"],
                    "real_x": samples[-1]["real_x"],
                    "real_y": samples[-1]["real_y"],
                }
                for agv_id, samples in export_paths.items()
                if samples
            },
            "algorithms": {
                "scheduler": getattr(SimConfig.scheduler_type, "value", str(SimConfig.scheduler_type)),
                "planner": getattr(SimConfig.planner_type, "value", str(SimConfig.planner_type)),
                "order_mode": getattr(SimConfig.order_mode, "value", str(SimConfig.order_mode)),
            },
            "orders": self._build_order_route_segments(orders or []),
            "paths": export_paths,
        }
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, ensure_ascii=False, indent=2)

        with open(csv_path, "w", encoding="utf-8") as csv_file:
            csv_file.write("agv_id,step,grid_x,grid_y,real_x,real_y\n")
            for agv_id in sorted(export_paths):
                for sample in export_paths[agv_id]:
                    csv_file.write(
                        f"{agv_id},{sample['step']},{sample['grid_x']},{sample['grid_y']},"
                        f"{sample['real_x']},{sample['real_y']}\n"
                    )

        with open(html_path, "w", encoding="utf-8") as html_file:
            html_file.write(self._build_agv_paths_html(payload))

        route_exports = self._export_route_figures(payload, str(routes_dir))

        self.add_runtime_log(
            f"[PathExport] Exported AGV paths to {json_path}, {csv_path}, {html_path}, "
            f"and route figures under {routes_dir}."
        )
        return {
            "json": json_path,
            "csv": csv_path,
            "html": html_path,
            "routes_dir": routes_dir,
            **route_exports,
        }

    def _next_export_suffix(self, export_root: Path) -> str:
        index = self._agv_path_export_count
        while True:
            suffix = f"run_{index:02d}"
            if not (export_root / f"agv_paths_{suffix}.json").exists() and not (export_root / "routes" / suffix).exists():
                return suffix
            index += 1

    def _build_agv_paths_html(self, payload: Dict[str, Any]) -> str:
        colors = [
            "#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6",
            "#ec4899", "#14b8a6", "#f97316", "#84cc16", "#06b6d4",
        ]
        color_map = {
            agv_id: colors[idx % len(colors)]
            for idx, agv_id in enumerate(sorted(payload["paths"]))
        }
        payload_json = json.dumps(payload, ensure_ascii=False)
        color_json = json.dumps(color_map, ensure_ascii=False)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AGV Path Viewer</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      display: grid;
      grid-template-columns: 1fr 280px;
      min-height: 100vh;
    }}
    #stage {{ padding: 24px; }}
    #map {{
      width: min(100%, 980px);
      aspect-ratio: {payload["map"]["width"]} / {payload["map"]["height"]};
      background:
        linear-gradient(to right, rgba(148,163,184,0.18) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(148,163,184,0.18) 1px, transparent 1px),
        #111827;
      background-size: calc(100% / {payload["map"]["width"]}) calc(100% / {payload["map"]["height"]});
      border: 1px solid rgba(148,163,184,0.3);
      border-radius: 16px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.35);
      overflow: hidden;
    }}
    svg {{ width: 100%; height: 100%; display: block; }}
    aside {{
      padding: 24px 20px;
      border-left: 1px solid rgba(148,163,184,0.15);
      background: rgba(15, 23, 42, 0.92);
    }}
    h1 {{ margin: 0 0 8px; font-size: 20px; }}
    .meta {{ font-size: 13px; color: #94a3b8; line-height: 1.6; margin-bottom: 20px; }}
    .legend-item {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; font-size: 14px; }}
    .swatch {{ width: 14px; height: 14px; border-radius: 999px; flex: 0 0 auto; }}
  </style>
</head>
<body>
  <div id="stage">
    <div id="map">
      <svg id="svg" viewBox="0 0 {payload["map"]["width"]} {payload["map"]["height"]}" preserveAspectRatio="none"></svg>
    </div>
  </div>
  <aside>
    <h1>AGV Paths</h1>
    <div class="meta">
      Scheduler: {payload["algorithms"]["scheduler"]}<br>
      Planner: {payload["algorithms"]["planner"]}<br>
      Order mode: {payload["algorithms"]["order_mode"]}<br>
      Map: {payload["map"]["width"]} x {payload["map"]["height"]}
    </div>
    <div id="legend"></div>
  </aside>
  <script>
    const payload = {payload_json};
    const colors = {color_json};
    const svg = document.getElementById('svg');
    const legend = document.getElementById('legend');
    const mapHeight = payload.map.height;
    Object.entries(payload.paths).forEach(([agvId, samples]) => {{
      const color = colors[agvId];
      const points = samples.map((sample) => `${{sample.real_x}},${{mapHeight - sample.real_y}}`).join(' ');
      const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      polyline.setAttribute('points', points);
      polyline.setAttribute('fill', 'none');
      polyline.setAttribute('stroke', color);
      polyline.setAttribute('stroke-width', '0.08');
      polyline.setAttribute('stroke-linecap', 'round');
      polyline.setAttribute('stroke-linejoin', 'round');
      svg.appendChild(polyline);
      if (samples.length) {{
        const start = samples[0];
        const end = samples[samples.length - 1];
        const startCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        startCircle.setAttribute('cx', start.real_x);
        startCircle.setAttribute('cy', mapHeight - start.real_y);
        startCircle.setAttribute('r', '0.12');
        startCircle.setAttribute('fill', '#f8fafc');
        startCircle.setAttribute('stroke', color);
        startCircle.setAttribute('stroke-width', '0.04');
        svg.appendChild(startCircle);
        const endCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        endCircle.setAttribute('cx', end.real_x);
        endCircle.setAttribute('cy', mapHeight - end.real_y);
        endCircle.setAttribute('r', '0.14');
        endCircle.setAttribute('fill', color);
        svg.appendChild(endCircle);
      }}
      const item = document.createElement('div');
      item.className = 'legend-item';
      item.innerHTML = `<span class="swatch" style="background:${{color}}"></span><span>AGV ${{agvId}} (${{samples.length}} samples)</span>`;
      legend.appendChild(item);
    }});
  </script>
</body>
</html>"""

    def _load_map_payload(self) -> Dict[str, Any]:
        map_path = Path(SimConfig.map_file)
        with map_path.open("r", encoding="utf-8") as map_file:
            return json.load(map_file)

    def _full_path_history_scope(self) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Any]]:
        full_paths = {
            agv_id: list(samples)
            for agv_id, samples in self._agv_path_history.items()
            if samples
        }
        return full_paths, {
            "mode": "full_history",
            "agv_count": len(full_paths),
        }

    def _build_order_route_segments(self, orders: List[Order]) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        for order in orders:
            order_id = getattr(order, "order_id", None)
            agv_id = getattr(order, "assigned_agv_id", None)
            start_step = getattr(order, "start_processing_step", None)
            end_step = getattr(order, "finished_step", None)
            if None in {order_id, agv_id, start_step, end_step}:
                continue

            agv_history = self._agv_path_history.get(int(agv_id), [])
            if not agv_history:
                continue

            samples = [
                sample for sample in agv_history
                if int(start_step) <= int(sample["step"]) <= int(end_step)
            ]
            if not samples:
                continue

            segments.append(
                {
                    "order_id": int(order_id),
                    "agv_id": int(agv_id),
                    "start_step": int(start_step),
                    "end_step": int(end_step),
                    "receiver_id": getattr(order, "receiver_id", None),
                    "goods_id": getattr(order, "goods_id", None),
                    "samples": samples,
                }
            )

        return sorted(segments, key=lambda item: item["order_id"])

    def _export_route_figures(self, payload: Dict[str, Any], routes_dir: str) -> Dict[str, str]:
        overview_png = os.path.join(routes_dir, "overview.png")

        self._plot_overview_figure(payload, overview_png)

        for agv_id in sorted(payload["paths"], key=lambda value: int(value)):
            agv_png = os.path.join(routes_dir, f"agv_{int(agv_id):02d}.png")
            self._plot_single_agv_figure(payload, agv_id, agv_png)

        for order_segment in payload.get("orders", []):
            order_png = os.path.join(routes_dir, f"order_{int(order_segment['order_id']):03d}.png")
            self._plot_single_order_figure(payload, order_segment, order_png)

        return {
            "overview_png": overview_png,
        }

    def _plot_overview_figure(self, payload: Dict[str, Any], output_path: str):
        fig, ax = self._create_base_figure(payload, figsize=(12, 9))
        sorted_ids = sorted(payload["paths"], key=lambda value: int(value))
        color_map = self._build_color_map(sorted_ids)

        for agv_id in sorted_ids:
            samples = payload["paths"][agv_id]
            if not samples:
                continue
            color = color_map[agv_id]
            xs = [sample["real_x"] for sample in samples]
            ys = [sample["real_y"] for sample in samples]
            ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.8)
            ax.scatter(xs[0], ys[0], s=20, color=color, edgecolors="white", linewidths=0.6, zorder=4)
            ax.scatter(xs[-1], ys[-1], s=28, color=color, marker="s", edgecolors="white", linewidths=0.6, zorder=4)

            pick_position = payload.get("pick_positions", {}).get(agv_id)
            if pick_position is not None:
                ax.scatter(
                    pick_position["real_x"],
                    pick_position["real_y"],
                    s=60,
                    color="#dc2626",
                    marker="^",
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=5,
                )

            drop_position = payload.get("drop_positions", {}).get(agv_id)
            if drop_position is not None:
                ax.scatter(
                    drop_position["real_x"],
                    drop_position["real_y"],
                    s=70,
                    color="#7c3aed",
                    marker="P",
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=5,
                )

        self._plot_conflicts(ax, payload.get("conflicts", []))
        ax.set_title(self._build_title(payload, None), fontsize=14, pad=18)
        fig.subplots_adjust(top=0.9)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.28)
        plt.close(fig)

    def _plot_single_agv_figure(self, payload: Dict[str, Any], agv_id: str, output_path: str):
        fig, ax = self._create_base_figure(payload, figsize=(10, 8))
        samples = payload["paths"][agv_id]
        color_map = self._build_color_map(sorted(payload["paths"], key=lambda value: int(value)))
        line_color = color_map[agv_id]
        xs = [sample["real_x"] for sample in samples]
        ys = [sample["real_y"] for sample in samples]

        if len(samples) >= 2:
            ax.plot(xs, ys, color=line_color, linewidth=2.2, solid_capstyle="round", solid_joinstyle="round")
        elif samples:
            ax.scatter(xs[0], ys[0], color=line_color, s=24, zorder=4)

        if samples:
            ax.scatter(xs[0], ys[0], s=42, color="white", edgecolors="black", linewidths=0.8, zorder=5)
            ax.text(xs[0], ys[0] + 0.35, "S", ha="center", va="bottom", fontsize=9, weight="bold")
            ax.scatter(xs[-1], ys[-1], s=52, color="black", marker="s", edgecolors="white", linewidths=0.8, zorder=5)
            ax.text(xs[-1], ys[-1] + 0.35, "E", ha="center", va="bottom", fontsize=9, weight="bold")

        pick_position = payload.get("pick_positions", {}).get(agv_id)
        if pick_position is not None:
            pick_x = pick_position["real_x"]
            pick_y = pick_position["real_y"]
            ax.scatter(
                pick_x,
                pick_y,
                s=150,
                facecolors="none",
                edgecolors="#b91c1c",
                linewidths=2.0,
                zorder=6,
            )
            ax.scatter(
                pick_x,
                pick_y,
                s=90,
                color="#dc2626",
                marker="^",
                edgecolors="white",
                linewidths=1.0,
                zorder=7,
            )
            ax.annotate(
                "Pick",
                xy=(pick_x, pick_y),
                xytext=(pick_x + 0.8, pick_y + 0.9),
                fontsize=9,
                fontweight="bold",
                color="#991b1b",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#991b1b",
                    "lw": 1.0,
                },
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "#fca5a5",
                    "lw": 0.8,
                },
                zorder=8,
            )

        drop_position = payload.get("drop_positions", {}).get(agv_id)
        if drop_position is not None:
            drop_x = drop_position["real_x"]
            drop_y = drop_position["real_y"]
            ax.scatter(
                drop_x,
                drop_y,
                s=150,
                facecolors="none",
                edgecolors="#6d28d9",
                linewidths=2.0,
                zorder=6,
            )
            ax.scatter(
                drop_x,
                drop_y,
                s=95,
                color="#7c3aed",
                marker="P",
                edgecolors="white",
                linewidths=1.0,
                zorder=7,
            )
            ax.annotate(
                "Drop",
                xy=(drop_x, drop_y),
                xytext=(drop_x + 0.8, drop_y - 0.9),
                fontsize=9,
                fontweight="bold",
                color="#5b21b6",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#5b21b6",
                    "lw": 1.0,
                },
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "#c4b5fd",
                    "lw": 0.8,
                },
                zorder=8,
            )

        agv_conflicts = [
            conflict for conflict in payload.get("conflicts", [])
            if int(agv_id) in conflict.get("agv_ids", [])
        ]
        self._plot_conflicts(ax, agv_conflicts)
        ax.set_title(self._build_title(payload, int(agv_id)), fontsize=14, pad=18)
        fig.subplots_adjust(top=0.9)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.28)
        plt.close(fig)

    def _plot_single_order_figure(self, payload: Dict[str, Any], order_segment: Dict[str, Any], output_path: str):
        fig, ax = self._create_base_figure(payload, figsize=(10, 8))

        order_id = int(order_segment["order_id"])
        agv_id = str(int(order_segment["agv_id"]))
        samples = order_segment.get("samples", [])
        color_map = self._build_color_map(sorted(payload["paths"], key=lambda value: int(value)))
        line_color = color_map.get(agv_id, "#2563eb")
        xs = [sample["real_x"] for sample in samples]
        ys = [sample["real_y"] for sample in samples]

        if len(samples) >= 2:
            ax.plot(xs, ys, color=line_color, linewidth=2.4, solid_capstyle="round", solid_joinstyle="round", zorder=4)
        elif samples:
            ax.scatter(xs[0], ys[0], color=line_color, s=36, zorder=4)

        if samples:
            start = samples[0]
            end = samples[-1]
            ax.scatter(start["real_x"], start["real_y"], s=46, color="white", edgecolors="black", linewidths=0.8, zorder=5)
            ax.text(start["real_x"], start["real_y"] + 0.35, "S", ha="center", va="bottom", fontsize=9, weight="bold")
            ax.scatter(end["real_x"], end["real_y"], s=56, color="black", marker="s", edgecolors="white", linewidths=0.8, zorder=5)
            ax.text(end["real_x"], end["real_y"] + 0.35, "E", ha="center", va="bottom", fontsize=9, weight="bold")

        order_conflicts = [
            conflict for conflict in payload.get("conflicts", [])
            if int(order_segment["start_step"]) <= int(conflict.get("step", -1)) <= int(order_segment["end_step"])
            and int(agv_id) in conflict.get("agv_ids", [])
        ]
        self._plot_conflicts(ax, order_conflicts)

        ax.set_title(
            f"{self._build_title(payload, int(agv_id))} | ORDER {order_id:03d} | "
            f"STEP {int(order_segment['start_step'])}-{int(order_segment['end_step'])}",
            fontsize=13,
            pad=18,
        )
        fig.subplots_adjust(top=0.9)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.28)
        plt.close(fig)

    def _build_color_map(self, agv_ids: List[str]) -> Dict[str, str]:
        palette = [
            "#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6",
            "#ec4899", "#14b8a6", "#f97316", "#84cc16", "#06b6d4",
        ]
        return {
            agv_id: palette[idx % len(palette)]
            for idx, agv_id in enumerate(agv_ids)
        }

    def _plot_conflicts(self, ax, conflicts: List[Dict[str, Any]]):
        if not conflicts:
            return

        xs = [conflict["real_x"] for conflict in conflicts]
        ys = [conflict["real_y"] for conflict in conflicts]
        ax.scatter(
            xs,
            ys,
            s=90,
            facecolors="#fee2e2",
            edgecolors="#b91c1c",
            linewidths=1.2,
            marker="X",
            zorder=9,
        )

    def _create_base_figure(self, payload: Dict[str, Any], figsize: Tuple[float, float]):
        fig, ax = plt.subplots(figsize=figsize)
        self._draw_map_background(ax, payload["map_layout"])

        map_width = payload["map"]["width"]
        map_height = payload["map"]["height"]
        ax.set_xlim(0, map_width)
        ax.set_ylim(0, map_height)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_xticks(range(0, map_width + 1, max(1, map_width // 10)))
        ax.set_yticks(range(0, map_height + 1, max(1, map_height // 10)))
        ax.tick_params(labelsize=8)
        ax.grid(False)
        ax.set_facecolor("#ffffff")
        fig.patch.set_facecolor("white")
        return fig, ax

    def _draw_map_background(self, ax, map_layout: Dict[str, Any]):
        map_width = map_layout["map"]["width"]
        map_height = map_layout["map"]["height"]

        for x in range(map_width + 1):
            ax.axvline(x, color="#e5e7eb", linewidth=0.5, zorder=0)
        for y in range(map_height + 1):
            ax.axhline(y, color="#e5e7eb", linewidth=0.5, zorder=0)

        for obstacle in map_layout.get("obstacles", []):
            if isinstance(obstacle, dict):
                x, y = obstacle["position"]
                size = obstacle.get("size", 1)
            else:
                x, y = obstacle
                size = 1
            ax.add_patch(Rectangle((x, y), size, size, facecolor="#4b5563", edgecolor="#374151", linewidth=0.7, zorder=1))

        for box in map_layout.get("boxes", []):
            x, y = box["position"]
            size = box.get("size", 1)
            ax.add_patch(Rectangle((x, y), size, size, facecolor="#d1d5db", edgecolor="#9ca3af", linewidth=0.7, zorder=1))

        for receiver in map_layout.get("receivers", []):
            x, y = receiver["position"]
            size = receiver.get("size", 1)
            ax.add_patch(Rectangle((x, y), size, size, facecolor="#dbeafe", edgecolor="#60a5fa", linewidth=0.9, zorder=1))

        for wait_zone in map_layout.get("wait_zones", []):
            x, y = wait_zone["position"]
            size = wait_zone.get("size", 1)
            ax.add_patch(Rectangle((x, y), size, size, facecolor="#dcfce7", edgecolor="#4ade80", linewidth=0.9, zorder=1))

    def _build_title(self, payload: Dict[str, Any], agv_id: Optional[int]) -> str:
        run_idx = self._agv_path_export_count
        algorithms = payload["algorithms"]
        prefix = f"Run {run_idx:02d}"
        if agv_id is not None:
            prefix += f" | AGV {agv_id:02d}"
        return (
            f"{prefix} | "
            f"{algorithms['scheduler'].upper()} + "
            f"{algorithms['planner'].upper()} + "
            f"{algorithms['order_mode'].upper()}"
        )


    # ================= Order Metrics =================
    def record_order_completed(self, order: Order):
        """
        Called exactly once when an order is finished.
        """
        if order.created_step is None or order.finished_step is None:
            return

        self.completed_orders += 1
        self.completed_task_time += (
            order.finished_step - order.created_step
        )
        if order.assigned_agv_id is not None and order.assigned_agv_id not in self._first_completed_order_step_by_agv:
            self._first_completed_order_step_by_agv[order.assigned_agv_id] = order.finished_step

    # ================= Computation Timer =================
    @contextmanager
    def computation_timer(self, category: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            if category not in self._computation_stats:
                self._computation_stats[category] = {
                    "total_time": 0.0,
                    "calls": 0,
                }
            stats = self._computation_stats[category]
            stats["total_time"] += time.perf_counter() - start
            stats["calls"] += 1

    # ================= Runtime Metrics =================
    def get_runtime_metrics(self, current_step: int) -> Dict[str, float]:
        """
        Metrics that can be queried during simulation.
        """
        success_rate = (
            self.completed_orders / self.total_orders
            if self.total_orders > 0
            else 0.0
        )

        throughput = (
            self.completed_orders / current_step
            if current_step > 0
            else 0.0
        )

        avg_task_time = (
            self.completed_task_time / self.completed_orders
            if self.completed_orders > 0
            else 0.0
        )

        scheduler = self._computation_stats["scheduler"]
        planner = self._computation_stats["planner"]
        scheduler_avg_time = (
            scheduler["total_time"] / scheduler["calls"]
            if scheduler["calls"] > 0
            else 0.0
        )
        planner_avg_time = (
            planner["total_time"] / planner["calls"]
            if planner["calls"] > 0
            else 0.0
        )

        return {
            "completed_orders": self.completed_orders,
            "success_rate": success_rate,
            "throughput": throughput,
            "avg_task_time": avg_task_time,
            "total_agv_collisions": float(self.total_agv_collisions),
            "scheduler_avg_time": scheduler_avg_time,
            "planner_avg_time": planner_avg_time,
        }

    # ================= Final Metrics =================
    def get_final_metrics(self, final_step: int) -> Dict[str, Any]:
        """
        Metrics collected after simulation ends.
        """
        avg_task_time = (
            self.completed_task_time / self.completed_orders
            if self.completed_orders > 0
            else 0.0
        )

        scheduler = self._computation_stats["scheduler"]
        planner = self._computation_stats["planner"]

        decision_total_time = (
            scheduler["total_time"] + planner["total_time"]
        )
        
        return {
            # ---------- Task ----------
            "Tasks Completed": self.completed_orders,
            "Task Success Rate": (
                self.completed_orders / self.total_orders
                if self.total_orders > 0
                else 0.0
            ),
            "Total Task Time": self.completed_task_time,
            "Avg Task Time": avg_task_time,

            # ---------- Throughput ----------
            "Throughput": (
                self.completed_orders / final_step
                if final_step > 0
                else 0.0
            ),
            # ---------- Collision ----------
            "Total AGV Collisions": self.total_agv_collisions,
            # ---------- Scheduler ----------
            "Scheduler Calls": scheduler["calls"],
            "Scheduler Total Time": scheduler["total_time"],
            "Scheduler Avg Time": (
                scheduler["total_time"] / scheduler["calls"]
                if scheduler["calls"] > 0
                else 0.0
            ),

            # ---------- Planner ----------
            "Planner Calls": planner["calls"],
            "Planner Total Time": planner["total_time"],
            "Planner Avg Time": (
                planner["total_time"] / planner["calls"]
                if planner["calls"] > 0
                else 0.0
            ),

            # ---------- Joint Decision ----------
            "Decision Total Time": decision_total_time,

            # ---------- Runtime ----------
            "Sim Steps": final_step,
        }
    
    def close(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None


# Global instance
global_logger = GlobalLogger()
