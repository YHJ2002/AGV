"""Center warehouse map view with stylized AGV and shelf rendering."""

from __future__ import annotations

from typing import Dict, Tuple

from gui.qt import (
    QBrush,
    QColor,
    QGraphicsEllipseItem,
    QGraphicsItemGroup,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QPainterPath,
    QPen,
    QPointF,
    QPolygonF,
    Qt,
)


class MapView(QGraphicsView):
    CELL = 34

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ref = QGraphicsScene(self)
        self.setScene(self.scene_ref)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#edf6ff"))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setAlignment(Qt.AlignCenter)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setMinimumSize(640, 480)
        self.map_size = {"width": 0, "height": 0}
        self.display_options = {
            "show_agv_id": True,
            "show_box_id": False,
            "show_receiver_id": False,
            "show_paths": True,
            "show_grid": True,
            "highlight_faults": True,
        }
        self.agv_items: Dict[int, dict] = {}
        self.shelf_items: Dict[int, dict] = {}
        self.receiver_labels: Dict[int, QGraphicsSimpleTextItem] = {}
        self.wait_zone_items = []
        self.receiver_items = []
        self.obstacle_items = []
        self.safe_path_items: list[QGraphicsPathItem] = []
        self.tile_items: list[QGraphicsRectItem] = []
        self.grid_items = []
        self.map_border_items = []
        self.selected_entity = None
        self.selected_entity_kind = None
        self.on_entity_selected = None

    def set_display_options(self, options: dict):
        self.display_options.update(options)
        self._refresh_label_visibility()
        self._redraw_background()
        self._update_safe_paths(getattr(self, "_last_safe_paths", {}))

    def load_snapshot(self, payload: dict):
        self.scene_ref.clear()
        self.agv_items.clear()
        self.shelf_items.clear()
        self.receiver_labels.clear()
        self.wait_zone_items.clear()
        self.receiver_items.clear()
        self.obstacle_items.clear()
        self.safe_path_items.clear()
        self.tile_items.clear()
        self.grid_items.clear()
        self.map_border_items.clear()

        self.map_size = payload.get("map_size", {"width": 0, "height": 0})
        self._redraw_background()

        for point in payload.get("obstacles", []):
            self.obstacle_items.append(self._add_obstacle(point))

        for shelf_id, info in payload.get("boxes", {}).items():
            item = self._add_shelf(int(shelf_id), info["pos"], info.get("size", 1))
            self.shelf_items[int(shelf_id)] = item

        for receiver_id, info in payload.get("receivers", {}).items():
            self.receiver_items.append(self._add_receiver(info["pos"], info.get("size", 1)))
            label = self._add_floating_label(info["pos"], f"R{receiver_id}", "#3b4d8d", top_offset=0.08)
            self.receiver_labels[int(receiver_id)] = label

        for _, info in payload.get("wait_zones", {}).items():
            self.wait_zone_items.append(self._add_wait_zone(info["pos"], info.get("size", 1)))

        for agv_id, info in payload.get("agvs", {}).items():
            item = self._add_agv(int(agv_id), info["pos"], info.get("size", 1))
            self.agv_items[int(agv_id)] = item

        self._refresh_label_visibility()
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def update_snapshot(self, payload: dict):
        agvs = payload.get("agvs", {})
        for agv_id, center in agvs.items():
            agv_id = int(agv_id)
            if agv_id in self.agv_items:
                self._move_group_to_center(self.agv_items[agv_id]["group"], center)

        for shelf_id, point in payload.get("boxes_on_shelf", {}).items():
            shelf_id = int(shelf_id)
            item = self.shelf_items.get(shelf_id)
            if item:
                item["group"].setVisible(True)
                self._move_group_to_center(item["group"], point)

        carrying = {int(box_id): pos for box_id, pos in payload.get("boxes_on_agv", {}).items()}
        carrying_agvs = set()
        for shelf_id, center in carrying.items():
            for agv_id, agv_item in self.agv_items.items():
                agv_center = self._group_center(agv_item["group"])
                if self._same_cell(agv_center, center):
                    self._set_agv_cargo_visible(agv_item, True)
                    carrying_agvs.add(agv_id)
                    break
            if shelf_id in self.shelf_items:
                self.shelf_items[shelf_id]["group"].setVisible(False)

        for agv_id, agv_item in self.agv_items.items():
            if agv_id not in carrying_agvs:
                self._set_agv_cargo_visible(agv_item, False)

        self._update_safe_paths(payload.get("safe_paths", {}))

    def mark_faults(self, faulty_ids: set[int]):
        for agv_id, item in self.agv_items.items():
            is_fault = agv_id in faulty_ids and self.display_options.get("highlight_faults", True)
            body_color = QColor("#dd5f4d") if is_fault else QColor("#2f68ff")
            accent_color = QColor("#772a23") if is_fault else QColor("#173582")
            item["body"].setBrush(QBrush(body_color))
            item["arrow"].setBrush(QBrush(accent_color))
            item["frame"].setPen(QPen(QColor("#5f1717") if is_fault else QColor("#0f172a"), 1.5))
            item["halo"].setBrush(QBrush(QColor(255, 107, 107, 75) if is_fault else QColor(255, 255, 255, 38)))

    def highlight_agv(self, agv_id: int | None):
        for current_id, item in self.agv_items.items():
            is_selected = agv_id is not None and current_id == agv_id
            item["frame"].setPen(QPen(QColor("#0ea5e9") if is_selected else QColor("#0f172a"), 2.4 if is_selected else 1.2))
            item["halo"].setBrush(QBrush(QColor(56, 189, 248, 95) if is_selected else QColor(255, 255, 255, 38)))

    def focus_on_agv(self, agv_id: int):
        item = self.agv_items.get(agv_id)
        if not item:
            return
        self.highlight_agv(agv_id)
        self.centerOn(item["group"])
        self.selected_entity = agv_id
        self.selected_entity_kind = "agv"

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        item = self.scene_ref.itemAt(scene_pos, self.transform())
        selected = self._resolve_selected_entity(item)
        if selected is not None:
            kind, payload = selected
            self.selected_entity_kind = kind
            self.selected_entity = payload.get("id")
            if kind == "agv":
                self.focus_on_agv(payload["id"])
            if callable(self.on_entity_selected):
                self.on_entity_selected(kind, payload)
        super().mousePressEvent(event)

    def _resolve_selected_entity(self, item):
        current = item
        while current is not None:
            for agv_id, agv_item in self.agv_items.items():
                if current in agv_item.values():
                    return ("agv", self._build_agv_payload(agv_id))
            for shelf_id, shelf_item in self.shelf_items.items():
                if current in shelf_item.values():
                    return ("shelf", self._build_shelf_payload(shelf_id))
            current = current.parentItem() if hasattr(current, "parentItem") else None
        return None

    def _build_agv_payload(self, agv_id: int) -> dict:
        item = self.agv_items.get(agv_id, {})
        return {
            "id": agv_id,
            "type": "agv",
            "position": self._group_center(item["group"]) if item else None,
            "is_fault": False,
        }

    def _build_shelf_payload(self, shelf_id: int) -> dict:
        item = self.shelf_items.get(shelf_id, {})
        return {
            "id": shelf_id,
            "type": "shelf",
            "position": self._group_center(item["group"]) if item else None,
        }

    def _redraw_background(self):
        for item in self.tile_items + self.grid_items:
            self.scene_ref.removeItem(item)
        self.tile_items.clear()
        self.grid_items.clear()

        width = self.map_size.get("width", 0)
        height = self.map_size.get("height", 0)
        self.setSceneRect(0, 0, width * self.CELL, height * self.CELL)

        for x in range(width):
            for y in range(height):
                rect = QGraphicsRectItem(x * self.CELL, y * self.CELL, self.CELL, self.CELL)
                color = "#e7e9ff" if (x + y) % 2 == 0 else "#dfe8ff"
                if x in (0, width - 1):
                    color = "#d5f3f7"
                rect.setBrush(QBrush(QColor(color)))
                rect.setPen(QPen(Qt.NoPen))
                rect.setZValue(-50)
                self.scene_ref.addItem(rect)
                self.tile_items.append(rect)

        border = QGraphicsRectItem(0, 0, width * self.CELL, height * self.CELL)
        border.setBrush(QBrush(Qt.NoBrush))
        border.setPen(QPen(QColor("#c8d8ef"), 1.4))
        border.setZValue(-30)
        self.scene_ref.addItem(border)
        self.map_border_items.append(border)

        if not self.display_options.get("show_grid", True):
            return

        pen = QPen(QColor(255, 255, 255, 210))
        pen.setWidth(1)
        pen.setCosmetic(True)
        for x in range(width + 1):
            line = self.scene_ref.addLine(x * self.CELL, 0, x * self.CELL, height * self.CELL, pen)
            line.setZValue(-40)
            self.grid_items.append(line)
        for y in range(height + 1):
            line = self.scene_ref.addLine(0, y * self.CELL, width * self.CELL, y * self.CELL, pen)
            line.setZValue(-40)
            self.grid_items.append(line)

    def _add_agv(self, agv_id: int, center: Tuple[float, float], size: float):
        group = QGraphicsItemGroup()
        self.scene_ref.addItem(group)

        pixel_size = self.CELL * max(0.64, size * 0.76)

        halo = QGraphicsEllipseItem(-pixel_size * 0.50, -pixel_size * 0.50, pixel_size, pixel_size)
        halo.setBrush(QBrush(QColor(32, 95, 180, 24)))
        halo.setPen(QPen(Qt.NoPen))
        group.addToGroup(halo)

        body_path = QPainterPath()
        body_path.addRoundedRect(-pixel_size * 0.33, -pixel_size * 0.24, pixel_size * 0.66, pixel_size * 0.48, 6, 6)
        body = QGraphicsPathItem(body_path)
        body.setBrush(QBrush(QColor("#2f68ff")))
        body.setPen(QPen(QColor("#173582"), 1.2))
        group.addToGroup(body)

        frame = QGraphicsRectItem(-pixel_size * 0.36, -pixel_size * 0.28, pixel_size * 0.72, pixel_size * 0.56)
        frame.setBrush(QBrush(Qt.NoBrush))
        frame.setPen(QPen(QColor("#0f172a"), 1.2))
        group.addToGroup(frame)

        top_plate = QGraphicsRectItem(-pixel_size * 0.14, -pixel_size * 0.15, pixel_size * 0.28, pixel_size * 0.30)
        top_plate.setBrush(QBrush(QColor("#d6ebff")))
        top_plate.setPen(QPen(QColor("#8bb9e8"), 1))
        group.addToGroup(top_plate)

        arrow = QGraphicsPolygonItem(
            QPolygonF([
                QPointF(0, -pixel_size * 0.28),
                QPointF(pixel_size * 0.13, -pixel_size * 0.06),
                QPointF(-pixel_size * 0.13, -pixel_size * 0.06),
            ])
        )
        arrow.setBrush(QBrush(QColor("#173582")))
        arrow.setPen(QPen(Qt.NoPen))
        group.addToGroup(arrow)

        for x_offset in (-0.34, 0.34):
            wheel = QGraphicsRectItem(pixel_size * x_offset - pixel_size * 0.05, -pixel_size * 0.20, pixel_size * 0.10, pixel_size * 0.40)
            wheel.setBrush(QBrush(QColor("#1e293b")))
            wheel.setPen(QPen(Qt.NoPen))
            group.addToGroup(wheel)

        cargo = QGraphicsRectItem(-pixel_size * 0.11, -pixel_size * 0.04, pixel_size * 0.22, pixel_size * 0.17)
        cargo.setBrush(QBrush(QColor("#f59e0b")))
        cargo.setPen(QPen(QColor("#9a6700"), 1))
        cargo.setVisible(False)
        group.addToGroup(cargo)

        label = QGraphicsSimpleTextItem(f"AGV {agv_id}")
        label.setBrush(QBrush(QColor("#1d4ed8")))
        label.setPos(-label.boundingRect().width() / 2, pixel_size * 0.34)
        label.setZValue(2)
        group.addToGroup(label)

        group.setZValue(10)
        self._move_group_to_center(group, center)
        return {
            "group": group,
            "body": body,
            "frame": frame,
            "arrow": arrow,
            "cargo": cargo,
            "halo": halo,
            "label": label,
        }

    def _add_shelf(self, shelf_id: int, center: Tuple[float, float], size: float):
        group = QGraphicsItemGroup()
        self.scene_ref.addItem(group)
        pixel_size = self.CELL * size * 0.86

        shadow = QGraphicsRectItem(-pixel_size * 0.50 + 2, -pixel_size * 0.38 + 2, pixel_size, pixel_size * 0.76)
        shadow.setBrush(QBrush(QColor(0, 0, 0, 20)))
        shadow.setPen(QPen(Qt.NoPen))
        group.addToGroup(shadow)

        frame = QGraphicsRectItem(-pixel_size * 0.50, -pixel_size * 0.38, pixel_size, pixel_size * 0.76)
        frame.setBrush(QBrush(QColor("#ffca70")))
        frame.setPen(QPen(QColor("#121723"), 1.8))
        group.addToGroup(frame)

        inner = QGraphicsRectItem(-pixel_size * 0.40, -pixel_size * 0.29, pixel_size * 0.80, pixel_size * 0.58)
        inner.setBrush(QBrush(QColor("#f59e0b")))
        inner.setPen(QPen(QColor("#8c4b00"), 1.2))
        group.addToGroup(inner)

        for ratio in (-0.15, 0.15):
            line = QGraphicsRectItem(-pixel_size * 0.40, pixel_size * ratio, pixel_size * 0.80, pixel_size * 0.03)
            line.setBrush(QBrush(QColor("#ffe1af")))
            line.setPen(QPen(Qt.NoPen))
            group.addToGroup(line)

        for ratio in (-0.18, 0.18):
            line = QGraphicsRectItem(pixel_size * ratio, -pixel_size * 0.30, pixel_size * 0.03, pixel_size * 0.60)
            line.setBrush(QBrush(QColor("#ffe1af")))
            line.setPen(QPen(Qt.NoPen))
            group.addToGroup(line)

        for x_sign in (-1, 1):
            for y_sign in (-1, 1):
                brace = self.scene_ref.addLine(
                    0,
                    0,
                    x_sign * pixel_size * 0.14,
                    y_sign * pixel_size * 0.14,
                    QPen(QColor("#121723"), 1.6),
                )
                brace.setParentItem(group)
                brace.setPos(x_sign * pixel_size * 0.50, y_sign * pixel_size * 0.38)

        label = QGraphicsSimpleTextItem(f"S{shelf_id}")
        label.setBrush(QBrush(QColor("#855400")))
        label.setPos(-label.boundingRect().width() / 2, pixel_size * 0.18)
        group.addToGroup(label)

        group.setZValue(4)
        self._move_group_to_center(group, center)
        return {"group": group, "label": label}

    def _add_receiver(self, center: Tuple[float, float], size: float):
        pixel_size = self.CELL * size
        group = QGraphicsItemGroup()
        self.scene_ref.addItem(group)
        base = QGraphicsRectItem(-pixel_size / 2, -pixel_size / 2, pixel_size, pixel_size)
        base.setBrush(QBrush(QColor(163, 181, 255, 44)))
        base.setPen(QPen(QColor("#8ba0ff"), 1.0))
        group.addToGroup(base)
        stripe = QGraphicsRectItem(-pixel_size / 2, -pixel_size * 0.10, pixel_size, pixel_size * 0.20)
        stripe.setBrush(QBrush(QColor("#7c4ee4")))
        stripe.setPen(QPen(Qt.NoPen))
        group.addToGroup(stripe)
        self._move_group_to_center(group, center)
        group.setZValue(1)
        return group

    def _add_wait_zone(self, center: Tuple[float, float], size: float):
        pixel_size = self.CELL * size
        group = QGraphicsItemGroup()
        self.scene_ref.addItem(group)
        outline = QGraphicsRectItem(-pixel_size / 2, -pixel_size / 2, pixel_size, pixel_size)
        outline.setBrush(QBrush(QColor(46, 184, 196, 22)))
        outline.setPen(QPen(QColor("#59b8c3"), 1.0, Qt.DashLine))
        group.addToGroup(outline)
        self._move_group_to_center(group, center)
        group.setZValue(0)
        return group

    def _add_obstacle(self, center: Tuple[float, float]):
        pixel_size = self.CELL * 0.86
        group = QGraphicsItemGroup()
        self.scene_ref.addItem(group)
        body = QGraphicsRectItem(-pixel_size / 2, -pixel_size / 2, pixel_size, pixel_size)
        body.setBrush(QBrush(QColor("#5b6475")))
        body.setPen(QPen(QColor("#384152"), 1.2))
        group.addToGroup(body)
        for ratio in (-0.18, 0.18):
            line = QGraphicsRectItem(-pixel_size / 2, ratio * pixel_size, pixel_size, pixel_size * 0.10)
            line.setBrush(QBrush(QColor("#6d778b")))
            line.setPen(QPen(Qt.NoPen))
            group.addToGroup(line)
        self._move_group_to_center(group, center)
        group.setZValue(3)
        return group

    def _add_floating_label(self, center: Tuple[float, float], text: str, color: str, top_offset: float):
        label = QGraphicsSimpleTextItem(text)
        label.setBrush(QBrush(QColor(color)))
        self.scene_ref.addItem(label)
        self._move_label_to_center(label, (center[0], center[1] + top_offset))
        label.setZValue(6)
        return label

    def _update_safe_paths(self, safe_paths: dict):
        self._last_safe_paths = safe_paths
        for item in self.safe_path_items:
            self.scene_ref.removeItem(item)
        self.safe_path_items.clear()

        if not self.display_options.get("show_paths", True):
            return

        for _, points in safe_paths.items():
            if not points:
                continue
            path = QPainterPath()
            start = self._to_scene_point(points[0])
            path.moveTo(start)
            for point in points[1:]:
                path.lineTo(self._to_scene_point(point))
            path_item = QGraphicsPathItem(path)
            pen = QPen(QColor(255, 149, 0, 118))
            pen.setWidth(4)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            path_item.setPen(pen)
            path_item.setZValue(0)
            self.scene_ref.addItem(path_item)
            self.safe_path_items.append(path_item)

    def _refresh_label_visibility(self):
        for item in self.agv_items.values():
            item["label"].setVisible(self.display_options.get("show_agv_id", True))
        for item in self.shelf_items.values():
            item["label"].setVisible(self.display_options.get("show_box_id", False))
        for label in self.receiver_labels.values():
            label.setVisible(self.display_options.get("show_receiver_id", False))

    def _move_group_to_center(self, group: QGraphicsItemGroup, center: Tuple[float, float]):
        point = self._to_scene_point(center)
        group.setPos(point)

    def _move_label_to_center(self, label: QGraphicsSimpleTextItem, center: Tuple[float, float]):
        point = self._to_scene_point(center)
        label.setPos(point.x() - label.boundingRect().width() / 2, point.y() - label.boundingRect().height() / 2)

    def _group_center(self, group: QGraphicsItemGroup) -> Tuple[float, float]:
        return self._to_grid_center(group.pos())

    def _set_agv_cargo_visible(self, agv_item: dict, visible: bool):
        agv_item["cargo"].setVisible(visible)

    def _same_cell(self, left: Tuple[float, float], right: Tuple[float, float]) -> bool:
        return round(left[0], 1) == round(right[0], 1) and round(left[1], 1) == round(right[1], 1)

    def _to_scene_point(self, center: Tuple[float, float]) -> QPointF:
        return QPointF(center[0] * self.CELL, (self.map_size["height"] - center[1]) * self.CELL)

    def _to_grid_center(self, point: QPointF) -> Tuple[float, float]:
        return (round(point.x() / self.CELL, 3), round(self.map_size["height"] - point.y() / self.CELL, 3))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.map_size.get("width") and self.map_size.get("height"):
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
