import { AGV } from './entities/agv.js';
import { Shelf } from './entities/shelf.js';
import { Box } from './entities/box.js';
import { Obstacle } from './entities/obstacle.js';
import { RestArea } from './entities/restArea.js';
import { ReceiveArea } from './entities/receiveArea.js';
import { updateMetrics } from './panel.js';
import { updateOrderPanel } from './orderPanel.js';

let ws = null;

function connectWebSocket(world) {
  ws = new WebSocket("ws://localhost:8765");

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log("get data: ", data);

    if (data.type === "reset") {
      // 后端进入重置流程时先清空旧场景，避免新 init 叠加到旧对象上。
      world.clear();
      updateOrderPanel([]);
      return;
    }

    if (data.type === "init") {
      world.addMap(data.map_size);

      if (data.boxes) {
        for (const boxId in data.boxes) {
          const box = data.boxes[boxId];
          const pos = box.pos;
          const size = box.size;
          world.addBox(new Box(parseInt(boxId), pos, size));
          world.addShelf(new Shelf(parseInt(boxId), pos, size));
        }
      }

      if (data.receivers) {
        for (const rid in data.receivers) {
          const receiver = data.receivers[rid];
          world.addReceiveArea(new ReceiveArea(rid, receiver.pos, receiver.size));
        }
      }

      if (data.agvs) {
        for (const agvId in data.agvs) {
          const agv = data.agvs[agvId];
          world.addAGV(new AGV(parseInt(agvId), agv.pos, agv.size));
        }
      }

      if (data.wait_zones) {
        for (const key in data.wait_zones) {
          const wait_zone = data.wait_zones[key];
          world.addRestArea(new RestArea(wait_zone.pos, wait_zone.size));
        }
      }

      if (data.obstacles) {
        data.obstacles.forEach(pos => {
          world.addObstacle(new Obstacle(pos));
        });
      }
    }

    if (data.type === "update") {
      if (data.agvs) {
        for (const key in data.agvs) {
          const pos = data.agvs[key];
          const agv = world.agvs.get(parseInt(key));
          if (agv) agv.update(pos);
        }
      }

      if (data.boxes_on_agv) {
        for (const [boxId, pos] of Object.entries(data.boxes_on_agv)) {
          const box = world.boxes.get(parseInt(boxId));
          if (box) box.update(pos, 0.55);
        }
      }

      if (data.boxes_on_shelf) {
        for (const [boxId, pos] of Object.entries(data.boxes_on_shelf)) {
          const box = world.boxes.get(parseInt(boxId));
          if (box) box.update(pos, 0.7);
        }
      }

      if (data.safe_paths) {
        world.safePathRenderer.updatePaths(data.safe_paths);
      }

      if (data.metrics) {
        updateMetrics(data.metrics);
      }

      if (data.orders) {
        updateOrderPanel(data.orders);
      }
    }

    if (data.type === "init" && data.orders) {
      updateOrderPanel(data.orders);
    }
  };
}

export { connectWebSocket, ws };
