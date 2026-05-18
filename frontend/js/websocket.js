import { AGV } from './entities/agv.js?v=20260512b';
import { Box } from './entities/box.js?v=20260512b';
import { Obstacle } from './entities/obstacle.js?v=20260512b';
import { ReceiveArea } from './entities/receiveArea.js?v=20260512b';
import { RestArea } from './entities/restArea.js?v=20260512b';
import { Shelf } from './entities/shelf.js?v=20260512b';

let ws = null;

function connectWebSocket(world, uiHandlers = {}) {
  const {
    updateAlgorithmConfig = () => {},
    updateMetrics = () => {},
    updateOrderPanel = () => {}
  } = uiHandlers;

  ws = new WebSocket('ws://localhost:8765');
  window.appSocket = ws;

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('get data: ', data);

    if (data.type === 'reset') {
      world.clear();
      updateOrderPanel([]);
      return;
    }

    if (data.type === 'algorithm_config') {
      updateAlgorithmConfig(data);
      return;
    }

    if (data.type === 'init') {
      if (data.algorithm_config) {
        updateAlgorithmConfig(data.algorithm_config);
      }

      world.addMap(data.map_size);

      if (data.boxes) {
        for (const boxId in data.boxes) {
          const box = data.boxes[boxId];
          world.addBox(new Box(parseInt(boxId, 10), box.pos, box.size));
          world.addShelf(new Shelf(parseInt(boxId, 10), box.pos, box.size));
        }
      }

      if (data.receivers) {
        for (const receiverId in data.receivers) {
          const receiver = data.receivers[receiverId];
          world.addReceiveArea(new ReceiveArea(receiverId, receiver.pos, receiver.size));
        }
      }

      if (data.agvs) {
        for (const agvId in data.agvs) {
          const agv = data.agvs[agvId];
          world.addAGV(new AGV(parseInt(agvId, 10), agv.pos, agv.size));
        }
      }

      if (data.wait_zones) {
        for (const key in data.wait_zones) {
          const waitZone = data.wait_zones[key];
          world.addRestArea(new RestArea(waitZone.pos, waitZone.size));
        }
      }

      if (data.obstacles) {
        data.obstacles.forEach((pos) => {
          world.addObstacle(new Obstacle(pos));
        });
      }

      if (data.orders) {
        updateOrderPanel(data.orders);
      }
      return;
    }

    if (data.type === 'update') {
      if (data.agvs) {
        for (const key in data.agvs) {
          const agv = world.agvs.get(parseInt(key, 10));
          if (agv) agv.update(data.agvs[key]);
        }
      }

      if (data.boxes_on_agv) {
        for (const [boxId, pos] of Object.entries(data.boxes_on_agv)) {
          const box = world.boxes.get(parseInt(boxId, 10));
          if (box) box.update(pos, 0.55);
        }
      }

      if (data.boxes_on_shelf) {
        for (const [boxId, pos] of Object.entries(data.boxes_on_shelf)) {
          const box = world.boxes.get(parseInt(boxId, 10));
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
  };
}

export { connectWebSocket, ws };
