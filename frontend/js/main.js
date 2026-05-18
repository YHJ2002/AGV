import { initOrderPanel, updateOrderPanel } from './orderPanel.js?v=20260512b';
import { initPanel, updateAlgorithmConfig, updateMetrics } from './panel.js?v=20260512b';
import { createScene, renderLoop } from './scene.js?v=20260512b';
import { connectWebSocket } from './websocket.js?v=20260512b';

const { scene, camera, renderer, world, controls, labelRenderer } = createScene();

initPanel();
initOrderPanel();

connectWebSocket(world, {
  updateAlgorithmConfig,
  updateMetrics,
  updateOrderPanel
});

renderLoop(renderer, labelRenderer, scene, camera, controls);
