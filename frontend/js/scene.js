// scene.js
import { SafePathRenderer } from './entities/safePathRenderer.js';
import { OrbitControls } from "https://unpkg.com/three@0.112/examples/jsm/controls/OrbitControls.js";
import * as THREE from 'three';
import { CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

function createScene() {
  // ---------------- Scene & Renderer ----------------
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x141319);

  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(25, 8, 20);
  camera.lookAt(15, 0, 15);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.getElementById('container').appendChild(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.domElement.style.position = 'absolute';
  labelRenderer.domElement.style.top = '0';
  labelRenderer.domElement.style.pointerEvents = 'none';
  document.getElementById('container').appendChild(labelRenderer.domElement);

  // ---------------- OrbitControls ----------------
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = false;
  controls.minDistance = 10;
  controls.maxDistance = 100;
  controls.maxPolarAngle = Math.PI / 2;

  // ---------------- Lighting ----------------
  const light = new THREE.DirectionalLight(0xffffff, 1);
  light.position.set(10, 20, 10);
  scene.add(light);

  const ambient = new THREE.AmbientLight(0xaaaaaa, 0.5);
  scene.add(ambient);

  // ---------------- Helpers ----------------
  const axesHelper = new THREE.AxesHelper(30);
  scene.add(axesHelper);

  // ---------------- World Container ----------------
  const world = {
    scene,
    mapSize: null,
    mapGrid: null,
    floor: null,
    agvs: new Map(),
    shelves: new Map(),
    boxes: new Map(),
    obstacles: new Map(),
    restAreas: new Map(),
    receiveAreas: new Map(),

    clear() {
      if (this.mapGrid) {
        this.scene.remove(this.mapGrid);
        this.mapGrid = null;
      }
      if (this.floor) {
        this.scene.remove(this.floor);
        this.floor = null;
      }

      for (const agv of this.agvs.values()) {
        this.scene.remove(agv.mesh);
      }
      for (const shelf of this.shelves.values()) {
        this.scene.remove(shelf.mesh);
      }
      for (const box of this.boxes.values()) {
        this.scene.remove(box.mesh);
      }
      for (const obstacle of this.obstacles.values()) {
        this.scene.remove(obstacle.mesh);
      }
      for (const restArea of this.restAreas.values()) {
        this.scene.remove(restArea.mesh);
      }
      for (const receiveArea of this.receiveAreas.values()) {
        this.scene.remove(receiveArea.mesh);
      }

      this.agvs.clear();
      this.shelves.clear();
      this.boxes.clear();
      this.obstacles.clear();
      this.restAreas.clear();
      this.receiveAreas.clear();
      this.mapSize = null;

      if (this.safePathRenderer) {
        this.safePathRenderer.updatePaths({});
      }
    },

    addMap(mapSize) {
      // 新 init 进来时先清空旧场景，保证 reset 幂等。
      this.clear();

      this.mapSize = mapSize;
      const grid = new THREE.Group();
      const material = new THREE.LineBasicMaterial({ color: 0xffffff });

      for (let x = 0; x <= mapSize.width; x++) {
        const points = [new THREE.Vector3(x, 0.01, 0), new THREE.Vector3(x, 0.01, mapSize.height)];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        grid.add(new THREE.Line(geometry, material));
      }

      for (let z = 0; z <= mapSize.height; z++) {
        const points = [new THREE.Vector3(0, 0.01, z), new THREE.Vector3(mapSize.width, 0.01, z)];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        grid.add(new THREE.Line(geometry, material));
      }

      this.scene.add(grid);
      this.mapGrid = grid;

      const geometry = new THREE.PlaneGeometry(mapSize.width, mapSize.height);
      const materialFloor = new THREE.MeshPhongMaterial({ color: 0xaca7fb });
      const floor = new THREE.Mesh(geometry, materialFloor);
      floor.rotation.x = -Math.PI / 2;
      floor.position.x = mapSize.width / 2;
      floor.position.z = mapSize.height / 2;
      this.scene.add(floor);
      this.floor = floor;
    },

    addAGV(agv) {
      this.agvs.set(agv.id, agv);
      this.scene.add(agv.mesh);
    },

    addShelf(shelf) {
      this.shelves.set(shelf.id, shelf);
      this.scene.add(shelf.mesh);
    },

    addBox(box) {
      this.boxes.set(box.id, box);
      this.scene.add(box.mesh);
    },

    addObstacle(obstacle, key = null) {
      const id = key || `${obstacle.mesh.position.x},${obstacle.mesh.position.z}`;
      this.obstacles.set(id, obstacle);
      this.scene.add(obstacle.mesh);
    },

    addRestArea(restArea, key = null) {
      const id = key || `${restArea.mesh.position.x},${restArea.mesh.position.z}`;
      this.restAreas.set(id, restArea);
      this.scene.add(restArea.mesh);
    },

    addReceiveArea(receiveArea, key = null) {
      const id = key || `${receiveArea.mesh.position.x},${receiveArea.mesh.position.z}`;
      this.receiveAreas.set(id, receiveArea);
      this.scene.add(receiveArea.mesh);
    }
  };

  world.safePathRenderer = new SafePathRenderer(scene);

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    labelRenderer.setSize(window.innerWidth, window.innerHeight);
  });

  window.sceneWorld = world;

  return { scene, camera, renderer, world, controls, labelRenderer };
}

function renderLoop(renderer, labelRenderer, scene, camera, controls) {
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  }
  animate();
}

export { createScene, renderLoop };
