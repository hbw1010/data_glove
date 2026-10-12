import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// MediaPipe HAND_CONNECTIONS: palm outline plus the five finger chains.
const HAND_CONNECTIONS = [
  [0, 1], [0, 5], [9, 13], [13, 17], [5, 9], [0, 17],
  [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20],
];

function dispose(object) {
  object.traverse((node) => {
    node.geometry?.dispose();
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    for (const material of materials) {
      if (!material) continue;
      for (const value of Object.values(material)) if (value?.isTexture) value.dispose();
      material.dispose();
    }
  });
}

function origin(object, xyz = [0, 0, 0], rpy = [0, 0, 0]) {
  object.position.fromArray(xyz);
  // URDF fixed-axis roll/pitch/yaw is Rz(yaw) Ry(pitch) Rx(roll).
  object.quaternion.setFromEuler(new THREE.Euler(...rpy, 'ZYX'));
}

async function visualObject(visual) {
  const rgba = visual.color ?? [0.65, 0.75, 0.85, 1];
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
    opacity: rgba[3] ?? 1, transparent: (rgba[3] ?? 1) < 1,
    roughness: 0.64, metalness: 0.12, side: THREE.DoubleSide,
  });
  let object;
  try {
    if (visual.mesh) {
      const url = new URL(visual.mesh, location.origin);
      if (url.origin !== location.origin || !url.pathname.startsWith('/assets/')) {
        throw new Error('拒绝非本项目模型资产地址');
      }
      const ext = url.pathname.split('.').pop().toLowerCase();
      if (ext === 'stl') object = new THREE.Mesh(await new STLLoader().loadAsync(url.href), material);
      else if (ext === 'obj') object = await new OBJLoader().loadAsync(url.href);
      else if (ext === 'dae') object = (await new ColladaLoader().loadAsync(url.href)).scene;
      else if (ext === 'gltf' || ext === 'glb') object = (await new GLTFLoader().loadAsync(url.href)).scene;
      else throw new Error(`不支持的网格格式：${ext}`);
      object.traverse((node) => {
        if (node.isMesh && node.material !== material) {
          const old = Array.isArray(node.material) ? node.material : [node.material];
          old.forEach((entry) => entry.dispose());
          node.material = material;
        }
      });
    } else {
      const geometry = visual.geometry;
      let meshGeometry;
      if (geometry?.type === 'box') meshGeometry = new THREE.BoxGeometry(...geometry.size);
      else if (geometry?.type === 'sphere') meshGeometry = new THREE.SphereGeometry(geometry.radius, 24, 16);
      else if (geometry?.type === 'cylinder') {
        meshGeometry = new THREE.CylinderGeometry(geometry.radius, geometry.radius, geometry.length, 24);
        meshGeometry.rotateX(Math.PI / 2);
      } else throw new Error(`不支持的基础几何：${geometry?.type ?? '未定义'}`);
      object = new THREE.Mesh(meshGeometry, material);
    }
    const frame = new THREE.Group();
    origin(frame, visual.xyz, visual.rpy);
    object.scale.fromArray(visual.scale ?? [1, 1, 1]);
    frame.add(object);
    return frame;
  } catch (error) {
    if (object) dispose(object);
    else material.dispose();
    throw error;
  }
}


export class ModelView {
  constructor(container, status) {
    this.container = container;
    this.status = status;
    this.generation = 0;
    this.root = null;
    this.joints = new Map();
    this.modelError = '';
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#101c2d');
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.001, 1000);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.append(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xe6f3ff, 0x445061, 2.6));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(1, -1, 2);
    this.scene.add(light);
    this.axes = new THREE.AxesHelper(0.05);
    this.scene.add(this.axes);
    this.renderFrame = null;
    this.inViewport = false;
    this.controls.addEventListener('change', () => this.invalidate());
    this.reset();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.intersectionObserver = new IntersectionObserver(([entry]) => {
      this.inViewport = entry.isIntersecting;
      if (this.inViewport) this.invalidate();
    });
    this.intersectionObserver.observe(container);
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) this.invalidate();
    });
  }

  invalidate() {
    if (this.renderFrame !== null || !this.inViewport || document.hidden) return;
    this.renderFrame = requestAnimationFrame(() => {
      this.renderFrame = null;
      if (!this.inViewport || document.hidden || !this.container.clientWidth || !this.container.clientHeight) return;
      // OrbitControls emits change while damping settles, scheduling only the next needed frame.
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  resize() {
    const { clientWidth: width, clientHeight: height } = this.container;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  reset() {
    let center = new THREE.Vector3();
    let radius = 0.18;
    if (this.root) {
      this.root.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(this.root);
      if (!box.isEmpty()) {
        center = box.getCenter(center);
        radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 0.04);
      }
    }
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(1, -1.6, 1).normalize().multiplyScalar(radius * 3.8));
    this.camera.near = Math.max(radius / 1000, 0.0001);
    this.camera.far = Math.max(radius * 100, 10);
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.invalidate();
  }

  clear(message = '未加载') {
    ++this.generation;
    if (this.root) {
      this.scene.remove(this.root);
      dispose(this.root);
    }
    this.root = null;
    this.joints.clear();
    this.modelError = '';
    this.key = null;
    this.pending = null;
    this.setStatus(message, false);
    this.invalidate();
  }

  setStatus(message, invalid) {
    this.status.textContent = message;
    this.status.classList.toggle('invalid', invalid);
  }

  hide(reason) {
    if (this.root?.visible) {
      this.root.visible = false;
      this.invalidate();
    }
    this.setStatus(this.modelError && this.modelError !== reason ? `${reason} · ${this.modelError}` : reason, true);
  }

  async load(hand, kind) {
    const key = `${hand}/${kind}`;
    if (this.pending?.key === key) return this.pending.promise;
    const pending = { key, promise: this.loadModel(hand, kind) };
    this.pending = pending;
    try { return await pending.promise; }
    finally { if (this.pending === pending) this.pending = null; }
  }

  async loadModel(hand, kind) {
    const key = `${hand}/${kind}`;
    if (this.key === key && this.root && !this.modelError) return true;
    this.clear('正在加载本地模型…');
    this.key = key;
    const generation = this.generation;
    const root = new THREE.Group();
    try {
      const response = await fetch(`/api/models/${encodeURIComponent(hand)}/${kind}`);
      const tree = await response.json();
      if (!response.ok) throw new Error(tree.detail ?? '模型接口请求失败');
      const links = new Map(tree.links.map((link) => [link.name, new THREE.Group()]));
      const joints = new Map();
      if (!links.has(tree.root)) throw new Error('模型缺少根链接');
      root.add(links.get(tree.root));
      for (const joint of tree.joints) {
        if (!links.has(joint.parent) || !links.has(joint.child)) throw new Error(`关节 ${joint.name} 的链接缺失`);
        const fixed = new THREE.Group();
        origin(fixed, joint.xyz, joint.rpy);
        const motion = new THREE.Group();
        fixed.add(motion);
        motion.add(links.get(joint.child));
        links.get(joint.parent).add(fixed);
        joints.set(joint.name, { object: motion, type: joint.type, axis: new THREE.Vector3(...(joint.axis ?? [1, 0, 0])).normalize() });
      }
      // Settle all mesh loads before disposal so a stale request cannot attach leaked assets later.
      const loaded = await Promise.allSettled(tree.links.flatMap((link) => link.visuals.map(async (visual) => {
        const object = await visualObject(visual);
        links.get(link.name).add(object);
      })));
      const failed = loaded.find((entry) => entry.status === 'rejected');
      if (failed) throw failed.reason;
      if (generation !== this.generation) { dispose(root); return false; }
      this.root = root;
      this.joints = joints;
      root.visible = false;
      this.scene.add(root);
      this.reset();
      this.setStatus('模型已加载，等待本帧数据', false);
      return true;
    } catch (error) {
      dispose(root);
      if (generation === this.generation) {
        this.modelError = `模型加载失败：${error.message}`;
        this.hide(this.modelError);
      }
      return false;
    }
  }

  pose(names, values) {
    if (!this.root) { this.hide(this.modelError || '模型尚未就绪'); return; }
    if (!Array.isArray(values) || values.length !== names.length || !values.every(Number.isFinite)) {
      this.hide('本帧关节数据缺失或包含非有限数值'); return;
    }
    const movable = [...this.joints].filter(([, joint]) => joint.type !== 'fixed');
    const nameSet = new Set(names);
    if (nameSet.size !== names.length || movable.length !== names.length || names.some((name) => !this.joints.has(name)) || movable.some(([name]) => !nameSet.has(name))) {
      this.hide('结果关节名称与当前手侧模型不匹配'); return;
    }
    names.forEach((name, index) => {
      const joint = this.joints.get(name);
      if (joint.type === 'revolute' || joint.type === 'continuous') joint.object.quaternion.setFromAxisAngle(joint.axis, values[index]);
      else if (joint.type === 'prismatic') joint.object.position.copy(joint.axis).multiplyScalar(values[index]);
    });
    this.root.visible = true;
    this.setStatus(`有效 · ${names.length} 关节`, false);
    this.invalidate();
  }

  skeleton(points) {
    if (!Array.isArray(points) || points.length !== 21 || !points.every((point) => Array.isArray(point) && point.length === 3 && point.every(Number.isFinite))) {
      this.hide('本帧关键点数据缺失或包含非有限数值'); return;
    }
    let first = false;
    if (this.key !== 'skeleton') {
      this.clear();
      this.key = 'skeleton';
      first = true;
      this.root = new THREE.Group();
      const geometry = new THREE.SphereGeometry(0.0025, 12, 8);
      this.pointObjects = points.map((_, index) => {
        const finger = Math.max(0, Math.floor((index - 1) / 4));
        const colors = [0xfbbf24, 0x38bdf8, 0xa78bfa, 0x34d399, 0xfb7185];
        const point = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: index === 0 ? 0xffffff : colors[finger] }));
        this.root.add(point);
        return point;
      });
      this.lineGeometry = new THREE.BufferGeometry();
      this.lineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(HAND_CONNECTIONS.length * 2 * 3), 3));
      this.root.add(new THREE.LineSegments(this.lineGeometry, new THREE.LineBasicMaterial({ color: 0xd8e8ff })));
      this.scene.add(this.root);
    }
    points.forEach((point, index) => this.pointObjects[index].position.fromArray(point));
    const positions = this.lineGeometry.attributes.position;
    let edge = 0;
    for (const [start, end] of HAND_CONNECTIONS) {
      positions.setXYZ(edge++, ...points[start]);
      positions.setXYZ(edge++, ...points[end]);
    }
    positions.needsUpdate = true;
    this.lineGeometry.computeBoundingSphere();
    this.root.visible = true;
    this.setStatus('有效 · 21 关键点 · MediaPipe 标准连线', false);
    if (first) this.reset();
    this.invalidate();
  }
}
