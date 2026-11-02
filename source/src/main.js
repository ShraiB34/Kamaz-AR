import '@google/model-viewer';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import QRCode from 'qrcode';
import { QRCameraTracker } from './qrPoseTracker.js';

// ===========================================================================
// 1. SUBSYSTEM HOTSPOTS METADATA (RUSSIAN)
// ===========================================================================
const HOTSPOTS = [
  {
    id: 'cab',
    num: '1',
    label: 'Шасси КАМАЗ-43118 (6×6)',
    pos: new THREE.Vector3(1.18, 2.05, 3.25),
    normal: new THREE.Vector3(0.45, 0.2, 0.87),
    camPos: new THREE.Vector3(7.8, 3.6, 8.2),
    camTarget: new THREE.Vector3(0.0, 1.8, 1.5),
    kicker: 'Узел 01 • Полноприводная база',
    desc: 'Вездеходное шасси КАМАЗ-43118 с колёсной формулой 6×6, системой централизованной подкачки шин, усиленным бампером и СГУ обеспечивает доставку узла связи в труднодоступные районы ЧС и паводков.',
  },
  {
    id: 'mast',
    num: '2',
    label: 'Телескопическая мачта + PTZ',
    pos: new THREE.Vector3(1.32, 4.95, 1.08),
    normal: new THREE.Vector3(0.0, 1.0, 0.0),
    camPos: new THREE.Vector3(8.2, 5.8, 6.6),
    camTarget: new THREE.Vector3(0.4, 3.4, 0.4),
    kicker: 'Узел 02 • Радиосвязь и видеомониторинг',
    desc: 'Пневматическая телескопическая мачта поднимает купольную PTZ-видеокамеру кругового обзора (360°, оптический зум) и антенны УКВ/ДМВ диапазона для оперативного контроля зоны ЧС.',
  },
  {
    id: 'kung',
    num: '3',
    label: 'Аппаратный КУНГ и трапы',
    pos: new THREE.Vector3(-1.35, 2.15, -1.10),
    normal: new THREE.Vector3(-1.0, 0.1, 0.0),
    camPos: new THREE.Vector3(-8.8, 3.6, 2.4),
    camTarget: new THREE.Vector3(0.0, 1.9, -0.9),
    kicker: 'Узел 03 • Оперативный зал связи',
    desc: 'Утеплённый фургон со скошенными верхними гранями вмещает автоматизированные рабочие места операторов, стойки КВ/УКВ радиостанций, сервер ВКС, цифровую АТС и откидные входные трапы с поручнями.',
  },
  {
    id: 'rack',
    num: '4',
    label: 'Платформа SUPER SKYRIDER',
    pos: new THREE.Vector3(0.0, 3.48, -0.98),
    normal: new THREE.Vector3(0.0, 1.0, 0.0),
    camPos: new THREE.Vector3(6.8, 6.6, 2.2),
    camTarget: new THREE.Vector3(0.0, 2.8, -1.0),
    kicker: 'Узел 04 • Крышевой комплекс и климат',
    desc: 'Экспедиционная платформа SUPER SKYRIDER на рифлёном алюминиевом настиле крыши, ограждения безопасности, штыревые антенны и два климатических/вентиляционных люка жизнеобеспечения.',
  },
  {
    id: 'vsat',
    num: '5',
    label: 'Спутниковая станция iNetVu',
    pos: new THREE.Vector3(0.0, 3.78, -3.62),
    normal: new THREE.Vector3(0.0, 0.85, -0.5),
    camPos: new THREE.Vector3(6.6, 5.8, -7.8),
    camTarget: new THREE.Vector3(0.0, 2.9, -2.6),
    kicker: 'Узел 05 • Спутниковый канал VSAT',
    desc: 'Моторизованная антенная система iNetVu (C-COM Satellite Systems) автоматически наводится на геостационарный спутник за 2–3 минуты, обеспечивая высокоскоростной Интернет, ВКС и закрытую телефонию.',
  },
  {
    id: 'rear',
    num: '6',
    label: 'Агрегатный отсек и аппарель',
    pos: new THREE.Vector3(0.0, 1.65, -4.55),
    normal: new THREE.Vector3(0.0, 0.15, -0.98),
    camPos: new THREE.Vector3(-6.2, 3.4, -9.2),
    camTarget: new THREE.Vector3(0.0, 1.8, -2.6),
    kicker: 'Узел 06 • Энергоснабжение и вводы',
    desc: 'Кормовой технический отсек с распашными дверями, откидной рабочей площадкой-аппарелью, кабельными барабанами, системой автономного электропитания и щитами внешних подключений.',
  },
];

// ===========================================================================
// 2. THREE.JS SCENE, RENDERER & LIGHTING
// ===========================================================================
const viewportEl = document.getElementById('viewport-area');
const threeCanvas = document.getElementById('three-canvas');
const videoEl = document.getElementById('ar-video');
const simBgCanvas = document.getElementById('ar-sim-bg');
const hudCanvas = document.getElementById('ar-hud-canvas');
const hotspotsLayer = document.getElementById('hotspots-layer');

const renderer = new THREE.WebGLRenderer({
  canvas: threeCanvas,
  antialias: true,
  alpha: true,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.02;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();

// Build realistic Outdoor Daylight Sky + Horizon + Asphalt Bounce Equirectangular Environment (PMREM)
function createOutdoorDaylightEnv() {
  const ew = 512, eh = 256;
  const c = document.createElement('canvas');
  c.width = ew;
  c.height = eh;
  const ctx = c.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, eh);
  grad.addColorStop(0.00, '#3b82f6'); // Deep clear sky zenith
  grad.addColorStop(0.36, '#93c5fd'); // Mid sky
  grad.addColorStop(0.49, '#e0f2fe'); // Bright horizon mist
  grad.addColorStop(0.51, '#64748b'); // Distant treeline / concrete horizon
  grad.addColorStop(0.62, '#334155'); // Proving ground tarmac bounce
  grad.addColorStop(1.00, '#1e293b'); // Deep ground shadow
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, ew, eh);

  // Soft sunlit cloud bands & warm solar disk highlight for realistic window & paint reflections!
  const sunGrad = ctx.createRadialGradient(ew * 0.28, eh * 0.24, 2, ew * 0.28, eh * 0.24, 68);
  sunGrad.addColorStop(0.0, 'rgba(255, 253, 240, 1.0)');
  sunGrad.addColorStop(0.25, 'rgba(254, 243, 199, 0.65)');
  sunGrad.addColorStop(1.0, 'rgba(255, 255, 255, 0.0)');
  ctx.fillStyle = sunGrad;
  ctx.fillRect(0, 0, ew, eh * 0.52);

  const tex = new THREE.CanvasTexture(c);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

const pmrem = new THREE.PMREMGenerator(renderer);
const envTex = pmrem.fromEquirectangular(createOutdoorDaylightEnv()).texture;
scene.environment = envTex;
scene.environmentIntensity = 0.88;

// Cameras: orbitCam for 3D mode, arCam for QR-AR mode
const AR_FOV = 54;
const orbitCam = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 150);
orbitCam.position.set(-9.6, 5.0, 9.8);

const arCam = new THREE.PerspectiveCamera(AR_FOV, 16 / 9, 0.04, 80);
arCam.position.set(0, 0, 0);
arCam.rotation.set(0, 0, 0);

const controls = new OrbitControls(orbitCam, threeCanvas);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.target.set(0.0, 2.05, -0.35);
controls.minDistance = 4.0;
controls.maxDistance = 28.0;
controls.maxPolarAngle = Math.PI * 0.492;

// Realistic Outdoor Daylight Lighting (Warm Sun + Cool Sky Hemisphere + Subtle Rim)
const hemiLight = new THREE.HemisphereLight(0xdbeafe, 0x334155, 0.72);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xfffbeb, 2.15);
dirLight.position.set(-9.5, 14.5, 10.5);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.near = 2;
dirLight.shadow.camera.far = 36;
const sc = 7.2;
dirLight.shadow.camera.left = -sc;
dirLight.shadow.camera.right = sc;
dirLight.shadow.camera.top = sc;
dirLight.shadow.camera.bottom = -sc;
dirLight.shadow.bias = -0.0004;
dirLight.shadow.normalBias = 0.015;
scene.add(dirLight);

const rimLight = new THREE.DirectionalLight(0xbae6fd, 0.45);
rimLight.position.set(9.0, 6.0, -9.0);
scene.add(rimLight);

// 3D Proving-Ground Asphalt Pad (inspired by the real Noginsk EMERCOM tarmac in photo_1..13)
function createAsphaltPadTexture() {
  const s = 512;
  const c = document.createElement('canvas');
  c.width = s;
  c.height = s;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#252b34';
  ctx.fillRect(0, 0, s, s);
  const imgData = ctx.getImageData(0, 0, s, s);
  const d = imgData.data;
  let seed = 1337;
  for (let i = 0; i < d.length; i += 4) {
    seed = (seed * 16807) % 2147483647;
    const n = ((seed % 19) - 9);
    d[i] = Math.max(18, Math.min(68, d[i] + n));
    d[i + 1] = Math.max(22, Math.min(72, d[i + 1] + n));
    d[i + 2] = Math.max(28, Math.min(80, d[i + 2] + n));
  }
  ctx.putImageData(imgData, 0, 0);
  // Subtle white proving-ground parking bay lines
  ctx.strokeStyle = 'rgba(226, 232, 240, 0.42)';
  ctx.lineWidth = 5;
  ctx.strokeRect(56, 38, s - 112, s - 76);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

const studioStage = new THREE.Group();
const groundGeo = new THREE.CircleGeometry(14.5, 72);
const groundMat = new THREE.MeshStandardMaterial({
  map: createAsphaltPadTexture(),
  roughness: 0.88,
  metalness: 0.06,
});
const groundMesh = new THREE.Mesh(groundGeo, groundMat);
groundMesh.rotation.x = -Math.PI / 2;
groundMesh.position.y = -0.005;
groundMesh.receiveShadow = true;
studioStage.add(groundMesh);
scene.add(studioStage);

// World container for 3D mode vs QR-Anchor container for AR mode
const world3DRoot = new THREE.Group();
scene.add(world3DRoot);

const qrAnchorRoot = new THREE.Group();
qrAnchorRoot.visible = false;
scene.add(qrAnchorRoot);

// Holographic pad lying flat on the 1.0 x 1.0 QR code in AR mode
const qrPadGroup = new THREE.Group();
const qrPadBorder = new THREE.Mesh(
  new THREE.RingGeometry(0.62, 0.67, 4),
  new THREE.MeshBasicMaterial({ color: 0x38bdf8, side: THREE.DoubleSide, transparent: true, opacity: 0.85 })
);
qrPadBorder.rotation.x = -Math.PI / 2;
qrPadBorder.rotation.z = Math.PI / 4;
qrPadBorder.position.y = 0.004;
qrPadGroup.add(qrPadBorder);

const qrShadowPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(1.45, 1.45),
  new THREE.MeshBasicMaterial({ color: 0x061326, transparent: true, opacity: 0.36, side: THREE.DoubleSide })
);
qrShadowPlane.rotation.x = -Math.PI / 2;
qrShadowPlane.position.y = 0.001;
qrPadGroup.add(qrShadowPlane);
qrAnchorRoot.add(qrPadGroup);

// Sub-carrier inside qrAnchorRoot that scales & rotates the 8.85m truck neatly onto the 1.0 unit QR square!
const qrModelCarrier = new THREE.Group();
const BASE_QR_TRUCK_SCALE = 0.145; // 8.85m * 0.145 ≈ 1.28 QR-widths long!
let userArScale = 1.15;
let userArRotSteps = 0;
let autoWallTiltRad = 0.0;
qrAnchorRoot.add(qrModelCarrier);

// ===========================================================================
// 3. LOAD RIGGED KAMAZ-43118 .GLB & ATTACH SIGNAL EFFECTS
// ===========================================================================
const vehicleRig = new THREE.Group();
world3DRoot.add(vehicleRig);

// Soft physical under-chassis & 6-tire contact Ambient Occlusion shadow (moves with truck in BOTH 3D & QR-AR!)
function createUnderbodyContactShadow() {
  const cw = 256, ch = 512;
  const c = document.createElement('canvas');
  c.width = cw;
  c.height = ch;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, cw, ch);

  // Under-chassis soft dark core
  const bodyG = ctx.createRadialGradient(cw / 2, ch * 0.52, 18, cw / 2, ch * 0.52, cw * 0.48);
  bodyG.addColorStop(0.0, 'rgba(0, 0, 0, 0.68)');
  bodyG.addColorStop(0.65, 'rgba(0, 0, 0, 0.36)');
  bodyG.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');
  ctx.save();
  ctx.scale(1.0, 2.05);
  ctx.fillStyle = bodyG;
  ctx.beginPath();
  ctx.arc(cw / 2, ch * 0.255, cw * 0.44, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // 6 tire contact dark spots
  const wheelSpots = [
    [cw * 0.22, ch * 0.18], [cw * 0.78, ch * 0.18],
    [cw * 0.22, ch * 0.66], [cw * 0.78, ch * 0.66],
    [cw * 0.22, ch * 0.80], [cw * 0.78, ch * 0.80],
  ];
  wheelSpots.forEach(([wx, wy]) => {
    const wg = ctx.createRadialGradient(wx, wy, 3, wx, wy, 28);
    wg.addColorStop(0.0, 'rgba(0, 0, 0, 0.85)');
    wg.addColorStop(0.6, 'rgba(0, 0, 0, 0.40)');
    wg.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');
    ctx.fillStyle = wg;
    ctx.beginPath();
    ctx.arc(wx, wy, 28, 0, Math.PI * 2);
    ctx.fill();
  });

  const tex = new THREE.CanvasTexture(c);
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(3.4, 9.6),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, opacity: 0.88 })
  );
  plane.rotation.x = -Math.PI / 2;
  plane.position.set(0.0, 0.008, -0.15);
  return plane;
}
vehicleRig.add(createUnderbodyContactShadow());

const rigNodes = {};
const beaconMats = [];
const signalFxGroup = new THREE.Group();
const vsatBeamGroup = new THREE.Group();
const mastWaveRings = [];

// Blue strobe point light over cab roof
const beaconLight = new THREE.PointLight(0x0ea5e9, 0, 7.5);
beaconLight.position.set(0, 3.25, 3.24);
vehicleRig.add(beaconLight);

const loader = new GLTFLoader();
loader.load('./models/kamaz_comms_hub.glb', (gltf) => {
  const model = gltf.scene;
  model.traverse((obj) => {
    if (obj.name) {
      rigNodes[obj.name] = obj;
    }
    if (obj.isMesh) {
      obj.castShadow = true;
      obj.receiveShadow = true;
      if (obj.material && obj.material.name === 'BeaconBlue') {
        beaconMats.push(obj.material);
      }
    }
  });

  vehicleRig.add(model);

  // Attach Holographic Satellite Uplink Beam directly to VSAT_Elevation dish!
  if (rigNodes['VSAT_Elevation']) {
    const beamCone = new THREE.Mesh(
      new THREE.CylinderGeometry(0.68, 0.12, 4.6, 28, 1, true),
      new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.18,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    beamCone.position.set(0, 2.55, -0.12);
    vsatBeamGroup.add(beamCone);

    for (let i = 0; i < 3; i++) {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.25 + i * 0.12, 0.014, 10, 36),
        new THREE.MeshBasicMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.75 })
      );
      ring.rotation.x = Math.PI / 2;
      ring.userData.phase = i / 3;
      vsatBeamGroup.add(ring);
    }
    rigNodes['VSAT_Elevation'].add(vsatBeamGroup);
  }

  // Attach Expanding VHF/UHF Radio Wave Rings to top of Mast_Stage2!
  if (rigNodes['Mast_Stage2']) {
    const mastFx = new THREE.Group();
    mastFx.position.set(-0.08, 2.15, 0.0);
    for (let i = 0; i < 3; i++) {
      const wave = new THREE.Mesh(
        new THREE.RingGeometry(0.18, 0.23, 40),
        new THREE.MeshBasicMaterial({
          color: i % 2 === 0 ? 0x38bdf8 : 0xf97316,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.7,
          depthWrite: false,
        })
      );
      wave.rotation.x = -Math.PI / 2;
      wave.userData.phase = i / 3;
      mastFx.add(wave);
      mastWaveRings.push(wave);
    }
    rigNodes['Mast_Stage2'].add(mastFx);
    signalFxGroup.add(mastFx);
  }

  buildHotspotDOM();
  window.__modelReady = true;
});

// ===========================================================================
// 4. INTERACTIVE STATE MACHINE (DEPLOY / STOW, BEACONS, WAVES, HOTSPOTS)
// ===========================================================================
let isDeployed = true;
let deployProg = 1.0; // 0 = stowed (travel), 1 = deployed (emergency comms active)
let wavesActive = false; // Off by default for pure photorealistic look; toggleable via dock!
let beaconActive = true;
let hotspotsVisible = true;
let currentMode = '3d'; // '3d' | 'qr-ar' | 'space-ar'
let selectedHotspotId = null;

// Camera smooth fly-to state
let camAnim = null;

function setStatus(text, isWarn = false) {
  document.getElementById('status-text').textContent = text;
  const dot = document.getElementById('status-dot');
  dot.classList.toggle('warn', isWarn);
}

// Create clickable HTML 3D Hotspots
const hotspotEls = [];
function buildHotspotDOM() {
  hotspotsLayer.innerHTML = '';
  HOTSPOTS.forEach((h) => {
    const el = document.createElement('button');
    el.className = 'hotspot-pin';
    el.innerHTML = `<span>${h.num}</span><span class="pin-tooltip">${h.label}</span>`;
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      selectHotspot(h);
    });
    hotspotsLayer.appendChild(el);
    hotspotEls.push({ data: h, el });
  });
}

function selectHotspot(h) {
  selectedHotspotId = h.id;
  hotspotEls.forEach(({ data, el }) => {
    el.classList.toggle('selected', data.id === h.id);
  });
  document.getElementById('spec-kicker').textContent = h.kicker;
  document.getElementById('spec-title').textContent = h.label;
  document.getElementById('spec-desc').textContent = h.desc;

  if (currentMode === '3d') {
    flyCameraTo(h.camPos, h.camTarget);
  }
}

function flyCameraTo(targetPos, targetLook) {
  const startLook = controls.target.clone();
  const endLook = targetLook.clone();
  const startSph = new THREE.Spherical().setFromVector3(orbitCam.position.clone().sub(startLook));
  const endSph = new THREE.Spherical().setFromVector3(targetPos.clone().sub(endLook));

  // Shortest angular path for theta around Y axis
  let dTheta = endSph.theta - startSph.theta;
  while (dTheta > Math.PI) dTheta -= Math.PI * 2;
  while (dTheta < -Math.PI) dTheta += Math.PI * 2;
  endSph.theta = startSph.theta + dTheta;

  camAnim = {
    startSph,
    endSph,
    startLook,
    endLook,
    t: 0,
  };
}

// Camera Presets
const CAM_PRESETS = {
  iso:   { pos: new THREE.Vector3(-9.6, 5.0, 9.8),  look: new THREE.Vector3(0.0, 2.05, -0.35) },
  front: { pos: new THREE.Vector3(0.0, 2.55, 10.4), look: new THREE.Vector3(0.0, 1.95, 1.8) },
  right: { pos: new THREE.Vector3(-9.2, 3.4, 1.8),  look: new THREE.Vector3(0.0, 1.95, -0.9) },
  roof:  { pos: new THREE.Vector3(7.4, 7.2, -7.2),  look: new THREE.Vector3(0.0, 2.95, -1.6) },
  rear:  { pos: new THREE.Vector3(-5.6, 3.5, -10.4),look: new THREE.Vector3(0.0, 1.9, -2.5) },
};

document.querySelectorAll('[data-cam]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const p = CAM_PRESETS[btn.dataset.cam];
    if (p && currentMode === '3d') flyCameraTo(p.pos, p.look);
  });
});

// Dock Buttons
const btnDeploy = document.getElementById('btn-toggle-deploy');
const deployPill = document.getElementById('deploy-state-pill');
btnDeploy.addEventListener('click', () => {
  isDeployed = !isDeployed;
  btnDeploy.textContent = isDeployed ? 'Свернуть в походное' : 'Развернуть узел связи';
  deployPill.textContent = isDeployed
    ? 'Статус: РАЗВЁРНУТ (РАБОТА ЧС)'
    : 'Статус: ПОХОДНОЕ ПОЛОЖЕНИЕ (МАРШ)';
});

const btnWaves = document.getElementById('btn-toggle-waves');
btnWaves.addEventListener('click', () => {
  wavesActive = !wavesActive;
  btnWaves.classList.toggle('active', wavesActive);
});

const btnBeacon = document.getElementById('btn-toggle-beacon');
btnBeacon.addEventListener('click', () => {
  beaconActive = !beaconActive;
  btnBeacon.classList.toggle('active', beaconActive);
});

const btnHotspots = document.getElementById('btn-toggle-hotspots');
btnHotspots.addEventListener('click', () => {
  hotspotsVisible = !hotspotsVisible;
  btnHotspots.classList.toggle('active', hotspotsVisible);
});

// ===========================================================================
// 5. QR-CODE GENERATION & PRINTABLE MARKER SHEET
// ===========================================================================
function getDefaultARUrl() {
  const url = new URL(window.location.href);
  url.searchParams.set('ar', '1');
  return url.toString();
}

let currentQRUrl = getDefaultARUrl();

async function renderAllQRCodes(targetUrl) {
  currentQRUrl = targetUrl;
  const sideCanvas = document.getElementById('sidebar-qr-canvas');
  const modalCanvas = document.getElementById('modal-qr-canvas');
  const urlInput = document.getElementById('custom-url-input');
  const urlDisplay = document.getElementById('modal-url-display');

  if (urlInput && document.activeElement !== urlInput) {
    urlInput.value = targetUrl;
  }
  if (urlDisplay) {
    urlDisplay.textContent = targetUrl;
  }

  await QRCode.toCanvas(sideCanvas, targetUrl, {
    width: 240,
    margin: 2,
    color: { dark: '#000000', light: '#FFFFFF' },
    errorCorrectionLevel: 'M',
  });

  await QRCode.toCanvas(modalCanvas, targetUrl, {
    width: 320,
    margin: 2,
    color: { dark: '#000000', light: '#FFFFFF' },
    errorCorrectionLevel: 'M',
  });
}

renderAllQRCodes(currentQRUrl);

async function downloadPrintableMarkerPNG() {
  const W = 1400, H = 1750;
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const ctx = c.getContext('2d');

  // Crisp white sheet with tactical MCHS border
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = '#0B1F3A';
  ctx.lineWidth = 18;
  ctx.strokeRect(48, 48, W - 96, H - 96);

  // Orange-Blue-Orange top banner
  ctx.fillStyle = '#F97316';
  ctx.fillRect(57, 57, W - 114, 22);
  ctx.fillStyle = '#0284C7';
  ctx.fillRect(57, 79, W - 114, 22);
  ctx.fillStyle = '#F97316';
  ctx.fillRect(57, 101, W - 114, 22);

  ctx.fillStyle = '#0B1F3A';
  ctx.textAlign = 'center';
  ctx.font = 'bold 48px sans-serif';
  ctx.fillText('МОБИЛЬНЫЙ УЗЕЛ СВЯЗИ ЧС', W / 2, 225);
  ctx.font = 'bold 30px sans-serif';
  ctx.fillStyle = '#0284C7';
  ctx.fillText('КАМАЗ-43118 (6×6) • ГОСНОМЕР 0828 ММ 18 • WebAR МАРКЕР', W / 2, 278);

  // High-contrast QR code in center
  const qrTemp = document.createElement('canvas');
  await QRCode.toCanvas(qrTemp, currentQRUrl, {
    width: 860,
    margin: 2,
    color: { dark: '#000000', light: '#FFFFFF' },
    errorCorrectionLevel: 'M',
  });

  const qx = (W - 860) / 2;
  const qy = 365;
  ctx.strokeStyle = '#0B1F3A';
  ctx.lineWidth = 10;
  ctx.strokeRect(qx - 24, qy - 24, 908, 908);
  ctx.drawImage(qrTemp, qx, qy, 860, 860);

  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 30px sans-serif';
  ctx.fillText('1. Отсканируйте камерой смартфона для открытия 3D / AR сцены', W / 2, 1385);
  ctx.fillText('2. Наведите AR-камеру на этот QR-код — 3D узел появится поверх листа!', W / 2, 1438);

  ctx.fillStyle = '#64748B';
  ctx.font = '22px monospace';
  ctx.fillText(currentQRUrl.slice(0, 78), W / 2, 1545);

  const link = document.createElement('a');
  link.download = 'QR_AR_Marker_Kamaz_Comms_Hub.png';
  link.href = c.toDataURL('image/png');
  link.click();
}

// Modal & Download bindings
const qrModal = document.getElementById('qr-modal');
document.getElementById('btn-open-qr-modal').addEventListener('click', () => qrModal.classList.add('open'));
document.getElementById('btn-close-qr-modal').addEventListener('click', () => qrModal.classList.remove('open'));
document.getElementById('btn-apply-url').addEventListener('click', () => {
  const val = document.getElementById('custom-url-input').value.trim();
  if (val) renderAllQRCodes(val);
});
document.getElementById('btn-dl-marker').addEventListener('click', downloadPrintableMarkerPNG);
document.getElementById('modal-btn-download').addEventListener('click', downloadPrintableMarkerPNG);
document.getElementById('btn-print-sheet').addEventListener('click', () => {
  qrModal.classList.add('open');
  setTimeout(() => window.print(), 200);
});
document.getElementById('modal-btn-print').addEventListener('click', () => window.print());

// Photo Lightbox
const photoModal = document.getElementById('photo-modal');
const lightboxImg = document.getElementById('lightbox-img');
document.querySelectorAll('.photo-thumb').forEach((img) => {
  img.addEventListener('click', () => {
    lightboxImg.src = img.src;
    photoModal.classList.add('open');
  });
});
document.getElementById('btn-close-photo').addEventListener('click', () => photoModal.classList.remove('open'));
photoModal.addEventListener('click', (e) => {
  if (e.target === photoModal) photoModal.classList.remove('open');
});

// ===========================================================================
// 6. REAL-TIME 6-DOF QR CAMERA TRACKER INTEGRATION
// ===========================================================================
const targetArPos = new THREE.Vector3();
const targetArQuat = new THREE.Quaternion();
let hasInitialLock = false;

const qrTracker = new QRCameraTracker({
  videoEl,
  simBgCanvasEl: simBgCanvas,
  hudCanvasEl: hudCanvas,
  onPose: (pos, quat) => {
    targetArPos.copy(pos);
    targetArQuat.copy(quat);

    // Detect if QR is nearly vertical (e.g. on laptop monitor / wall: marker +Y points toward camera +Z)
    // If vertical, smoothly tilt carrier by -58 deg around X & yaw -36 deg so viewer sees a glorious 3/4 side view instead of just the roof!
    const markerUpInCam = new THREE.Vector3(0, 1, 0).applyQuaternion(quat);
    const verticalness = Math.max(0, Math.min(1, (Math.abs(markerUpInCam.z) - 0.72) / 0.22));
    autoWallTiltRad += (verticalness - autoWallTiltRad) * 0.25;
    updateQrCarrierTransform();

    if (!qrAnchorRoot.visible || !hasInitialLock) {
      qrAnchorRoot.position.copy(pos);
      qrAnchorRoot.quaternion.copy(quat);
      qrAnchorRoot.visible = true;
      hasInitialLock = true;
    }
    setStatus('QR-код распознан! 3D мобильный узел связи привязан к маркеру', false);
  },
  onLost: () => {
    if (qrAnchorRoot.visible) {
      qrAnchorRoot.visible = false;
      hasInitialLock = false;
    }
    setStatus('Поиск QR-кода... Наведите камеру на любой QR-маркер', true);
  },
  onStatus: (_code, msg) => {
    setStatus(msg, false);
  },
});

function updateQrCarrierTransform() {
  const s = BASE_QR_TRUCK_SCALE * userArScale;
  qrModelCarrier.scale.set(s, s, s);
  const baseYaw = -0.62 + userArRotSteps * (Math.PI / 2); // Heroic 3/4 diagonal angle on QR!
  const wallPitch = -autoWallTiltRad * 1.02; // Tilts stage into 3/4 view when QR is vertical on a monitor!
  qrModelCarrier.rotation.set(wallPitch, baseYaw, 0, 'XYZ');
}
updateQrCarrierTransform();

document.getElementById('ar-scale-slider').addEventListener('input', (e) => {
  userArScale = parseFloat(e.target.value);
  document.getElementById('ar-scale-val').textContent = `${userArScale.toFixed(2)}×`;
  updateQrCarrierTransform();
});

document.getElementById('btn-ar-rot').addEventListener('click', () => {
  userArRotSteps = (userArRotSteps + 1) % 4;
  updateQrCarrierTransform();
});

document.getElementById('btn-ar-flip-cam').addEventListener('click', async () => {
  const nextFacing = qrTracker.facingMode === 'environment' ? 'user' : 'environment';
  try {
    simBgCanvas.style.display = 'none';
    videoEl.style.display = 'block';
    await qrTracker.startCamera(nextFacing);
  } catch (err) {
    setStatus(`Камера недоступна (${err.message}) — включена симуляция`, true);
    videoEl.style.display = 'none';
    simBgCanvas.style.display = 'block';
    await qrTracker.startSimulation(currentQRUrl);
  }
});

document.getElementById('btn-ar-sim').addEventListener('click', async () => {
  videoEl.style.display = 'none';
  simBgCanvas.style.display = 'block';
  await qrTracker.startSimulation(currentQRUrl);
});

// ===========================================================================
// 7. MODE SWITCHING (3D <-> QR-AR CAMERA <-> SPACE AR)
// ===========================================================================
const modeTabs = document.querySelectorAll('.mode-tab[data-mode]');
const qrArHudBar = document.getElementById('qr-ar-hud-bar');
const camPresetsEl = document.getElementById('cam-presets');
const spaceArView = document.getElementById('space-ar-view');
const bottomDock = document.getElementById('bottom-dock');

async function switchMode(mode) {
  currentMode = mode;
  modeTabs.forEach((t) => t.classList.toggle('active', t.dataset.mode === mode));

  if (mode === '3d') {
    qrTracker.stop();
    videoEl.style.display = 'none';
    simBgCanvas.style.display = 'none';
    hudCanvas.style.display = 'none';
    qrArHudBar.classList.remove('visible');
    spaceArView.classList.remove('visible');
    camPresetsEl.style.display = '';
    bottomDock.style.display = 'flex';
    viewportEl.style.background = 'radial-gradient(circle at 50% 42%, #162944 0%, #07101e 78%)';

    studioStage.visible = true;
    qrAnchorRoot.visible = false;
    world3DRoot.add(vehicleRig);
    vehicleRig.position.set(0, 0, 0);
    vehicleRig.rotation.set(0, 0, 0);
    vehicleRig.scale.set(1, 1, 1);
    controls.enabled = true;
    setStatus('Режим 3D: Вращайте модель, открывайте отсеки и изучайте узлы', false);
  } else if (mode === 'qr-ar') {
    spaceArView.classList.remove('visible');
    camPresetsEl.style.display = 'none';
    bottomDock.style.display = 'flex';
    qrArHudBar.classList.add('visible');
    hudCanvas.style.display = 'block';
    viewportEl.style.background = '#000';

    controls.enabled = false;
    studioStage.visible = false;
    qrModelCarrier.add(vehicleRig);
    vehicleRig.position.set(0, 0, 0);
    vehicleRig.rotation.set(0, 0, 0);
    vehicleRig.scale.set(1, 1, 1);
    hasInitialLock = false;
    qrAnchorRoot.visible = false;

    try {
      simBgCanvas.style.display = 'none';
      videoEl.style.display = 'block';
      await qrTracker.startCamera('environment');
    } catch (_err) {
      // Seamless automatic fallback to real jsQR synthetic tabletop simulation when no camera exists!
      videoEl.style.display = 'none';
      simBgCanvas.style.display = 'block';
      await qrTracker.startSimulation(currentQRUrl);
    }
  } else if (mode === 'space-ar') {
    qrTracker.stop();
    videoEl.style.display = 'none';
    simBgCanvas.style.display = 'none';
    hudCanvas.style.display = 'none';
    qrArHudBar.classList.remove('visible');
    camPresetsEl.style.display = 'none';
    bottomDock.style.display = 'none';
    spaceArView.classList.add('visible');
    setStatus('Безмаркерный AR: нажмите оранжевую кнопку для размещения в комнате', false);
  }
}

modeTabs.forEach((tab) => {
  tab.addEventListener('click', () => switchMode(tab.dataset.mode));
});

// Handle ?ar=1 when user arrives directly by scanning the QR code on their smartphone!
const urlParams = new URLSearchParams(window.location.search);
const mobileSheet = document.getElementById('mobile-ar-sheet');
if (urlParams.get('ar') === '1') {
  mobileSheet.classList.add('visible');
}

document.getElementById('sheet-btn-qr-ar').addEventListener('click', () => {
  mobileSheet.classList.remove('visible');
  switchMode('qr-ar');
});
document.getElementById('sheet-btn-space-ar').addEventListener('click', () => {
  mobileSheet.classList.remove('visible');
  switchMode('space-ar');
  const mv = document.getElementById('native-model-viewer');
  if (mv && typeof mv.activateAR === 'function') {
    try { mv.activateAR(); } catch (_e) { /* user can tap slot button */ }
  }
});
document.getElementById('sheet-btn-close').addEventListener('click', () => {
  mobileSheet.classList.remove('visible');
});

// ===========================================================================
// 8. RESIZE & MAIN 60-FPS RENDER / ANIMATION LOOP
// ===========================================================================
function handleResize() {
  const w = viewportEl.clientWidth || window.innerWidth;
  const h = viewportEl.clientHeight || window.innerHeight;
  renderer.setSize(w, h, false);
  orbitCam.aspect = w / h;
  orbitCam.updateProjectionMatrix();
  arCam.aspect = w / h;
  arCam.updateProjectionMatrix();
}
window.addEventListener('resize', handleResize);
handleResize();

const clock = new THREE.Clock();
const tmpVec = new THREE.Vector3();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const elapsed = clock.elapsedTime;

  // 1. Smoothly interpolate Deploy <-> Stow state (deployProg 0..1)
  const targetDeploy = isDeployed ? 1.0 : 0.0;
  deployProg += (targetDeploy - deployProg) * Math.min(1.0, dt * 3.6);
  const p = deployProg;

  if (rigNodes['Mast_Stage1']) {
    rigNodes['Mast_Stage1'].position.y = 0.22 + p * 1.43;
  }
  if (rigNodes['Mast_Stage2']) {
    rigNodes['Mast_Stage2'].position.y = 0.14 + p * 1.11;
    // Gentle PTZ surveillance sweep when deployed
    rigNodes['Mast_Stage2'].rotation.y = p * Math.sin(elapsed * 0.7) * 0.45;
  }
  if (rigNodes['VSAT_Elevation']) {
    const elevRad = THREE.MathUtils.degToRad(-34.0 * p);
    rigNodes['VSAT_Elevation'].rotation.set(elevRad, 0, 0);
  }
  if (rigNodes['VSAT_Azimuth']) {
    const azRad = THREE.MathUtils.degToRad(22.0 * p) + p * Math.sin(elapsed * 0.35) * 0.08;
    rigNodes['VSAT_Azimuth'].rotation.set(0, azRad, 0);
  }
  if (rigNodes['SideDoor_F']) {
    rigNodes['SideDoor_F'].rotation.set(0, THREE.MathUtils.degToRad(+86.0 * p), 0);
  }
  if (rigNodes['SideDoor_R']) {
    rigNodes['SideDoor_R'].rotation.set(0, THREE.MathUtils.degToRad(-86.0 * p), 0);
  }
  if (rigNodes['RearDoor_L']) {
    rigNodes['RearDoor_L'].rotation.set(0, THREE.MathUtils.degToRad(-92.0 * p), 0);
  }
  if (rigNodes['RearDoor_R']) {
    rigNodes['RearDoor_R'].rotation.set(0, THREE.MathUtils.degToRad(+92.0 * p), 0);
  }
  if (rigNodes['SideStairs']) {
    rigNodes['SideStairs'].scale.setScalar(Math.max(0.001, p));
    rigNodes['SideStairs'].visible = p > 0.04;
  }
  if (rigNodes['RearPlatform']) {
    rigNodes['RearPlatform'].rotation.set(THREE.MathUtils.degToRad((1.0 - p) * 88.0), 0, 0);
    rigNodes['RearPlatform'].visible = p > 0.03;
  }

  // 2. Holographic VSAT Beam & VHF Mast Radio Waves
  const showFx = wavesActive && p > 0.65;
  vsatBeamGroup.visible = showFx;
  if (showFx) {
    vsatBeamGroup.children.forEach((ch) => {
      if (ch.userData.phase !== undefined) {
        const u = (elapsed * 0.65 + ch.userData.phase) % 1.0;
        ch.position.set(0, 0.35 + u * 4.1, -0.12);
        const scRing = 0.45 + u * 1.65;
        ch.scale.set(scRing, scRing, 1);
        ch.material.opacity = (1.0 - u) * 0.78;
      }
    });
  }

  mastWaveRings.forEach((ring) => {
    ring.visible = showFx;
    if (showFx) {
      const u = (elapsed * 0.55 + ring.userData.phase) % 1.0;
      const scW = 0.5 + u * 8.5;
      ring.scale.set(scW, scW, 1);
      ring.material.opacity = (1.0 - u) * 0.65;
    }
  });

  // 3. Flashing Cobalt-Blue Emergency Lightbar (СГУ)
  if (beaconActive) {
    const strobe = Math.sin(elapsed * 18.0) > 0.15 ? 1.0 : 0.18;
    beaconLight.intensity = strobe * 3.2;
    beaconMats.forEach((m) => {
      if (m.emissive) m.emissive.setRGB(0.06 * strobe, 0.48 * strobe, 1.0 * strobe);
    });
  } else {
    beaconLight.intensity = 0;
    beaconMats.forEach((m) => {
      if (m.emissive) m.emissive.setRGB(0.03, 0.12, 0.28);
    });
  }

  // 4. Mode-specific camera & rendering
  const w = viewportEl.clientWidth;
  const h = viewportEl.clientHeight;

  if (currentMode === '3d') {
    if (camAnim) {
      camAnim.t = Math.min(1.0, camAnim.t + dt * 2.4);
      const ease = 1 - Math.pow(1 - camAnim.t, 3);
      controls.target.lerpVectors(camAnim.startLook, camAnim.endLook, ease);
      const curSph = new THREE.Spherical(
        THREE.MathUtils.lerp(camAnim.startSph.radius, camAnim.endSph.radius, ease),
        THREE.MathUtils.lerp(camAnim.startSph.phi, camAnim.endSph.phi, ease),
        THREE.MathUtils.lerp(camAnim.startSph.theta, camAnim.endSph.theta, ease)
      );
      orbitCam.position.setFromSpherical(curSph).add(controls.target);
      if (camAnim.t >= 1.0) camAnim = null;
    }
    controls.update();

    // Update 3D HTML Hotspots
    hotspotsLayer.style.display = hotspotsVisible ? 'block' : 'none';
    if (hotspotsVisible) {
      const toCam = new THREE.Vector3();
      hotspotEls.forEach(({ data, el }) => {
        tmpVec.copy(data.pos);
        if (data.id === 'mast') {
          tmpVec.y = 3.25 + p * 1.70;
        }
        toCam.copy(orbitCam.position).sub(tmpVec).normalize();
        const facing = data.normal ? data.normal.dot(toCam) : 1.0;

        tmpVec.project(orbitCam);
        if (tmpVec.z > 1.0 || tmpVec.z < -1.0) {
          el.style.display = 'none';
        } else {
          const sx = (tmpVec.x * 0.5 + 0.5) * w;
          const sy = (-tmpVec.y * 0.5 + 0.5) * h;
          el.style.display = 'flex';
          el.style.opacity = facing > -0.15 ? '1' : '0.25';
          el.style.left = `${sx}px`;
          el.style.top = `${sy}px`;
        }
      });
    }

    renderer.render(scene, orbitCam);
  } else if (currentMode === 'qr-ar') {
    hotspotsLayer.style.display = 'none';
    qrTracker.update(elapsed, w, h, AR_FOV);

    if (qrAnchorRoot.visible) {
      qrAnchorRoot.position.lerp(targetArPos, 0.42);
      qrAnchorRoot.quaternion.slerp(targetArQuat, 0.42);
    }

    renderer.render(scene, arCam);
  }
}

animate();
