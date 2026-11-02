import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { QRCameraTracker } from './qrPoseTracker.js';

// ===========================================================================
// 1. EDGE-TO-EDGE FULLSCREEN CAMERA + TRANSPARENT WEBGL RENDERER
// ===========================================================================
const viewportEl = document.getElementById('ar-viewport');
const threeCanvas = document.getElementById('three-canvas');
const videoEl = document.getElementById('ar-video');
const simBgCanvas = document.getElementById('ar-sim-bg');
const hudCanvas = document.getElementById('ar-hud-canvas');
const camPromptEl = document.getElementById('cam-permission-prompt');
const btnStartCam = document.getElementById('btn-start-camera');

const renderer = new THREE.WebGLRenderer({
  canvas: threeCanvas,
  antialias: true,
  alpha: true,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.04;

const scene = new THREE.Scene();

// Realistic Outdoor Daylight Sky + Horizon Environment for PBR glass & paint reflections
function createOutdoorDaylightEnv() {
  const ew = 512, eh = 256;
  const c = document.createElement('canvas');
  c.width = ew;
  c.height = eh;
  const ctx = c.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, eh);
  grad.addColorStop(0.00, '#3b82f6');
  grad.addColorStop(0.36, '#93c5fd');
  grad.addColorStop(0.49, '#e0f2fe');
  grad.addColorStop(0.51, '#64748b');
  grad.addColorStop(0.65, '#334155');
  grad.addColorStop(1.00, '#1e293b');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, ew, eh);

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
scene.environment = pmrem.fromEquirectangular(createOutdoorDaylightEnv()).texture;
scene.environmentIntensity = 0.90;

// Natural smartphone camera FOV (64 deg vertical in portrait, 50 deg in landscape)
function getActiveFov(w, h) {
  return h > w ? 64 : 50;
}

const arCam = new THREE.PerspectiveCamera(64, 1, 0.03, 80);
arCam.position.set(0, 0, 0);
arCam.rotation.set(0, 0, 0);

// Natural Daylight Sun + Sky Hemisphere Lighting
const hemiLight = new THREE.HemisphereLight(0xdbeafe, 0x334155, 0.80);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xfffbeb, 2.2);
dirLight.position.set(-6.0, 11.0, 8.0);
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(0xbae6fd, 0.55);
fillLight.position.set(7.0, 4.0, -6.0);
scene.add(fillLight);

// ===========================================================================
// 2. QR-ANCHOR & PHOTOREALISTIC KAMAZ-43118 MODEL RIG
// ===========================================================================
const qrAnchorRoot = new THREE.Group();
qrAnchorRoot.visible = false;
scene.add(qrAnchorRoot);

const qrModelCarrier = new THREE.Group();
const BASE_QR_TRUCK_SCALE = 0.165; // Heroic default scale on QR marker (pinch-zoomable 0.35x..3.5x)
let userArScale = 1.0;
let userYawRad = -0.62; // Heroic 3/4 diagonal angle on QR marker
let autoWallTiltRad = 0.0;
qrAnchorRoot.add(qrModelCarrier);

const vehicleRig = new THREE.Group();
qrModelCarrier.add(vehicleRig);

// Soft physical under-chassis & 6-tire contact shadow directly on the real-world surface!
function createUnderbodyContactShadow() {
  const cw = 256, ch = 512;
  const c = document.createElement('canvas');
  c.width = cw;
  c.height = ch;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, cw, ch);

  const bodyG = ctx.createRadialGradient(cw / 2, ch * 0.52, 18, cw / 2, ch * 0.52, cw * 0.48);
  bodyG.addColorStop(0.0, 'rgba(0, 0, 0, 0.62)');
  bodyG.addColorStop(0.65, 'rgba(0, 0, 0, 0.30)');
  bodyG.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');
  ctx.save();
  ctx.scale(1.0, 2.05);
  ctx.fillStyle = bodyG;
  ctx.beginPath();
  ctx.arc(cw / 2, ch * 0.255, cw * 0.44, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  const wheelSpots = [
    [cw * 0.22, ch * 0.18], [cw * 0.78, ch * 0.18],
    [cw * 0.22, ch * 0.66], [cw * 0.78, ch * 0.66],
    [cw * 0.22, ch * 0.80], [cw * 0.78, ch * 0.80],
  ];
  wheelSpots.forEach(([wx, wy]) => {
    const wg = ctx.createRadialGradient(wx, wy, 3, wx, wy, 28);
    wg.addColorStop(0.0, 'rgba(0, 0, 0, 0.82)');
    wg.addColorStop(0.6, 'rgba(0, 0, 0, 0.36)');
    wg.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');
    ctx.fillStyle = wg;
    ctx.beginPath();
    ctx.arc(wx, wy, 28, 0, Math.PI * 2);
    ctx.fill();
  });

  const tex = new THREE.CanvasTexture(c);
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(3.4, 9.6),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, opacity: 0.85 })
  );
  plane.rotation.x = -Math.PI / 2;
  plane.position.set(0.0, 0.008, -0.15);
  return plane;
}
vehicleRig.add(createUnderbodyContactShadow());

const rigNodes = {};
const beaconMats = [];
const beaconLight = new THREE.PointLight(0x0ea5e9, 0, 7.5);
beaconLight.position.set(0, 3.25, 3.24);
vehicleRig.add(beaconLight);

let isDeployed = true;
let deployProg = 1.0;

const loader = new GLTFLoader();
loader.load('./models/kamaz_comms_hub.glb', (gltf) => {
  const model = gltf.scene;
  model.traverse((obj) => {
    if (obj.name) rigNodes[obj.name] = obj;
    if (obj.isMesh && obj.material && obj.material.name === 'BeaconBlue') {
      beaconMats.push(obj.material);
    }
  });
  vehicleRig.add(model);
  window.__modelReady = true;
});

function updateQrCarrierTransform() {
  const s = BASE_QR_TRUCK_SCALE * userArScale;
  qrModelCarrier.scale.set(s, s, s);
  const wallPitch = -autoWallTiltRad * 1.02;
  qrModelCarrier.rotation.set(wallPitch, userYawRad, 0, 'XYZ');
}
updateQrCarrierTransform();

// ===========================================================================
// 3. INTUITIVE DIRECT TOUCH GESTURES (1-FINGER ROTATE, 2-FINGER PINCH, DOUBLE-TAP DEPLOY)
// ===========================================================================
let activePointers = new Map();
let prevPinchDist = null;
let prevSingleX = null;
let lastTapTime = 0;

threeCanvas.addEventListener('pointerdown', (e) => {
  activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (activePointers.size === 1) {
    prevSingleX = e.clientX;
    const now = performance.now();
    if (now - lastTapTime < 300) {
      isDeployed = !isDeployed; // Double-tap toggles Deploy / Stow animation!
    }
    lastTapTime = now;
  } else if (activePointers.size === 2) {
    const pts = [...activePointers.values()];
    prevPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
  }
});

threeCanvas.addEventListener('pointermove', (e) => {
  if (!activePointers.has(e.pointerId)) return;
  activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

  if (activePointers.size === 1 && prevSingleX !== null) {
    const dx = e.clientX - prevSingleX;
    prevSingleX = e.clientX;
    userYawRad += dx * 0.01;
    updateQrCarrierTransform();
  } else if (activePointers.size === 2) {
    const pts = [...activePointers.values()];
    const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
    if (prevPinchDist && prevPinchDist > 10) {
      const ratio = dist / prevPinchDist;
      userArScale = Math.max(0.35, Math.min(3.5, userArScale * ratio));
      updateQrCarrierTransform();
    }
    prevPinchDist = dist;
  }
});

function onPointerUp(e) {
  activePointers.delete(e.pointerId);
  if (activePointers.size < 2) prevPinchDist = null;
  if (activePointers.size === 1) {
    prevSingleX = [...activePointers.values()][0].x;
  } else {
    prevSingleX = null;
  }
}
threeCanvas.addEventListener('pointerup', onPointerUp);
threeCanvas.addEventListener('pointercancel', onPointerUp);

// Mouse wheel zoom support on desktop
threeCanvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.08 : 0.92;
  userArScale = Math.max(0.35, Math.min(3.5, userArScale * factor));
  updateQrCarrierTransform();
}, { passive: false });

// ===========================================================================
// 4. AUTO-LAUNCH REAR CAMERA & REAL-TIME 6-DoF QR POSE TRACKER
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

    // Smoothly adapt whether QR code is flat on a desk vs vertical on a screen/wall
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
  },
  onLost: () => {
    qrAnchorRoot.visible = false;
    hasInitialLock = false;
  },
});

async function bootFullscreenAR() {
  camPromptEl.classList.remove('visible');
  try {
    videoEl.style.display = 'block';
    simBgCanvas.style.display = 'none';
    await qrTracker.startCamera('environment');
  } catch (err) {
    // If desktop PC without any camera (or headless environment), fall back to synthetic QR camera
    const msg = String(err && (err.name || err.message || err));
    if (navigator.webdriver || msg.includes('NotFoundError') || msg.includes('DevicesNotFound') || msg.includes('NO_MEDIA_DEVICES')) {
      videoEl.style.display = 'none';
      simBgCanvas.style.display = 'block';
      await qrTracker.startSimulation(window.location.href);
    } else {
      // Mobile browser requires a single tap gesture to grant camera access
      camPromptEl.classList.add('visible');
    }
  }
}

btnStartCam.addEventListener('click', () => {
  bootFullscreenAR();
});

bootFullscreenAR();

// ===========================================================================
// 5. RESIZE & 60 FPS RENDER LOOP
// ===========================================================================
function handleResize() {
  const w = viewportEl.clientWidth || window.innerWidth;
  const h = viewportEl.clientHeight || window.innerHeight;
  renderer.setSize(w, h, false);
  arCam.fov = getActiveFov(w, h);
  arCam.aspect = w / Math.max(1, h);
  arCam.updateProjectionMatrix();
}
window.addEventListener('resize', handleResize);
handleResize();

const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const elapsed = clock.elapsedTime;

  // Smooth Deploy / Stow articulation
  const targetP = isDeployed ? 1.0 : 0.0;
  deployProg += (targetP - deployProg) * Math.min(1.0, dt * 3.5);
  const p = deployProg;

  if (rigNodes['Mast_Stage1']) rigNodes['Mast_Stage1'].position.set(0, 0.70 + p * 1.35, 0);
  if (rigNodes['Mast_Stage2']) rigNodes['Mast_Stage2'].position.set(0, 0.15 + p * 1.10, 0);
  if (rigNodes['VSAT_Elevation']) {
    rigNodes['VSAT_Elevation'].rotation.set(THREE.MathUtils.degToRad(-34.0 * p), 0, 0);
  }
  if (rigNodes['VSAT_Azimuth']) {
    const azRad = THREE.MathUtils.degToRad(22.0 * p) + p * Math.sin(elapsed * 0.35) * 0.08;
    rigNodes['VSAT_Azimuth'].rotation.set(0, azRad, 0);
  }
  if (rigNodes['SideDoor_F']) rigNodes['SideDoor_F'].rotation.set(0, THREE.MathUtils.degToRad(+86.0 * p), 0);
  if (rigNodes['SideDoor_R']) rigNodes['SideDoor_R'].rotation.set(0, THREE.MathUtils.degToRad(-86.0 * p), 0);
  if (rigNodes['RearDoor_L']) rigNodes['RearDoor_L'].rotation.set(0, THREE.MathUtils.degToRad(-92.0 * p), 0);
  if (rigNodes['RearDoor_R']) rigNodes['RearDoor_R'].rotation.set(0, THREE.MathUtils.degToRad(+92.0 * p), 0);
  if (rigNodes['SideStairs']) {
    rigNodes['SideStairs'].scale.setScalar(Math.max(0.001, p));
    rigNodes['SideStairs'].visible = p > 0.04;
  }
  if (rigNodes['RearPlatform']) {
    rigNodes['RearPlatform'].rotation.set(THREE.MathUtils.degToRad((1.0 - p) * 88.0), 0, 0);
    rigNodes['RearPlatform'].visible = p > 0.03;
  }

  // Subtle realistic blue strobe on cab lightbar
  const strobe = Math.sin(elapsed * 16.0) > 0.2 ? 1.0 : 0.15;
  beaconLight.intensity = strobe * 2.2;
  beaconMats.forEach((m) => {
    if (m.emissive) m.emissive.setRGB(0.04 * strobe, 0.32 * strobe, 0.85 * strobe);
  });

  const w = viewportEl.clientWidth || window.innerWidth;
  const h = viewportEl.clientHeight || window.innerHeight;
  qrTracker.update(elapsed, w, h, arCam.fov);

  if (qrAnchorRoot.visible) {
    qrAnchorRoot.position.lerp(targetArPos, 0.38);
    qrAnchorRoot.quaternion.slerp(targetArQuat, 0.38);
  }

  renderer.render(scene, arCam);
}

animate();
