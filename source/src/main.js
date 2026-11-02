import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { QRCameraTracker } from './qrPoseTracker.js';

// ===========================================================================
// 1. FULLSCREEN TRANSPARENT THREE.JS RENDERER & DAYLIGHT PBR ENVIRONMENT
// ===========================================================================
const viewportEl = document.getElementById('ar-viewport');
const videoEl = document.getElementById('ar-video');
const camCanvas = document.getElementById('ar-cam-canvas');
const threeCanvas = document.getElementById('three-canvas');
const hudCanvas = document.getElementById('ar-hud-canvas');

const uiOverlayEl = document.getElementById('ui-overlay');
const camPromptEl = document.getElementById('cam-permission-prompt');
const btnStartCam = document.getElementById('btn-start-camera');
const btnToggleDeploy = document.getElementById('btn-toggle-deploy');
const deployBtnLabel = document.getElementById('deploy-btn-label');
const btnPlaceSurface = document.getElementById('btn-place-surface');
const surfaceBtnLabel = document.getElementById('surface-btn-label');
const iosArLink = document.getElementById('ios-ar-link');

const renderer = new THREE.WebGLRenderer({
  canvas: threeCanvas,
  antialias: true,
  alpha: true,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;
renderer.xr.enabled = true;

const scene = new THREE.Scene();

function createOutdoorDaylightEnv() {
  const w = 256, h = 128;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0.00, '#5a91c8');
  g.addColorStop(0.32, '#a8c8e8');
  g.addColorStop(0.49, '#e4ecf3');
  g.addColorStop(0.51, '#686b6e');
  g.addColorStop(0.75, '#46494c');
  g.addColorStop(1.00, '#2a2c2f');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  const sun = ctx.createRadialGradient(w * 0.28, h * 0.22, 2, w * 0.28, h * 0.22, 34);
  sun.addColorStop(0.0, 'rgba(255, 253, 240, 1.0)');
  sun.addColorStop(0.4, 'rgba(255, 246, 218, 0.5)');
  sun.addColorStop(1.0, 'rgba(255, 246, 218, 0.0)');
  ctx.fillStyle = sun;
  ctx.fillRect(0, 0, w, h * 0.5);

  const tex = new THREE.CanvasTexture(c);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromEquirectangular(createOutdoorDaylightEnv()).texture;
scene.environmentIntensity = 0.90;

function getActiveFov(w, h) {
  return h > w ? 64 : 50;
}

const arCam = new THREE.PerspectiveCamera(64, 1, 0.03, 80);
arCam.position.set(0, 0, 0);
arCam.rotation.set(0, 0, 0);

const hemiLight = new THREE.HemisphereLight(0xdbeafe, 0x334155, 0.80);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xfffbeb, 2.2);
dirLight.position.set(-6.0, 11.0, 8.0);
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(0xbae6fd, 0.55);
fillLight.position.set(7.0, 4.0, -6.0);
scene.add(fillLight);

// ===========================================================================
// 2. ANCHOR ROOTS & PHOTOREALISTIC KAMAZ-43118 MODEL RIG
// ===========================================================================
const qrAnchorRoot = new THREE.Group();
qrAnchorRoot.visible = false;
scene.add(qrAnchorRoot);

const surfaceAnchorRoot = new THREE.Group();
surfaceAnchorRoot.visible = false;
scene.add(surfaceAnchorRoot);

const modelCarrier = new THREE.Group();
const BASE_QR_TRUCK_SCALE = 0.165;
const BASE_SURFACE_TRUCK_SCALE = 0.086;
let userArScale = 1.0;
let userYawRad = -0.62;
let surfaceOffsetX = 0.05;
let surfaceOffsetY = -0.22;
let autoWallTiltRad = 0.0;
qrAnchorRoot.add(modelCarrier);

const vehicleRig = new THREE.Group();
modelCarrier.add(vehicleRig);

// Soft physical under-chassis & 6-tire contact shadow directly on the surface
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

// WebXR Hit-Test Surface Reticle Ring
const xrReticle = new THREE.Mesh(
  new THREE.RingGeometry(0.14, 0.19, 36).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.9, side: THREE.DoubleSide })
);
xrReticle.matrixAutoUpdate = false;
xrReticle.visible = false;
scene.add(xrReticle);

const rigNodes = {};
const beaconMats = [];
const beaconLight = new THREE.PointLight(0x0ea5e9, 0, 7.5);
beaconLight.position.set(0, 3.25, 3.24);
vehicleRig.add(beaconLight);

let isDeployed = true;
let deployProg = 1.0;

// Placement modes: 'qr' | 'in-cam-surface' | 'webxr'
let placementMode = 'qr';
let xrSession = null;
let xrHitTestSource = null;
let xrHitTestRequested = false;

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

function syncDeployButtonUI() {
  const ico = btnToggleDeploy.querySelector('.btn-ico');
  if (isDeployed) {
    deployBtnLabel.textContent = 'Свернуть в походное';
    btnToggleDeploy.classList.add('primary-orange');
    btnToggleDeploy.classList.remove('stowed-state');
    if (ico) ico.style.transform = 'rotate(180deg)';
    iosArLink.setAttribute('href', './models/kamaz_comms_hub.usdz');
  } else {
    deployBtnLabel.textContent = 'Развёрнутое положение';
    btnToggleDeploy.classList.remove('primary-orange');
    btnToggleDeploy.classList.add('stowed-state');
    if (ico) ico.style.transform = 'rotate(0deg)';
    iosArLink.setAttribute('href', './models/kamaz_comms_hub_stowed.usdz');
  }
}
syncDeployButtonUI();

function syncSurfaceButtonUI() {
  if (placementMode === 'qr') {
    surfaceBtnLabel.textContent = 'Поместить на поверхность';
    btnPlaceSurface.classList.remove('surface-active');
  } else {
    surfaceBtnLabel.textContent = 'Вернуться к QR-коду';
    btnPlaceSurface.classList.add('surface-active');
  }
}
syncSurfaceButtonUI();

function updateCarrierTransform() {
  if (placementMode === 'qr') {
    const s = BASE_QR_TRUCK_SCALE * userArScale;
    modelCarrier.scale.set(s, s, s);
    const wallPitch = -autoWallTiltRad * 1.02;
    modelCarrier.rotation.set(wallPitch, userYawRad, 0, 'XYZ');
  } else if (placementMode === 'in-cam-surface') {
    const s = BASE_SURFACE_TRUCK_SCALE * userArScale;
    modelCarrier.scale.set(s, s, s);
    modelCarrier.rotation.set(0, userYawRad, 0, 'XYZ');
    surfaceAnchorRoot.position.set(surfaceOffsetX, surfaceOffsetY, -1.85);
    surfaceAnchorRoot.rotation.set(THREE.MathUtils.degToRad(18), 0, 0, 'XYZ');
  } else if (placementMode === 'webxr') {
    const s = 0.11 * userArScale; // ~1m tabletop/floor default, pinch-scalable up to 3.5x
    modelCarrier.scale.set(s, s, s);
    modelCarrier.rotation.set(0, userYawRad, 0, 'XYZ');
  }
}
updateCarrierTransform();

// ===========================================================================
// 3. BOTTOM BUTTON ACTIONS (DEPLOY/STOW + PLACE ON SURFACE)
// ===========================================================================
btnToggleDeploy.addEventListener('click', (e) => {
  e.stopPropagation();
  isDeployed = !isDeployed;
  syncDeployButtonUI();
});

async function startWebXRSurfaceAR() {
  if (!navigator.xr || !navigator.xr.isSessionSupported) return false;
  let supported = false;
  try {
    supported = await navigator.xr.isSessionSupported('immersive-ar');
  } catch (_) {
    return false;
  }
  if (!supported) return false;

  try {
    const session = await navigator.xr.requestSession('immersive-ar', {
      requiredFeatures: ['hit-test'],
      optionalFeatures: ['dom-overlay'],
      domOverlay: { root: uiOverlayEl },
    });

    xrSession = session;
    xrHitTestRequested = false;
    xrHitTestSource = null;
    placementMode = 'webxr';
    qrTracker.qrTrackingEnabled = false;
    qrAnchorRoot.visible = false;

    surfaceAnchorRoot.add(modelCarrier);
    surfaceAnchorRoot.visible = false;
    surfaceAnchorRoot.rotation.set(0, 0, 0);
    updateCarrierTransform();
    syncSurfaceButtonUI();

    session.addEventListener('select', () => {
      if (xrReticle.visible) {
        surfaceAnchorRoot.position.setFromMatrixPosition(xrReticle.matrix);
        surfaceAnchorRoot.quaternion.setFromRotationMatrix(xrReticle.matrix);
        surfaceAnchorRoot.visible = true;
      }
    });

    session.addEventListener('end', () => {
      xrSession = null;
      xrHitTestSource = null;
      xrHitTestRequested = false;
      xrReticle.visible = false;
      returnToQRMode();
    });

    await renderer.xr.setSession(session);
    return true;
  } catch (_) {
    return false;
  }
}

function enterInCameraSurfaceMode() {
  placementMode = 'in-cam-surface';
  qrTracker.qrTrackingEnabled = false;
  qrAnchorRoot.visible = false;
  hudCanvas.getContext('2d').clearRect(0, 0, hudCanvas.width, hudCanvas.height);

  surfaceOffsetX = 0.05;
  surfaceOffsetY = -0.22;
  surfaceAnchorRoot.add(modelCarrier);
  surfaceAnchorRoot.visible = true;
  updateCarrierTransform();
  syncSurfaceButtonUI();
}

function returnToQRMode() {
  if (xrSession) {
    xrSession.end().catch(() => {});
    return;
  }
  placementMode = 'qr';
  surfaceAnchorRoot.visible = false;
  qrAnchorRoot.add(modelCarrier);
  qrAnchorRoot.visible = false;
  hasInitialLock = false;
  qrTracker.qrTrackingEnabled = true;
  qrTracker.ensurePlaying();
  updateCarrierTransform();
  syncSurfaceButtonUI();
}

btnPlaceSurface.addEventListener('click', async (e) => {
  e.stopPropagation();

  if (placementMode !== 'qr') {
    returnToQRMode();
    return;
  }

  // 1. iPhone / iPad Safari & Chrome: Official Apple ARKit Quick Look (<a rel="ar"> with static USDZ)
  const supportsIOSQuickLook = Boolean(iosArLink && iosArLink.relList && iosArLink.relList.supports && iosArLink.relList.supports('ar'));
  if (supportsIOSQuickLook) {
    syncDeployButtonUI();
    iosArLink.click();
    return;
  }

  // 2. Android Chrome / Samsung Internet: Live WebXR Hit-Test Surface AR (keeps bottom buttons active!)
  const startedXR = await startWebXRSurfaceAR();
  if (startedXR) return;

  // 3. Android Scene Viewer fallback (for Android browsers with ARCore but without WebXR)
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isInAppWebView = /Telegram|VKClient|Instagram|FBAN|FBAV/i.test(navigator.userAgent);
  if (isAndroid && !isInAppWebView) {
    const modelFile = isDeployed ? 'models/kamaz_comms_hub.glb' : 'models/kamaz_comms_hub_stowed.glb';
    const absGlbUrl = new URL(modelFile, window.location.href).href;
    const fallbackUrl = window.location.href;
    const title = 'Мобильный узел связи ЧС КАМАЗ-43118';
    const intentUrl =
      `intent://arvr.google.com/scene-viewer/1.0?file=${encodeURIComponent(absGlbUrl)}&mode=ar_preferred&title=${encodeURIComponent(title)}` +
      `#Intent;scheme=https;package=com.google.android.googlequicksearchbox;action=android.intent.action.VIEW;` +
      `S.browser_fallback_url=${encodeURIComponent(fallbackUrl)};end;`;
    window.location.href = intentUrl;
    return;
  }

  // 4. Universal In-Camera Surface Mode (Telegram in-app browser, desktop, or devices without ARKit/ARCore)
  enterInCameraSurfaceMode();
});

// ===========================================================================
// 4. TOUCH GESTURES (1-FINGER ROTATE/DRAG, 2-FINGER PINCH SCALE)
// ===========================================================================
const activePointers = new Map();
let prevPinchDist = null;
let prevSinglePos = null;

threeCanvas.addEventListener('pointerdown', (e) => {
  qrTracker.ensurePlaying();
  activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (activePointers.size === 1) {
    prevSinglePos = { x: e.clientX, y: e.clientY };
  } else if (activePointers.size === 2) {
    const pts = [...activePointers.values()];
    prevPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
  }
});

threeCanvas.addEventListener('pointermove', (e) => {
  if (!activePointers.has(e.pointerId)) return;
  activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

  if (activePointers.size === 1 && prevSinglePos) {
    const dx = e.clientX - prevSinglePos.x;
    const dy = e.clientY - prevSinglePos.y;
    prevSinglePos = { x: e.clientX, y: e.clientY };
    userYawRad += dx * 0.01;
    if (placementMode === 'in-cam-surface') {
      surfaceOffsetY = Math.max(-0.55, Math.min(0.35, surfaceOffsetY - dy * 0.0018));
    }
    updateCarrierTransform();
  } else if (activePointers.size === 2) {
    const pts = [...activePointers.values()];
    const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
    if (prevPinchDist && prevPinchDist > 10) {
      const ratio = dist / prevPinchDist;
      userArScale = Math.max(0.35, Math.min(3.5, userArScale * ratio));
      updateCarrierTransform();
    }
    prevPinchDist = dist;
  }
});

function onPointerUp(e) {
  activePointers.delete(e.pointerId);
  if (activePointers.size < 2) prevPinchDist = null;
  if (activePointers.size === 1) {
    const pt = [...activePointers.values()][0];
    prevSinglePos = { x: pt.x, y: pt.y };
  } else {
    prevSinglePos = null;
  }
}
threeCanvas.addEventListener('pointerup', onPointerUp);
threeCanvas.addEventListener('pointercancel', onPointerUp);

threeCanvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.08 : 0.92;
  userArScale = Math.max(0.35, Math.min(3.5, userArScale * factor));
  updateCarrierTransform();
}, { passive: false });

// Automatically resume camera stream when returning from native AR Quick Look / Scene Viewer
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) qrTracker.ensurePlaying();
});
window.addEventListener('pageshow', () => qrTracker.ensurePlaying());
window.addEventListener('focus', () => qrTracker.ensurePlaying());

// ===========================================================================
// 5. AUTO-LAUNCH REAR CAMERA & REAL-TIME 6-DoF QR POSE TRACKER
// ===========================================================================
const targetArPos = new THREE.Vector3();
const targetArQuat = new THREE.Quaternion();
let hasInitialLock = false;

const qrTracker = new QRCameraTracker({
  videoEl,
  camCanvasEl: camCanvas,
  hudCanvasEl: hudCanvas,
  onPose: (pos, quat) => {
    if (placementMode !== 'qr') return;
    targetArPos.copy(pos);
    targetArQuat.copy(quat);

    const markerUpInCam = new THREE.Vector3(0, 1, 0).applyQuaternion(quat);
    const verticalness = Math.max(0, Math.min(1, (Math.abs(markerUpInCam.z) - 0.72) / 0.22));
    autoWallTiltRad += (verticalness - autoWallTiltRad) * 0.25;
    updateCarrierTransform();

    if (!qrAnchorRoot.visible || !hasInitialLock) {
      qrAnchorRoot.position.copy(pos);
      qrAnchorRoot.quaternion.copy(quat);
      qrAnchorRoot.visible = true;
      hasInitialLock = true;
    }
  },
  onLost: () => {
    if (placementMode !== 'qr') return;
    qrAnchorRoot.visible = false;
    hasInitialLock = false;
  },
});

async function bootFullscreenAR() {
  camPromptEl.classList.remove('visible');
  try {
    await qrTracker.startCamera('environment');
  } catch (err) {
    const msg = String(err && (err.name || err.message || err));
    if (navigator.webdriver || msg.includes('NotFoundError') || msg.includes('DevicesNotFound') || msg.includes('NO_MEDIA_DEVICES')) {
      await qrTracker.startSimulation(window.location.href);
    } else {
      camPromptEl.classList.add('visible');
    }
  }
}

btnStartCam.addEventListener('click', () => {
  bootFullscreenAR();
});

bootFullscreenAR();

// ===========================================================================
// 6. RESIZE & 60 FPS RENDER LOOP (UNIFIED CAMERA + WEBXR LOOP)
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

renderer.setAnimationLoop((timestamp, xrFrame) => {
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

  // Blue emergency beacon strobe
  const strobe = Math.sin(elapsed * 16.0) > 0.2 ? 1.0 : 0.15;
  beaconLight.intensity = strobe * 2.2;
  beaconMats.forEach((m) => {
    if (m.emissive) m.emissive.setRGB(0.04 * strobe, 0.32 * strobe, 0.85 * strobe);
  });

  if (xrFrame && placementMode === 'webxr') {
    const refSpace = renderer.xr.getReferenceSpace();
    const session = renderer.xr.getSession();

    if (!xrHitTestRequested && session) {
      session.requestReferenceSpace('viewer').then((viewerSpace) => {
        session.requestHitTestSource({ space: viewerSpace }).then((source) => {
          xrHitTestSource = source;
        });
      });
      xrHitTestRequested = true;
    }

    if (xrHitTestSource && refSpace) {
      const hits = xrFrame.getHitTestResults(xrHitTestSource);
      if (hits.length > 0) {
        const hitPose = hits[0].getPose(refSpace);
        xrReticle.visible = true;
        xrReticle.matrix.fromArray(hitPose.transform.matrix);
        // Auto-place immediately on first detected real surface so the truck appears right away,
        // and user can tap anywhere else to move it!
        if (!surfaceAnchorRoot.visible) {
          surfaceAnchorRoot.position.setFromMatrixPosition(xrReticle.matrix);
          surfaceAnchorRoot.quaternion.setFromRotationMatrix(xrReticle.matrix);
          surfaceAnchorRoot.visible = true;
        }
      } else {
        xrReticle.visible = false;
      }
    }

    renderer.render(scene, arCam);
    return;
  }

  const w = viewportEl.clientWidth || window.innerWidth;
  const h = viewportEl.clientHeight || window.innerHeight;
  qrTracker.update(elapsed, w, h, arCam.fov);

  if (placementMode === 'qr' && qrAnchorRoot.visible) {
    qrAnchorRoot.position.lerp(targetArPos, 0.38);
    qrAnchorRoot.quaternion.slerp(targetArQuat, 0.38);
  }

  renderer.render(scene, arCam);
});
