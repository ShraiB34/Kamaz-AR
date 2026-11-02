import jsQR from 'jsqr';
import QRCode from 'qrcode';
import * as THREE from 'three';

function solve8x8(A, b) {
  const n = 8;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(M[r][col]);
      if (v > maxVal) {
        maxVal = v;
        maxRow = r;
      }
    }
    if (maxVal < 1e-11) return null;
    if (maxRow !== col) {
      const tmp = M[col];
      M[col] = M[maxRow];
      M[maxRow] = tmp;
    }

    const piv = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= piv;

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      if (Math.abs(factor) < 1e-14) continue;
      for (let c = col; c <= n; c++) {
        M[r][c] -= factor * M[col][c];
      }
    }
  }

  return M.map((row) => row[n]);
}

function invertTranspose3x3(R) {
  const [a, b, c, d, e, f, g, h, i] = R;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const D = -(b * i - c * h);
  const E = a * i - c * g;
  const F = -(a * h - b * g);
  const G = b * f - c * e;
  const H = -(a * f - c * d);
  const I = a * e - b * d;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-10) return null;
  const invDet = 1.0 / det;
  return [
    A * invDet, B * invDet, C * invDet,
    D * invDet, E * invDet, F * invDet,
    G * invDet, H * invDet, I * invDet,
  ];
}

function vidH_safe(v) {
  return Math.max(1, v);
}

/**
 * Computes Three.js 6-DoF Pose (position, quaternion) of a 1.0 x 1.0 unit QR marker
 * lying in the local XZ plane (+Y up out of the QR code) from 4 jsQR pixel corners.
 */
export function computePoseFromQRCorners(cornersVid, vidW, vidH, viewW, viewH, fovDeg) {
  const coverScale = Math.max(viewW / vidW, viewH / vidH_safe(vidH));
  const dispW = vidW * coverScale;
  const dispH = vidH * coverScale;
  const cropX = (dispW - viewW) * 0.5;
  const cropY = (dispH - viewH) * 0.5;

  const fy = (viewH * 0.5) / Math.tan(THREE.MathUtils.degToRad(fovDeg * 0.5));
  const fx = fy;
  const cx = viewW * 0.5;
  const cy = viewH * 0.5;

  const modelPts = [
    [-0.5, -0.5],
    [+0.5, -0.5],
    [+0.5, +0.5],
    [-0.5, +0.5],
  ];

  const A = [];
  const b = [];
  const viewPts = [];

  for (let i = 0; i < 4; i++) {
    const u = cornersVid[i].x * coverScale - cropX;
    const v = cornersVid[i].y * coverScale - cropY;
    viewPts.push({ u, v });

    const xn = (u - cx) / fx;
    const yn = -(v - cy) / fy;
    const [X, Z] = modelPts[i];

    A.push([X, Z, 1, 0, 0, 0, -X * xn, -Z * xn]);
    b.push(xn);
    A.push([0, 0, 0, X, Z, 1, -X * yn, -Z * yn]);
    b.push(yn);
  }

  const h = solve8x8(A, b);
  if (!h) return null;

  let m00 = h[0],  m01 = h[1],  m02 = h[2];
  let m10 = h[3],  m11 = h[4],  m12 = h[5];
  let m20 = -h[6], m21 = -h[7], m22 = -1.0;

  if (m22 > 0) {
    m00 = -m00; m01 = -m01; m02 = -m02;
    m10 = -m10; m11 = -m11; m12 = -m12;
    m20 = -m20; m21 = -m21; m22 = -m22;
  }

  const norm1 = Math.hypot(m00, m10, m20);
  const norm2 = Math.hypot(m01, m11, m21);
  const scale = 0.5 * (norm1 + norm2);
  if (scale < 1e-6) return null;

  let r1x = m00 / scale, r1y = m10 / scale, r1z = m20 / scale;
  let r3x = m01 / scale, r3y = m11 / scale, r3z = m21 / scale;
  let r2x = r3y * r1z - r3z * r1y;
  let r2y = r3z * r1x - r3x * r1z;
  let r2z = r3x * r1y - r3y * r1x;

  let R = [
    r1x, r2x, r3x,
    r1y, r2y, r3y,
    r1z, r2z, r3z,
  ];

  for (let iter = 0; iter < 5; iter++) {
    const invT = invertTranspose3x3(R);
    if (!invT) break;
    for (let k = 0; k < 9; k++) {
      R[k] = 0.5 * (R[k] + invT[k]);
    }
  }

  const tx = m02 / scale;
  const ty = m12 / scale;
  const tz = m22 / scale;

  if (tz > -0.15 || tz < -30.0 || !Number.isFinite(tx)) return null;

  const basisMat = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(R[0], R[3], R[6]),
    new THREE.Vector3(R[1], R[4], R[7]),
    new THREE.Vector3(R[2], R[5], R[8])
  );
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(basisMat);
  const position = new THREE.Vector3(tx, ty, tz);

  return { position, quaternion, viewPts };
}

/**
 * Fullscreen Canvas Camera Renderer + jsQR 6-DoF Tracker.
 * Renders the camera stream onto `camCanvasEl` so `<video>` never appears visibly in the DOM
 * (100% preventing iOS Safari / Android browser Play/Pause media overlays).
 */
export class QRCameraTracker {
  constructor({ videoEl, camCanvasEl, hudCanvasEl, onPose, onLost }) {
    this.videoEl = videoEl;
    this.camCanvasEl = camCanvasEl;
    this.camCtx = camCanvasEl.getContext('2d', { alpha: false });
    this.hudCanvasEl = hudCanvasEl;
    this.hudCtx = hudCanvasEl.getContext('2d');

    this.procCanvas = document.createElement('canvas');
    this.procCtx = this.procCanvas.getContext('2d', { willReadFrequently: true });

    this.onPose = onPose;
    this.onLost = onLost;

    this.active = false;
    this.simMode = false;
    this.qrTrackingEnabled = true;
    this.facingMode = 'environment';
    this.stream = null;
    this.frameCount = 0;
    this.holdFrames = 0;
    this.maxHoldFrames = 24;

    this.simCanvas = document.createElement('canvas');
    this.simCanvas.width = 720;
    this.simCanvas.height = 1280;
    this.simCtx = this.simCanvas.getContext('2d');
    this.simQrImg = null;
  }

  async prepareSimQR(urlText) {
    const qrDataUrl = await QRCode.toDataURL(urlText, {
      width: 360,
      margin: 2,
      color: { dark: '#000000', light: '#FFFFFF' },
      errorCorrectionLevel: 'M',
    });
    const img = new Image();
    await new Promise((res) => {
      img.onload = res;
      img.src = qrDataUrl;
    });
    this.simQrImg = img;
  }

  async startCamera(facingMode = 'environment') {
    this.stop();
    this.facingMode = facingMode;
    this.simMode = false;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('NO_MEDIA_DEVICES');
    }

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.videoEl.controls = false;
    this.videoEl.muted = true;
    this.videoEl.playsInline = true;
    this.videoEl.disablePictureInPicture = true;
    this.videoEl.disableRemotePlayback = true;
    this.videoEl.setAttribute('playsinline', '');
    this.videoEl.setAttribute('webkit-playsinline', '');
    this.videoEl.srcObject = this.stream;
    await this.videoEl.play();

    this.active = true;
    this.holdFrames = 0;
    return true;
  }

  ensurePlaying() {
    if (this.stream && this.videoEl && this.videoEl.paused) {
      this.videoEl.play().catch(() => {});
    }
  }

  async startSimulation(urlText) {
    this.stop();
    await this.prepareSimQR(urlText);
    this.simMode = true;
    this.active = true;
    this.holdFrames = 0;
  }

  stop() {
    this.active = false;
    this.simMode = false;
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.videoEl) {
      this.videoEl.srcObject = null;
    }
    if (this.hudCtx && this.hudCanvasEl) {
      this.hudCtx.clearRect(0, 0, this.hudCanvasEl.width, this.hudCanvasEl.height);
    }
  }

  renderSyntheticTabletop(timeSec, viewW, viewH) {
    const isPortrait = viewH > viewW;
    const W = isPortrait ? 720 : 1280;
    const H = isPortrait ? 1280 : 720;
    if (this.simCanvas.width !== W || this.simCanvas.height !== H) {
      this.simCanvas.width = W;
      this.simCanvas.height = H;
    }
    const ctx = this.simCtx;

    const grad = ctx.createRadialGradient(W * 0.5, H * 0.5, 40, W * 0.5, H * 0.5, Math.max(W, H) * 0.72);
    grad.addColorStop(0, '#3a424e');
    grad.addColorStop(1, '#1c222b');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // Only draw synthetic QR card when QR tracking is enabled (clean surface when in Surface Mode!)
    if (this.simQrImg && this.qrTrackingEnabled) {
      ctx.save();
      const cx = W * 0.5 + Math.sin(timeSec * 0.55) * 24;
      const cy = H * 0.55 + Math.cos(timeSec * 0.45) * 14;
      const rot = Math.sin(timeSec * 0.35) * 0.16;
      const cardR = Math.min(W, H) * 0.21;
      ctx.translate(cx, cy);
      ctx.scale(1.0, 0.68);
      ctx.rotate(rot);

      ctx.fillStyle = 'rgba(0, 0, 0, 0.38)';
      ctx.fillRect(-cardR + 6, -cardR + 8, cardR * 2, cardR * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(-cardR, -cardR, cardR * 2, cardR * 2);

      const qrR = cardR * 0.80;
      ctx.drawImage(this.simQrImg, -qrR, -qrR, qrR * 2, qrR * 2);
      ctx.restore();
    }
  }

  update(timeSec, viewW, viewH, fovDeg) {
    if (!this.active) return;
    this.frameCount++;

    if (this.hudCanvasEl.width !== viewW || this.hudCanvasEl.height !== viewH) {
      this.hudCanvasEl.width = viewW;
      this.hudCanvasEl.height = viewH;
    }
    const hctx = this.hudCtx;
    hctx.clearRect(0, 0, viewW, viewH);

    let srcEl = this.videoEl;
    let srcW = this.videoEl.videoWidth || 0;
    let srcH = this.videoEl.videoHeight || 0;

    if (this.simMode) {
      this.renderSyntheticTabletop(timeSec, viewW, viewH);
      srcEl = this.simCanvas;
      srcW = this.simCanvas.width;
      srcH = this.simCanvas.height;
    } else if (srcW < 32 || srcH < 32) {
      return;
    }

    // Paint live camera (or synthetic tabletop) onto Retina canvas with exact object-fit: cover
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.round(viewW * dpr);
    const ch = Math.round(viewH * dpr);
    if (this.camCanvasEl.width !== cw || this.camCanvasEl.height !== ch) {
      this.camCanvasEl.width = cw;
      this.camCanvasEl.height = ch;
    }
    const cover = Math.max(cw / srcW, ch / vidH_safe(srcH));
    const dw = srcW * cover;
    const dh = srcH * cover;
    this.camCtx.drawImage(srcEl, (cw - dw) * 0.5, (ch - dh) * 0.5, dw, dh);

    if (!this.qrTrackingEnabled) {
      return;
    }

    // 840px jsQR pass for fast long-range phone detection
    const targetW = Math.min(840, srcW);
    const scaleDown = targetW / srcW;
    const targetH = Math.round(srcH * scaleDown);
    if (this.procCanvas.width !== targetW || this.procCanvas.height !== targetH) {
      this.procCanvas.width = targetW;
      this.procCanvas.height = targetH;
    }

    this.procCtx.drawImage(srcEl, 0, 0, targetW, targetH);
    const imgData = this.procCtx.getImageData(0, 0, targetW, targetH);
    const invMode = this.frameCount % 5 === 0 ? 'attemptBoth' : 'dontInvert';
    const qr = jsQR(imgData.data, targetW, targetH, { inversionAttempts: invMode });

    if (qr && qr.location) {
      const invScale = 1.0 / scaleDown;
      const cornersVid = [
        { x: qr.location.topLeftCorner.x * invScale,     y: qr.location.topLeftCorner.y * invScale },
        { x: qr.location.topRightCorner.x * invScale,    y: qr.location.topRightCorner.y * invScale },
        { x: qr.location.bottomRightCorner.x * invScale, y: qr.location.bottomRightCorner.y * invScale },
        { x: qr.location.bottomLeftCorner.x * invScale,  y: qr.location.bottomLeftCorner.y * invScale },
      ];

      const pose = computePoseFromQRCorners(cornersVid, srcW, srcH, viewW, viewH, fovDeg);
      if (pose) {
        this.holdFrames = this.maxHoldFrames;
        this.onPose?.(pose.position, pose.quaternion);
        return;
      }
    }

    if (this.holdFrames > 0) {
      this.holdFrames--;
    } else {
      this.drawMinimalScanCorners(hctx, viewW, viewH);
      this.onLost?.();
    }
  }

  drawMinimalScanCorners(ctx, w, h) {
    const boxSize = Math.min(w, h) * 0.52;
    const x0 = (w - boxSize) * 0.5;
    const y0 = (h - boxSize) * 0.46;
    const len = 26;

    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';

    const corners = [
      [x0, y0 + len, x0, y0, x0 + len, y0],
      [x0 + boxSize - len, y0, x0 + boxSize, y0, x0 + boxSize, y0 + len],
      [x0 + boxSize, y0 + boxSize - len, x0 + boxSize, y0 + boxSize, x0 + boxSize - len, y0 + boxSize],
      [x0 + len, y0 + boxSize, x0, y0 + boxSize, x0, y0 + boxSize - len],
    ];
    for (const c of corners) {
      ctx.beginPath();
      ctx.moveTo(c[0], c[1]);
      ctx.lineTo(c[2], c[3]);
      ctx.lineTo(c[4], c[5]);
      ctx.stroke();
    }
    ctx.restore();
  }
}
