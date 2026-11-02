import jsQR from 'jsqr';
import * as THREE from 'three';
import QRCode from 'qrcode';

function vidH_safe(h) {
  return h > 0 ? h : 1;
}

/**
 * Solve 8x8 linear system A * x = b with partial pivoting.
 */
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

    const pivot = M[col][col];
    for (let r = col + 1; r < n; r++) {
      const factor = M[r][col] / pivot;
      for (let c = col; c <= n; c++) {
        M[r][c] -= factor * M[col][c];
      }
    }
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= M[i][j] * x[j];
    }
    x[i] = sum / M[i][i];
  }
  return x;
}

/**
 * Invert 3x3 matrix (array of 9 floats in row-major order).
 */
function invertTranspose3x3(R) {
  const [r00, r01, r02, r10, r11, r12, r20, r21, r22] = R;
  const c00 = r11 * r22 - r12 * r21;
  const c01 = -(r10 * r22 - r12 * r20);
  const c02 = r10 * r21 - r11 * r20;

  const c10 = -(r01 * r22 - r02 * r21);
  const c11 = r00 * r22 - r02 * r20;
  const c12 = -(r00 * r21 - r01 * r20);

  const c20 = r01 * r12 - r02 * r11;
  const c21 = -(r00 * r12 - r02 * r10);
  const c22 = r00 * r11 - r01 * r10;

  const det = r00 * c00 + r01 * c01 + r02 * c02;
  if (Math.abs(det) < 1e-11) return null;
  const invDet = 1.0 / det;
  // Cofactor matrix divided by det IS already (R^-1)^T !
  return [
    c00 * invDet, c01 * invDet, c02 * invDet,
    c10 * invDet, c11 * invDet, c12 * invDet,
    c20 * invDet, c21 * invDet, c22 * invDet,
  ];
}

/**
 * Exact 6-DoF planar homography pose estimation from 4 QR corners.
 * Marker local frame:
 *   Origin (0,0,0) = center of QR code
 *   +X = left -> right of QR code
 *   +Y = UP out of the QR code surface toward viewer (wheels sit at Y = 0!)
 *   +Z = top -> bottom of QR code
 */
export function computePoseFromQRCorners(cornersVid, vidW, vidH, viewW, viewH, fovDeg) {
  const coverScale = Math.max(viewW / vidW, viewH / vidH);
  const fPx = (viewH * 0.5) / Math.tan((fovDeg * Math.PI) / 360.0);

  const srcXZ = [
    [-0.5, -0.5], // topLeft
    [+0.5, -0.5], // topRight
    [+0.5, +0.5], // bottomRight
    [-0.5, +0.5], // bottomLeft
  ];

  const viewPts = cornersVid.map((pt) => {
    const uView = (pt.x - vidW * 0.5) * coverScale + viewW * 0.5;
    const vView = (pt.y - vidH * 0.5) * coverScale + viewH * 0.5;
    return {
      u: uView,
      v: vView,
      xn: (uView - viewW * 0.5) / fPx,
      yn: (viewH * 0.5 - vView) / fPx,
    };
  });

  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const [xm, zm] = srcXZ[i];
    const { xn, yn } = viewPts[i];
    A.push([xm, zm, 1, 0, 0, 0, -xm * xn, -zm * xn]);
    b.push(xn);
    A.push([0, 0, 0, xm, zm, 1, -xm * yn, -zm * yn]);
    b.push(yn);
  }

  const h = solve8x8(A, b);
  if (!h) return null;

  // M = diag(1, 1, -1) * H; enforce M[2][2] < 0 (marker in front of Three.js camera, Z_c < 0)
  let sgn = -1.0 > 0 ? 1.0 : 1.0;
  // M row 2 is [-h[6], -h[7], -1.0]; M[2][2] = -1.0 < 0 already! Check if h3_z had opposite sign:
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
  // r2 = r3 x r1 (right-handed: +Z x +X = +Y UP!)
  let r2x = r3y * r1z - r3z * r1y;
  let r2y = r3z * r1x - r3x * r1z;
  let r2z = r3x * r1y - r3y * r1x;

  let R = [
    r1x, r2x, r3x,
    r1y, r2y, r3y,
    r1z, r2z, r3z,
  ];

  // 5-step polar decomposition orthogonalization onto SO(3)
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

  // Reject implausible poses (too close or behind camera or absurdly far)
  if (tz > -0.18 || tz < -25.0 || !Number.isFinite(tx)) return null;

  const basisMat = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(R[0], R[3], R[6]),
    new THREE.Vector3(R[1], R[4], R[7]),
    new THREE.Vector3(R[2], R[5], R[8])
  );
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(basisMat);
  const position = new THREE.Vector3(tx, ty, tz);

  return {
    position,
    quaternion,
    viewPts,
  };
}

/**
 * Manages live camera stream (or interactive synthetic QR tabletop simulator) + jsQR tracking.
 */
export class QRCameraTracker {
  constructor({ videoEl, simBgCanvasEl, hudCanvasEl, onPose, onLost, onStatus }) {
    this.videoEl = videoEl;
    this.simBgCanvasEl = simBgCanvasEl;
    this.simBgCtx = simBgCanvasEl ? simBgCanvasEl.getContext('2d') : null;
    this.hudCanvasEl = hudCanvasEl;
    this.hudCtx = hudCanvasEl.getContext('2d');

    this.procCanvas = document.createElement('canvas');
    this.procCtx = this.procCanvas.getContext('2d', { willReadFrequently: true });

    this.onPose = onPose;
    this.onLost = onLost;
    this.onStatus = onStatus;

    this.active = false;
    this.simMode = false;
    this.facingMode = 'environment';
    this.stream = null;
    this.frameCount = 0;
    this.holdFrames = 0;
    this.maxHoldFrames = 14;
    this.lastDecodedText = '';

    // Synthetic simulator assets
    this.simCanvas = document.createElement('canvas');
    this.simCanvas.width = 960;
    this.simCanvas.height = 540;
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
      throw new Error('Браузер не поддерживает доступ к камере (требуется HTTPS).');
    }

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.videoEl.srcObject = this.stream;
    this.videoEl.setAttribute('playsinline', 'true');
    await this.videoEl.play();

    this.active = true;
    this.holdFrames = 0;
    this.onStatus?.('camera_ready', 'Наведите камеру на любой QR-код (или маркер справа)');
    return true;
  }

  async startSimulation(urlText) {
    this.stop();
    await this.prepareSimQR(urlText);
    this.simMode = true;
    this.active = true;
    this.holdFrames = 0;
    this.onStatus?.('sim_ready', 'Демо-симуляция: виртуальная камера наведена на печатный QR-маркер');
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
    if (this.simBgCtx && this.simBgCanvasEl) {
      this.simBgCtx.clearRect(0, 0, this.simBgCanvasEl.width, this.simBgCanvasEl.height);
    }
  }

  renderSyntheticTabletop(timeSec) {
    const W = this.simCanvas.width;
    const H = this.simCanvas.height;
    const ctx = this.simCtx;

    // Realistic desk surface background
    const grad = ctx.createRadialGradient(W * 0.5, H * 0.45, 50, W * 0.5, H * 0.5, W * 0.72);
    grad.addColorStop(0, '#2a3444');
    grad.addColorStop(1, '#111824');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // Subtle tactical desk grid lines
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.12)';
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 48) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += 48) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Gently orbiting / tilting QR card on the table so jsQR detects real pixels!
    if (this.simQrImg) {
      ctx.save();
      const cx = W * 0.5 + Math.sin(timeSec * 0.65) * 42;
      const cy = H * 0.63 + Math.cos(timeSec * 0.5) * 16;
      const rot = Math.sin(timeSec * 0.45) * 0.22;
      ctx.translate(cx, cy);
      ctx.scale(1.0, 0.70 + Math.sin(timeSec * 0.4) * 0.05); // realistic tabletop foreshortening!
      ctx.rotate(rot);

      // White printed A5 marker sheet shadow + sheet
      ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
      ctx.fillRect(-142, -142, 296, 296);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(-148, -148, 296, 296);
      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 5;
      ctx.strokeRect(-138, -138, 276, 276);

      ctx.drawImage(this.simQrImg, -112, -112, 224, 224);
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
      this.renderSyntheticTabletop(timeSec);
      srcEl = this.simCanvas;
      srcW = this.simCanvas.width;
      srcH = this.simCanvas.height;

      // Draw synthetic camera feed onto background canvas BEHIND Three.js canvas
      if (this.simBgCanvasEl && this.simBgCtx) {
        if (this.simBgCanvasEl.width !== viewW || this.simBgCanvasEl.height !== viewH) {
          this.simBgCanvasEl.width = viewW;
          this.simBgCanvasEl.height = viewH;
        }
        const cover = Math.max(viewW / srcW, viewH / vidH_safe(srcH));
        const dw = srcW * cover;
        const dh = srcH * cover;
        this.simBgCtx.drawImage(srcEl, (viewW - dw) * 0.5, (viewH - dh) * 0.5, dw, dh);
      }
    } else if (srcW < 32 || srcH < 32) {
      return;
    }

    // Downsample to ~640px wide for ultra-fast 60 FPS jsQR detection
    const targetW = Math.min(640, srcW);
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
        this.lastDecodedText = qr.data || '';
        this.drawDetectedReticle(hctx, pose.viewPts);
        this.onPose?.(pose.position, pose.quaternion, this.lastDecodedText);
        return;
      }
    }

    // Brief occlusion hold so fast hand movement doesn't flicker
    if (this.holdFrames > 0) {
      this.holdFrames--;
    } else {
      this.drawScanningGuide(hctx, viewW, viewH, timeSec);
      this.onLost?.();
    }
  }

  drawDetectedReticle(ctx, pts) {
    ctx.save();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.85)';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(pts[0].u, pts[0].v);
    ctx.lineTo(pts[1].u, pts[1].v);
    ctx.lineTo(pts[2].u, pts[2].v);
    ctx.lineTo(pts[3].u, pts[3].v);
    ctx.closePath();
    ctx.stroke();

    // Glowing rescue-orange corner dots
    pts.forEach((p, idx) => {
      ctx.fillStyle = idx === 0 ? '#f97316' : '#38bdf8';
      ctx.beginPath();
      ctx.arc(p.u, p.v, 5.5, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  }

  drawScanningGuide(ctx, w, h, t) {
    const boxSize = Math.min(w, h) * 0.56;
    const x0 = (w - boxSize) * 0.5;
    const y0 = (h - boxSize) * 0.5;
    const len = 32;

    ctx.save();
    ctx.strokeStyle = 'rgba(249, 115, 22, 0.88)';
    ctx.lineWidth = 3.5;
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

    // Animated scan line
    const sy = y0 + ((Math.sin(t * 2.4) * 0.5 + 0.5) * boxSize);
    const grad = ctx.createLinearGradient(x0, sy, x0 + boxSize, sy);
    grad.addColorStop(0, 'rgba(56, 189, 248, 0)');
    grad.addColorStop(0.5, 'rgba(56, 189, 248, 0.85)');
    grad.addColorStop(1, 'rgba(56, 189, 248, 0)');
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(x0 + 8, sy);
    ctx.lineTo(x0 + boxSize - 8, sy);
    ctx.stroke();
    ctx.restore();
  }
}
