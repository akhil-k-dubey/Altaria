/**
 * GestureTracker - MediaPipe Tasks-Vision HandLandmarker & 3D Metric Gesture Recognizer
 * Migrated from legacy @mediapipe/hands to official @mediapipe/tasks-vision HandLandmarker.
 */

export const DEFAULT_GESTURE_CONFIG = Object.freeze({
  pinchCloseRatio: 0.40,
  pinchOpenRatio: 0.50,
  middlePinchSelectRatio: 0.38,
  fingerExtendedRatio: 1.14,
  dwellDurationMs: 540,
  fistHoldDurationMs: 360,
  domSelectCooldownMs: 650,
  partSelectCooldownMs: 500,
  pinholeFocusScale: 1.15,
  minSpreadMeters: 0.15,
  spreadRangeMeters: 0.40,
  orbitSensitivityX: 5.2,
  orbitSensitivityY: 4.2
});

class AdaptiveVectorSmoother {
  constructor(minAlpha = 0.22, maxAlpha = 0.78, velocityScale = 14.0) {
    this.minAlpha = minAlpha;
    this.maxAlpha = maxAlpha;
    this.velocityScale = velocityScale;
    this.x = null;
    this.y = null;
    this.z = null;
  }

  update(nx, ny, nz = 0) {
    if (this.x === null) {
      this.x = nx;
      this.y = ny;
      this.z = nz;
      return { x: this.x, y: this.y, z: this.z };
    }
    const vel = Math.hypot(nx - this.x, ny - this.y);
    const alpha = Math.min(this.maxAlpha, Math.max(this.minAlpha, this.minAlpha + vel * this.velocityScale));
    this.x += (nx - this.x) * alpha;
    this.y += (ny - this.y) * alpha;
    this.z += (nz - this.z) * alpha;
    return { x: this.x, y: this.y, z: this.z };
  }

  reset() {
    this.x = null;
    this.y = null;
    this.z = null;
  }
}

export class GestureTracker {
  constructor(callbacks, config = {}) {
    this.callbacks = callbacks || {};
    this.config = { ...DEFAULT_GESTURE_CONFIG, ...config };
    this.active = false;
    this.webcamStream = null;
    this.animId = null;
    this.handLandmarker = null;

    this.cursorSmoother = new AdaptiveVectorSmoother(0.25, 0.85, 18.0);
    this.orbitHandSmoother = new AdaptiveVectorSmoother(0.3, 0.85, 16.0);
    this.twoHandSpreadSmoother = new AdaptiveVectorSmoother(0.2, 0.65, 10.0);

    this.isPinchGrabbing = false;
    this.lastPinchOrbitPos = null;
    this.dwellPartIndex = -1;
    this.dwellDomEl = null;
    this.dwellStartMs = 0;
    this.dwellTriggered = false;
    this.lastPinchSelectTime = 0;
    this.fistHoldStartMs = 0;
    this.fistTriggered = false;
    this._ringCircumference = null;

    this.connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [0, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20],
      [5, 9], [9, 13], [13, 17]
    ];
  }

  _getRingCircumference(progressEl) {
    if (this._ringCircumference !== null) return this._ringCircumference;
    let circ = 0;
    if (progressEl) {
      if (typeof progressEl.getTotalLength === 'function') {
        try {
          circ = progressEl.getTotalLength();
        } catch (_) {
          circ = 0;
        }
      }
      if (!circ || !Number.isFinite(circ)) {
        const r = parseFloat(progressEl.getAttribute?.('r') || '21');
        circ = 2 * Math.PI * (Number.isFinite(r) && r > 0 ? r : 21);
      }
      progressEl.style.strokeDasharray = circ.toFixed(2);
    }
    this._ringCircumference = circ > 0 ? circ : (2 * Math.PI * 21);
    return this._ringCircumference;
  }

  dist3D(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = (a.z || 0) - (b.z || 0);
    return Math.hypot(dx, dy, dz);
  }

  dist2D(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  isFingerExtended3D(wlm, tipIdx, pipIdx, wristIdx = 0) {
    const wrist = wlm[wristIdx];
    const tipDist = this.dist3D(wlm[tipIdx], wrist);
    const pipDist = this.dist3D(wlm[pipIdx], wrist);
    return tipDist > pipDist * this.config.fingerExtendedRatio;
  }

  drawSkeleton(ctx, landmarks, width, height, accentColor = '#00f3ff') {
    ctx.save();
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = Math.max(1.5, width / 260);
    ctx.shadowColor = accentColor;
    ctx.shadowBlur = 6;

    ctx.beginPath();
    for (const [i, j] of this.connections) {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      ctx.moveTo(p1.x * width, p1.y * height);
      ctx.lineTo(p2.x * width, p2.y * height);
    }
    ctx.stroke();

    const tipRad = Math.max(2.5, width / 140);
    const jointRad = Math.max(1.6, width / 220);
    for (let i = 0; i < landmarks.length; i++) {
      const pt = landmarks[i];
      const isTip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
      ctx.fillStyle = isTip ? '#ffaa00' : '#ffffff';
      ctx.beginPath();
      ctx.arc(pt.x * width, pt.y * height, isTip ? tipRad : jointRad, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  handleResults(results, frameStartMs) {
    const canvas = document.getElementById('webcam-canvas');
    const video = document.getElementById('webcam-video');
    if (!canvas) return;
    if (video && video.videoWidth > 0 && video.videoHeight > 0) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
    }

    const ctx = canvas.getContext('2d');
    const badge = document.getElementById('gesture-state-badge');
    const latencyBadge = document.getElementById('hand-latency-badge');
    const countBadge = document.getElementById('hand-count-badge');
    const screenCursor = document.getElementById('hand-screen-cursor');
    const cursorRing = document.getElementById('hand-cursor-ring');
    const cursorLabel = document.getElementById('hand-cursor-label');
    const cursorProgress = document.getElementById('hand-cursor-progress');

    const cfg = this.config;
    const ringCirc = this._getRingCircumference(cursorProgress);
    const now = performance.now();

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const latencyMs = Math.max(2, Math.round(now - frameStartMs));
    if (latencyBadge) latencyBadge.textContent = `LATENCY: ${latencyMs} ms // TASKS-VISION`;

    const screenHands = results.landmarks || [];
    const worldHands = results.worldLandmarks || [];
    const numHands = screenHands.length;

    if (countBadge) countBadge.textContent = `HANDS: ${numHands}`;

    if (numHands === 0) {
      this.isPinchGrabbing = false;
      this.lastPinchOrbitPos = null;
      this.dwellPartIndex = -1;
      this.dwellDomEl = null;
      this.dwellStartMs = 0;
      this.dwellTriggered = false;
      this.fistHoldStartMs = 0;
      this.fistTriggered = false;
      if (screenCursor) screenCursor.classList.add('hidden');
      if (badge) {
        badge.textContent = 'READY // SHOW HAND';
        badge.className = 'font-mono-tech text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-200 border border-cyan-400/40';
      }
      return;
    }

    if (numHands >= 2) {
      const s1 = screenHands[0];
      const s2 = screenHands[1];
      const w1 = worldHands[0] || s1;
      const w2 = worldHands[1] || s2;

      this.drawSkeleton(ctx, s1, W, H, '#ffaa00');
      this.drawSkeleton(ctx, s2, W, H, '#ffaa00');

      const p1 = s1[9];
      const p2 = s2[9];

      ctx.save();
      ctx.strokeStyle = '#ffaa00';
      ctx.lineWidth = Math.max(2.0, W / 240);
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(p1.x * W, p1.y * H);
      ctx.lineTo(p2.x * W, p2.y * H);
      ctx.stroke();
      ctx.restore();

      const metricPalm1 = Math.max(0.06, this.dist3D(w1[0], w1[9]));
      const metricPalm2 = Math.max(0.06, this.dist3D(w2[0], w2[9]));
      const sensorPalm1 = Math.max(0.03, this.dist2D(s1[0], s1[9]));
      const sensorPalm2 = Math.max(0.03, this.dist2D(s2[0], s2[9]));

      const Z1 = (cfg.pinholeFocusScale * metricPalm1) / sensorPalm1;
      const Z2 = (cfg.pinholeFocusScale * metricPalm2) / sensorPalm2;

      const camAspect = W / Math.max(1, H);
      const camP1 = { x: (p1.x - 0.5) * Z1 * camAspect, y: (p1.y - 0.5) * Z1, z: Z1 };
      const camP2 = { x: (p2.x - 0.5) * Z2 * camAspect, y: (p2.y - 0.5) * Z2, z: Z2 };

      const metricDistance3D = this.dist3D(camP1, camP2);
      const smoothedDistance = this.twoHandSpreadSmoother.update(metricDistance3D, 0).x;
      const mappedExplosion = Math.max(
        0,
        Math.min(1, (smoothedDistance - cfg.minSpreadMeters) / cfg.spreadRangeMeters)
      );

      if (this.callbacks.onExplosionChange) {
        this.callbacks.onExplosionChange(mappedExplosion);
      }

      const midCamX = (camP1.x + camP2.x) * 0.5;
      const midCamY = (camP1.y + camP2.y) * 0.5;
      if (this.callbacks.onParallax) {
        this.callbacks.onParallax(-midCamX * 1.4, -midCamY * 1.2);
      }

      if (screenCursor) screenCursor.classList.add('hidden');
      this.isPinchGrabbing = false;
      this.lastPinchOrbitPos = null;
      this.dwellPartIndex = -1;
      this.dwellDomEl = null;
      this.dwellStartMs = 0;
      this.dwellTriggered = false;
      this.fistHoldStartMs = 0;
      this.fistTriggered = false;

      const pct = Math.round(mappedExplosion * 100);
      const distCm = Math.round(smoothedDistance * 100);
      if (badge) {
        badge.textContent = `🙌 2-HAND METRIC: ${distCm}cm (${pct}%)`;
        badge.className = 'font-mono-tech text-[10px] px-1.5 py-0.5 rounded bg-amber-500/30 text-amber-200 border border-amber-400';
      }
      return;
    }

    const slm = screenHands[0];
    const wlm = worldHands[0] || slm;

    const indexTipScreen = slm[8];
    const palmCenterScreen = slm[9];

    const wristWorld = wlm[0];
    const thumbTipWorld = wlm[4];
    const indexTipWorld = wlm[8];
    const middleTipWorld = wlm[12];
    const palmCenterWorld = wlm[9];

    const metricPalmScale = Math.max(0.06, this.dist3D(wristWorld, palmCenterWorld));
    const metricPinchDistance = this.dist3D(thumbTipWorld, indexTipWorld);
    const metricMiddlePinchDistance = this.dist3D(thumbTipWorld, middleTipWorld);

    const pinchRatio = metricPinchDistance / metricPalmScale;
    const middlePinchRatio = metricMiddlePinchDistance / metricPalmScale;

    const indexExt = this.isFingerExtended3D(wlm, 8, 6);
    const middleExt = this.isFingerExtended3D(wlm, 12, 10);
    const ringExt = this.isFingerExtended3D(wlm, 16, 14);
    const pinkyExt = this.isFingerExtended3D(wlm, 20, 18);
    const extendedCount = [indexExt, middleExt, ringExt, pinkyExt].filter(Boolean).length;

    if (!this.isPinchGrabbing && pinchRatio < cfg.pinchCloseRatio) {
      this.isPinchGrabbing = true;
      if (this.callbacks.onSound) this.callbacks.onSound(1420, 0.05);
    } else if (this.isPinchGrabbing && pinchRatio > cfg.pinchOpenRatio) {
      this.isPinchGrabbing = false;
      this.lastPinchOrbitPos = null;
    }

    const rawScreenNormX = Math.max(0, Math.min(1, 1.0 - indexTipScreen.x));
    const rawScreenNormY = Math.max(0, Math.min(1, indexTipScreen.y));
    const smoothCursor = this.cursorSmoother.update(rawScreenNormX, rawScreenNormY);
    const screenX = smoothCursor.x * window.innerWidth;
    const screenY = smoothCursor.y * window.innerHeight;

    if (this.callbacks.onParallax) {
      this.callbacks.onParallax(((1.0 - palmCenterScreen.x) - 0.5) * 1.1, -(palmCenterScreen.y - 0.5) * 0.75);
    }

    if (screenCursor) {
      screenCursor.classList.remove('hidden');
      screenCursor.style.transform = `translate3d(${screenX.toFixed(1)}px, ${screenY.toFixed(1)}px, 0)`;
    }

    if (this.isPinchGrabbing) {
      this.dwellPartIndex = -1;
      this.dwellDomEl = null;
      this.drawSkeleton(ctx, slm, W, H, '#ffaa00');

      ctx.save();
      ctx.strokeStyle = '#ffaa00';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(slm[4].x * W, slm[4].y * H, Math.max(8, W / 45), 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      const grabPoint = this.orbitHandSmoother.update(1.0 - palmCenterScreen.x, palmCenterScreen.y);
      if (this.lastPinchOrbitPos && this.callbacks.onOrbitDelta) {
        const dx = grabPoint.x - this.lastPinchOrbitPos.x;
        const dy = grabPoint.y - this.lastPinchOrbitPos.y;
        this.callbacks.onOrbitDelta(dx * cfg.orbitSensitivityX, dy * cfg.orbitSensitivityY);
      }
      this.lastPinchOrbitPos = { x: grabPoint.x, y: grabPoint.y };

      if (cursorRing) cursorRing.className = 'relative w-10 h-10 rounded-full border-2 border-amber-400 bg-amber-500/20 shadow-[0_0_22px_#ffaa00] flex items-center justify-center scale-90 transition-all duration-75';
      if (cursorLabel) {
        cursorLabel.textContent = '🤏 PINCH GRAB // ORBITING';
        cursorLabel.className = 'mt-1 px-2 py-0.5 rounded bg-black/85 border border-amber-400 font-mono-tech text-[10px] text-amber-300 whitespace-nowrap text-center';
      }
      if (cursorProgress) cursorProgress.style.strokeDashoffset = ringCirc.toFixed(2);

      if (badge) {
        const pinchCm = (metricPinchDistance * 100).toFixed(1);
        badge.textContent = `🤏 3D PINCH: ${pinchCm}cm`;
        badge.className = 'font-mono-tech text-[10px] px-1.5 py-0.5 rounded bg-amber-500/30 text-amber-200 border border-amber-400';
      }
      return;
    }

    this.lastPinchOrbitPos = null;

    if (extendedCount === 0 && pinchRatio >= cfg.pinchCloseRatio) {
      this.dwellPartIndex = -1;
      this.dwellDomEl = null;
      this.drawSkeleton(ctx, slm, W, H, '#38bdf8');
      if (this.fistHoldStartMs === 0) {
        this.fistHoldStartMs = now;
        this.fistTriggered = false;
      }
      const elapsedMs = now - this.fistHoldStartMs;
      const progress = Math.min(1, elapsedMs / cfg.fistHoldDurationMs);
      if (cursorProgress) cursorProgress.style.strokeDashoffset = (ringCirc * (1 - progress)).toFixed(2);
      if (cursorLabel) cursorLabel.textContent = '✊ FIST // ASSEMBLING CORE';

      if (!this.fistTriggered && elapsedMs >= cfg.fistHoldDurationMs && this.callbacks.onAssembleSnap) {
        this.fistTriggered = true;
        this.callbacks.onAssembleSnap();
      }

      if (badge) {
        badge.textContent = '✊ FIST: ASSEMBLE 0%';
        badge.className = 'font-mono-tech text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/30 text-cyan-100 border border-cyan-400';
      }
      return;
    }
    this.fistHoldStartMs = 0;
    this.fistTriggered = false;

    this.drawSkeleton(ctx, slm, W, H, '#00f3ff');
    if (cursorRing) cursorRing.className = 'relative w-12 h-12 rounded-full border-2 border-cyan-400 shadow-[0_0_18px_#00f3ff] flex items-center justify-center transition-all duration-75';
    if (cursorLabel) cursorLabel.className = 'mt-1 px-2 py-0.5 rounded bg-black/80 border border-cyan-400/60 font-mono-tech text-[10px] text-cyan-200 whitespace-nowrap text-center';

    const rawPointEl = typeof document.elementFromPoint === 'function'
      ? document.elementFromPoint(screenX, screenY)
      : null;
    const domDwellTarget = rawPointEl && typeof rawPointEl.closest === 'function'
      ? rawPointEl.closest('[data-dwell-target]')
      : null;

    if (domDwellTarget) {
      if (cursorRing) cursorRing.classList.add('border-amber-300', 'scale-110');
      this.dwellPartIndex = -1;
      if (this.dwellDomEl !== domDwellTarget) {
        this.dwellDomEl = domDwellTarget;
        this.dwellStartMs = now;
        this.dwellTriggered = false;
      }

      const elapsedMs = now - this.dwellStartMs;
      const dwellRatio = Math.min(1, elapsedMs / cfg.dwellDurationMs);
      if (cursorProgress) cursorProgress.style.strokeDashoffset = (ringCirc * (1 - dwellRatio)).toFixed(2);
      if (cursorLabel) cursorLabel.textContent = domDwellTarget.getAttribute('data-dwell-label') || '🎯 PINCH / DWELL SELECT';

      const dwellComplete = !this.dwellTriggered && elapsedMs >= cfg.dwellDurationMs;
      const pinchSelect = middlePinchRatio < cfg.middlePinchSelectRatio;
      if ((dwellComplete || pinchSelect) && now - this.lastPinchSelectTime > cfg.domSelectCooldownMs) {
        this.lastPinchSelectTime = now;
        this.dwellTriggered = true;
        this.dwellStartMs = now;
        domDwellTarget.click();
      }

      if (badge) {
        badge.textContent = '🎯 HUD TARGET LOCKED';
        badge.className = 'font-mono-tech text-[10px] px-1.5 py-0.5 rounded bg-amber-500/25 text-amber-200 border border-amber-400/60';
      }
      return;
    }

    this.dwellDomEl = null;

    let hitIdx = -1;
    if (this.callbacks.onRaycast) {
      hitIdx = this.callbacks.onRaycast(screenX, screenY);
    }

    if (hitIdx !== -1) {
      if (cursorRing) cursorRing.classList.add('border-amber-300', 'scale-110');

      if (this.dwellPartIndex !== hitIdx) {
        this.dwellPartIndex = hitIdx;
        this.dwellStartMs = now;
        this.dwellTriggered = false;
      }

      const elapsedMs = now - this.dwellStartMs;
      const dwellRatio = Math.min(1, elapsedMs / cfg.dwellDurationMs);
      if (cursorProgress) cursorProgress.style.strokeDashoffset = (ringCirc * (1 - dwellRatio)).toFixed(2);

      const dwellComplete = !this.dwellTriggered && elapsedMs >= cfg.dwellDurationMs;
      const pinchSelect = middlePinchRatio < cfg.middlePinchSelectRatio;
      if ((dwellComplete || pinchSelect) && now - this.lastPinchSelectTime > cfg.partSelectCooldownMs) {
        this.lastPinchSelectTime = now;
        this.dwellTriggered = true;
        if (this.callbacks.onSelectPart) this.callbacks.onSelectPart(hitIdx);
      }

      if (badge) {
        badge.textContent = '🎯 TARGET LOCKED';
        badge.className = 'font-mono-tech text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/25 text-emerald-200 border border-emerald-400/60';
      }
    } else {
      this.dwellPartIndex = -1;
      this.dwellStartMs = 0;
      this.dwellTriggered = false;
      if (cursorProgress) cursorProgress.style.strokeDashoffset = ringCirc.toFixed(2);
      if (cursorLabel) cursorLabel.textContent = extendedCount >= 4 ? '🖐️ OPEN PALM // HOVER' : '☝️ LASER POINTER';

      if (badge) {
        badge.textContent = '☝️ LASER POINT ACTIVE';
        badge.className = 'font-mono-tech text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/25 text-emerald-200 border border-emerald-400/60';
      }
    }
  }

  async initHandLandmarker() {
    if (this.handLandmarker) return this.handLandmarker;

    let visionModule;
    try {
      visionModule = await import('./vendor/vision_bundle.mjs');
    } catch (_) {
      visionModule = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs');
    }

    const { FilesetResolver, HandLandmarker } = visionModule;

    let vision;
    try {
      vision = await FilesetResolver.forVisionTasks('/data/wasm');
    } catch (_) {
      vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
    }

    try {
      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/data/models/hand_landmarker.task',
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.55
      });
    } catch (err) {
      console.warn('[!] GPU delegate or local model failed, attempting CPU fallback:', err);
      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'CPU'
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.55
      });
    }

    return this.handLandmarker;
  }

  async toggle() {
    const webcamHud = document.getElementById('webcam-hud');
    const dot = document.getElementById('tracking-dot');
    const label = document.getElementById('tracking-btn-label');
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const screenCursor = document.getElementById('hand-screen-cursor');

    if (this.active) {
      this.active = false;
      if (this.animId) cancelAnimationFrame(this.animId);
      if (this.webcamStream) {
        this.webcamStream.getTracks().forEach((t) => t.stop());
        this.webcamStream = null;
      }
      this.isPinchGrabbing = false;
      this.dwellPartIndex = -1;
      this.dwellDomEl = null;
      this.dwellStartMs = 0;
      this.dwellTriggered = false;
      this.fistHoldStartMs = 0;
      if (screenCursor) screenCursor.classList.add('hidden');
      if (webcamHud) webcamHud.classList.add('hidden');
      if (dot) dot.className = 'w-2.5 h-2.5 rounded-full bg-slate-500';
      if (label) label.textContent = 'ENABLE HAND / HEAD TRACKING';
      return false;
    }

    if (!video || !canvas || !navigator.mediaDevices?.getUserMedia) {
      return false;
    }

    try {
      if (label) label.textContent = 'BOOTING NEURAL HAND TRACKER...';
      this.webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 360 }, frameRate: { ideal: 60 } },
        audio: false
      });
      video.srcObject = this.webcamStream;
      await video.play();

      canvas.width = video.videoWidth || canvas.clientWidth || 640;
      canvas.height = video.videoHeight || canvas.clientHeight || 360;

      await this.initHandLandmarker();

      this.active = true;
      this.cursorSmoother.reset();
      this.orbitHandSmoother.reset();
      this.twoHandSpreadSmoother.reset();

      if (webcamHud) webcamHud.classList.remove('hidden');
      if (dot) dot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]';
      if (label) label.textContent = '21-PT HANDLANDMARKER ONLINE';

      let lastVideoTime = -1;
      const pumpVideoFrame = () => {
        if (!this.active) return;

        if (video.currentTime !== lastVideoTime && video.readyState >= 2 && this.handLandmarker) {
          lastVideoTime = video.currentTime;
          const frameStartTimestamp = performance.now();
          try {
            const results = this.handLandmarker.detectForVideo(video, frameStartTimestamp);
            this.handleResults(results, frameStartTimestamp);
          } catch (e) {
            console.warn('[!] Frame detection warning:', e);
          }
        }
        this.animId = requestAnimationFrame(pumpVideoFrame);
      };

      pumpVideoFrame();
      return true;
    } catch (err) {
      console.error('Webcam Hand Tracking Error:', err);
      if (label) label.textContent = 'ENABLE HAND / HEAD TRACKING';
      return false;
    }
  }
}