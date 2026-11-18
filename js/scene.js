/**
 * SceneManager - Three.js 3D Viewport, Camera, Quantum Grid & Pepper's Ghost Projections
 * Driven by structured configuration with zero inline magic numbers.
 */

export const DEFAULT_SCENE_CONFIG = Object.freeze({
  camera: Object.freeze({
    fov: 45,
    near: 0.1,
    far: 1000,
    initialDistance: 8.5,
    baseElevationRatio: 0.17,
    lookAtTarget: Object.freeze([0, 0, 0])
  }),
  renderer: Object.freeze({
    maxPixelRatio: 2,
    clearColorHex: 0x02060d,
    pyramidClearColorHex: 0x000000,
    clearAlpha: 1,
    toneMappingExposure: 1.0
  }),
  postprocessing: Object.freeze({
    bloomEnabled: true,
    bloomStrength: 0.28,
    bloomRadius: 0.34,
    bloomThreshold: 0.85
  }),
  lighting: Object.freeze({
    ambientColor: 0x00f3ff,
    ambientIntensity: 0.65,
    keyColor: 0x00f3ff,
    keyIntensity: 1.3,
    keyPosition: Object.freeze([6, 10, 8]),
    rimColor: 0xffaa00,
    rimIntensity: 1.1,
    rimDistance: 25,
    rimPosition: Object.freeze([-7, -5, -6])
  }),
  environment: Object.freeze({
    floorOffsetY: -3.2,
    polarRadius: 6.5,
    polarRadials: 16,
    polarCircles: 6,
    polarDivisions: 64,
    polarColor1: 0x0077ff,
    polarColor2: 0x00f3ff,
    polarOpacity: 0.22,
    particleCount: 240,
    particleSpread: Object.freeze([18, 12, 18]),
    particleColor: 0x00f3ff,
    particleSize: 0.045,
    particleOpacity: 0.45,
    particleRotationRadPerSec: 0.036
  }),
  pyramid: Object.freeze({
    quadrantSizeRatio: 0.40,
    centerGapRatio: 0.12
  })
});

export class SceneManager {
  constructor(container, config = {}) {
    this.container = container;
    this.config = {
      camera: { ...DEFAULT_SCENE_CONFIG.camera, ...(config.camera || {}) },
      renderer: { ...DEFAULT_SCENE_CONFIG.renderer, ...(config.renderer || {}) },
      postprocessing: { ...DEFAULT_SCENE_CONFIG.postprocessing, ...(config.postprocessing || {}) },
      lighting: { ...DEFAULT_SCENE_CONFIG.lighting, ...(config.lighting || {}) },
      environment: { ...DEFAULT_SCENE_CONFIG.environment, ...(config.environment || {}) },
      pyramid: { ...DEFAULT_SCENE_CONFIG.pyramid, ...(config.pyramid || {}) }
    };

    this.scene = new THREE.Scene();

    const camCfg = this.config.camera;
    this.camera = new THREE.PerspectiveCamera(
      camCfg.fov,
      window.innerWidth / Math.max(1, window.innerHeight),
      camCfg.near,
      camCfg.far
    );
    this.cameraDistance = camCfg.initialDistance;
    this.targetCameraDistance = camCfg.initialDistance;
    this.minCameraDistance = 2.0;
    this.maxCameraDistance = 22.0;
    this.lookAtTarget = new THREE.Vector3(...camCfg.lookAtTarget);
    this.headParallaxOffset = { x: 0, y: 0 };

    const renCfg = this.config.renderer;
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    const initW = Math.max(1, window.innerWidth);
    const initH = Math.max(1, window.innerHeight);
    const initPR = Math.min(window.devicePixelRatio || 1, renCfg.maxPixelRatio);
    this.renderer.setSize(initW, initH);
    this.renderer.setPixelRatio(initPR);
    if (THREE.ACESFilmicToneMapping !== undefined) {
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = renCfg.toneMappingExposure ?? 1.05;
    }
    if (THREE.sRGBEncoding !== undefined) {
      this.renderer.outputEncoding = THREE.sRGBEncoding;
    }

    this._clearColorRaw = new THREE.Color(renCfg.clearColorHex);
    this._clearColorLinear = this._clearColorRaw.clone();
    if (typeof this._clearColorLinear.convertSRGBToLinear === 'function') {
      this._clearColorLinear.convertSRGBToLinear();
    }
    this.renderer.setClearColor(this._clearColorRaw, renCfg.clearAlpha);
    if (this.container) {
      this.container.appendChild(this.renderer.domElement);
    }

    this.composer = null;
    this.renderPass = null;
    this.bloomPass = null;
    this.hdrRenderTarget = null;
    this._setupPostProcessing(initW, initH, initPR);

    this.masterGroup = new THREE.Group();
    this.scene.add(this.masterGroup);

    this.polarGrid = null;
    this.pyramidMode = false;
    this.pyramidParallaxMode = 'symmetric';

    this._onWindowResize = () => {
      const w = Math.max(1, window.innerWidth);
      const h = Math.max(1, window.innerHeight);
      const pr = Math.min(window.devicePixelRatio || 1, this.config.renderer.maxPixelRatio);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setPixelRatio(pr);
      this.renderer.setSize(w, h);
      if (this.composer && typeof this.composer.setPixelRatio === 'function') {
        this.composer.setPixelRatio(pr);
      }
      if (this.composer && typeof this.composer.setSize === 'function') {
        this.composer.setSize(w, h);
      }
      if (this.bloomPass && typeof this.bloomPass.setSize === 'function') {
        this.bloomPass.setSize(w, h);
      }
    };

    this.setupLighting();
    this.setupEnvironment();
    this.setupResizeListener();
  }

  _setupPostProcessing(width, height, pixelRatio = 1) {
    const ppCfg = this.config.postprocessing;
    if (!ppCfg || !ppCfg.bloomEnabled) return;
    if (
      typeof THREE.EffectComposer !== 'function' ||
      typeof THREE.RenderPass !== 'function' ||
      typeof THREE.UnrealBloomPass !== 'function'
    ) {
      return;
    }
    try {
      const isWebGL2 = Boolean(this.renderer.capabilities?.isWebGL2);
      const rtType =
        THREE.HalfFloatType && (isWebGL2 || this.renderer.getContext?.()?.getExtension?.('OES_texture_half_float'))
          ? THREE.HalfFloatType
          : THREE.UnsignedByteType;
      const rtW = Math.max(1, Math.round(width * pixelRatio));
      const rtH = Math.max(1, Math.round(height * pixelRatio));
      if (typeof THREE.WebGLRenderTarget === 'function' && THREE.RGBAFormat !== undefined) {
        this.hdrRenderTarget = new THREE.WebGLRenderTarget(rtW, rtH, {
          minFilter: THREE.LinearFilter,
          magFilter: THREE.LinearFilter,
          format: THREE.RGBAFormat,
          type: rtType
        });
        this.composer = new THREE.EffectComposer(this.renderer, this.hdrRenderTarget);
      } else {
        this.composer = new THREE.EffectComposer(this.renderer);
      }
      if (typeof this.composer.setPixelRatio === 'function') {
        this.composer.setPixelRatio(pixelRatio);
      }
      if (typeof this.composer.setSize === 'function') {
        this.composer.setSize(width, height);
      }
      this.renderPass = new THREE.RenderPass(this.scene, this.camera);
      this.composer.addPass(this.renderPass);

      const resVec = new THREE.Vector2(width, height);
      this.bloomPass = new THREE.UnrealBloomPass(
        resVec,
        ppCfg.bloomStrength ?? 0.38,
        ppCfg.bloomRadius ?? 0.42,
        ppCfg.bloomThreshold ?? 0.78
      );
      this.bloomPass.renderToScreen = true;
      this.composer.addPass(this.bloomPass);
    } catch (e) {
      console.warn('[!] Postprocessing HDR bloom unavailable, using standard renderer:', e);
      this.composer = null;
      this.renderPass = null;
      this.bloomPass = null;
      this.hdrRenderTarget?.dispose?.();
      this.hdrRenderTarget = null;
    }
  }

  setupLighting() {
    const lCfg = this.config.lighting;

    const ambientLight = new THREE.AmbientLight(lCfg.ambientColor, lCfg.ambientIntensity);
    this.scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(lCfg.keyColor, lCfg.keyIntensity);
    keyLight.position.set(...lCfg.keyPosition);
    this.scene.add(keyLight);

    const rimLight = new THREE.PointLight(lCfg.rimColor, lCfg.rimIntensity, lCfg.rimDistance);
    rimLight.position.set(...lCfg.rimPosition);
    this.scene.add(rimLight);
  }

  setupEnvironment() {
    const envCfg = this.config.environment;

    this.floorGroup = new THREE.Group();
    this.floorGroup.position.y = envCfg.floorOffsetY;
    this.polarGrid = new THREE.PolarGridHelper(
      envCfg.polarRadius,
      envCfg.polarRadials,
      envCfg.polarCircles,
      envCfg.polarDivisions,
      envCfg.polarColor1,
      envCfg.polarColor2
    );
    this.polarGrid.material.transparent = true;
    this.polarGrid.material.opacity = envCfg.polarOpacity;
    this.floorGroup.add(this.polarGrid);
    this.scene.add(this.floorGroup);

    const particleCount = envCfg.particleCount;
    const [spreadX, spreadY, spreadZ] = envCfg.particleSpread;
    const particleGeo = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
      particlePositions[i] = (Math.random() - 0.5) * spreadX;
      particlePositions[i + 1] = (Math.random() - 0.5) * spreadY;
      particlePositions[i + 2] = (Math.random() - 0.5) * spreadZ;
    }
    particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    const particleMat = new THREE.PointsMaterial({
      color: envCfg.particleColor,
      size: envCfg.particleSize,
      transparent: true,
      opacity: envCfg.particleOpacity,
      blending: THREE.AdditiveBlending
    });
    this.ambientParticles = new THREE.Points(particleGeo, particleMat);
    this.scene.add(this.ambientParticles);
  }

  setupResizeListener() {
    window.addEventListener('resize', this._onWindowResize);
  }

  fitToAssembly(targetGroup = this.masterGroup, { updateZoom = true, immediate = true } = {}) {
    if (!targetGroup || typeof THREE.Box3 !== 'function') return;

    const prevRotX = targetGroup.rotation.x;
    const prevRotY = targetGroup.rotation.y;
    const prevRotZ = targetGroup.rotation.z;
    targetGroup.rotation.set(0, 0, 0);
    targetGroup.updateMatrixWorld(true);

    const box = new THREE.Box3();
    const tmpBox = new THREE.Box3();
    let hasMesh = false;

    for (const child of targetGroup.children) {
      if (child?.userData?.persistentOverlay === true) continue;
      if (!child.visible) continue;
      child.traverse?.((node) => {
        if (node.isMesh && node.geometry) {
          if (!node.geometry.boundingBox && typeof node.geometry.computeBoundingBox === 'function') {
            node.geometry.computeBoundingBox();
          }
          if (node.geometry.boundingBox && typeof tmpBox.copy === 'function' && typeof tmpBox.applyMatrix4 === 'function' && node.matrixWorld) {
            tmpBox.copy(node.geometry.boundingBox).applyMatrix4(node.matrixWorld);
          } else {
            tmpBox.setFromObject(node);
          }
          if (!tmpBox.isEmpty()) {
            box.union(tmpBox);
            hasMesh = true;
          }
        }
      });
    }

    targetGroup.rotation.set(prevRotX, prevRotY, prevRotZ);
    targetGroup.updateMatrixWorld(true);

    if (!hasMesh || box.isEmpty()) return;

    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);
    const radius = Math.max(0.55, Number.isFinite(sphere.radius) ? sphere.radius : 1.0);

    const vFovRad = (this.camera.fov * Math.PI) / 180.0;
    const aspect = Math.max(0.5, this.camera.aspect || (window.innerWidth / Math.max(1, window.innerHeight)));
    const hFovRad = 2.0 * Math.atan(Math.tan(vFovRad * 0.5) * aspect);
    const halfMinFov = Math.min(vFovRad, hFovRad) * 0.5;
    const fitDist = (radius * 1.78) / Math.max(0.18, Math.tan(halfMinFov));

    this.minCameraDistance = Math.max(1.8, radius * 1.35);
    this.maxCameraDistance = Math.max(fitDist * 2.5, radius * 7.5, 16.0);
    this.lookAtTarget.copy(sphere.center);

    const clampedFit = Math.max(this.minCameraDistance, Math.min(this.maxCameraDistance, fitDist));
    if (updateZoom) {
      this.targetCameraDistance = clampedFit;
      if (immediate) {
        this.cameraDistance = clampedFit;
      }
    } else {
      this.cameraDistance = Math.max(this.minCameraDistance, Math.min(this.maxCameraDistance, this.cameraDistance));
      this.targetCameraDistance = this.cameraDistance;
    }

    if (this.floorGroup) {
      this.floorGroup.position.y = Math.min(-1.0, box.min.y - radius * 0.48);
    }
    if (this.polarGrid) {
      const targetGridRadius = Math.max(3.6, radius * 2.8);
      const scaleRatio = targetGridRadius / Math.max(0.5, this.config.environment.polarRadius);
      this.polarGrid.scale.setScalar(scaleRatio);
    }
  }

  renderStandard() {
    const w = Math.max(1, window.innerWidth);
    const h = Math.max(1, window.innerHeight);
    const camCfg = this.config.camera;
    const renCfg = this.config.renderer;

    this.renderer.setViewport(0, 0, w, h);
    this.renderer.setScissorTest(false);
    const useComposer = Boolean(this.composer && typeof this.composer.render === 'function');
    const clearCol = useComposer ? this._clearColorLinear : this._clearColorRaw;
    this.renderer.setClearColor(clearCol ?? renCfg.clearColorHex, renCfg.clearAlpha);

    this.camera.aspect = w / h;
    const elevY = (camCfg.baseElevationRatio || 0.17) * this.cameraDistance;
    this.camera.position.x = this.lookAtTarget.x + this.headParallaxOffset.x;
    this.camera.position.y = this.lookAtTarget.y + elevY + this.headParallaxOffset.y;
    this.camera.position.z = this.lookAtTarget.z + this.cameraDistance;
    this.camera.lookAt(this.lookAtTarget);
    this.camera.updateProjectionMatrix();

    if (useComposer) {
      this.composer.render();
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  renderPyramid4Way() {
    const w = window.innerWidth;
    const h = Math.max(1, window.innerHeight);
    const pyrCfg = this.config.pyramid;
    const renCfg = this.config.renderer;

    const size = Math.min(w, h) * pyrCfg.quadrantSizeRatio;
    const cx = w * 0.5;
    const cy = h * 0.5;
    const gap = size * pyrCfg.centerGapRatio;

    this.renderer.setScissorTest(true);
    this.renderer.setClearColor(renCfg.pyramidClearColorHex, renCfg.clearAlpha);
    this.renderer.clear();

    const views = [
      {
        x: cx - size * 0.5,
        y: cy + gap,
        w: size,
        h: size,
        rotZ: 0,
        azimuth: 0,
        nominalCam: [0, 0, this.cameraDistance]
      },
      {
        x: cx - size * 0.5,
        y: cy - gap - size,
        w: size,
        h: size,
        rotZ: Math.PI,
        azimuth: Math.PI,
        nominalCam: [0, 0, -this.cameraDistance]
      },
      {
        x: cx - gap - size,
        y: cy - size * 0.5,
        w: size,
        h: size,
        rotZ: Math.PI * 0.5,
        azimuth: -Math.PI * 0.5,
        nominalCam: [-this.cameraDistance, 0, 0]
      },
      {
        x: cx + gap,
        y: cy - size * 0.5,
        w: size,
        h: size,
        rotZ: -Math.PI * 0.5,
        azimuth: Math.PI * 0.5,
        nominalCam: [this.cameraDistance, 0, 0]
      }
    ];

    const isSymmetric = this.pyramidParallaxMode === 'symmetric';
    const px = isSymmetric ? (this.headParallaxOffset.x || 0) : 0;
    const py = isSymmetric ? (this.headParallaxOffset.y || 0) : 0;

    views.forEach((v) => {
      this.renderer.setViewport(v.x, v.y, v.w, v.h);
      this.renderer.setScissor(v.x, v.y, v.w, v.h);

      const isRotatedQuarterTurn = Math.abs(Math.abs(v.rotZ) - Math.PI * 0.5) < 0.01;
      this.camera.aspect = isRotatedQuarterTurn ? (v.h / v.w) : (v.w / v.h);

      const dX = px * Math.cos(v.azimuth);
      const dZ = -px * Math.sin(v.azimuth);
      const dY = py;

      this.camera.position.set(
        this.lookAtTarget.x + v.nominalCam[0] + dX,
        this.lookAtTarget.y + v.nominalCam[1] + dY,
        this.lookAtTarget.z + v.nominalCam[2] + dZ
      );
      this.camera.up.set(Math.sin(v.rotZ), Math.cos(v.rotZ), 0);
      this.camera.lookAt(this.lookAtTarget);
      this.camera.updateProjectionMatrix();

      this.renderer.render(this.scene, this.camera);
    });

    this.camera.up.set(0, 1, 0);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render(deltaSec = 0.016) {
    const dt = Math.min(0.1, Math.max(0.001, deltaSec || 0.016));
    if (this.ambientParticles) {
      this.ambientParticles.rotation.y += (this.config.environment.particleRotationRadPerSec || 0.036) * dt;
    }
    if (this.pyramidMode) {
      this.renderPyramid4Way();
    } else {
      this.renderStandard();
    }
  }

  dispose() {
    window.removeEventListener('resize', this._onWindowResize);
    this.bloomPass?.dispose?.();
    this.hdrRenderTarget?.dispose?.();
    this.composer?.renderTarget1?.dispose?.();
    this.composer?.renderTarget2?.dispose?.();
    if (this.scene && typeof this.scene.traverse === 'function') {
      this.scene.traverse((obj) => {
        if (obj.geometry && typeof obj.geometry.dispose === 'function') {
          obj.geometry.dispose();
        }
        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m?.dispose?.());
          } else if (typeof obj.material.dispose === 'function') {
            obj.material.dispose();
          }
        }
      });
    }
    if (this.renderer) {
      this.renderer.dispose();
    }
  }
}