/**
 * DiagnosticPlatesManager - Border-Free Volumetric Light Hologram Engine
 * ZERO Rectangular Cards / Zero Flat Picture Boards / Zero Square Frames.
 * Converts real-world diagnostic imagery into border-free 3D particle constellations
 * with smooth radial falloff and traveling photon pulses.
 */

export const DEFAULT_PLATES_CONFIG = Object.freeze({
  spriteSize: 64,
  arcSpanRadians: Math.PI * 1.15,
  arcStartOffset: -Math.PI * 0.575,
  baseArcRadius: 2.35,
  alternatingRadiusStep: 0.32,
  offsetX: 0.0,
  baseOffsetY: 0.45,
  alternatingOffsetY: 0.28,
  depthScaleZ: 0.35,
  leaderAnchorOffsetRatio: 0.34,
  cloudDimensions: Object.freeze({
    width: 1.65,
    height: 1.20,
    depthScale: 0.45,
    sampleWidth: 84,
    minSampleHeight: 32,
    defaultAspect: 0.75,
    maxRadialNorm: 0.92,
    envelopeFalloffSpan: 0.35,
    minNormLuminance: 0.08,
    minAlpha: 0.22,
    minEnvelope: 0.05,
    luminanceCenterOffset: 0.35,
    phosphorBlend: 0.36,
    phosphorRGB: Object.freeze([0.0, 0.92, 1.0])
  }),
  styles: Object.freeze({
    baseColorHex: 0x00f3ff,
    hoverColorHex: 0x38bdf8,
    selectedColorHex: 0xffaa00,
    dashSize: 0.12,
    gapSize: 0.06,
    pulseRadius: 0.038,
    basePointSize: 0.032,
    hoverPointSize: 0.037,
    selectedPointSize: 0.042,
    selectedScale: 1.08,
    hoverScale: 1.04,
    baseScale: 1.0
  }),
  animation: Object.freeze({
    timeScale: 0.0015,
    phaseStep: 1.3,
    floatAmplitudeY: 0.05,
    pulseSpeed: 0.85,
    pulsePhaseStep: 0.3,
    cloudSwingSpeed: 0.7,
    cloudSwingAmplitude: 0.12
  }),
  proxyEndpoint: '/api/proxy-image'
});

function createSoftGlowSprite(size = DEFAULT_PLATES_CONFIG.spriteSize) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const half = size * 0.5;

  const grad = ctx.createRadialGradient(half, half, 0, half, half, half);
  grad.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
  grad.addColorStop(0.25, 'rgba(0, 243, 255, 0.85)');
  grad.addColorStop(0.6, 'rgba(0, 119, 255, 0.3)');
  grad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

const PARTICLE_SPRITE = createSoftGlowSprite();

export class DiagnosticPlatesManager {
  constructor(scene, masterGroup, config = {}) {
    this.scene = scene;
    this.masterGroup = masterGroup;
    this.config = {
      ...DEFAULT_PLATES_CONFIG,
      ...config,
      cloudDimensions: {
        ...DEFAULT_PLATES_CONFIG.cloudDimensions,
        ...(config.cloudDimensions || {})
      },
      styles: {
        ...DEFAULT_PLATES_CONFIG.styles,
        ...(config.styles || {})
      },
      animation: {
        ...DEFAULT_PLATES_CONFIG.animation,
        ...(config.animation || {})
      }
    };

    this.platesGroup = new THREE.Group();
    this.platesGroup.userData.persistentOverlay = true;
    this.masterGroup.add(this.platesGroup);

    this.plateItems = [];
    this.visible = false;
    this.volumetricMode = false;
    this.selectedPartIdx = -1;
    this.hoveredPartIdx = -1;
    this._buildGeneration = 0;

    this._tmpExplosionDelta = new THREE.Vector3();
    this._tmpPlateAnchor = new THREE.Vector3();
  }

  _disposeObjectTree(node) {
    if (!node) return;
    if (Array.isArray(node.children)) {
      for (const child of [...node.children]) {
        this._disposeObjectTree(child);
      }
    }
    if (node.geometry && typeof node.geometry.dispose === 'function') {
      node.geometry.dispose();
    }
    if (node.material) {
      if (Array.isArray(node.material)) {
        node.material.forEach((m) => m?.dispose?.());
      } else if (typeof node.material.dispose === 'function') {
        node.material.dispose();
      }
    }
  }

  clear() {
    this._buildGeneration++;
    while (this.platesGroup.children.length > 0) {
      const child = this.platesGroup.children[0];
      this.platesGroup.remove(child);
      this._disposeObjectTree(child);
    }
    this.plateItems = [];
  }

  setVisible(visible) {
    this.visible = Boolean(visible);
    this.platesGroup.visible = this.visible;
  }

  setVolumetricMode(isVolumetric) {
    this.volumetricMode = Boolean(isVolumetric);
    this.setVisible(this.volumetricMode);
  }

  _resolveProxyUrl(imgUrl) {
    const raw = String(imgUrl || '').trim();
    if (!raw) return '';
    if (raw.startsWith('http://') || raw.startsWith('https://')) {
      return `${this.config.proxyEndpoint}?url=${encodeURIComponent(raw)}`;
    }
    return raw;
  }

  _computeAssemblyRadius(activePartObjects) {
    if (!Array.isArray(activePartObjects) || activePartObjects.length === 0) {
      return 1.25;
    }
    let maxRad = 0.75;
    for (const item of activePartObjects) {
      const pos = item.assembledVec || item.group?.position;
      if (!pos) continue;
      const geo = item.subMeshes?.[0]?.mesh?.geometry;
      if (geo) {
        if (!geo.boundingBox && typeof geo.computeBoundingBox === 'function') {
          geo.computeBoundingBox();
        }
        const bb = geo.boundingBox;
        if (bb && Number.isFinite(bb.max.x)) {
          const r = Math.hypot(
            pos.x + (Math.abs(pos.x + bb.max.x) > Math.abs(pos.x + bb.min.x) ? bb.max.x : bb.min.x),
            pos.y + (Math.abs(pos.y + bb.max.y) > Math.abs(pos.y + bb.min.y) ? bb.max.y : bb.min.y),
            pos.z + (Math.abs(pos.z + bb.max.z) > Math.abs(pos.z + bb.min.z) ? bb.max.z : bb.min.z)
          );
          if (Number.isFinite(r)) maxRad = Math.max(maxRad, r);
          continue;
        }
      }
      const s = Array.isArray(item.data?.scale) ? item.data.scale : [0.7, 0.7, 0.7];
      const sr = Math.hypot(Number(s[0]) || 0.7, Number(s[1]) || 0.7, Number(s[2]) || 0.7) * 0.5;
      maxRad = Math.max(maxRad, pos.length() + sr);
    }
    return Math.max(0.75, maxRad);
  }

  /**
   * Builds border-free 3D holographic particle clouds from real reference imagery.
   */
  buildPlates(blueprint, activePartObjects) {
    this.clear();
    if (this.masterGroup && this.platesGroup.parent !== this.masterGroup) {
      this.masterGroup.add(this.platesGroup);
    }
    if (!blueprint) return;

    const currentGen = this._buildGeneration;
    const references = blueprint.references || [];
    const wikiSummary = blueprint.wikiSummary || {};
    const parts = blueprint.parts || [];

    if (parts.length === 0) return;

    const cfg = this.config;
    const st = cfg.styles;
    const cd = cfg.cloudDimensions;

    const assemblyRadius = this._computeAssemblyRadius(activePartObjects);
    const layoutScale = Math.max(0.75, Math.min(2.6, assemblyRadius / 1.25));
    const scaledWidth = cd.width * Math.min(1.65, layoutScale);
    const scaledHeight = cd.height * Math.min(1.65, layoutScale);
    const scaledDepth = cd.depthScale * Math.min(1.65, layoutScale);
    const scaledPulseRad = st.pulseRadius * Math.min(1.5, layoutScale);

    parts.forEach((part, idx) => {
      let refImage = part.referenceImage;
      if (!refImage && references.length > 0) {
        refImage = references[idx % references.length];
      }
      if (!refImage && wikiSummary.thumbnail) {
        refImage = {
          title: wikiSummary.title || part.name,
          thumbUrl: wikiSummary.thumbnail,
          originalUrl: wikiSummary.originalImage || wikiSummary.thumbnail,
          caption: wikiSummary.description || part.role
        };
      }

      if (!refImage || (!refImage.thumbUrl && !refImage.url && !refImage.originalUrl)) return;

      const primaryRawUrl = refImage.thumbUrl || refImage.url || refImage.originalUrl;
      const fallbackRawUrl = (refImage.originalUrl && refImage.originalUrl !== primaryRawUrl)
        ? refImage.originalUrl
        : '';

      const partObj = activePartObjects[idx];
      const partPos = partObj ? partObj.assembledVec : new THREE.Vector3(0, 0, 0);

      const plateGroup = new THREE.Group();

      const angle = (idx / Math.max(1, parts.length)) * cfg.arcSpanRadians + cfg.arcStartOffset;
      const radius = (cfg.baseArcRadius + (idx % 2) * cfg.alternatingRadiusStep) * layoutScale;
      const platePosX = partPos.x + Math.sin(angle) * radius + cfg.offsetX * layoutScale;
      const platePosY = partPos.y + (cfg.baseOffsetY + (idx % 2 === 0 ? cfg.alternatingOffsetY : -cfg.alternatingOffsetY)) * layoutScale;
      const platePosZ = partPos.z + Math.cos(angle) * radius * cfg.depthScaleZ;

      plateGroup.position.set(platePosX, platePosY, platePosZ);
      plateGroup.rotation.y = -angle * 0.5;

      const anchorSignX = platePosX >= partPos.x ? -1 : 1;
      const leaderAnchorOffsetX = anchorSignX * scaledWidth * cfg.leaderAnchorOffsetRatio;

      const baseHexCol = part.color ? new THREE.Color(part.color) : new THREE.Color(st.baseColorHex);
      const lineMat = new THREE.LineDashedMaterial({
        color: baseHexCol.clone(),
        dashSize: st.dashSize * layoutScale,
        gapSize: st.gapSize * layoutScale,
        transparent: true,
        opacity: 0.26,
        blending: THREE.AdditiveBlending
      });
      const lineGeo = new THREE.BufferGeometry().setFromPoints([
        partPos.clone(),
        plateGroup.position.clone()
      ]);
      const leaderLine = new THREE.Line(lineGeo, lineMat);
      leaderLine.computeLineDistances();
      this.platesGroup.add(leaderLine);

      const pulseGeo = new THREE.SphereGeometry(scaledPulseRad, 10, 10);
      const pulseMat = new THREE.MeshBasicMaterial({
        color: baseHexCol.clone(),
        transparent: true,
        opacity: 0.9,
        blending: THREE.AdditiveBlending
      });
      const pulseMesh = new THREE.Mesh(pulseGeo, pulseMat);
      pulseMesh.position.copy(partPos);
      this.platesGroup.add(pulseMesh);

      const loadUrl = this._resolveProxyUrl(primaryRawUrl);
      const fallbackUrl = fallbackRawUrl ? this._resolveProxyUrl(fallbackRawUrl) : '';
      const img = new Image();
      img.crossOrigin = 'anonymous';

      const itemRecord = {
        partIdx: idx,
        partData: part,
        basePartColor: baseHexCol,
        plateGroup,
        pointCloud: null,
        leaderLine,
        lineMat,
        lineGeo,
        pulseMesh,
        pulseMat,
        basePos: plateGroup.position.clone(),
        leaderAnchorOffsetX,
        layoutScale,
        refImage,
        loadUrl
      };

      img.onload = () => {
        if (this._buildGeneration !== currentGen) return;
        try {
          const pointCloud = this.createBorderFreeParticleCloud(
            img,
            scaledWidth,
            scaledHeight,
            scaledDepth,
            part.color || null,
            layoutScale
          );
          if (pointCloud) {
            plateGroup.add(pointCloud);
            itemRecord.pointCloud = pointCloud;
            this.updateVisualStates(this.selectedPartIdx, this.hoveredPartIdx);
          }
        } catch (e) {
          console.warn('[!] Particle cloud creation warning:', e);
        }
      };
      img.onerror = () => {
        if (fallbackUrl && img.dataset.fallbackAttempted !== '1') {
          img.dataset.fallbackAttempted = '1';
          img.src = fallbackUrl;
        }
      };
      img.src = loadUrl;

      this.plateItems.push(itemRecord);
      this.platesGroup.add(plateGroup);
    });

    this.platesGroup.visible = this.visible;
    this.updateVisualStates(this.selectedPartIdx, this.hoveredPartIdx);
  }

  /**
   * Converts a 2D technical image into a Border-Free 3D Volumetric Particle Cloud
   * with contrast-adaptive luminance depth projection.
   */
  createBorderFreeParticleCloud(
    imageElement,
    width = this.config.cloudDimensions.width,
    height = this.config.cloudDimensions.height,
    depthScale = this.config.cloudDimensions.depthScale,
    tintHex = null,
    layoutScale = 1.0
  ) {
    const cd = this.config.cloudDimensions;
    const st = this.config.styles;

    const canvas = document.createElement('canvas');
    const natW = imageElement.naturalWidth || imageElement.width || cd.sampleWidth;
    const natH = imageElement.naturalHeight || imageElement.height || Math.round(cd.sampleWidth * cd.defaultAspect);
    const sampleW = Math.max(48, Math.min(cd.sampleWidth, natW));
    const aspectRatio = natW > 0 && natH > 0 ? (natH / natW) : cd.defaultAspect;
    const sampleH = Math.max(cd.minSampleHeight, Math.round(sampleW * aspectRatio));
    canvas.width = sampleW;
    canvas.height = sampleH;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, sampleW, sampleH);
    const imgData = ctx.getImageData(0, 0, sampleW, sampleH).data;

    // Pass 1: measure image luminance bounds inside valid radial envelope for adaptive dynamic range
    const lumSamples = [];
    for (let y = 0; y < sampleH; y++) {
      const ny = (y / sampleH - 0.5) * 2.0;
      for (let x = 0; x < sampleW; x++) {
        const nx = (x / sampleW - 0.5) * 2.0;
        if (Math.hypot(nx, ny) > cd.maxRadialNorm) continue;
        const idx = (y * sampleW + x) * 4;
        const a = imgData[idx + 3] / 255.0;
        if (a < cd.minAlpha) continue;
        const r = imgData[idx] / 255.0;
        const g = imgData[idx + 1] / 255.0;
        const b = imgData[idx + 2] / 255.0;
        lumSamples.push(0.2126 * r + 0.7152 * g + 0.0722 * b);
      }
    }
    if (lumSamples.length === 0) return null;
    lumSamples.sort((a, b) => a - b);
    const lumLow = lumSamples[Math.floor(lumSamples.length * 0.05)] ?? 0.0;
    const lumHigh = lumSamples[Math.min(lumSamples.length - 1, Math.floor(lumSamples.length * 0.95))] ?? 1.0;
    const lumSpan = Math.max(0.20, lumHigh - lumLow);

    let [pR, pG, pB] = cd.phosphorRGB;
    if (tintHex && window.THREE) {
      const tc = new THREE.Color(tintHex);
      pR = tc.r * 0.5 + pR * 0.5;
      pG = tc.g * 0.5 + pG * 0.5;
      pB = tc.b * 0.5 + pB * 0.5;
    }

    const positions = [];
    const colors = [];
    const phos = cd.phosphorBlend;

    for (let y = 0; y < sampleH; y++) {
      const ny = (y / sampleH - 0.5) * 2.0;
      for (let x = 0; x < sampleW; x++) {
        const nx = (x / sampleW - 0.5) * 2.0;
        const rNorm = Math.hypot(nx, ny);
        if (rNorm > cd.maxRadialNorm) continue;

        const envelope = Math.max(
          0.0,
          Math.min(1.0, (cd.maxRadialNorm - rNorm) / cd.envelopeFalloffSpan)
        );
        if (envelope < cd.minEnvelope) continue;

        const idx = (y * sampleW + x) * 4;
        const a = imgData[idx + 3] / 255.0;
        if (a < cd.minAlpha) continue;

        const r = imgData[idx] / 255.0;
        const g = imgData[idx + 1] / 255.0;
        const b = imgData[idx + 2] / 255.0;

        const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        const normLum = Math.max(0.0, Math.min(1.0, (lum - lumLow) / lumSpan));
        if (normLum < cd.minNormLuminance) continue;

        const px = nx * 0.5 * width;
        const py = -ny * 0.5 * height;
        const pz = (normLum - cd.luminanceCenterOffset) * depthScale * envelope;

        positions.push(px, py, pz);

        colors.push(
          (r * (1.0 - phos) + pR * phos) * envelope,
          (g * (1.0 - phos) + pG * phos) * envelope,
          (b * (1.0 - phos) + pB * phos) * envelope
        );
      }
    }

    if (positions.length === 0) return null;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: st.basePointSize * Math.min(1.5, layoutScale),
      map: PARTICLE_SPRITE,
      vertexColors: true,
      transparent: true,
      opacity: 0.88,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    return new THREE.Points(geometry, material);
  }

  update(now, activePartObjects) {
    if (!this.visible || this.plateItems.length === 0) return;

    const anim = this.config.animation;
    const t = now * anim.timeScale;

    this.plateItems.forEach((item) => {
      const partObj = activePartObjects[item.partIdx];
      if (!partObj) return;

      // Both platesGroup and partObj.group share masterGroup local space
      const partLocalPos = partObj.group.position;
      this._tmpExplosionDelta.subVectors(partLocalPos, partObj.assembledVec);

      const hoverFloat = Math.sin(t + item.partIdx * anim.phaseStep) * anim.floatAmplitudeY;
      item.plateGroup.position.set(
        item.basePos.x + this._tmpExplosionDelta.x * 0.45,
        item.basePos.y + this._tmpExplosionDelta.y * 0.45 + hoverFloat,
        item.basePos.z + this._tmpExplosionDelta.z * 0.45
      );

      this._tmpPlateAnchor.copy(item.plateGroup.position);
      this._tmpPlateAnchor.x += item.leaderAnchorOffsetX;

      item.lineGeo.setFromPoints([partLocalPos, this._tmpPlateAnchor]);
      item.lineGeo.attributes.position.needsUpdate = true;
      if (typeof item.leaderLine.computeLineDistances === 'function') {
        item.leaderLine.computeLineDistances();
      }

      const pulseT = (t * anim.pulseSpeed + item.partIdx * anim.pulsePhaseStep) % 1.0;
      item.pulseMesh.position.lerpVectors(partLocalPos, this._tmpPlateAnchor, pulseT);

      if (item.pointCloud) {
        item.pointCloud.rotation.y = Math.sin(t * anim.cloudSwingSpeed + item.partIdx) * anim.cloudSwingAmplitude;
      }
    });
  }

  setPalette(paletteColors) {
    if (!paletteColors) return;
    const toColor = (c, fallbackHex) => {
      if (c && typeof c.getHex === 'function') return c.clone();
      if (c !== undefined && c !== null) return new THREE.Color(c);
      return new THREE.Color(fallbackHex);
    };

    const rimCol = toColor(paletteColors.rim || paletteColors.base, this.config.styles.baseColorHex);
    const hoverCol = toColor(paletteColors.hover || paletteColors.rim, this.config.styles.hoverColorHex);
    const selectedCol = toColor(paletteColors.selected, this.config.styles.selectedColorHex);

    this.config.styles.baseColorHex = rimCol.getHex();
    this.config.styles.hoverColorHex = hoverCol.getHex();
    this.config.styles.selectedColorHex = selectedCol.getHex();
    this.config.cloudDimensions.phosphorRGB = [rimCol.r, rimCol.g, rimCol.b];

    const tintCol = new THREE.Color(0xffffff).lerp(rimCol, 0.35);
    this.plateItems.forEach((item) => {
      if (item.pointCloud?.material) {
        item.pointCloud.material.color.copy(tintCol);
      }
    });
    this.updateVisualStates(this.selectedPartIdx, this.hoveredPartIdx);
  }

  updateVisualStates(selectedIdx, hoveredIdx) {
    this.selectedPartIdx = selectedIdx;
    this.hoveredPartIdx = hoveredIdx;
    const st = this.config.styles;

    this.plateItems.forEach((item) => {
      const idx = item.partIdx;
      const isSelected = idx === selectedIdx;
      const isHovered = idx === hoveredIdx;
      const ptScale = Math.min(1.5, item.layoutScale || 1.0);

      if (isSelected) {
        item.lineMat.color.setHex(st.selectedColorHex);
        item.lineMat.opacity = 0.55;
        item.pulseMat.color.setHex(st.selectedColorHex);
        item.plateGroup.scale.setScalar(st.selectedScale);
        if (item.pointCloud?.material) {
          item.pointCloud.material.opacity = 1.0;
          item.pointCloud.material.size = st.selectedPointSize * ptScale;
        }
      } else if (isHovered) {
        item.lineMat.color.setHex(st.hoverColorHex);
        item.lineMat.opacity = 0.42;
        item.pulseMat.color.setHex(st.hoverColorHex);
        item.plateGroup.scale.setScalar(st.hoverScale);
        if (item.pointCloud?.material) {
          item.pointCloud.material.opacity = 0.95;
          item.pointCloud.material.size = st.hoverPointSize * ptScale;
        }
      } else {
        if (item.basePartColor) {
          item.lineMat.color.copy(item.basePartColor);
          item.pulseMat.color.copy(item.basePartColor);
        } else {
          item.lineMat.color.setHex(st.baseColorHex);
          item.pulseMat.color.setHex(st.baseColorHex);
        }
        item.lineMat.opacity = selectedIdx === -1 ? 0.26 : 0.09;
        item.plateGroup.scale.setScalar(st.baseScale);
        if (item.pointCloud?.material) {
          item.pointCloud.material.opacity = selectedIdx === -1 ? 0.8 : 0.2;
          item.pointCloud.material.size = st.basePointSize * ptScale;
        }
      }
    });
  }
}