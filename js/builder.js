/**
 * HologramBuilder - True Tony Stark Holographic Light-Field CAD Engine
 * Zero Rectangular Planes / Cards / Cones
 * Features Custom GLSL Fresnel Silhouette Shader, Tomographic Laser Sweep,
 * Dual-Layer Volumetric Particle Constellation, and Structural Edge Lattice Wireframe.
 */

export const HOLOGRAM_PALETTES = Object.freeze({
  cyan: Object.freeze({
    base: '#00e5ff',
    rim: '#38bdf8',
    laser: '#e0f2fe',
    selected: '#ffaa00',
    selectedRim: '#ffe082',
    hover: '#38bdf8',
    hoverRim: '#e0f7ff',
    laserSelected: '#ffd166'
  }),
  arc_blue: Object.freeze({
    base: '#0055ff',
    rim: '#00e5ff',
    laser: '#cffafe',
    selected: '#ffb300',
    selectedRim: '#fff0a8',
    hover: '#60a5fa',
    hoverRim: '#dbeafe',
    laserSelected: '#ffd166'
  }),
  stark_gold: Object.freeze({
    base: '#f59e0b',
    rim: '#fef08a',
    laser: '#fffbeb',
    selected: '#38bdf8',
    selectedRim: '#e0f7ff',
    hover: '#fcd34d',
    hoverRim: '#ffffff',
    laserSelected: '#7dd3fc'
  }),
  emerald: Object.freeze({
    base: '#059669',
    rim: '#34d399',
    laser: '#d1fae5',
    selected: '#fbbf24',
    selectedRim: '#fef3c7',
    hover: '#2dd4bf',
    hoverRim: '#ecfdf5',
    laserSelected: '#fde68a'
  })
});

export const TOPOLOGY_UNIT_SPECS = Object.freeze({
  hollow_shell: 'hollow_shell',
  rotary_disc: 'rotary_disc',
  cylindrical_core: 'cylindrical_core',
  conical_taper: 'conical_taper',
  structural_frame: 'structural_frame',
  branched_conduit: 'branched_conduit',
  contoured_solid: 'contoured_solid'
});

export const DEFAULT_BUILDER_CONFIG = Object.freeze({
  defaultPalette: 'cyan',
  targetPointsCount: 2000,
  wireframeDensityCap: 1600,
  edgeCreaseAngleDeg: 22,
  cappingWireOpacity: 0.085,
  masterHullWireOpacity: 0.038,
  masterHullCoreAlpha: 0.016,
  defaultCoreAlpha: 0.026,
  conduitSegments: 20,
  conduitRadiusRatio: 0.0018,
  conduitRadialSegments: 6,
  pulseCount: 4,
  spawnDurationSec: 1.25,
  sweepSpeed: 1.4
});

function createGlowSpriteTexture(size = 64) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const half = size * 0.5;

  const grad = ctx.createRadialGradient(half, half, 0, half, half, half);
  grad.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
  grad.addColorStop(0.14, 'rgba(140, 248, 255, 0.72)');
  grad.addColorStop(0.38, 'rgba(0, 170, 255, 0.16)');
  grad.addColorStop(0.72, 'rgba(0, 90, 240, 0.02)');
  grad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

const GLOW_SPRITE = createGlowSpriteTexture();

export const VectorBondGLSL = Object.freeze({
  uniforms: Object.freeze({
    uTime: { value: 0.0 },
    uColor: { value: new THREE.Color('#38bdf8') },
    uExcitedColor: { value: new THREE.Color('#00f3ff') },
    uSelectedColor: { value: new THREE.Color('#ffaa00') },
    uSelected: { value: 0.0 },
    uHovered: { value: 0.0 },
    uBondProgress: { value: 1.0 },
    uOpacity: { value: 0.85 }
  }),
  vertexShader: `
    attribute vec3 aOrigin;
    attribute vec3 aTarget;
    attribute float aSegmentT;
    attribute float aBondPhase;

    uniform float uBondProgress;
    uniform float uTime;

    varying float vSegmentT;
    varying float vBondPhase;
    varying float vLocalP;

    void main() {
      vSegmentT = aSegmentT;
      vBondPhase = aBondPhase;

      // Local bond growth wavefront
      float startT = aBondPhase * 0.40;
      float localP = clamp((uBondProgress - startT) / 0.60, 0.0, 1.0);
      vLocalP = localP;

      // Animate line extension from Origin toward Target (like chemical bonds linking)
      vec3 currentPos = (aSegmentT < 0.5) 
        ? aOrigin 
        : mix(aOrigin, aTarget, localP);

      vec4 mvPosition = modelViewMatrix * vec4(currentPos, 1.0);
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: `
    uniform vec3 uColor;
    uniform vec3 uExcitedColor;
    uniform vec3 uSelectedColor;
    uniform float uSelected;
    uniform float uHovered;
    uniform float uTime;
    uniform float uBondProgress;
    uniform float uOpacity;

    varying float vSegmentT;
    varying float vBondPhase;
    varying float vLocalP;

    void main() {
      if (vLocalP <= 0.002) discard;

      vec3 baseCol = (uSelected > 0.5) ? uSelectedColor : uColor;
      vec3 excitedCol = (uSelected > 0.5) ? vec3(1.0, 0.95, 0.7) : uExcitedColor;

      // Traveling photon pulse along the bond
      float pulse = sin(uTime * 12.0 - vSegmentT * 10.0 + vBondPhase * 6.28318);
      float pulseIntensity = max(0.0, pulse);

      // Flash at the leading tip of the growing bond
      float tipFlash = (vSegmentT > 0.5 && vLocalP < 0.96) ? 2.2 : 1.0;

      vec3 col = mix(baseCol, excitedCol, pulseIntensity * 0.65) * tipFlash;
      float alpha = clamp(vLocalP * uOpacity * (0.65 + 0.35 * pulseIntensity), 0.0, 0.95);

      gl_FragColor = vec4(col, alpha);
    }
  `
});

export const HologramGLSL = Object.freeze({
  uniforms: Object.freeze({
    uTime: { value: 0.0 },
    uColor: { value: new THREE.Color('#00e5ff') },
    uRimColor: { value: new THREE.Color('#38bdf8') },
    uSelectedColor: { value: new THREE.Color('#ffaa00') },
    uSelectedRimColor: { value: new THREE.Color('#ffe082') },
    uHoverColor: { value: new THREE.Color('#38bdf8') },
    uHoverRimColor: { value: new THREE.Color('#e0f7ff') },
    uLaserSelectedColor: { value: new THREE.Color('#ffd166') },
    uRimPower: { value: 2.8 },
    uCoreAlpha: { value: 0.10 },
    uRimAlpha: { value: 0.95 },
    uSweepY: { value: 0.0 },
    uSweepWidth: { value: 30.0 },
    uSweepIntensity: { value: 0.72 },
    uLaserColor: { value: new THREE.Color('#d1fae5') },
    uOpacity: { value: 1.0 },
    uFormProgress: { value: 1.0 },
    uSelected: { value: 0.0 },
    uHovered: { value: 0.0 },
    uIsolateDim: { value: 0.0 },
    uEnableLaser: { value: 1.0 },
    uEnableScanlines: { value: 1.0 },
    uCurvatureStrength: { value: 1.0 },
    uMicroEtchStrength: { value: 1.0 },
    uMeshScale: { value: 1.0 },
    uSliceOrigin: { value: new THREE.Vector3(0, 0, 0) },
    uSliceAxis: { value: new THREE.Vector3(0, 1, 0) },
    uDepthSliceY: { value: 0.0 },
    uDepthSliceHalfWidth: { value: 0.14 },
    uSliceCutaway: { value: 0.0 },
    uDepthSliceStrength: { value: 1.0 }
  }),
  vertexShader: `
    attribute float aCurvature;
    attribute float aVertexLum;
    uniform float uMeshScale;

    varying vec3 vNormal;
    varying vec3 vLocalNormal;
    varying vec3 vLocalPos;
    varying vec3 vViewPosition;
    varying vec3 vWorldPosition;
    varying vec2 vUv;
    varying float vCurvature;
    varying float vVertexLum;
    varying float vRelViewDepth;

    void main() {
      vUv = uv;
      vCurvature = aCurvature;
      vVertexLum = aVertexLum;
      vLocalPos = position;
      vLocalNormal = (dot(normal, normal) > 1e-8) ? normalize(normal) : vec3(0.0, 1.0, 0.0);
      vec3 viewNorm = normalMatrix * vLocalNormal;
      vNormal = (dot(viewNorm, viewNorm) > 1e-8) ? normalize(viewNorm) : vec3(0.0, 0.0, 1.0);
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      vViewPosition = -mvPosition.xyz;
      vec4 centerMv = modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0);
      vRelViewDepth = clamp((mvPosition.z - centerMv.z) / max(uMeshScale, 1e-4), -1.0, 1.0);
      vec4 worldPos = modelMatrix * vec4(position, 1.0);
      vWorldPosition = worldPos.xyz;
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: `
    uniform float uTime;
    uniform vec3 uColor;
    uniform vec3 uRimColor;
    uniform vec3 uSelectedColor;
    uniform vec3 uSelectedRimColor;
    uniform vec3 uHoverColor;
    uniform vec3 uHoverRimColor;
    uniform vec3 uLaserSelectedColor;
    uniform float uRimPower;
    uniform float uCoreAlpha;
    uniform float uRimAlpha;
    uniform float uSweepY;
    uniform float uSweepWidth;
    uniform float uSweepIntensity;
    uniform vec3 uLaserColor;
    uniform float uOpacity;
    uniform float uSelected;
    uniform float uHovered;
    uniform float uIsolateDim;
    uniform float uEnableLaser;
    uniform float uEnableScanlines;
    uniform float uCurvatureStrength;
    uniform float uMicroEtchStrength;
    uniform float uMeshScale;
    uniform vec3 uSliceOrigin;
    uniform vec3 uSliceAxis;
    uniform float uDepthSliceY;
    uniform float uDepthSliceHalfWidth;
    uniform float uSliceCutaway;
    uniform float uDepthSliceStrength;
    uniform float uFormProgress;

    varying vec3 vNormal;
    varying vec3 vLocalNormal;
    varying vec3 vLocalPos;
    varying vec3 vViewPosition;
    varying vec3 vWorldPosition;
    varying vec2 vUv;
    varying float vCurvature;
    varying float vVertexLum;
    varying float vRelViewDepth;

    vec2 computePlaneFW(vec2 uvCoord) {
      #if defined(GL_OES_standard_derivatives) || __VERSION__ >= 300
        return max(fwidth(uvCoord), vec2(0.008));
      #else
        return vec2(0.025);
      #endif
    }

    float computeScalarFW(float sVal) {
      #if defined(GL_OES_standard_derivatives) || __VERSION__ >= 300
        return max(fwidth(sVal), 0.004);
      #else
        return 0.012;
      #endif
    }

    vec3 evalPlaneEtch(vec2 uvPlane, float curv) {
      vec2 g1 = uvPlane * 16.0;
      vec2 fw1 = computePlaneFW(g1);
      vec2 d1 = abs(fract(g1 - 0.5) - 0.5) / fw1;
      float lineDist1 = min(d1.x, d1.y);
      float fade1 = 1.0 - smoothstep(0.32, 0.85, max(fw1.x, fw1.y));
      float scribe1 = (1.0 - smoothstep(0.0, 1.35, lineDist1)) * fade1;
      float nodeTick = (1.0 - smoothstep(0.0, 1.85, max(d1.x, d1.y))) * fade1;

      float warpMix = smoothstep(0.10, 0.65, abs(curv));
      vec2 g2 = vec2(
        uvPlane.x * 52.0 + warpMix * (curv * 2.2),
        uvPlane.y * 52.0 + warpMix * (curv * 3.4 + 0.35 * sin(uvPlane.x * 18.0))
      );
      vec2 fw2 = computePlaneFW(g2);
      float fade2 = 1.0 - smoothstep(0.28, 0.78, max(fw2.x, fw2.y));
      float striaWave = cos(g2.y * 6.2831853);
      float crossHash = cos(g2.x * 6.2831853);
      float grainMask = mix(1.0, 0.68 + 0.32 * crossHash, 1.0 - warpMix);
      float microGrain = (0.5 + 0.5 * striaWave) * grainMask * fade2;

      float etchH = (-0.85 * scribe1) + (0.45 * striaWave * grainMask * fade2);
      return vec3(etchH, clamp(scribe1 * 0.78 + nodeTick * 0.85, 0.0, 1.25), microGrain);
    }

    void main() {
      vec3 viewDir = normalize(vViewPosition);
      vec3 normal = (dot(vNormal, vNormal) > 1e-6) ? normalize(vNormal) : vec3(0.0, 0.0, 1.0);

      float safeScale = max(uMeshScale, 1e-4);
      vec3 q = vLocalPos / safeScale;
      vec3 objN = (dot(vLocalNormal, vLocalNormal) > 1e-6) ? normalize(vLocalNormal) : vec3(0.0, 1.0, 0.0);
      vec3 triW = pow(abs(objN), vec3(6.0));
      triW /= max(dot(triW, vec3(1.0)), 1e-5);

      vec3 etchVal =
        evalPlaneEtch(q.yz, vCurvature) * triW.x +
        evalPlaneEtch(q.xz, vCurvature) * triW.y +
        evalPlaneEtch(q.xy, vCurvature) * triW.z;

      float etchHeight = etchVal.x;
      float scribeMask = clamp(etchVal.y, 0.0, 1.2) * uMicroEtchStrength;
      float microGrain = clamp(etchVal.z, 0.0, 1.0);

      vec3 microNormal = normal;
      #if defined(GL_OES_standard_derivatives) || __VERSION__ >= 300
        if (uMicroEtchStrength > 0.01) {
          float dHdx = dFdx(etchHeight);
          float dHdy = dFdy(etchHeight);
          vec3 pDx = -dFdx(vViewPosition);
          vec3 pDy = -dFdy(vViewPosition);
          vec3 r1 = cross(pDy, normal);
          vec3 r2 = cross(normal, pDx);
          float det = dot(pDx, r1);
          if (abs(det) > 1e-7) {
            vec3 surfGrad = sign(det) * (dHdx * r1 + dHdy * r2);
            vec3 bumpVec = surfGrad * (safeScale * 0.0015 * uMicroEtchStrength);
            float maxBump = 0.42 * abs(det);
            float bumpLen = length(bumpVec);
            if (bumpLen > maxBump && bumpLen > 1e-12) {
              bumpVec *= (maxBump / bumpLen);
            }
            vec3 rawN = abs(det) * normal - bumpVec;
            float len2 = dot(rawN, rawN);
            if (len2 > 1e-10) {
              microNormal = rawN * inversesqrt(len2);
            }
          }
        }
      #endif

      // View-space dual key/fill sculptural lighting so fine surface features read in 3D
      vec3 effN = gl_FrontFacing ? microNormal : -microNormal;
      vec3 keyDir = normalize(vec3(0.45, 0.72, 0.52));
      vec3 fillDir = normalize(vec3(-0.62, -0.28, 0.36));
      float keyDiffuse = max(dot(effN, keyDir), 0.0);
      float fillDiffuse = max(dot(effN, fillDir), 0.0);
      float sculptLight = 0.42 + 0.56 * keyDiffuse + 0.20 * fillDiffuse;

      // Neural vertex-color luminance micro-detail extracted from TripoSR albedo
      float texLum = clamp(vVertexLum, 0.16, 1.0);
      float texMod = 0.62 + 0.66 * texLum;

      float cosTheta = abs(dot(normal, viewDir));
      float fresnel = pow(1.0 - cosTheta, uRimPower);
      float microCos = abs(dot(microNormal, viewDir));
      float microFresnel = pow(1.0 - microCos, max(1.75, uRimPower * 0.78));
      float grainGlint = max(0.0, microFresnel - fresnel * 0.68) * (0.42 + 0.58 * microGrain) * uMicroEtchStrength;

      float screenRidge = 0.0;
      #if defined(GL_OES_standard_derivatives) || __VERSION__ >= 300
        vec3 dNdx = dFdx(vNormal);
        vec3 dNdy = dFdy(vNormal);
        vec3 dPdx = dFdx(vViewPosition);
        vec3 dPdy = dFdy(vViewPosition);
        float pixelSpan = max(1e-4, length(dPdx) + length(dPdy));
        screenRidge = clamp((length(dNdx) + length(dNdy)) / (pixelSpan * 22.0), 0.0, 1.0);
      #endif

      float ridgeMask = smoothstep(0.28, 0.90, vCurvature) * uCurvatureStrength;
      float sharpCrest = clamp(max(ridgeMask, screenRidge * 0.55 * uCurvatureStrength), 0.0, 1.0);
      float cavityMask = smoothstep(-0.05, -0.72, vCurvature) * uCurvatureStrength;
      float cavityAtten = clamp(1.0 - cavityMask * 0.78, 0.18, 1.0);

      vec3 baseCol = uColor * (0.64 + 0.50 * texLum);
      vec3 rimCol = uRimColor;

      if (uSelected > 0.5) {
        baseCol = mix(baseCol, uSelectedColor * (0.48 + 0.52 * texLum), 0.34);
        rimCol = mix(uRimColor, uSelectedRimColor, 0.86) * 1.08;
      } else if (uHovered > 0.5) {
        baseCol = mix(baseCol, uHoverColor * (0.62 + 0.42 * texLum), 0.30);
        rimCol = mix(uRimColor, uHoverRimColor, 0.74) * 1.05;
      }

      vec3 color = mix(baseCol, rimCol, clamp(fresnel + grainGlint * 0.32, 0.0, 1.0));
      color *= sculptLight * cavityAtten * mix(1.0, 0.72 + 0.42 * microGrain, 0.60 * uMicroEtchStrength);

      // Push crisp silhouette crests & laser scribe ribs smoothly into bloom without flooding faces
      vec3 ridgeHighlight = max(rimCol, vec3(0.32)) * (0.96 * (0.62 + 0.38 * texLum));
      color = mix(color, ridgeHighlight, clamp(sharpCrest * 0.44 + scribeMask * 0.24 + grainGlint * 0.20, 0.0, 0.80));

      float partDensityBias = mix(0.45, 1.0, smoothstep(0.02, 0.08, uCoreAlpha));
      float isInnerWall = gl_FrontFacing ? 0.0 : 1.0;
      float viewDepthBias = mix(0.42, 1.05, smoothstep(-0.65, 0.45, vRelViewDepth));
      viewDepthBias *= mix(1.0, 0.55, isInnerWall);

      float sliceCoord = dot(vWorldPosition - uSliceOrigin, uSliceAxis);
      float strataPhase = (sliceCoord / safeScale) * 16.0;
      float strataFW = computeScalarFW(strataPhase);
      float strataDist = abs(fract(strataPhase - 0.5) - 0.5) / strataFW;
      float strataFade = 1.0 - smoothstep(0.32, 0.85, strataFW);
      float strataLine = (1.0 - smoothstep(0.0, 1.35, strataDist)) * strataFade * uDepthSliceStrength;

      float signedSlice = sliceCoord - uDepthSliceY;
      float sliceDist = abs(signedSlice);
      float sliceFW = computeScalarFW(signedSlice);
      float safeHalfW = max(0.04 * safeScale, uDepthSliceHalfWidth * safeScale);
      float slabWindow = exp(-(sliceDist * sliceDist) / (safeHalfW * safeHalfW)) * uDepthSliceStrength;
      float sliceTrace = (1.0 - smoothstep(0.0, sliceFW * 2.4 + 0.004, sliceDist)) * uDepthSliceStrength;

      float cutGhost = 1.0;
      float cutawayLip = 0.0;
      if (uSliceCutaway > 0.5) {
        float upperClip = smoothstep(0.0, 0.035 * safeScale, signedSlice);
        float upperFloor = mix(0.05, 0.11, partDensityBias);
        cutGhost = mix(1.0, upperFloor, upperClip);
        cutawayLip = sliceTrace * 1.35;
      }

      color *= mix(0.70, 1.05, viewDepthBias);
      color = mix(color, ridgeHighlight, clamp(strataLine * (0.14 + 0.22 * isInnerWall) + (sliceTrace * 0.58 + cutawayLip * 0.42), 0.0, 0.85));

      float laserBeam = 0.0;
      if (uEnableLaser > 0.5) {
        float distToSweep = abs(sliceCoord - uSweepY);
        laserBeam = exp(-distToSweep * distToSweep * uSweepWidth) * uSweepIntensity;
        float laserEtchBoost = laserBeam * (0.78 + scribeMask * 0.85 + microGrain * 0.40 * uMicroEtchStrength);
        vec3 laserCol = (uSelected > 0.5) ? uLaserSelectedColor : uLaserColor;
        color += laserCol * (laserEtchBoost + sliceTrace * 0.42 + cutawayLip * 0.34);
      } else if (sliceTrace > 0.01) {
        vec3 sliceCol = (uSelected > 0.5) ? uLaserSelectedColor : uLaserColor;
        color += sliceCol * (sliceTrace * 0.58 + cutawayLip * 0.36);
      }

      float scanlines = 1.0;
      if (uEnableScanlines > 0.5) {
        float baseScan = 0.84 + 0.16 * sin(sliceCoord * 72.0 - uTime * 6.5);
        float cavityFringes = 0.68 + 0.32 * sin(sliceCoord * 165.0 + vCurvature * 14.0 - uTime * 8.0);
        scanlines = mix(baseScan, cavityFringes, cavityMask * 0.85);
      }
      float flicker = 0.97 + 0.03 * sin(uTime * 32.0 + vWorldPosition.x * 12.0);

      float slabBoost = mix(1.0, 1.35, slabWindow * partDensityBias);
      float strataAlpha = strataLine * mix(0.045, 0.14, max(isInnerWall, slabWindow));
      float core = uCoreAlpha * cavityAtten * texMod * mix(1.0, 0.78 + 0.36 * microGrain, 0.50 * uMicroEtchStrength) * viewDepthBias;
      float etchAlpha = (scribeMask * 0.12 + grainGlint * 0.22) * partDensityBias * (0.45 + 0.55 * cavityAtten);
      float rim = (fresnel * uRimAlpha + sharpCrest * 0.32 + etchAlpha) * mix(1.0, 0.52, cavityMask) * viewDepthBias;

      if (uSelected > 0.5) {
        core = min(0.052, uCoreAlpha * 1.30) * cavityAtten * texMod * viewDepthBias;
        rim = (fresnel * min(1.08, uRimAlpha * 1.05) + sharpCrest * 0.36 + etchAlpha * 1.12) * mix(1.0, 0.56, cavityMask) * viewDepthBias;
      } else if (uHovered > 0.5) {
        core = min(0.046, uCoreAlpha * 1.18) * cavityAtten * texMod * viewDepthBias;
        rim = (fresnel * min(1.02, uRimAlpha * 1.02) + sharpCrest * 0.33 + etchAlpha * 1.08) * mix(1.0, 0.54, cavityMask) * viewDepthBias;
      }

      float sliceRimAlpha = (sliceTrace * 0.40 + cutawayLip * 0.34) * mix(0.75, 1.05, partDensityBias);
      float alpha = ((core + rim + strataAlpha) * slabBoost + sliceRimAlpha + laserBeam * 0.16 * (1.0 + scribeMask * 0.45)) * scanlines * flicker * cutGhost * uOpacity;

      // Compress high-energy additive accumulation smoothly to prevent solid whiteout while keeping razor-sharp rim glow
      color = color / (1.0 + color * 0.32);

      // Laser-sintering molecular crystallization plane (reveals pure vector bonds above sweep front)
      if (uFormProgress < 0.999) {
        float normY = vLocalPos.y / safeScale;
        float sweepPlane = mix(-0.95, 0.95, uFormProgress);
        float deltaY = normY - sweepPlane;
        if (deltaY > 0.0) {
          discard;
        }
        float spark = 1.0 - smoothstep(0.0, 0.05, abs(deltaY));
        vec3 sparkCol = (uSelected > 0.5) ? uLaserSelectedColor : uLaserColor;
        color += sparkCol * spark * 2.8;
        alpha = max(alpha, spark * 0.92);
      }

      // Strongly attenuate internal backfaces so hollowed hulls & selected parts stay X-ray transparent
      if (!gl_FrontFacing) {
        color *= 0.42;
        alpha *= 0.14;
      }

      if (uIsolateDim > 0.5) {
        alpha = alpha * 0.04;
      }

      gl_FragColor = vec4(color, clamp(alpha, 0.0, 0.58));
    }
  `
});

export class HologramBuilder {
  constructor(masterGroup, config = {}) {
    this.masterGroup = masterGroup;
    this.config = { ...DEFAULT_BUILDER_CONFIG, ...config };
    this.activePartObjects = [];
    this.plyLoader = window.THREE && window.THREE.PLYLoader ? new window.THREE.PLYLoader() : null;

    this.conduitsGroup = new THREE.Group();
    this.masterGroup.add(this.conduitsGroup);
    this.activeConduits = [];
    this.currentExplosionScalar = 0.0;
    this.masterHullIndex = -1;
    this._conduitParentIndex = -1;

    this.sweepY = 0.0;
    this.sweepDir = 1.0;
    this.sweepSpeed = this.config.sweepSpeed;
    this.laserEnabled = true;

    this.sliceCutawayEnabled = false;
    this.sliceY = 0.0;
    this.assemblyMinY = -0.85;
    this.assemblyMaxY = 0.85;
    this.assemblyRadius = 1.15;
    this.sliceRingGroup = null;
    this.sliceRingMat = null;
    this.sliceTickMat = null;
    this.sliceTicksMesh = null;
    this._sliceWorldOrigin = new THREE.Vector3(0, 0, 0);
    this._sliceWorldQuat = THREE.Quaternion ? new THREE.Quaternion() : null;
    this._sliceWorldAxis = new THREE.Vector3(0, 1, 0);

    this._tmpDir = new THREE.Vector3();
    this._tmpAnchorA = new THREE.Vector3();
    this._tmpAnchorB = new THREE.Vector3();
    this._tmpMid = new THREE.Vector3();

    this.currentPalette = this.config.defaultPalette;
    this.isBudgetConstrained = false;
    this.onMeshAttached = null;

    this.initSliceRing();
  }

  getPaletteColors(colorKey = this.currentPalette) {
    const entry = HOLOGRAM_PALETTES[colorKey] || HOLOGRAM_PALETTES.cyan;
    return {
      base: new THREE.Color(entry.base),
      rim: new THREE.Color(entry.rim),
      laser: new THREE.Color(entry.laser),
      selected: new THREE.Color(entry.selected),
      selectedRim: new THREE.Color(entry.selectedRim),
      hover: new THREE.Color(entry.hover),
      hoverRim: new THREE.Color(entry.hoverRim),
      laserSelected: new THREE.Color(entry.laserSelected)
    };
  }

  _computeGeometryRadius(geometry) {
    if (!geometry) return 0.5;
    if (!geometry.boundingSphere && typeof geometry.computeBoundingSphere === 'function') {
      geometry.computeBoundingSphere();
    }
    const r = geometry.boundingSphere?.radius;
    return Number.isFinite(r) && r > 1e-4 ? r : 0.5;
  }

  _computeGeometryVolume(geometry) {
    if (!geometry) return 0.0;
    if (!geometry.boundingBox && typeof geometry.computeBoundingBox === 'function') {
      geometry.computeBoundingBox();
    }
    const bb = geometry.boundingBox;
    if (!bb) return 0.0;
    const dx = Math.max(1e-4, bb.max.x - bb.min.x);
    const dy = Math.max(1e-4, bb.max.y - bb.min.y);
    const dz = Math.max(1e-4, bb.max.z - bb.min.z);
    return dx * dy * dz;
  }

  _getPartRadius(item) {
    if (!item) return 0.45;
    const primarySub = item.subMeshes?.[0];
    if (primarySub?.mesh?.geometry) {
      return this._computeGeometryRadius(primarySub.mesh.geometry);
    }
    const s = this.safeVec3Array(item.data?.scale, [0.7, 0.7, 0.7]);
    return Math.max(0.15, Math.hypot(s[0], s[1], s[2]) * 0.32);
  }

  _recomputeHullRoles() {
    if (!this.activePartObjects.length) {
      this.masterHullIndex = -1;
      return;
    }

    const hasExplicitTag = this.activePartObjects.some(
      (item) => typeof item.data?.isMasterShell === 'boolean'
    );

    let hullIdx = -1;
    if (hasExplicitTag) {
      hullIdx = this.activePartObjects.findIndex((item) => item.data?.isMasterShell === true);
    } else {
      const hollowIndices = [];
      this.activePartObjects.forEach((item, idx) => {
        if (item.data?.topology === 'hollow_shell') {
          hollowIndices.push(idx);
        }
      });
      if (
        hollowIndices.length > 0 &&
        (hollowIndices.length < this.activePartObjects.length || this.activePartObjects.length === 1)
      ) {
        let bestVol = -1;
        hollowIndices.forEach((idx) => {
          const geo = this.activePartObjects[idx]?.subMeshes?.[0]?.mesh?.geometry;
          const vol = geo ? this._computeGeometryVolume(geo) : 0.25;
          if (vol > bestVol) {
            bestVol = vol;
            hullIdx = idx;
          }
        });
      }
    }
    this.masterHullIndex = hullIdx;

    this.activePartObjects.forEach((item, idx) => {
      const isMasterHull =
        typeof item.data?.isMasterShell === 'boolean'
          ? item.data.isMasterShell
          : (idx === this.masterHullIndex && this.masterHullIndex >= 0);

      item.subMeshes?.forEach((sub) => {
        const vertCount = sub.mesh?.geometry?.attributes?.position?.count || 1000;
        const densityFactor = Math.min(
          1.0,
          this.config.wireframeDensityCap / Math.max(this.config.wireframeDensityCap, vertCount)
        );
        sub.baseWireOpacity =
          (isMasterHull ? this.config.masterHullWireOpacity : this.config.cappingWireOpacity) *
          (0.28 + 0.72 * densityFactor);
        sub.basePointOpacity = isMasterHull ? 0.22 : 0.34;

        if (sub.shaderMat?.uniforms) {
          sub.shaderMat.uniforms.uCoreAlpha.value = isMasterHull
            ? this.config.masterHullCoreAlpha
            : this.config.defaultCoreAlpha;
          sub.shaderMat.uniforms.uRimAlpha.value = isMasterHull ? 0.90 : 1.04;
          sub.shaderMat.uniforms.uRimPower.value = isMasterHull ? 2.65 : 2.95;
        }
        if (sub.pointsMat && sub.basePointSize) {
          sub.pointsMat.size = isMasterHull ? sub.basePointSize * 0.85 : sub.basePointSize;
        }
      });
    });
  }

  initSliceRing() {
    if (this.sliceRingGroup) {
      this.masterGroup.remove(this.sliceRingGroup);
      this._disposeObjectTree(this.sliceRingGroup);
    }

    const pal = this.getPaletteColors(this.currentPalette);
    this.sliceRingGroup = new THREE.Group();
    if (typeof THREE.LineLoop === 'function' && typeof THREE.LineBasicMaterial === 'function') {
      const segs = 72;
      const ringPts = [];
      for (let i = 0; i < segs; i++) {
        const theta = (i / segs) * Math.PI * 2;
        ringPts.push(new THREE.Vector3(Math.cos(theta), 0, Math.sin(theta)));
      }
      const ringGeo = new THREE.BufferGeometry().setFromPoints(ringPts);
      this.sliceRingMat = new THREE.LineBasicMaterial({
        color: this.sliceCutawayEnabled ? pal.selected.clone() : pal.rim.clone(),
        transparent: true,
        opacity: this.sliceCutawayEnabled ? 0.82 : 0.34,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
      const innerLoop = new THREE.LineLoop(ringGeo, this.sliceRingMat);
      this.sliceRingGroup.add(innerLoop);

      const tickPts = [];
      const tickCount = 16;
      for (let i = 0; i < tickCount; i++) {
        const theta = (i / tickCount) * Math.PI * 2;
        const rIn = 1.03;
        const rOut = (i % 4 === 0) ? 1.18 : 1.10;
        tickPts.push(
          new THREE.Vector3(Math.cos(theta) * rIn, 0, Math.sin(theta) * rIn),
          new THREE.Vector3(Math.cos(theta) * rOut, 0, Math.sin(theta) * rOut)
        );
      }
      const tickGeo = new THREE.BufferGeometry().setFromPoints(tickPts);
      this.sliceTickMat = new THREE.LineBasicMaterial({
        color: this.sliceCutawayEnabled ? pal.selectedRim.clone() : pal.laser.clone(),
        transparent: true,
        opacity: this.sliceCutawayEnabled ? 0.90 : 0.42,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
      this.sliceTicksMesh = new THREE.LineSegments(tickGeo, this.sliceTickMat);
      this.sliceRingGroup.add(this.sliceTicksMesh);
    }

    if (this.sliceRingGroup.scale) this.sliceRingGroup.scale.set(this.assemblyRadius, 1, this.assemblyRadius);
    if (this.sliceRingGroup.position) this.sliceRingGroup.position.y = this.sliceY;
    this.sliceRingGroup.visible = this.laserEnabled || this.sliceCutawayEnabled;
    this.masterGroup.add(this.sliceRingGroup);
  }

  fitSliceRingToAssembly() {
    if (!this.activePartObjects || this.activePartObjects.length === 0) {
      this.assemblyMinY = -0.85;
      this.assemblyMaxY = 0.85;
      this.assemblyRadius = 1.15;
      if (this.sliceRingGroup?.scale) {
        this.sliceRingGroup.scale.set(this.assemblyRadius, 1, this.assemblyRadius);
      }
      return;
    }

    let minY = Infinity;
    let maxY = -Infinity;
    let maxRad = 0.55;

    this.activePartObjects.forEach((item) => {
      const pos = item.group.position;
      const primarySub = item.subMeshes && item.subMeshes[0];
      const geo = primarySub && primarySub.mesh ? primarySub.mesh.geometry : null;
      if (geo) {
        if (!geo.boundingBox && typeof geo.computeBoundingBox === 'function') geo.computeBoundingBox();
        const bb = geo.boundingBox;
        if (bb && Number.isFinite(bb.min.y) && Number.isFinite(bb.max.y)) {
          minY = Math.min(minY, pos.y + bb.min.y);
          maxY = Math.max(maxY, pos.y + bb.max.y);
          const rx = Math.max(Math.abs(pos.x + bb.min.x), Math.abs(pos.x + bb.max.x));
          const rz = Math.max(Math.abs(pos.z + bb.min.z), Math.abs(pos.z + bb.max.z));
          maxRad = Math.max(maxRad, Math.hypot(rx, rz));
          return;
        }
      }
      const s = this.safeVec3Array(item.data?.scale, [0.8, 0.8, 0.8]);
      const halfY = Math.max(0.15, Math.abs(s[1]) * 0.5);
      const halfXZ = Math.max(0.15, Math.hypot(s[0], s[2]) * 0.5);
      minY = Math.min(minY, pos.y - halfY);
      maxY = Math.max(maxY, pos.y + halfY);
      maxRad = Math.max(maxRad, Math.hypot(pos.x, pos.z) + halfXZ);
    });

    if (Number.isFinite(minY) && Number.isFinite(maxY) && maxY > minY) {
      const pad = Math.max(0.06, (maxY - minY) * 0.05);
      this.assemblyMinY = minY - pad;
      this.assemblyMaxY = maxY + pad;
    }
    this.assemblyRadius = Math.max(0.75, maxRad * 1.18);
    if (this.sliceCutawayEnabled) {
      this.sliceY = Math.max(this.assemblyMinY, Math.min(this.assemblyMaxY, this.sliceY));
    }
    if (this.sliceRingGroup?.scale) {
      this.sliceRingGroup.scale.set(this.assemblyRadius, 1, this.assemblyRadius);
    }
  }

  setDepthSliceCutaway(enabled) {
    this.sliceCutawayEnabled = Boolean(enabled);
    if (this.sliceCutawayEnabled) {
      this.sliceY = (this.assemblyMinY + this.assemblyMaxY) * 0.5;
    }
    this.updateSliceVisualStyle();
    return this.sliceCutawayEnabled;
  }

  adjustDepthSliceOffset(deltaNormalized) {
    if (!this.sliceCutawayEnabled) {
      this.setDepthSliceCutaway(true);
    }
    const span = Math.max(0.25, this.assemblyMaxY - this.assemblyMinY);
    this.sliceY = Math.max(
      this.assemblyMinY,
      Math.min(this.assemblyMaxY, this.sliceY + deltaNormalized * span)
    );
    return Math.round(((this.sliceY - this.assemblyMinY) / span) * 100);
  }

  updateSliceVisualStyle() {
    const pal = this.getPaletteColors(this.currentPalette);
    if (this.sliceRingMat) {
      if (this.sliceCutawayEnabled) {
        this.sliceRingMat.color.copy(pal.selected);
        this.sliceRingMat.opacity = 0.82;
      } else {
        this.sliceRingMat.color.copy(pal.rim);
        this.sliceRingMat.opacity = 0.34;
      }
    }
    if (this.sliceTickMat) {
      if (this.sliceCutawayEnabled) {
        this.sliceTickMat.color.copy(pal.selectedRim);
        this.sliceTickMat.opacity = 0.90;
      } else {
        this.sliceTickMat.color.copy(pal.laser);
        this.sliceTickMat.opacity = 0.42;
      }
    }
    if (this.sliceRingGroup) {
      this.sliceRingGroup.visible = this.laserEnabled || this.sliceCutawayEnabled;
    }
  }

  setRenderBudget(isConstrained) {
    this.isBudgetConstrained = Boolean(isConstrained);
    this.activePartObjects.forEach((item) => {
      item.subMeshes?.forEach((sub) => {
        if (sub.wireMesh) {
          sub.wireMesh.visible = !this.isBudgetConstrained;
        }
        if (sub.shaderMat?.uniforms?.uMicroEtchStrength) {
          sub.shaderMat.uniforms.uMicroEtchStrength.value = this.isBudgetConstrained ? 0.75 : 1.0;
        }
        if (sub.shaderMat?.uniforms?.uDepthSliceStrength) {
          sub.shaderMat.uniforms.uDepthSliceStrength.value = this.isBudgetConstrained ? 0.80 : 1.0;
        }
      });
    });
  }

  computeVoxelDecimatedPointsGeometry(geometry, targetPoints = this.config.targetPointsCount) {
    if (!geometry || !geometry.attributes || !geometry.attributes.position) {
      return geometry;
    }
    const pos = geometry.attributes.position.array;
    if (pos.length / 3 <= targetPoints * 1.25) {
      return geometry;
    }

    if (!geometry.boundingBox) geometry.computeBoundingBox();
    const bb = geometry.boundingBox;
    const lx = Math.max(bb.max.x - bb.min.x, 0.001);
    const ly = Math.max(bb.max.y - bb.min.y, 0.001);
    const lz = Math.max(bb.max.z - bb.min.z, 0.001);
    const area = 2 * (lx * ly + ly * lz + lx * lz);
    const pitch = Math.sqrt(area / targetPoints);

    const cells = new Map();
    for (let i = 0; i < pos.length; i += 3) {
      const x = pos[i], y = pos[i + 1], z = pos[i + 2];
      const gx = Math.floor((x - bb.min.x) / pitch);
      const gy = Math.floor((y - bb.min.y) / pitch);
      const gz = Math.floor((z - bb.min.z) / pitch);
      const key = `${gx},${gy},${gz}`;
      if (!cells.has(key)) {
        cells.set(key, [x, y, z, 1]);
      } else {
        const c = cells.get(key);
        c[0] += x; c[1] += y; c[2] += z; c[3] += 1;
      }
    }

    const decimatedArray = new Float32Array(cells.size * 3);
    let idx = 0;
    for (const [_, c] of cells) {
      decimatedArray[idx++] = c[0] / c[3];
      decimatedArray[idx++] = c[1] / c[3];
      decimatedArray[idx++] = c[2] / c[3];
    }

    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute('position', new THREE.BufferAttribute(decimatedArray, 3));
    return pGeo;
  }

  safeVec3Array(arr, fallback = [0, 0, 0]) {
    if (!Array.isArray(arr) || arr.length < 3) return fallback;
    return [
      Number.isFinite(Number(arr[0])) ? Number(arr[0]) : fallback[0],
      Number.isFinite(Number(arr[1])) ? Number(arr[1]) : fallback[1],
      Number.isFinite(Number(arr[2])) ? Number(arr[2]) : fallback[2]
    ];
  }

  setPalette(paletteKey) {
    this.currentPalette = paletteKey;
    const pal = this.getPaletteColors(paletteKey);

    this.activePartObjects.forEach((item) => {
      item.subMeshes.forEach((sub) => {
        const originCol = sub.partColor || pal.base;
        const blendedBase = originCol.clone().lerp(pal.base, 0.58);
        sub.baseColor.copy(blendedBase);
        sub.rimColor.copy(pal.rim);
        sub.selectedColor.copy(pal.selected);
        sub.selectedRimColor.copy(pal.selectedRim);
        sub.hoverColor.copy(pal.hover);
        sub.hoverRimColor.copy(pal.hoverRim);

        const u = sub.shaderMat?.uniforms;
        if (u) {
          u.uColor.value.copy(blendedBase);
          u.uRimColor.value.copy(pal.rim);
          u.uLaserColor.value.copy(pal.laser);
          u.uSelectedColor.value.copy(pal.selected);
          u.uSelectedRimColor.value.copy(pal.selectedRim);
          u.uHoverColor.value.copy(pal.hover);
          u.uHoverRimColor.value.copy(pal.hoverRim);
          u.uLaserSelectedColor.value.copy(pal.laserSelected);
        }
        if (sub.wireMat) sub.wireMat.color.copy(pal.rim);
        if (sub.pointsMat) sub.pointsMat.color.copy(pal.rim);
      });

      if (item.trajMat) {
        item.trajMat.color.copy(pal.rim);
      }
    });

    this.updateSliceVisualStyle();
    if (this._lastSelectedIndex !== undefined) {
      this.updateVisualStates(this._lastSelectedIndex, this._lastHoveredIndex ?? -1, Boolean(this._lastIsIsolated));
    }
  }

  computeCurvatureAttribute(geometry) {
    if (!geometry || !geometry.attributes || !geometry.attributes.position) return;
    if (geometry.attributes.aCurvature) return;

    if (!geometry.attributes.normal) {
      geometry.computeVertexNormals();
    }

    const posAttr = geometry.attributes.position;
    const normAttr = geometry.attributes.normal;
    if (!posAttr || !posAttr.array) return;
    if (!normAttr || !normAttr.array) {
      geometry.setAttribute('aCurvature', new THREE.BufferAttribute(new Float32Array(posAttr.count || 1), 1));
      return;
    }
    const pos = posAttr.array;
    const norm = normAttr.array;
    const vertCount = posAttr.count;

    if (vertCount === 0) return;

    const curvOut = new Float32Array(vertCount);
    let canonCount = vertCount;
    let canonMap = null;

    if (!geometry.index) {
      if (!geometry.boundingBox) geometry.computeBoundingBox();
      const bb = geometry.boundingBox;
      const maxExtent = bb
        ? Math.max(bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z, 1e-6)
        : 1.0;
      const invQuant = 10000.0 / maxExtent;

      canonMap = new Int32Array(vertCount);
      const spatialHash = new Map();
      let nextId = 0;
      for (let i = 0; i < vertCount; i++) {
        const kx = Math.round(pos[i * 3] * invQuant);
        const ky = Math.round(pos[i * 3 + 1] * invQuant);
        const kz = Math.round(pos[i * 3 + 2] * invQuant);
        const key = `${kx},${ky},${kz}`;
        let cid = spatialHash.get(key);
        if (cid === undefined) {
          cid = nextId++;
          spatialHash.set(key, cid);
        }
        canonMap[i] = cid;
      }
      canonCount = nextId;
    }

    const cPos = canonMap ? new Float32Array(canonCount * 3) : pos;
    const cNorm = canonMap ? new Float32Array(canonCount * 3) : norm;
    if (canonMap) {
      for (let i = 0; i < vertCount; i++) {
        const c = canonMap[i];
        cPos[c * 3] = pos[i * 3];
        cPos[c * 3 + 1] = pos[i * 3 + 1];
        cPos[c * 3 + 2] = pos[i * 3 + 2];
        cNorm[c * 3] += norm[i * 3];
        cNorm[c * 3 + 1] += norm[i * 3 + 1];
        cNorm[c * 3 + 2] += norm[i * 3 + 2];
      }
      for (let c = 0; c < canonCount; c++) {
        const nx = cNorm[c * 3];
        const ny = cNorm[c * 3 + 1];
        const nz = cNorm[c * 3 + 2];
        const len = Math.hypot(nx, ny, nz);
        if (len > 1e-6) {
          cNorm[c * 3] = nx / len;
          cNorm[c * 3 + 1] = ny / len;
          cNorm[c * 3 + 2] = nz / len;
        }
      }
    }

    const lap = new Float32Array(canonCount * 3);
    const deg = new Float32Array(canonCount);
    const idxArray = geometry.index ? geometry.index.array : null;
    const triCount = idxArray ? idxArray.length : vertCount;

    for (let f = 0; f + 2 < triCount; f += 3) {
      const a = canonMap ? canonMap[f] : idxArray[f];
      const b = canonMap ? canonMap[f + 1] : idxArray[f + 1];
      const c = canonMap ? canonMap[f + 2] : idxArray[f + 2];
      if (a < 0 || b < 0 || c < 0 || a >= canonCount || b >= canonCount || c >= canonCount) continue;

      let dx = cPos[b * 3] - cPos[a * 3];
      let dy = cPos[b * 3 + 1] - cPos[a * 3 + 1];
      let dz = cPos[b * 3 + 2] - cPos[a * 3 + 2];
      lap[a * 3] += dx; lap[a * 3 + 1] += dy; lap[a * 3 + 2] += dz; deg[a]++;
      lap[b * 3] -= dx; lap[b * 3 + 1] -= dy; lap[b * 3 + 2] -= dz; deg[b]++;

      dx = cPos[c * 3] - cPos[b * 3];
      dy = cPos[c * 3 + 1] - cPos[b * 3 + 1];
      dz = cPos[c * 3 + 2] - cPos[b * 3 + 2];
      lap[b * 3] += dx; lap[b * 3 + 1] += dy; lap[b * 3 + 2] += dz; deg[b]++;
      lap[c * 3] -= dx; lap[c * 3 + 1] -= dy; lap[c * 3 + 2] -= dz; deg[c]++;

      dx = cPos[a * 3] - cPos[c * 3];
      dy = cPos[a * 3 + 1] - cPos[c * 3 + 1];
      dz = cPos[a * 3 + 2] - cPos[c * 3 + 2];
      lap[c * 3] += dx; lap[c * 3 + 1] += dy; lap[c * 3 + 2] += dz; deg[c]++;
      lap[a * 3] -= dx; lap[a * 3 + 1] -= dy; lap[a * 3 + 2] -= dz; deg[a]++;
    }

    const rawCurv = new Float32Array(canonCount);
    const smoothCurv = new Float32Array(canonCount);
    for (let i = 0; i < canonCount; i++) {
      const d = deg[i];
      if (d > 0) {
        const lx = lap[i * 3] / d;
        const ly = lap[i * 3 + 1] / d;
        const lz = lap[i * 3 + 2] / d;
        const val = -(lx * cNorm[i * 3] + ly * cNorm[i * 3 + 1] + lz * cNorm[i * 3 + 2]);
        rawCurv[i] = Number.isFinite(val) ? val : 0.0;
        smoothCurv[i] = rawCurv[i];
      }
    }

    const acc = new Float32Array(canonCount);
    for (let pass = 0; pass < 4; pass++) {
      acc.fill(0.0);
      for (let f = 0; f + 2 < triCount; f += 3) {
        const a = canonMap ? canonMap[f] : idxArray[f];
        const b = canonMap ? canonMap[f + 1] : idxArray[f + 1];
        const c = canonMap ? canonMap[f + 2] : idxArray[f + 2];
        if (a < 0 || b < 0 || c < 0 || a >= canonCount || b >= canonCount || c >= canonCount) continue;

        acc[a] += smoothCurv[b] + smoothCurv[c];
        acc[b] += smoothCurv[a] + smoothCurv[c];
        acc[c] += smoothCurv[a] + smoothCurv[b];
      }
      for (let i = 0; i < canonCount; i++) {
        const d = deg[i];
        if (d > 0) {
          smoothCurv[i] = 0.5 * smoothCurv[i] + 0.5 * (acc[i] / d);
        }
      }
    }

    const combined = new Float32Array(canonCount);
    let sumSq = 0.0;
    for (let i = 0; i < canonCount; i++) {
      const val = 0.65 * rawCurv[i] + 0.65 * (rawCurv[i] - smoothCurv[i]);
      combined[i] = val;
      sumSq += val * val;
    }

    const rms = Math.sqrt(sumSq / Math.max(1, canonCount));
    const invScale = 1.0 / Math.max(1e-6, rms * 1.4);

    if (canonMap) {
      const cCurv = new Float32Array(canonCount);
      for (let i = 0; i < canonCount; i++) {
        cCurv[i] = Math.tanh(combined[i] * invScale);
      }
      for (let i = 0; i < vertCount; i++) {
        curvOut[i] = cCurv[canonMap[i]];
      }
    } else {
      for (let i = 0; i < vertCount; i++) {
        curvOut[i] = Math.tanh(combined[i] * invScale);
      }
    }

    geometry.setAttribute('aCurvature', new THREE.BufferAttribute(curvOut, 1));
  }

  computeVertexLuminanceAttribute(geometry) {
    if (!geometry || !geometry.attributes || !geometry.attributes.position) return;
    if (geometry.attributes.aVertexLum) return;

    const vertCount = geometry.attributes.position.count;
    if (!vertCount) return;

    const lumOut = new Float32Array(vertCount);
    const colorAttr = geometry.attributes.color;

    if (colorAttr && colorAttr.array && colorAttr.count === vertCount && (colorAttr.itemSize || 3) >= 3) {
      const arr = colorAttr.array;
      const stride = colorAttr.itemSize || 3;
      let maxComp = 0.0;
      const probeStep = Math.max(1, Math.floor(vertCount / 512));
      for (let i = 0; i < vertCount; i += probeStep) {
        const off = i * stride;
        const m = Math.max(arr[off] || 0, arr[off + 1] || 0, arr[off + 2] || 0);
        if (m > maxComp) maxComp = m;
      }
      const scaleInv = maxComp > 1.5 ? (1.0 / 255.0) : 1.0;

      const rawLum = new Float32Array(vertCount);
      const hist = new Uint32Array(256);
      for (let i = 0; i < vertCount; i++) {
        const off = i * stride;
        const r = Math.max(0.0, Math.min(1.0, (arr[off] || 0) * scaleInv));
        const g = Math.max(0.0, Math.min(1.0, (arr[off + 1] || 0) * scaleInv));
        const b = Math.max(0.0, Math.min(1.0, (arr[off + 2] || 0) * scaleInv));
        const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        rawLum[i] = lum;
        const bin = Math.max(0, Math.min(255, Math.round(lum * 255)));
        hist[bin]++;
      }

      const target05 = Math.max(0, Math.floor(vertCount * 0.05));
      const target95 = Math.min(vertCount - 1, Math.floor(vertCount * 0.95));
      let cum = 0;
      let bin05 = 0;
      let bin95 = 255;
      let found05 = false;
      for (let b = 0; b < 256; b++) {
        cum += hist[b];
        if (!found05 && cum > target05) {
          bin05 = b;
          found05 = true;
        }
        if (cum > target95) {
          bin95 = b;
          break;
        }
      }

      const p05 = bin05 / 255.0;
      const p95 = bin95 / 255.0;
      const span = p95 - p05;
      if (span > 0.04) {
        const invSpan = 1.0 / span;
        for (let i = 0; i < vertCount; i++) {
          const normL = Math.max(0.0, Math.min(1.0, (rawLum[i] - p05) * invSpan));
          lumOut[i] = 0.18 + 0.82 * normL;
        }
      } else {
        lumOut.fill(0.78);
      }
    } else {
      lumOut.fill(0.78);
    }

    geometry.setAttribute('aVertexLum', new THREE.BufferAttribute(lumOut, 1));
  }

  createVectorBondGeometry(geometry) {
    let rawPos = null;
    if (typeof THREE.EdgesGeometry === 'function') {
      try {
        const edgesGeo = new THREE.EdgesGeometry(geometry, Number(this.config.edgeCreaseAngleDeg ?? 22));
        if (edgesGeo?.attributes?.position?.array) {
          rawPos = edgesGeo.attributes.position.array;
        }
      } catch (_) { }
    }
    if (!rawPos && geometry?.attributes?.position?.array) {
      const vArr = geometry.attributes.position.array;
      const segs = [];
      for (let i = 0; i < vArr.length - 5; i += 6) {
        segs.push(vArr[i], vArr[i+1], vArr[i+2], vArr[i+3], vArr[i+4], vArr[i+5]);
      }
      rawPos = new Float32Array(segs);
    }
    if (!rawPos || rawPos.length < 6) {
      rawPos = new Float32Array([0, -0.2, 0, 0, 0.2, 0]);
    }

    const numPoints = rawPos.length / 3;
    const numSegments = Math.floor(numPoints / 2);
    const posOut = new Float32Array(numSegments * 6);
    const originOut = new Float32Array(numSegments * 6);
    const targetOut = new Float32Array(numSegments * 6);
    const segTOut = new Float32Array(numSegments * 2);
    const phaseOut = new Float32Array(numSegments * 2);

    for (let s = 0; s < numSegments; s++) {
      const idx = s * 6;
      const x0 = rawPos[idx];
      const y0 = rawPos[idx + 1];
      const z0 = rawPos[idx + 2];
      const x1 = rawPos[idx + 3];
      const y1 = rawPos[idx + 4];
      const z1 = rawPos[idx + 5];

      posOut[idx] = x0; posOut[idx + 1] = y0; posOut[idx + 2] = z0;
      posOut[idx + 3] = x1; posOut[idx + 4] = y1; posOut[idx + 5] = z1;

      originOut[idx] = x0; originOut[idx + 1] = y0; originOut[idx + 2] = z0;
      originOut[idx + 3] = x0; originOut[idx + 4] = y0; originOut[idx + 5] = z0;

      targetOut[idx] = x1; targetOut[idx + 1] = y1; targetOut[idx + 2] = z1;
      targetOut[idx + 3] = x1; targetOut[idx + 4] = y1; targetOut[idx + 5] = z1;

      const tIdx = s * 2;
      segTOut[tIdx] = 0.0;
      segTOut[tIdx + 1] = 1.0;

      const midY = (y0 + y1) * 0.5;
      const hash = Math.abs(Math.sin((s + 1) * 12.9898 + (x0 + z1) * 78.233)) % 1.0;
      const normY = Math.max(0.0, Math.min(1.0, midY * 0.8 + 0.5));
      const phase = Math.min(1.0, Math.max(0.0, normY * 0.7 + hash * 0.3));
      phaseOut[tIdx] = phase;
      phaseOut[tIdx + 1] = phase;
    }

    const bondGeo = new THREE.BufferGeometry();
    bondGeo.setAttribute('position', new THREE.BufferAttribute(posOut, 3));
    bondGeo.setAttribute('aOrigin', new THREE.BufferAttribute(originOut, 3));
    bondGeo.setAttribute('aTarget', new THREE.BufferAttribute(targetOut, 3));
    bondGeo.setAttribute('aSegmentT', new THREE.BufferAttribute(segTOut, 1));
    bondGeo.setAttribute('aBondPhase', new THREE.BufferAttribute(phaseOut, 1));

    return bondGeo;
  }

  createHologramMeshFromGeometry(geometry, colorHex, partIndex, customPointsGeometry = null, partData = null) {
    geometry.computeVertexNormals();
    this.computeCurvatureAttribute(geometry);
    this.computeVertexLuminanceAttribute(geometry);

    if (!geometry.boundingBox) geometry.computeBoundingBox();
    const bb = geometry.boundingBox;
    const rawExtent = (bb && Number.isFinite(bb.max.x - bb.min.x))
      ? Math.max(bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z)
      : 1.0;
    const meshExtent = (rawExtent > 1e-4 && Number.isFinite(rawExtent)) ? rawExtent : 1.0;
    const meshRadius = this._computeGeometryRadius(geometry);

    const initialPal = this.getPaletteColors(this.currentPalette);
    const partColor = colorHex ? new THREE.Color(colorHex) : initialPal.base.clone();
    const baseColor = partColor.clone().lerp(initialPal.base, 0.25);
    const rimColor = initialPal.rim.clone();
    const selectedColor = initialPal.selected.clone();
    const selectedRimColor = initialPal.selectedRim.clone();
    const hoverColor = initialPal.hover.clone();
    const hoverRimColor = initialPal.hoverRim.clone();

    const shaderMat = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.clone(HologramGLSL.uniforms),
      vertexShader: HologramGLSL.vertexShader,
      fragmentShader: HologramGLSL.fragmentShader,
      extensions: { derivatives: true },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide
    });

    shaderMat.uniforms.uColor.value.copy(baseColor);
    shaderMat.uniforms.uRimColor.value.copy(rimColor);
    shaderMat.uniforms.uLaserColor.value.copy(initialPal.laser);
    shaderMat.uniforms.uSelectedColor.value.copy(selectedColor);
    shaderMat.uniforms.uSelectedRimColor.value.copy(selectedRimColor);
    shaderMat.uniforms.uHoverColor.value.copy(hoverColor);
    shaderMat.uniforms.uHoverRimColor.value.copy(hoverRimColor);
    shaderMat.uniforms.uLaserSelectedColor.value.copy(initialPal.laserSelected);
    shaderMat.uniforms.uEnableLaser.value = this.laserEnabled ? 1.0 : 0.0;
    shaderMat.uniforms.uMeshScale.value = meshExtent;
    shaderMat.uniforms.uMicroEtchStrength.value = this.isBudgetConstrained ? 0.75 : 1.0;
    shaderMat.uniforms.uDepthSliceStrength.value = this.isBudgetConstrained ? 0.80 : 1.0;
    shaderMat.uniforms.uSliceCutaway.value = this.sliceCutawayEnabled ? 1.0 : 0.0;
    shaderMat.uniforms.uDepthSliceY.value = this.sliceCutawayEnabled ? this.sliceY : this.sweepY;

    const isMasterHull = typeof partData?.isMasterShell === 'boolean'
      ? partData.isMasterShell
      : (this.masterHullIndex >= 0 && partIndex === this.masterHullIndex);
    shaderMat.uniforms.uCoreAlpha.value = isMasterHull ? this.config.masterHullCoreAlpha : this.config.defaultCoreAlpha;
    shaderMat.uniforms.uRimAlpha.value = isMasterHull ? 0.90 : 1.04;
    shaderMat.uniforms.uRimPower.value = isMasterHull ? 2.65 : 2.95;

    const mesh = new THREE.Mesh(geometry, shaderMat);
    mesh.userData.partIndex = partIndex;

    const basePointSize = Math.max(0.008, Math.min(0.018, meshRadius * 0.016));
    const basePointOpacity = isMasterHull ? 0.22 : 0.34;
    const pointsGeo = customPointsGeometry || this.computeVoxelDecimatedPointsGeometry(geometry, this.config.targetPointsCount);
    const pointsMat = new THREE.PointsMaterial({
      size: isMasterHull ? basePointSize * 0.85 : basePointSize,
      map: GLOW_SPRITE,
      color: rimColor,
      transparent: true,
      opacity: basePointOpacity,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const pointsObj = new THREE.Points(pointsGeo, pointsMat);
    pointsObj.userData.partIndex = partIndex;
    mesh.add(pointsObj);

    const vertCount = geometry.attributes.position ? geometry.attributes.position.count : 1000;
    const densityFactor = Math.min(1.0, this.config.wireframeDensityCap / Math.max(this.config.wireframeDensityCap, vertCount));
    const baseWireOpacity = (isMasterHull ? this.config.masterHullWireOpacity : this.config.cappingWireOpacity) * (0.28 + 0.72 * densityFactor);

    const creaseAngleDeg = Number(this.config.edgeCreaseAngleDeg ?? 22);
    const canEdgeLines =
      typeof THREE.EdgesGeometry === 'function' &&
      typeof THREE.LineSegments === 'function';

    let bondGeo;
    let bondMat;
    let bondMesh;

    if (canEdgeLines && typeof THREE.ShaderMaterial === 'function' && typeof VectorBondGLSL === 'object') {
      bondGeo = this.createVectorBondGeometry(geometry);
      bondMat = new THREE.ShaderMaterial({
        uniforms: THREE.UniformsUtils.clone(VectorBondGLSL.uniforms),
        vertexShader: VectorBondGLSL.vertexShader,
        fragmentShader: VectorBondGLSL.fragmentShader,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        depthTest: true
      });
      bondMat.uniforms.uColor.value.copy(rimColor);
      bondMat.uniforms.uExcitedColor.value.copy(initialPal.laser);
      bondMat.uniforms.uSelectedColor.value.copy(selectedColor);
      bondMat.uniforms.uOpacity.value = baseWireOpacity * 2.4;
      bondMat.uniforms.uBondProgress.value = 1.0;
      // Provide standard color property for backward-compatibility with updateVisualStates
      bondMat.color = bondMat.uniforms.uColor.value;
      bondMesh = new THREE.LineSegments(bondGeo, bondMat);
    } else {
      bondMat = new THREE.LineBasicMaterial({
        color: rimColor,
        transparent: true,
        opacity: baseWireOpacity,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        depthTest: true
      });
      bondMesh = new THREE.LineSegments(geometry, bondMat);
    }

    bondMesh.userData.partIndex = partIndex;
    if (this.isBudgetConstrained) {
      bondMesh.visible = false;
    }
    mesh.add(bondMesh);

    return {
      mesh,
      shaderMat,
      pointsObj,
      pointsMat,
      bondMesh,
      bondMat,
      wireMesh: bondMesh,
      wireMat: bondMat,
      baseWireOpacity,
      basePointSize,
      basePointOpacity,
      partColor,
      baseColor,
      rimColor,
      selectedColor,
      selectedRimColor,
      hoverColor,
      hoverRimColor
    };
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
    for (const child of [...this.masterGroup.children]) {
      if (child?.userData?.persistentOverlay === true) {
        continue;
      }
      this.masterGroup.remove(child);
      this._disposeObjectTree(child);
    }
    this.conduitsGroup = new THREE.Group();
    this.masterGroup.add(this.conduitsGroup);
    this.activeConduits = [];
    this.activePartObjects = [];
    this.masterHullIndex = -1;
    this._conduitParentIndex = -1;
    this._lastConduitKey = '';
    this.sliceY = 0.0;
    this.assemblyMinY = -0.85;
    this.assemblyMaxY = 0.85;
    this.assemblyRadius = 1.15;
    this.sliceRingGroup = null;
    this.sliceRingMat = null;
    this.sliceTickMat = null;
    this.sliceTicksMesh = null;
    this.initSliceRing();
  }

  _buildCustomLatticeGeometry(positions, indices, sx, sy, sz) {
    const geo = new THREE.BufferGeometry();
    const count = positions.length;
    const scaled = new Float32Array(count);
    for (let i = 0; i < count; i += 3) {
      scaled[i] = positions[i] * sx;
      scaled[i + 1] = positions[i + 1] * sy;
      scaled[i + 2] = positions[i + 2] * sz;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(scaled, 3));
    if (indices && indices.length > 0 && typeof geo.setIndex === 'function') {
      const idxArr = (count / 3 > 65535) ? new Uint32Array(indices) : new Uint16Array(indices);
      geo.setIndex(new THREE.BufferAttribute(idxArr, 1));
    }

    // Explicit normals for compatibility with minimal mock & test environments
    const normals = new Float32Array(count);
    if (indices && indices.length > 0) {
      for (let i = 0; i < indices.length; i += 3) {
        const i0 = indices[i] * 3;
        const i1 = indices[i + 1] * 3;
        const i2 = indices[i + 2] * 3;
        const ax = scaled[i1] - scaled[i0];
        const ay = scaled[i1 + 1] - scaled[i0 + 1];
        const az = scaled[i1 + 2] - scaled[i0 + 2];
        const bx = scaled[i2] - scaled[i0];
        const by = scaled[i2 + 1] - scaled[i0 + 1];
        const bz = scaled[i2 + 2] - scaled[i0 + 2];
        const nx = ay * bz - az * by;
        const ny = az * bx - ax * bz;
        const nz = ax * by - ay * bx;
        normals[i0] += nx; normals[i0 + 1] += ny; normals[i0 + 2] += nz;
        normals[i1] += nx; normals[i1 + 1] += ny; normals[i1 + 2] += nz;
        normals[i2] += nx; normals[i2 + 1] += ny; normals[i2 + 2] += nz;
      }
      for (let i = 0; i < count; i += 3) {
        const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]);
        if (len > 1e-6) {
          normals[i] /= len; normals[i + 1] /= len; normals[i + 2] /= len;
        } else {
          normals[i + 1] = 1.0;
        }
      }
    } else {
      for (let i = 0; i < count; i += 3) {
        normals[i + 1] = 1.0;
      }
    }
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));

    if (typeof geo.computeVertexNormals === 'function') geo.computeVertexNormals();
    if (typeof geo.computeBoundingBox === 'function') geo.computeBoundingBox();
    if (typeof geo.computeBoundingSphere === 'function') geo.computeBoundingSphere();
    return geo;
  }

  _buildRotaryDiscLattice(sx, sy, sz) {
    const pos = [];
    const idx = [];
    const hubCount = 8;
    for (let i = 0; i < hubCount; i++) {
      const a = (i * 2 * Math.PI) / hubCount;
      pos.push(Math.cos(a) * 0.16, 0.05, Math.sin(a) * 0.16);
      pos.push(Math.cos(a) * 0.16, -0.05, Math.sin(a) * 0.16);
    }
    const midCount = 12;
    for (let i = 0; i < midCount; i++) {
      const a = (i * 2 * Math.PI) / midCount;
      pos.push(Math.cos(a) * 0.36, 0.03, Math.sin(a) * 0.36);
      pos.push(Math.cos(a) * 0.36, -0.03, Math.sin(a) * 0.36);
    }
    const outCount = 16;
    for (let i = 0; i < outCount; i++) {
      const a = (i * 2 * Math.PI) / outCount;
      pos.push(Math.cos(a) * 0.54, 0.015, Math.sin(a) * 0.54);
      pos.push(Math.cos(a) * 0.54, -0.015, Math.sin(a) * 0.54);
    }
    pos.push(0, 0.05, 0);
    pos.push(0, -0.05, 0);
    const topCenterIdx = pos.length / 3 - 2;
    const btmCenterIdx = pos.length / 3 - 1;

    for (let i = 0; i < hubCount; i++) {
      const next = (i + 1) % hubCount;
      const t0 = i * 2;
      const b0 = i * 2 + 1;
      const t1 = next * 2;
      const b1 = next * 2 + 1;
      idx.push(topCenterIdx, t0, t1);
      idx.push(btmCenterIdx, b1, b0);
      idx.push(t0, b0, t1);
      idx.push(t1, b0, b1);
    }
    const midBase = hubCount * 2;
    const outBase = midBase + midCount * 2;
    for (let i = 0; i < midCount; i++) {
      const nextMid = (i + 1) % midCount;
      const outI = Math.floor((i * outCount) / midCount);
      const outNext = Math.floor(((i + 1) * outCount) / midCount);
      const mt0 = midBase + i * 2;
      const mb0 = midBase + i * 2 + 1;
      const mt1 = midBase + nextMid * 2;
      const mb1 = midBase + nextMid * 2 + 1;
      const ot0 = outBase + outI * 2;
      const ob0 = outBase + outI * 2 + 1;
      const ot1 = outBase + outNext * 2;
      const ob1 = outBase + outNext * 2 + 1;
      idx.push(mt0, ot0, mt1);
      idx.push(mt1, ot0, ot1);
      idx.push(mb0, mb1, ob0);
      idx.push(mb1, ob1, ob0);
    }
    return this._buildCustomLatticeGeometry(pos, idx, sx, sy, sz);
  }

  _buildHelicalCoreLattice(sx, sy, sz) {
    const pos = [];
    const idx = [];
    const tiers = 5;
    const segs = 8;
    const radius = 0.42;

    for (let t = 0; t < tiers; t++) {
      const y = -0.48 + (t / (tiers - 1)) * 0.96;
      const twist = (t * Math.PI) / segs;
      for (let s = 0; s < segs; s++) {
        const a = (s * 2 * Math.PI) / segs + twist;
        pos.push(Math.cos(a) * radius, y, Math.sin(a) * radius);
      }
      pos.push(0, y, 0);
    }

    for (let t = 0; t < tiers - 1; t++) {
      const base0 = t * (segs + 1);
      const base1 = (t + 1) * (segs + 1);
      const center0 = base0 + segs;
      const center1 = base1 + segs;

      for (let s = 0; s < segs; s++) {
        const next = (s + 1) % segs;
        const p00 = base0 + s;
        const p01 = base0 + next;
        const p10 = base1 + s;
        const p11 = base1 + next;

        idx.push(p00, p01, p11);
        idx.push(p00, p11, p10);
        idx.push(p00, center0, p10);
        idx.push(p10, center0, center1);
      }
    }
    return this._buildCustomLatticeGeometry(pos, idx, sx, sy, sz);
  }

  _buildConicalTaperLattice(sx, sy, sz) {
    const pos = [];
    const idx = [];
    const tiers = 5;
    const segs = 12;
    const radii = [0.55, 0.42, 0.32, 0.24, 0.18];
    const yLevels = [-0.50, -0.22, 0.05, 0.30, 0.50];

    for (let t = 0; t < tiers; t++) {
      const y = yLevels[t];
      const r = radii[t];
      for (let s = 0; s < segs; s++) {
        const a = (s * 2 * Math.PI) / segs;
        pos.push(Math.cos(a) * r, y, Math.sin(a) * r);
      }
    }

    for (let t = 0; t < tiers - 1; t++) {
      const base0 = t * segs;
      const base1 = (t + 1) * segs;
      for (let s = 0; s < segs; s++) {
        const next = (s + 1) % segs;
        const p00 = base0 + s;
        const p01 = base0 + next;
        const p10 = base1 + s;
        const p11 = base1 + next;
        idx.push(p00, p01, p11);
        idx.push(p00, p11, p10);
      }
    }
    return this._buildCustomLatticeGeometry(pos, idx, sx, sy, sz);
  }

  _buildGeodesicShellLattice(sx, sy, sz) {
    const t = (1.0 + Math.sqrt(5.0)) / 2.0;
    const v = [
      -1,  t,  0,   1,  t,  0,  -1, -t,  0,   1, -t,  0,
       0, -1,  t,   0,  1,  t,   0, -1, -t,   0,  1, -t,
       t,  0, -1,   t,  0,  1,  -t,  0, -1,  -t,  0,  1
    ];
    const pos = [];
    const scaleFactor = 0.52 / Math.hypot(1, t, 0);
    for (let i = 0; i < v.length; i += 3) {
      pos.push(v[i] * scaleFactor, v[i + 1] * scaleFactor, v[i + 2] * scaleFactor);
    }
    const idx = [
      0, 11, 5,   0, 5, 1,   0, 1, 7,   0, 7, 10,  0, 10, 11,
      1, 5, 9,    5, 11, 4,  11, 10, 2, 10, 7, 6,   7, 1, 8,
      3, 9, 4,    3, 4, 2,   3, 2, 6,   3, 6, 8,    3, 8, 9,
      4, 9, 5,    2, 4, 11,  6, 2, 10,  8, 6, 7,    9, 8, 1
    ];
    return this._buildCustomLatticeGeometry(pos, idx, sx, sy, sz);
  }

  _buildSpaceFrameTrussLattice(sx, sy, sz) {
    const pos = [
      -0.45, -0.35, -0.45,   0.45, -0.35, -0.45,   0.45,  0.35, -0.45,  -0.45,  0.35, -0.45,
      -0.45, -0.35,  0.45,   0.45, -0.35,  0.45,   0.45,  0.35,  0.45,  -0.45,  0.35,  0.45,
       0.0,  -0.35,  0.0,    0.0,   0.35,  0.0,
      -0.45,  0.0,   0.0,    0.45,  0.0,   0.0,
       0.0,   0.0,  -0.45,   0.0,   0.0,   0.45,
       0.0,   0.0,   0.0
    ];
    const idx = [
      0, 1, 8,   1, 5, 8,   5, 4, 8,   4, 0, 8,
      3, 9, 2,   2, 9, 6,   6, 9, 7,   7, 9, 3,
      0, 1, 14,  1, 2, 14,  2, 3, 14,  3, 0, 14,
      4, 5, 14,  5, 6, 14,  6, 7, 14,  7, 4, 14,
      0, 4, 14,  1, 5, 14,  2, 6, 14,  3, 7, 14
    ];
    return this._buildCustomLatticeGeometry(pos, idx, sx, sy, sz);
  }

  _buildBranchingConduitLattice(sx, sy, sz) {
    const pos = [];
    const idx = [];
    const rings = 8;
    const segs = 6;
    const r = 0.14;

    for (let k = 0; k < rings; k++) {
      const t = (k / (rings - 1)) * Math.PI * 0.9;
      const cx = Math.cos(t) * 0.38;
      const cy = Math.sin(t) * 0.38 - 0.15;
      const cz = (k - rings / 2) * 0.05;
      for (let s = 0; s < segs; s++) {
        const a = (s * 2 * Math.PI) / segs;
        pos.push(cx + Math.cos(a) * r, cy + Math.sin(a) * r, cz);
      }
    }

    for (let k = 0; k < rings - 1; k++) {
      const base0 = k * segs;
      const base1 = (k + 1) * segs;
      for (let s = 0; s < segs; s++) {
        const next = (s + 1) % segs;
        idx.push(base0 + s, base0 + next, base1 + next);
        idx.push(base0 + s, base1 + next, base1 + s);
      }
    }
    return this._buildCustomLatticeGeometry(pos, idx, sx, sy, sz);
  }

  _buildContouredClusterLattice(sx, sy, sz) {
    const pos = [];
    const idx = [];
    const rings = 7;
    const segs = 10;

    for (let i = 0; i < rings; i++) {
      const v = (i / (rings - 1)) * Math.PI;
      const y = -Math.cos(v) * 0.48;
      const r = Math.sin(v) * (0.42 + 0.10 * Math.sin(v * 2.0));
      for (let j = 0; j < segs; j++) {
        const u = (j * 2 * Math.PI) / segs;
        const x = Math.cos(u) * r * (1.0 + 0.12 * Math.cos(u * 2.0));
        const z = Math.sin(u) * r * (1.0 - 0.08 * Math.sin(u));
        pos.push(x, y, z);
      }
    }

    for (let i = 0; i < rings - 1; i++) {
      const b0 = i * segs;
      const b1 = (i + 1) * segs;
      for (let j = 0; j < segs; j++) {
        const next = (j + 1) % segs;
        idx.push(b0 + j, b0 + next, b1 + next);
        idx.push(b0 + j, b1 + next, b1 + j);
      }
    }
    return this._buildCustomLatticeGeometry(pos, idx, sx, sy, sz);
  }

  createPlaceholderGeometry(topology, scaleArr = [0.75, 0.75, 0.75]) {
    const [sx, sy, sz] = this.safeVec3Array(scaleArr, [0.75, 0.75, 0.75]).map((v) =>
      Math.max(0.15, Math.min(2.5, Math.abs(v)))
    );

    switch (topology) {
      case TOPOLOGY_UNIT_SPECS.rotary_disc:
        return this._buildRotaryDiscLattice(sx, sy, sz);
      case TOPOLOGY_UNIT_SPECS.cylindrical_core:
        return this._buildHelicalCoreLattice(sx, sy, sz);
      case TOPOLOGY_UNIT_SPECS.conical_taper:
        return this._buildConicalTaperLattice(sx, sy, sz);
      case TOPOLOGY_UNIT_SPECS.structural_frame:
        return this._buildSpaceFrameTrussLattice(sx, sy, sz);
      case TOPOLOGY_UNIT_SPECS.branched_conduit:
        return this._buildBranchingConduitLattice(sx, sy, sz);
      case TOPOLOGY_UNIT_SPECS.hollow_shell:
        return this._buildGeodesicShellLattice(sx, sy, sz);
      case TOPOLOGY_UNIT_SPECS.contoured_solid:
      default:
        return this._buildContouredClusterLattice(sx, sy, sz);
    }
  }

  startProgressiveBuild(header) {
    this.clear();
    if (!this.plyLoader && window.THREE && window.THREE.PLYLoader) {
      this.plyLoader = new window.THREE.PLYLoader();
    }
    return this.activePartObjects;
  }

  addPartProgressive(part, idx) {
    if (this.activePartObjects.some((p) => p.index === idx)) {
      return this.activePartObjects.find((p) => p.index === idx);
    }

    if (!this.plyLoader && window.THREE && window.THREE.PLYLoader) {
      this.plyLoader = new window.THREE.PLYLoader();
    }

    const pal = this.getPaletteColors(this.currentPalette);
    const partGroup = new THREE.Group();
    const assembledVec = new THREE.Vector3(...this.safeVec3Array(part.assembledPos, [0, 0, 0]));
    const explodedOffset = new THREE.Vector3(...this.safeVec3Array(part.explodedVec, [0, 0, 0]));

    partGroup.position.copy(assembledVec);

    const placeholderGeo = this.createPlaceholderGeometry(part.topology, part.scale);
    const placeholderInst = this.createHologramMeshFromGeometry(placeholderGeo, part.color, idx, null, part);
    partGroup.add(placeholderInst.mesh);
    partGroup.scale.set(0.94, 0.94, 0.94);

    if (this.config.sound && typeof this.config.sound.playSelect === 'function') {
      try { this.config.sound.playSelect(); } catch (_) { }
    }

    const endPos = assembledVec.clone().add(explodedOffset);
    const trajGeo = new THREE.BufferGeometry().setFromPoints([assembledVec, endPos]);
    const trajMat = new THREE.LineDashedMaterial({
      color: pal.rim.clone(),
      dashSize: 0.14,
      gapSize: 0.09,
      transparent: true,
      opacity: 0.0,
      blending: THREE.AdditiveBlending
    });
    const trajLine = new THREE.Line(trajGeo, trajMat);
    trajLine.computeLineDistances();
    this.masterGroup.add(trajLine);

    this.masterGroup.add(partGroup);

    const partObj = {
      index: idx,
      data: part,
      group: partGroup,
      subMeshes: [placeholderInst],
      assembledVec,
      explodedOffset,
      trajLine,
      trajMat,
      spawnProgress: 0.0,
      isPlaceholder: true
    };
    this.activePartObjects.push(partObj);
    this._recomputeHullRoles();
    this.fitSliceRingToAssembly();
    this._lastConduitKey = '';

    if (part.meshUrl) {
      this.attachPartMesh(idx, part.meshUrl, part.textureUrl, part.normalMapUrl);
    }

    return partObj;
  }

  attachPartMesh(partIndex, meshUrl, textureUrl = null, normalMapUrl = null) {
    const item = this.activePartObjects.find((p) => p.index === partIndex);
    if (!item) return;

    item.data.meshUrl = meshUrl;
    item.data.textureUrl = textureUrl;
    item.data.normalMapUrl = normalMapUrl;

    if (!this.plyLoader && window.THREE && window.THREE.PLYLoader) {
      this.plyLoader = new window.THREE.PLYLoader();
    }
    if (!this.plyLoader || !meshUrl) return;

    let pointsUrl = '';
    try {
      const parsedUrl = new URL(meshUrl, window.location.origin);
      const lastDot = parsedUrl.pathname.lastIndexOf('.');
      const cleanPath = lastDot > 0 ? parsedUrl.pathname.slice(0, lastDot) : parsedUrl.pathname;
      pointsUrl = `${cleanPath}_points.json${parsedUrl.search}`;
    } catch (_) {
      const qIdx = meshUrl.indexOf('?');
      const pathOnly = qIdx >= 0 ? meshUrl.slice(0, qIdx) : meshUrl;
      const queryPart = qIdx >= 0 ? meshUrl.slice(qIdx) : '';
      const dotIdx = pathOnly.lastIndexOf('.');
      const stem = dotIdx > 0 ? pathOnly.slice(0, dotIdx) : pathOnly;
      pointsUrl = `${stem}_points.json${queryPart}`;
    }

    const fetchPointsPromise = fetch(pointsUrl)
      .then((res) => (res.ok ? res.json() : null))
      .catch(() => null);

    this.plyLoader.load(
      meshUrl,
      async (geometry) => {
        for (const ch of [...item.group.children]) {
          if (ch.userData && ch.userData.isCouplingNode) continue;
          item.group.remove(ch);
          this._disposeObjectTree(ch);
        }

        let customPointsGeo = null;
        try {
          const pointsData = await fetchPointsPromise;
          if (pointsData && Array.isArray(pointsData.points) && pointsData.points.length > 0) {
            customPointsGeo = new THREE.BufferGeometry();
            customPointsGeo.setAttribute(
              'position',
              new THREE.BufferAttribute(new Float32Array(pointsData.points), 3)
            );
          }
        } catch (_) { }

        const inst = this.createHologramMeshFromGeometry(geometry, item.data.color, partIndex, customPointsGeo, item.data);
        item.group.add(inst.mesh);
        item.subMeshes = [inst];
        item.isPlaceholder = false;
        // Trigger high-resolution neural mesh vector bond and laser crystallization sweep!
        item.spawnProgress = 0.0;
        if (this.config.sound && typeof this.config.sound.playChime === 'function') {
          try { this.config.sound.playChime(1560, 0.06, 'sine'); } catch (_) { }
        }
        this._recomputeHullRoles();
        this.fitSliceRingToAssembly();
        this._lastConduitKey = '';
        this.updateConduits();
        if (this._lastSelectedIndex !== undefined) {
          this.updateVisualStates(this._lastSelectedIndex, this._lastHoveredIndex ?? -1, Boolean(this._lastIsIsolated));
        }
        if (typeof this.onMeshAttached === 'function') {
          this.onMeshAttached(partIndex, item);
        }
      },
      undefined,
      (err) => {
        console.warn(`[!] Progressive mesh load issue for part ${partIndex} (${meshUrl}):`, err);
      }
    );
  }

  updateLayoutPositions(repairedParts) {
    if (!Array.isArray(repairedParts)) return;

    repairedParts.forEach((part, idx) => {
      const item = this.activePartObjects.find((p) => p.index === idx || p.data.id === part.id);
      if (!item) return;

      const newPos = new THREE.Vector3(...this.safeVec3Array(part.assembledPos, item.assembledVec.toArray()));
      const newExp = new THREE.Vector3(...this.safeVec3Array(part.explodedVec, item.explodedOffset.toArray()));

      item.assembledVec.copy(newPos);
      item.explodedOffset.copy(newExp);
      item.data.assembledPos = [newPos.x, newPos.y, newPos.z];
      item.data.explodedVec = [newExp.x, newExp.y, newExp.z];
      if (Array.isArray(part.scale)) item.data.scale = [...part.scale];
      if (typeof part.isMasterShell === 'boolean') item.data.isMasterShell = part.isMasterShell;

      item.group.position.copy(newPos).addScaledVector(newExp, this.currentExplosionScalar || 0.0);

      if (item.trajLine) {
        const endPos = newPos.clone().add(newExp);
        item.trajLine.geometry.setFromPoints([newPos, endPos]);
        item.trajLine.computeLineDistances();
      }

      if (part.meshUrl && Array.isArray(part.scale)) {
        const s0 = Number(part.scale[0] ?? 1.0);
        const prevS = Number(item.data._appliedScale ?? 1.0);
        if (Math.abs(s0 - prevS) > 1e-3) {
          item.data._appliedScale = s0;
          try {
            const parsedUrl = new URL(part.meshUrl, window.location.origin);
            parsedUrl.searchParams.set('s', String(Math.round(s0 * 1000)));
            this.attachPartMesh(idx, parsedUrl.toString());
          } catch (_) {
            const cleanMeshUrl = String(part.meshUrl).split('?')[0];
            this.attachPartMesh(idx, `${cleanMeshUrl}?s=${Math.round(s0 * 1000)}`);
          }
        }
      }
    });
    this._recomputeHullRoles();
    this._lastConduitKey = '';
    this.updateConduits();
    this.fitSliceRingToAssembly();
  }

  build(blueprint) {
    this.startProgressiveBuild(blueprint);
    const parts = Array.isArray(blueprint.parts) ? blueprint.parts : [];
    parts.forEach((part, idx) => {
      this.addPartProgressive(part, idx);
    });
    this.updateConduits();
    return this.activePartObjects;
  }

  createCouplingNode(colorHex, unitRadius = 0.04) {
    const pal = this.getPaletteColors(this.currentPalette);
    const nodeGroup = new THREE.Group();
    nodeGroup.userData.isCouplingNode = true;
    const rimCol = new THREE.Color(colorHex || pal.rim);

    const ringGeo = new THREE.TorusGeometry(unitRadius, unitRadius * 0.10, 8, 16);
    const ringMat = new THREE.MeshBasicMaterial({
      color: rimCol,
      transparent: true,
      opacity: 0.45,
      wireframe: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    nodeGroup.add(ringMesh);

    const pinGeo = new THREE.CylinderGeometry(unitRadius * 0.06, unitRadius * 0.06, unitRadius * 0.42, 6);
    const pinMat = new THREE.MeshBasicMaterial({
      color: pal.laserSelected.clone(),
      transparent: true,
      opacity: 0.55,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    for (let i = 0; i < 4; i++) {
      const pin = new THREE.Mesh(pinGeo, pinMat);
      const angle = (i * Math.PI) / 2;
      pin.position.set(Math.cos(angle) * unitRadius * 0.85, Math.sin(angle) * unitRadius * 0.85, unitRadius * 0.25);
      pin.rotation.x = Math.PI / 2;
      nodeGroup.add(pin);
    }

    return { nodeGroup, ringMat, pinMat };
  }

  _clearConduits() {
    for (const conduit of this.activeConduits) {
      if (conduit.parentCoupling?.nodeGroup) {
        conduit.parentCoupling.nodeGroup.parent?.remove(conduit.parentCoupling.nodeGroup);
        this._disposeObjectTree(conduit.parentCoupling.nodeGroup);
      }
      if (conduit.childCoupling?.nodeGroup) {
        conduit.childCoupling.nodeGroup.parent?.remove(conduit.childCoupling.nodeGroup);
        this._disposeObjectTree(conduit.childCoupling.nodeGroup);
      }
    }
    if (this.conduitsGroup) {
      for (const ch of [...this.conduitsGroup.children]) {
        this.conduitsGroup.remove(ch);
        this._disposeObjectTree(ch);
      }
    }
    this.activeConduits = [];
    this._conduitParentIndex = -1;
    this._lastConduitKey = '';
  }

  updateConduits() {
    if (this.activePartObjects.length < 2) return;

    const curArch = this.activeBlueprint?.architecture || '';
    const titleL = (this.activeBlueprint?.title || '').toLowerCase();
    const isLayeredNetwork = curArch === 'graph_layered_network' ||
      titleL.includes('network') || titleL.includes('neural');

    const hullIndex = (this.masterHullIndex >= 0 && this.masterHullIndex < this.activePartObjects.length)
      ? this.masterHullIndex
      : 0;

    let connectionPairs = [];
    if (isLayeredNetwork) {
      for (let i = 0; i < this.activePartObjects.length - 1; i++) {
        connectionPairs.push({ pIdx: i, cIdx: i + 1 });
      }
    } else {
      for (let i = 0; i < this.activePartObjects.length; i++) {
        if (i === hullIndex) continue;
        connectionPairs.push({ pIdx: hullIndex, cIdx: i });
      }
    }

    const scalar = this.currentExplosionScalar || 0.0;
    const stateParts = [scalar.toFixed(4), String(hullIndex), curArch];
    for (const item of this.activePartObjects) {
      const p = item.group.position;
      stateParts.push(`${p.x.toFixed(4)},${p.y.toFixed(4)},${p.z.toFixed(4)}`);
    }
    const key = stateParts.join('|');
    if (key === this._lastConduitKey && this.activeConduits.length > 0) {
      return;
    }
    this._lastConduitKey = key;

    const pal = this.getPaletteColors(this.currentPalette);
    const unitCouplingRad = Math.max(0.014, Math.min(0.042, this.assemblyRadius * 0.020));
    const tubeRadius = Math.max(0.0014, Math.min(0.0042, this.assemblyRadius * this.config.conduitRadiusRatio));

    const expectedKeys = new Set(connectionPairs.map((cp) => `${cp.pIdx}->${cp.cIdx}`));
    const activeKeys = new Set(this.activeConduits.map((c) => `${c.parentIndex}->${c.childIndex}`));
    if (expectedKeys.size !== activeKeys.size || [...expectedKeys].some((k) => !activeKeys.has(k))) {
      this._clearConduits();
    }

    for (const pair of connectionPairs) {
      const pIdx = pair.pIdx;
      const cIdx = pair.cIdx;
      const parentObj = this.activePartObjects[pIdx];
      const childObj = this.activePartObjects[cIdx];
      if (!parentObj || !childObj) continue;

      const pRad = this._getPartRadius(parentObj);
      const cRad = this._getPartRadius(childObj);

      let conduit = this.activeConduits.find((c) => c.parentIndex === pIdx && c.childIndex === cIdx);
      if (!conduit) {
        const parentCoupling = this.createCouplingNode(parentObj.data.color, unitCouplingRad);
        const childCoupling = this.createCouplingNode(childObj.data.color, unitCouplingRad);
        parentObj.group.add(parentCoupling.nodeGroup);
        childObj.group.add(childCoupling.nodeGroup);

        const tubeMat = new THREE.MeshBasicMaterial({
          color: childObj.data.color ? new THREE.Color(childObj.data.color) : pal.rim.clone(),
          wireframe: false,
          transparent: true,
          opacity: isLayeredNetwork ? 0.22 : 0.14,
          blending: THREE.AdditiveBlending,
          depthWrite: false
        });
        const tubeMesh = new THREE.Mesh(new THREE.BufferGeometry(), tubeMat);
        this.conduitsGroup.add(tubeMesh);

        const pulseCount = this.config.pulseCount;
        const pulsePositions = new Float32Array(pulseCount * 3);
        const pulseGeo = new THREE.BufferGeometry();
        pulseGeo.setAttribute('position', new THREE.BufferAttribute(pulsePositions, 3));
        const pulseMat = new THREE.PointsMaterial({
          size: Math.max(0.018, unitCouplingRad * 0.90),
          map: GLOW_SPRITE,
          color: pal.laserSelected.clone(),
          transparent: true,
          opacity: 0.85,
          blending: THREE.AdditiveBlending,
          depthWrite: false
        });
        const pulsePoints = new THREE.Points(pulseGeo, pulseMat);
        this.conduitsGroup.add(pulsePoints);

        conduit = {
          parentIndex: pIdx,
          childIndex: cIdx,
          parentObj,
          childObj,
          parentCoupling,
          childCoupling,
          tubeMesh,
          tubeMat,
          pulsePoints,
          pulseGeo,
          pulseMat,
          pulseCount,
          curve: null
        };
        this.activeConduits.push(conduit);
      }

      const pLocal = parentObj.group.position;
      const cLocal = childObj.group.position;
      this._tmpDir.subVectors(cLocal, pLocal);
      const dist = this._tmpDir.length();

      if (dist < 0.025) {
        conduit.tubeMesh.visible = false;
        conduit.pulsePoints.visible = false;
        conduit.parentCoupling.nodeGroup.visible = false;
        conduit.childCoupling.nodeGroup.visible = false;
        conduit.curve = null;
        continue;
      }

      this._tmpDir.multiplyScalar(1.0 / dist);
      conduit.tubeMesh.visible = true;
      conduit.parentCoupling.nodeGroup.visible = true;
      conduit.childCoupling.nodeGroup.visible = true;

      const offsetA = Math.min(dist * 0.35, Math.max(0.02, pRad * 0.55));
      const offsetB = Math.min(dist * 0.35, Math.max(0.02, cRad * 0.55));

      this._tmpAnchorA.copy(pLocal).addScaledVector(this._tmpDir, offsetA);
      this._tmpAnchorB.copy(cLocal).addScaledVector(this._tmpDir, -offsetB);

      conduit.parentCoupling.nodeGroup.position.copy(this._tmpDir).multiplyScalar(offsetA);
      conduit.childCoupling.nodeGroup.position.copy(this._tmpDir).multiplyScalar(-offsetB);

      const c1 = this._tmpAnchorA.clone().addScaledVector(this._tmpDir, dist * 0.22);
      const c2 = this._tmpAnchorB.clone().addScaledVector(this._tmpDir, -dist * 0.22);
      this._tmpMid.addVectors(this._tmpAnchorA, this._tmpAnchorB).multiplyScalar(0.5);
      this._tmpMid.y -= Math.max(0.03, dist * 0.06) * Math.max(0.15, 1.0 - scalar * 0.55);

      const curve = new THREE.CatmullRomCurve3(
        [this._tmpAnchorA.clone(), c1, this._tmpMid.clone(), c2, this._tmpAnchorB.clone()],
        false,
        'centripetal',
        0.5
      );
      conduit.curve = curve;

      if (conduit.tubeMesh.geometry) {
        conduit.tubeMesh.geometry.dispose();
      }
      conduit.tubeMesh.geometry = new THREE.TubeGeometry(
        curve,
        this.config.conduitSegments,
        tubeRadius,
        this.config.conduitRadialSegments,
        false
      );
      conduit.tubeMat.opacity = isLayeredNetwork ? 0.22 : Math.min(0.18, 0.04 + scalar * 0.14);
      conduit.pulsePoints.visible = isLayeredNetwork || (scalar > 0.02);
    }
    if (this._lastSelectedIndex !== undefined) {
      this.updateVisualStates(this._lastSelectedIndex, this._lastHoveredIndex ?? -1, Boolean(this._lastIsIsolated));
    }
  }

  animateConduitPulses(timeSec) {
    this.activeConduits.forEach((conduit) => {
      if (!conduit.curve || !conduit.pulsePoints.visible) return;

      const posAttr = conduit.pulseGeo.attributes.position;
      const arr = posAttr.array;
      for (let k = 0; k < conduit.pulseCount; k++) {
        const u = ((timeSec * 0.55 + k / conduit.pulseCount) % 1.0);
        const pt = conduit.curve.getPointAt(u);
        arr[k * 3] = pt.x;
        arr[k * 3 + 1] = pt.y;
        arr[k * 3 + 2] = pt.z;
      }
      posAttr.needsUpdate = true;
    });
  }

  update(now, delta) {
    const timeSec = now * 0.001;
    const dt = Math.min(0.1, Math.max(0.001, delta || 0.016));
    const dtFrameScale = dt * 60.0;

    if (this.laserEnabled) {
      const scaledSpeed = Math.max(0.6, this.sweepSpeed * Math.max(0.6, this.assemblyRadius));
      this.sweepY += this.sweepDir * scaledSpeed * dt;
      const minLim = Math.min(-0.85, this.assemblyMinY - 0.18);
      const maxLim = Math.max(0.85, this.assemblyMaxY + 0.18);
      if (this.sweepY > maxLim) {
        this.sweepY = maxLim;
        this.sweepDir = -1.0;
      } else if (this.sweepY < minLim) {
        this.sweepY = minLim;
        this.sweepDir = 1.0;
      }
    }

    const activeSliceCoord = this.sliceCutawayEnabled ? this.sliceY : this.sweepY;
    if (typeof this.masterGroup.getWorldPosition === 'function') {
      this.masterGroup.getWorldPosition(this._sliceWorldOrigin);
    }
    if (this._sliceWorldQuat && typeof this.masterGroup.getWorldQuaternion === 'function') {
      this.masterGroup.getWorldQuaternion(this._sliceWorldQuat);
      this._sliceWorldAxis.set(0, 1, 0).applyQuaternion(this._sliceWorldQuat).normalize();
    } else {
      this._sliceWorldAxis.set(0, 1, 0);
    }

    if (this.sliceRingGroup) {
      this.sliceRingGroup.visible = this.laserEnabled || this.sliceCutawayEnabled;
      this.sliceRingGroup.position.y = activeSliceCoord;
      if (this.sliceTicksMesh) {
        this.sliceTicksMesh.rotation.y += dt * (this.sliceCutawayEnabled ? 0.45 : 0.25);
      }
    }

    this.activePartObjects.forEach((item) => {
      let bondP = 1.0;
      let formP = 1.0;
      let isForming = false;

      let gentleScale = 1.0;
      if (item.spawnProgress !== undefined && item.spawnProgress < 1.0) {
        item.spawnProgress = Math.min(1.0, item.spawnProgress + dt / this.config.spawnDurationSec);
        const p = item.spawnProgress;
        isForming = p < 0.999;

        // Zero-G magnetic settling scale (anchored in space, no crude ballooning)
        gentleScale = 0.94 + 0.06 * (1.0 - Math.pow(1.0 - p, 2));
        item.group.scale.setScalar(gentleScale);

        // Vector bonds weave first, then holographic skin crystallizes behind the laser front
        bondP = Math.min(1.0, p * 1.35);
        formP = Math.max(0.0, (p - 0.22) / 0.78);
      }

      // Live Kinematic Rotational Motion
      const spd = item.data?.rotSpeed;
      let rotX = 0, rotY = 0, rotZ = 0;
      const curArch = this.activeBlueprint?.architecture || '';
      const isNonRotary = ['graph_layered_network', 'organic_brain', 'organic_heart', 'organic_anatomical'].includes(curArch) ||
        (this.activeBlueprint?.title || '').toLowerCase().includes('brain') ||
        (this.activeBlueprint?.title || '').toLowerCase().includes('heart') ||
        (this.activeBlueprint?.title || '').toLowerCase().includes('neural');
      const isRotaryMachine = (curArch === 'coaxial_rotary') || (!curArch && item.data?.topology === 'rotary_disc');

      if (!isNonRotary && Array.isArray(spd) && spd.some((v) => Math.abs(v) > 1e-4)) {
        rotX = (spd[0] || 0) * dtFrameScale;
        rotY = (spd[1] || 0) * dtFrameScale;
        rotZ = (spd[2] || 0) * dtFrameScale;
      } else if (!isNonRotary && isRotaryMachine && item.data?.topology === 'rotary_disc') {
        // Rotary discs in rotary machines rotate around their mechanical axis
        const dir = (item.index % 2 === 0) ? 1.0 : -1.0;
        rotZ = dir * 0.028 * dtFrameScale;
      }
      if (rotX || rotY || rotZ) {
        item.group.rotation.x += rotX;
        item.group.rotation.y += rotY;
        item.group.rotation.z += rotZ;
      }

      // Living biological pulsing for branched conduits, cardiac chambers, and brain nuclei
      const isBio = ['organic_brain', 'organic_heart', 'organic_anatomical'].includes(curArch) ||
        item.data?.topology === 'branched_conduit' ||
        (this.activeBlueprint?.title || '').toLowerCase().includes('brain') ||
        (this.activeBlueprint?.title || '').toLowerCase().includes('heart');
      if (isBio) {
        const beatCycle = Math.sin(timeSec * 4.8 + item.index * 0.75);
        const pulse = 1.0 + (beatCycle > 0 ? 0.025 * Math.pow(beatCycle, 2.0) : -0.012 * Math.abs(beatCycle));
        const finalS = gentleScale * pulse;
        item.group.scale.set(finalS, finalS, finalS);
      }

      item.subMeshes.forEach((sub) => {
        if (sub.shaderMat && sub.shaderMat.uniforms) {
          sub.shaderMat.uniforms.uTime.value = timeSec;
          sub.shaderMat.uniforms.uSweepY.value = this.sweepY;
          sub.shaderMat.uniforms.uEnableLaser.value = this.laserEnabled ? 1.0 : 0.0;
          sub.shaderMat.uniforms.uSliceOrigin.value.copy(this._sliceWorldOrigin);
          sub.shaderMat.uniforms.uSliceAxis.value.copy(this._sliceWorldAxis);
          sub.shaderMat.uniforms.uDepthSliceY.value = activeSliceCoord;
          sub.shaderMat.uniforms.uSliceCutaway.value = this.sliceCutawayEnabled ? 1.0 : 0.0;
          if (sub.shaderMat.uniforms.uFormProgress) {
            sub.shaderMat.uniforms.uFormProgress.value = formP;
          }
        }

        if (sub.bondMat && sub.bondMat.uniforms) {
          sub.bondMat.uniforms.uTime.value = timeSec;
          if (sub.bondMat.uniforms.uBondProgress) {
            sub.bondMat.uniforms.uBondProgress.value = bondP;
          }
        }

        if (sub.pointsObj) {
          sub.pointsObj.rotation.y = Math.sin(timeSec * 0.4 + item.index) * 0.02;
        }
        if (sub.pointsMat && sub.basePointOpacity) {
          const ionize = isForming ? 1.45 : 1.0;
          sub.pointsMat.opacity = sub.basePointOpacity * ionize;
        }
      });
    });

    this.animateConduitPulses(timeSec);
  }

  updateExplosion(scalar) {
    this.currentExplosionScalar = scalar;
    this.activePartObjects.forEach((item) => {
      item.group.position.copy(item.assembledVec).addScaledVector(item.explodedOffset, scalar);

      if (item.trajMat) {
        item.trajMat.opacity = Math.min(0.24, scalar * 0.30);
      }
    });

    this.fitSliceRingToAssembly();
    this.updateConduits();
  }

  updateVisualStates(selectedIndex, hoveredIndex, isIsolated) {
    this._lastSelectedIndex = selectedIndex;
    this._lastHoveredIndex = hoveredIndex;
    this._lastIsIsolated = isIsolated;
    const pal = this.getPaletteColors(this.currentPalette);

    this.activePartObjects.forEach((item, idx) => {
      const isSelected = idx === selectedIndex;
      const isHovered = idx === hoveredIndex;
      const dimmedByIsolate = isIsolated && selectedIndex !== -1 && !isSelected;

      item.subMeshes.forEach((sub) => {
        const baseWire = sub.baseWireOpacity !== undefined ? sub.baseWireOpacity : 0.085;
        const basePt = sub.basePointSize !== undefined ? sub.basePointSize : 0.015;
        const basePtOp = sub.basePointOpacity !== undefined ? sub.basePointOpacity : 0.34;

        if (sub.shaderMat && sub.shaderMat.uniforms) {
          sub.shaderMat.uniforms.uSelected.value = isSelected ? 1.0 : 0.0;
          sub.shaderMat.uniforms.uHovered.value = isHovered ? 1.0 : 0.0;
          sub.shaderMat.uniforms.uIsolateDim.value = dimmedByIsolate ? 1.0 : 0.0;
        }

        if (sub.bondMat && sub.bondMat.uniforms) {
          sub.bondMat.uniforms.uSelected.value = isSelected ? 1.0 : 0.0;
          sub.bondMat.uniforms.uHovered.value = isHovered ? 1.0 : 0.0;
        }

        if (sub.wireMat) {
          if (isSelected) {
            sub.wireMat.color.copy(sub.selectedRimColor || pal.selectedRim);
            sub.wireMat.opacity = Math.min(0.28, Math.max(0.06, baseWire * 1.8));
          } else if (isHovered) {
            sub.wireMat.color.copy(sub.hoverRimColor || pal.hoverRim);
            sub.wireMat.opacity = Math.min(0.22, Math.max(0.05, baseWire * 1.45));
          } else if (dimmedByIsolate) {
            sub.wireMat.color.copy(sub.rimColor);
            sub.wireMat.opacity = 0.025;
          } else {
            sub.wireMat.color.copy(sub.rimColor);
            sub.wireMat.opacity = baseWire;
          }
        }

        if (sub.pointsMat) {
          if (isSelected) {
            sub.pointsMat.color.copy(sub.selectedColor || pal.selected);
            sub.pointsMat.opacity = Math.min(0.48, basePtOp + 0.12);
            sub.pointsMat.size = basePt * 1.08;
          } else if (isHovered) {
            sub.pointsMat.color.copy(sub.hoverColor || pal.hover);
            sub.pointsMat.opacity = Math.min(0.44, basePtOp + 0.08);
            sub.pointsMat.size = basePt * 1.04;
          } else if (dimmedByIsolate) {
            sub.pointsMat.opacity = 0.025;
            sub.pointsMat.size = basePt * 0.65;
          } else {
            sub.pointsMat.color.copy(sub.rimColor);
            sub.pointsMat.opacity = basePtOp;
            sub.pointsMat.size = basePt;
          }
        }
      });
    });

    const activeHubIdx = this.masterHullIndex >= 0 ? this.masterHullIndex : 0;
    const expScalar = this.currentExplosionScalar || 0;
    this.activeConduits.forEach((conduit) => {
      const isSelected = selectedIndex >= 0 && (selectedIndex === activeHubIdx || conduit.childIndex === selectedIndex);
      const isHovered = hoveredIndex >= 0 && conduit.childIndex === hoveredIndex;

      if (isSelected) {
        conduit.tubeMat.color.copy(pal.selected);
        conduit.tubeMat.opacity = Math.min(0.24, 0.06 + expScalar * 0.18);
        if (conduit.pulseMat) conduit.pulseMat.opacity = 0.82;
        conduit.parentCoupling.ringMat.color.copy(pal.selected);
        conduit.childCoupling.ringMat.color.copy(pal.selected);
      } else if (isHovered) {
        conduit.tubeMat.color.copy(pal.hover);
        conduit.tubeMat.opacity = Math.min(0.22, 0.06 + expScalar * 0.16);
        if (conduit.pulseMat) conduit.pulseMat.opacity = 0.75;
        conduit.parentCoupling.ringMat.color.copy(pal.hover);
        conduit.childCoupling.ringMat.color.copy(pal.hover);
      } else if (isIsolated && selectedIndex !== -1) {
        conduit.tubeMat.opacity = 0.02;
        if (conduit.pulseMat) conduit.pulseMat.opacity = 0.04;
      } else {
        const baseCol = conduit.childObj.data.color ? new THREE.Color(conduit.childObj.data.color) : pal.rim.clone();
        conduit.tubeMat.color.copy(baseCol);
        conduit.tubeMat.opacity = Math.min(0.18, 0.04 + expScalar * 0.14);
        if (conduit.pulseMat) conduit.pulseMat.opacity = 0.68;
        conduit.parentCoupling.ringMat.color.copy(baseCol);
        conduit.childCoupling.ringMat.color.copy(baseCol);
      }
    });
  }
}