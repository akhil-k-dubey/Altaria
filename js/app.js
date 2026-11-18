/**
 * App - Main Coordinator for M.A.R.K. Holo-CAD
 * Purely modular, zero hardcoded 3D geometry.
 */

import { StarkSoundFX } from './audio.js';
import { SceneManager } from './scene.js';
import { HologramBuilder } from './builder.js';
import { DiagnosticPlatesManager } from './diagnostic_plates.js';
import { GestureTracker } from './gestures.js';
import { UIManager } from './ui.js';
import { VLMClient } from './api.js';

class HoloCADApp {
  constructor() {
    this.container = document.getElementById('canvas-container');
    this.sound = new StarkSoundFX();
    this.scene = new SceneManager(this.container);
    this.builder = new HologramBuilder(this.scene.masterGroup, { sound: this.sound });
    this.diagnosticPlates = new DiagnosticPlatesManager(this.scene.scene, this.scene.masterGroup);
    this.ui = new UIManager();
    this.vlm = new VLMClient();

    this.currentBlueprint = null;
    this.selectedPartIndex = 0;
    this.hoveredPartIndex = -1;
    this.isolateMode = false;
    this.autoRotate = true;
    this.isSwitchingCandidate = false;
    this.activeWebcamFallbackInfo = null;
    this.activeStreamQuery = '';
    this.activePreGenGate = null;
    this._gateLastDwellInside = false;
    this._explosionDirty = false;
    this._userAdjustedZoom = false;

    this.builder.onMeshAttached = () => {
      this._explosionDirty = true;
      this.scene.fitToAssembly(this.scene.masterGroup, { updateZoom: !this._userAdjustedZoom });
    };

    this.currentExplosion = 0.0;
    this.targetExplosion = 0.0;
    this.targetRotationY = 0.45;
    this.targetRotationX = 0.18;
    this.isDragging = false;
    this.prevPointer = { x: 0, y: 0 };
    this.pointerDownPos = { x: 0, y: 0 };

    this.raycaster = new THREE.Raycaster();
    this.pointerNDC = new THREE.Vector2();

    this.lastFrameTime = performance.now();
    this.lastAnimTime = performance.now();
    this.frameCount = 0;

    this.initGestures();
    this.setupEventListeners();
  }

  initGestures() {
    this.gestures = new GestureTracker({
      onExplosionChange: (val) => this.setExplosionScalar(val, false),
      onParallax: (px, py) => {
        this.scene.headParallaxOffset.x = px;
        this.scene.headParallaxOffset.y = py;
      },
      onOrbitDelta: (dx, dy) => {
        this.targetRotationY += dx;
        this.targetRotationX = Math.max(-1.25, Math.min(1.25, this.targetRotationX + dy));
      },
      onAssembleSnap: () => {
        if (this.targetExplosion > 0.05) {
          this.setExplosionScalar(0, true);
          this.ui.showStatusToast('FIST GESTURE // CORE ASSEMBLED (0%)');
        }
      },
      onRaycast: (sx, sy) => this.getRaycastPartIndex(sx, sy),
      onSelectPart: (idx) => this.selectPart(idx, true),
      onSound: (freq, dur) => this.sound.playChime(freq, dur)
    });
  }

  getRaycastPartIndex(clientX, clientY) {
    this.pointerNDC.x = (clientX / window.innerWidth) * 2 - 1;
    this.pointerNDC.y = -(clientY / window.innerHeight) * 2 + 1;
    this.raycaster.setFromCamera(this.pointerNDC, this.scene.camera);

    const meshes = [];
    this.builder.activePartObjects.forEach((item) => {
      item.subMeshes.forEach((s) => {
        if (s.mesh) meshes.push(s.mesh);
      });
    });

    const intersects = this.raycaster.intersectObjects(meshes, false);
    if (intersects.length > 0) {
      const hitMesh = intersects[0].object;
      if (typeof hitMesh.userData.partIndex === 'number') {
        return hitMesh.userData.partIndex;
      }
    }
    return -1;
  }

  selectPart(index, playAudio = true) {
    this.selectedPartIndex = index;
    if (playAudio) this.sound.playSelect();

    if (index === -1 || !this.builder.activePartObjects[index]) {
      this.ui.showFloatingCallout(null, -1);
      this.builder.updateVisualStates(-1, this.hoveredPartIndex, this.isolateMode);
      this.diagnosticPlates.updateVisualStates(-1, this.hoveredPartIndex);
      return;
    }

    const part = this.builder.activePartObjects[index].data;
    this.ui.showFloatingCallout(part, index, () => this.toggleVolumetricRelief());
    this.builder.updateVisualStates(index, this.hoveredPartIndex, this.isolateMode);
    this.diagnosticPlates.updateVisualStates(index, this.hoveredPartIndex);

    this.builder.activePartObjects.forEach((_, i) => {
      const card = document.getElementById(`part-card-${i}`);
      if (card) {
        card.className = i === index
          ? 'w-full text-left p-2 rounded border border-amber-400 bg-amber-500/20 shadow-[0_0_12px_rgba(255,170,0,0.2)] transition flex items-center justify-between gap-2.5'
          : 'w-full text-left p-2 rounded border border-cyan-500/25 bg-cyan-950/35 hover:bg-cyan-900/45 transition flex items-center justify-between gap-2.5';
      }
    });

    this.syncEscapeHatchForPart(index);
  }

  syncEscapeHatchForPart(partIndex = this.selectedPartIndex, switchingIndex = -1) {
    if (this.activePreGenGate && switchingIndex < 0) {
      const gatePart = Number.isFinite(Number(this.activePreGenGate.partIndex))
        ? Number(this.activePreGenGate.partIndex)
        : 0;
      if (partIndex === gatePart || partIndex < 0) {
        return;
      }
    }

    const targetIdx =
      partIndex >= 0 && this.builder.activePartObjects[partIndex]
        ? partIndex
        : (this.builder.activePartObjects.length > 0 ? 0 : -1);

    const partObj = targetIdx >= 0 ? this.builder.activePartObjects[targetIdx] : null;
    const partData = partObj?.data || null;
    const bp = this.currentBlueprint || {};

    let candidates = [];
    if (partData && Array.isArray(partData.candidates) && partData.candidates.length > 0) {
      candidates = partData.candidates;
    } else if (Array.isArray(bp.candidates) && bp.candidates.length > 0) {
      candidates = bp.candidates;
    } else if (Array.isArray(bp.references) && bp.references.length > 0) {
      candidates = bp.references.slice(0, this.ui.config.maxCandidateThumbnails);
    }

    if (candidates.length === 0) {
      this.ui.hideCandidateEscapeHatch();
      return;
    }

    const activeUrl =
      partData?.referenceImage?.thumbUrl ||
      partData?.referenceImage?.originalUrl ||
      candidates[0]?.thumbUrl ||
      '';

    let verifyBadge = 'FUNNEL S4';
    let verifyTone = 'cyan';
    if (switchingIndex >= 0) {
      verifyBadge = 'RECONSTRUCTING...';
      verifyTone = 'amber';
    } else if (partData?.referenceImage?.userPreferred) {
      verifyBadge = '★ USER LOCKED';
      verifyTone = 'amber';
    } else if (partData?.verified3D) {
      const conf = Number(partData?.verification?.confidence);
      verifyBadge = Number.isFinite(conf) ? `S5 ✓ ${Math.round(conf)}/10` : '3D VERIFIED';
      verifyTone = 'emerald';
    } else if (partData?.usedHarmonicFallback) {
      verifyBadge = 'HARMONIC FALLBACK';
      verifyTone = 'amber';
    }

    const opMode = bp?.operatingMode?.mode || '';
    const modeBadge = targetIdx <= 0 && opMode
      ? this.ui.formatOperatingModeLabel(opMode)
      : '';

    const partNum = String(targetIdx + 1).padStart(2, '0');
    const targetLabel = partData
      ? `${partNum} · ${partData.name || 'SUB-SYSTEM'}`
      : (this.activeStreamQuery || bp.title || 'ASSEMBLY');

    const shownCount = Math.min(candidates.length, this.ui.config.maxCandidateThumbnails);
    this.ui.showCandidateEscapeHatch(
      candidates,
      activeUrl,
      {
        label: `FUNNEL TOP-${shownCount} CANDIDATES`,
        targetLabel,
        modeBadge,
        verifyBadge,
        verifyTone,
        switchingIndex,
        allowWebcamRescue: false
      },
      (cand, candIdx) => this.switchCandidate(targetIdx, cand, candIdx)
    );
  }

  async switchCandidate(partIndex, candidate, candidateIndex = 0) {
    if (this.isSwitchingCandidate || !candidate) return;
    const targetIdx = partIndex >= 0 ? partIndex : 0;
    const bp = this.currentBlueprint || {};
    const partObj = this.builder.activePartObjects[targetIdx];
    const partData = partObj?.data || bp.parts?.[targetIdx];
    if (!partData) return;

    this.isSwitchingCandidate = true;
    this.syncEscapeHatchForPart(targetIdx, candidateIndex);
    this.ui.showStatusToast(`CANDIDATE #${candidateIndex + 1} // RECONSTRUCTING "${(partData.name || 'PART').toUpperCase()}"...`);
    this.sound.playChime(820, 0.06);

    try {
      const query = bp.queryPrompt || this.activeStreamQuery || bp.title || '';
      const res = await this.vlm.selectCandidate({
        query,
        blueprintTitle: bp.title || '',
        partIndex: targetIdx,
        partId: partData.id || '',
        partName: partData.name || '',
        candidate
      });

      partData.meshUrl = res.meshUrl;
      partData.referenceImage = res.referenceImage || candidate;
      partData.verification = res.verification || null;
      partData.verified3D = Boolean(res.verified3D);
      partData.usedHarmonicFallback = Boolean(res.usedHarmonicFallback);

      if (Array.isArray(res.parts) && res.parts[targetIdx]) {
        const updatedPart = res.parts[targetIdx];
        partData.assembledPos = updatedPart.assembledPos || partData.assembledPos;
        partData.explodedVec = updatedPart.explodedVec || partData.explodedVec;
        partData.scale = updatedPart.scale || partData.scale;
        bp.parts = res.parts.map((rp, idx) => ({ ...(bp.parts?.[idx] || {}), ...rp }));
        if (bp.parts[targetIdx]) {
          bp.parts[targetIdx] = partData;
        }
      }
      if (res.geometryValidation) {
        bp.geometryValidation = res.geometryValidation;
      }

      const selUrl = partData.referenceImage?.thumbUrl || partData.referenceImage?.originalUrl;
      if (Array.isArray(partData.candidates)) {
        partData.candidates = partData.candidates.map((c) => {
          const match = (c.thumbUrl && c.thumbUrl === selUrl) || (c.originalUrl && c.originalUrl === selUrl);
          return { ...c, userPreferred: match };
        });
      }
      if (Array.isArray(bp.candidates)) {
        bp.candidates = bp.candidates.map((c) => {
          const match = (c.thumbUrl && c.thumbUrl === selUrl) || (c.originalUrl && c.originalUrl === selUrl);
          return match ? { ...c, userPreferred: true } : c;
        });
      }

      this.builder.attachPartMesh(targetIdx, res.meshUrl, null, null);
      if (Array.isArray(res.parts)) {
        this.builder.updateLayoutPositions(res.parts);
      }
      this._explosionDirty = true;
      this.scene.fitToAssembly(this.scene.masterGroup, { updateZoom: false });
      this.ui.appendPartCard(partData, targetIdx, targetIdx === this.selectedPartIndex, (idx) => this.selectPart(idx, true));
      this.diagnosticPlates.buildPlates(bp, this.builder.activePartObjects);

      if (this.selectedPartIndex === targetIdx) {
        this.ui.showFloatingCallout(partData, targetIdx, () => this.toggleVolumetricRelief());
      }
      if (!res.usedHarmonicFallback) {
        this.activeWebcamFallbackInfo = null;
        this.ui.hideWebcamFallbackBanner();
      }

      this.sound.playSelect();
      this.ui.showStatusToast(`LOCKED CANDIDATE #${candidateIndex + 1} FOR ${(partData.name || 'PART').toUpperCase()} (VOTE SAVED)`);
    } catch (err) {
      console.error('[!] Candidate switch failed:', err);
      this.ui.showStatusToast(`CANDIDATE SWITCH ERROR: ${err.message}`);
    } finally {
      this.isSwitchingCandidate = false;
      this.syncEscapeHatchForPart(targetIdx, -1);
    }
  }

  async handleWebcamFallbackCapture() {
    if (!this.gestures.active) {
      this.ui.showStatusToast('ACTIVATING CAMERA // AIM AT OBJECT & CLICK "SNAP OBJECT NOW"');
      const active = await this.gestures.toggle();
      if (active && this.activeWebcamFallbackInfo) {
        this.activeWebcamFallbackInfo.webcamActive = true;
        this.ui.showWebcamFallbackBanner(
          {
            ...this.activeWebcamFallbackInfo,
            message: 'Camera active — hold object in frame and click "SNAP OBJECT NOW" to reverse-engineer in 3D.'
          },
          () => this.handleWebcamFallbackCapture(),
          () => { this.activeWebcamFallbackInfo = null; }
        );
      } else if (!active) {
        this.ui.showStatusToast('CAMERA UNAVAILABLE // TRY SWITCHING A CANDIDATE THUMBNAIL OR QUERY');
      }
      return;
    }

    this.activeWebcamFallbackInfo = null;
    this.ui.hideWebcamFallbackBanner();
    this.captureWebcamAndSynthesize();
  }

  toggleDiagnosticTable() {
    const nextVis = !this.diagnosticPlates.visible;
    this.diagnosticPlates.setVisible(nextVis);
    const dot = document.getElementById('diagnostic-dot');
    if (dot) {
      dot.className = nextVis
        ? 'w-2 h-2 rounded-full bg-cyan-300 shadow-[0_0_6px_#00f3ff]'
        : 'w-2 h-2 rounded-full bg-slate-500';
    }
    this.ui.showStatusToast(nextVis ? 'DIAGNOSTIC TABLE SCANS ACTIVE' : 'DIAGNOSTIC SCANS HIDDEN');
    this.sound.playChime(640, 0.08);
  }

  toggleVolumetricRelief() {
    const nextVol = !this.diagnosticPlates.volumetricMode;
    this.diagnosticPlates.setVolumetricMode(nextVol);
    const dot = document.getElementById('volumetric-dot');
    if (dot) {
      dot.className = nextVol
        ? 'w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_6px_#ffaa00]'
        : 'w-2 h-2 rounded-full bg-slate-500';
    }
    const label = document.getElementById('volumetric-btn-label');
    if (label) {
      label.textContent = nextVol ? 'SHOW 2D FLAT SCAN' : 'EXTRUDE 3D SCAN';
    }
    this.ui.showStatusToast(nextVol ? '3D VOLUMETRIC EXTENSION ACTIVE' : 'FLAT PHOTOGRAPHIC SCAN ACTIVE');
    this.sound.playChime(780, 0.08);
  }

  setExplosionScalar(val, playSweep = false) {
    const clamped = Math.max(0, Math.min(1, Number(val) || 0));
    if (playSweep && Math.abs(clamped - this.targetExplosion) > 0.15) {
      this.sound.playExplodeSweep(clamped > this.targetExplosion);
    }
    this.targetExplosion = clamped;
    this.ui.updateExplosionUI(clamped);
  }

  loadBlueprint(blueprint) {
    if (!blueprint || !Array.isArray(blueprint.parts)) return;
    this.activePreGenGate = null;
    this._gateLastDwellInside = false;
    this.ui.resolvePreGenCandidateGate({});
    this.currentBlueprint = blueprint;
    this.activeWebcamFallbackInfo = null;
    this._userAdjustedZoom = false;
    this.ui.hideWebcamFallbackBanner();
    this.builder.build(blueprint);
    this._explosionDirty = true;
    this.scene.fitToAssembly(this.scene.masterGroup, { updateZoom: true });
    this.diagnosticPlates.buildPlates(blueprint, this.builder.activePartObjects);
    this.ui.updateAssemblyHeader(blueprint);
    this.ui.populatePartsList(blueprint.parts, 0, (idx) => this.selectPart(idx, true));
    this.setExplosionScalar(0, false);
    this.selectPart(0, false);
  }

  async synthesize(conceptPrompt, imageBase64 = null) {
    const text = (conceptPrompt || '').trim();
    if (!text && !imageBase64) return;

    this.activeStreamQuery = text;
    this.activeWebcamFallbackInfo = null;
    this.activePreGenGate = null;
    this._gateLastDwellInside = false;
    this._userAdjustedZoom = false;
    this.ui.resolvePreGenCandidateGate({});
    this.ui.hideWebcamFallbackBanner();
    this.ui.hideCandidateEscapeHatch();

    const btnText = document.getElementById('vlm-btn-text');
    const spinner = document.getElementById('vlm-spinner');
    if (btnText) btnText.textContent = 'QWEN VLM STREAMING...';
    if (spinner) spinner.classList.remove('hidden');

    this.ui.showStatusToast(`QWEN VLM // DECONSTRUCTING "${text ? text.toUpperCase() : 'CAMERA CAPTURE'}"...`);

    let accumulatedTokens = '';
    const streamingBlueprint = {
      queryPrompt: text,
      title: text ? text.toUpperCase() : 'VISUAL CAPTURE',
      subtitle: 'MATERIALIZING SUB-ASSEMBLIES...',
      overview: 'Synthesizing 3D engineering system with Qwen VLM...',
      operatingMode: null,
      parts: []
    };
    this.currentBlueprint = streamingBlueprint;

    try {
      await this.vlm.synthesizeConceptStream(text, imageBase64, (eventType, data) => {
        if (eventType === 'token') {
          accumulatedTokens += data.token || '';
          this.ui.setReasoningStream(data.token, accumulatedTokens);
        } else if (eventType === 'status') {
          this.ui.showStatusToast(data.message || 'Processing CAD sub-systems...');
        } else if (eventType === 'header') {
          streamingBlueprint.title = data.title || streamingBlueprint.title;
          streamingBlueprint.subtitle = data.subtitle || streamingBlueprint.subtitle;
          streamingBlueprint.overview = data.overview || streamingBlueprint.overview;
          this.ui.updateAssemblyHeader(streamingBlueprint);
          this.builder.startProgressiveBuild(streamingBlueprint);
          this._explosionDirty = true;
          const partsListEl = document.getElementById('parts-list');
          if (partsListEl) partsListEl.innerHTML = '';
        } else if (eventType === 'part_discovered') {
          const part = data.part;
          const idx = data.partIndex;
          streamingBlueprint.parts[idx] = part;

          this.builder.addPartProgressive(part, idx);
          this._explosionDirty = true;
          this.scene.fitToAssembly(this.scene.masterGroup, { updateZoom: false });
          this.ui.appendPartCard(part, idx, idx === 0, (selectedIdx) => this.selectPart(selectedIdx, true));
          this.sound.playChime(640 + idx * 75, 0.05);
          this.ui.showStatusToast(`MATERIALIZED // ${(part?.name || 'SUB-ASSEMBLY').toUpperCase()}`);

          if (idx === 0) {
            this.selectPart(0, false);
          }
        } else if (eventType === 'funnel_stage') {
          const stageTag = String(data.stage || 'funnel').toUpperCase();
          this.ui.setEscapeHatchStatus(`FUNNEL ${stageTag}`, 'cyan');
        } else if (eventType === 'candidates_ready') {
          streamingBlueprint.candidates = data.candidates || [];
          streamingBlueprint.funnelStats = data.stats || {};
          streamingBlueprint.operatingMode = data.operatingMode || null;
          if (!this.activePreGenGate) {
            this.syncEscapeHatchForPart(this.selectedPartIndex);
          }
        } else if (eventType === 'candidate_gate') {
          this.activePreGenGate = {
            gateId: data.gateId,
            mode: data.mode || 'human_gate_spotlight',
            partIndex: Number.isFinite(Number(data.partIndex)) ? Number(data.partIndex) : 0,
            submitting: false
          };
          this._gateLastDwellInside = false;
          this.sound.playChime(980, 0.06);

          const modeLabel = this.ui.formatOperatingModeLabel(data.mode) || '2s SPOTLIGHT';
          this.ui.showStatusToast(`${modeLabel} // CONFIRM CANDIDATE OR AUTO-LOCK`);

          this.ui.showPreGenCandidateGate(data, {
            onSelect: async (cand, idx) => {
              if (!this.activePreGenGate || this.activePreGenGate.gateId !== data.gateId) return;
              this.activePreGenGate.submitting = true;
              try {
                await this.vlm.confirmCandidate({
                  gateId: data.gateId,
                  action: 'select',
                  candidateIndex: idx,
                  candidate: cand
                });
                this.sound.playSelect();
                this.ui.showStatusToast(`LOCKED #${idx + 1} // BUILDING 3D...`);
              } catch (err) {
                console.warn('[!] Candidate gate select warning:', err);
                this.ui.showStatusToast(`GATE SELECT ERROR: ${err.message}`);
              }
            },
            onPauseChange: (paused) => {
              if (
                !this.activePreGenGate ||
                this.activePreGenGate.gateId !== data.gateId ||
                this.activePreGenGate.submitting
              ) {
                return;
              }
              this.vlm.confirmCandidate({
                gateId: data.gateId,
                action: paused ? 'pause' : 'resume'
              }).catch((err) => {
                console.warn('[!] Candidate gate pause/resume warning:', err);
              });
            },
            onReject: async () => {
              if (!this.activePreGenGate || this.activePreGenGate.gateId !== data.gateId) return;
              this.activePreGenGate.submitting = true;
              try {
                await this.vlm.confirmCandidate({
                  gateId: data.gateId,
                  action: 'reject',
                  reason: 'User rejected web candidates at HUD gate'
                });
                this.sound.playChime(520, 0.08);
                this.ui.showStatusToast('WEB CANDIDATES REJECTED // OPENING CAMERA RESCUE');
              } catch (err) {
                console.warn('[!] Candidate gate reject warning:', err);
                this.ui.showStatusToast(`GATE REJECT ERROR: ${err.message}`);
              }
            }
          });
        } else if (eventType === 'candidate_gate_resolved') {
          this.activePreGenGate = null;
          this._gateLastDwellInside = false;
          this.ui.resolvePreGenCandidateGate(data);

          if (data.outcome === 'select' && data.selectedCandidate) {
            const selUrl = data.selectedCandidate.thumbUrl || data.selectedCandidate.originalUrl || '';
            const markPref = (list) => (
              Array.isArray(list)
                ? list.map((c) => {
                    const match = (c.thumbUrl && c.thumbUrl === selUrl) || (c.originalUrl && c.originalUrl === selUrl);
                    if (match) return { ...c, userPreferred: true };
                    const copy = { ...c };
                    delete copy.userPreferred;
                    return copy;
                  })
                : list
            );
            streamingBlueprint.candidates = markPref(streamingBlueprint.candidates);
            if (streamingBlueprint.parts?.[0]) {
              streamingBlueprint.parts[0].referenceImage = data.selectedCandidate;
              streamingBlueprint.parts[0].candidates = markPref(streamingBlueprint.parts[0].candidates);
            }
            const p0Obj = this.builder.activePartObjects[0]?.data;
            if (p0Obj) {
              p0Obj.referenceImage = data.selectedCandidate;
              p0Obj.candidates = markPref(p0Obj.candidates);
            }
            this.syncEscapeHatchForPart(this.selectedPartIndex);
          } else if (data.outcome === 'rescue_webcam' || data.outcome === 'reject') {
            this.ui.showStatusToast(data.reason || 'SHOW OBJECT TO CAMERA OR UPLOAD PHOTO');
          }
        } else if (eventType === 'verification_update') {
          const maxAtt = data.maxAttempts || 1;
          if (data.phase === 'attempt') {
            this.ui.setEscapeHatchStatus(`S5 #${data.attempt}/${maxAtt} BUILD...`, 'amber');
          } else if (data.phase === 'verdict' && data.verdict) {
            if (data.verdict.plausible) {
              this.ui.setEscapeHatchStatus(`S5 ✓ ${Math.round(data.verdict.confidence || 0)}/10`, 'emerald');
            } else {
              this.ui.setEscapeHatchStatus(`S5 ✗ #${data.attempt}/${maxAtt}`, 'amber');
            }
          }
        } else if (eventType === 'mesh_progress') {
          this.ui.updatePartMeshStatus(data.partIndex, data.status, data.progress);
        } else if (eventType === 'mesh_ready') {
          const idx = data.partIndex;
          const partObj = this.builder.activePartObjects[idx];
          const partRef = partObj?.data || streamingBlueprint.parts[idx];
          if (partRef) {
            partRef.meshUrl = data.meshUrl;
            partRef.referenceImage = data.referenceImage || partRef.referenceImage;
            partRef.candidates = data.candidates || partRef.candidates;
            partRef.verification = data.verification || null;
            partRef.verified3D = Boolean(data.verified3D);
            partRef.usedHarmonicFallback = Boolean(data.usedHarmonicFallback);
            this.ui.appendPartCard(partRef, idx, idx === this.selectedPartIndex, (selectedIdx) => this.selectPart(selectedIdx, true));
          }

          this.builder.attachPartMesh(idx, data.meshUrl, data.textureUrl, data.normalMapUrl);
          this._explosionDirty = true;
          this.scene.fitToAssembly(this.scene.masterGroup, { updateZoom: false });
          const badgeState = data.usedHarmonicFallback ? 'harmonic' : (data.verified3D ? 'verified' : 'ready');
          this.ui.updatePartMeshStatus(idx, badgeState, 100);
          this.sound.playChime(900 + idx * 50, 0.04);

          if (this.selectedPartIndex === idx && partRef) {
            this.ui.showFloatingCallout(partRef, idx, () => this.toggleVolumetricRelief());
            this.syncEscapeHatchForPart(idx);
          } else if (idx === 0) {
            this.syncEscapeHatchForPart(this.selectedPartIndex);
          }
        } else if (eventType === 'references_ready') {
          streamingBlueprint.wikiSummary = data.wikiSummary;
          streamingBlueprint.references = data.references;
          if (Array.isArray(data.candidates) && data.candidates.length > 0) {
            streamingBlueprint.candidates = data.candidates;
          }
          if (data.operatingMode) {
            streamingBlueprint.operatingMode = data.operatingMode;
          }
          this.ui.updateAssemblyHeader(streamingBlueprint);
          this.diagnosticPlates.buildPlates(streamingBlueprint, this.builder.activePartObjects);
          if (!this.activePreGenGate) {
            this.syncEscapeHatchForPart(this.selectedPartIndex);
          }
        } else if (eventType === 'webcam_fallback_needed') {
          this.activeWebcamFallbackInfo = { ...data, webcamActive: this.gestures.active };
          this.ui.showWebcamFallbackBanner(
            this.activeWebcamFallbackInfo,
            () => this.handleWebcamFallbackCapture(),
            () => { this.activeWebcamFallbackInfo = null; }
          );
          this.ui.showStatusToast(data.message || 'SHOW OBJECT TO WEBCAM OR PICK A CANDIDATE');
        } else if (eventType === 'geometry_validated') {
          this.builder.updateLayoutPositions(data.parts);
          this._explosionDirty = true;
          this.scene.fitToAssembly(this.scene.masterGroup, { updateZoom: false });
          this.ui.showStatusToast('GEOMETRY ENGINE // 3D NON-PENETRATION & RELAXATION VERIFIED');
        } else if (eventType === 'complete') {
          this.currentBlueprint = data.blueprint;
          this._explosionDirty = true;
          this.scene.fitToAssembly(this.scene.masterGroup, { updateZoom: true });
          this.diagnosticPlates.buildPlates(data.blueprint, this.builder.activePartObjects);
          this.ui.updateAssemblyHeader(data.blueprint);
          const safeIdx = this.selectedPartIndex >= 0 && this.selectedPartIndex < (data.blueprint.parts || []).length ? this.selectedPartIndex : 0;
          this.selectPart(safeIdx, false);
          this.setExplosionScalar(0.28, true);
          this.sound.playSelect();
          this.ui.showStatusToast(`SYNTHESIZED 3D CAD // ${(data.blueprint?.title || 'ASSEMBLY').toUpperCase()}`);
        } else if (eventType === 'error') {
          console.error('[!] Streaming error event:', data);
          this.ui.showStatusToast(`SYNTHESIS ERROR: ${data.error}`);
        }
      });
    } catch (err) {
      console.warn('[!] Streaming failed, falling back to REST:', err);
      try {
        const bp = await this.vlm.synthesizeConcept(text, imageBase64);
        this.loadBlueprint(bp);
        this.setExplosionScalar(0.35, true);
        this.sound.playSelect();
        this.ui.showStatusToast(`SYNTHESIZED 3D CAD // ${(bp?.title || 'ASSEMBLY').toUpperCase()}`);
      } catch (fallbackErr) {
        console.error('[!] Fallback synthesis also failed:', fallbackErr);
        this.ui.showStatusToast(`SYNTHESIS FAILED: ${fallbackErr.message}`);
      }
    } finally {
      this.activePreGenGate = null;
      this._gateLastDwellInside = false;
      this.ui.resolvePreGenCandidateGate({});
      if (btnText) btnText.textContent = 'SYNTHESIZE 3D';
      if (spinner) spinner.classList.add('hidden');
      this.ui.hideReasoningStream();
    }
  }

  setupEventListeners() {
    this.container.addEventListener('pointerdown', (e) => {
      this.isDragging = true;
      this.prevPointer = { x: e.clientX, y: e.clientY };
      this.pointerDownPos = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('pointermove', (e) => {
      if (this.isDragging) {
        const dx = e.clientX - this.prevPointer.x;
        const dy = e.clientY - this.prevPointer.y;
        this.targetRotationY += dx * 0.0075;
        this.targetRotationX = Math.max(-1.25, Math.min(1.25, this.targetRotationX + dy * 0.0075));
        this.prevPointer = { x: e.clientX, y: e.clientY };
        return;
      }

      if (e.target === this.scene.renderer.domElement) {
        const hitIdx = this.getRaycastPartIndex(e.clientX, e.clientY);
        if (hitIdx !== this.hoveredPartIndex) {
          this.hoveredPartIndex = hitIdx;
          this.container.style.cursor = hitIdx !== -1 ? 'pointer' : 'grab';
          this.builder.updateVisualStates(this.selectedPartIndex, this.hoveredPartIndex, this.isolateMode);
        }
      }
    });

    window.addEventListener('pointerup', (e) => {
      if (!this.isDragging) return;
      this.isDragging = false;
      const moveDist = Math.hypot(e.clientX - this.pointerDownPos.x, e.clientY - this.pointerDownPos.y);
      if (moveDist < 6 && e.target === this.scene.renderer.domElement) {
        const hitIdx = this.getRaycastPartIndex(e.clientX, e.clientY);
        if (hitIdx !== -1) {
          this.selectPart(hitIdx, true);
        }
      }
    });

    this.container.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (e.shiftKey) {
        const pct = this.builder.adjustDepthSliceOffset(-Math.sign(e.deltaY) * 0.045);
        this.syncDepthSliceHUD();
        this.ui.showStatusToast(`DEPTH SLICE PLANE // ${pct}% ELEVATION`);
        return;
      }
      this._userAdjustedZoom = true;
      const minDist = this.scene.minCameraDistance || 2.0;
      const maxDist = this.scene.maxCameraDistance || 22.0;
      const step = e.deltaY * 0.0006 * Math.max(2.0, this.scene.cameraDistance);
      this.scene.cameraDistance = Math.max(minDist, Math.min(maxDist, this.scene.cameraDistance + step));
      this.scene.targetCameraDistance = this.scene.cameraDistance;
    }, { passive: false });

    const slider = document.getElementById('explosion-slider');
    if (slider) {
      slider.addEventListener('input', (e) => {
        this.setExplosionScalar(Number(e.target.value) / 100, false);
      });
    }

    document.getElementById('btn-assemble')?.addEventListener('click', () => {
      this.setExplosionScalar(0, true);
    });

    document.getElementById('btn-explode')?.addEventListener('click', () => {
      this.setExplosionScalar(1, true);
    });

    document.getElementById('btn-laser-sweep')?.addEventListener('click', () => {
      this.builder.laserEnabled = !this.builder.laserEnabled;
      this.builder.updateSliceVisualStyle();
      const dot = document.getElementById('laser-dot');
      if (dot) {
        dot.className = this.builder.laserEnabled
          ? 'w-2 h-2 rounded-full bg-cyan-300 shadow-[0_0_6px_#00f3ff]'
          : 'w-2 h-2 rounded-full bg-slate-500';
      }
      this.sound.playChime(this.builder.laserEnabled ? 920 : 540, 0.06);
      this.ui.showStatusToast(this.builder.laserEnabled ? 'TOMOGRAPHIC LASER SWEEP ACTIVE' : 'LASER SWEEP SUSPENDED');
    });

    document.getElementById('btn-depth-slice')?.addEventListener('click', () => {
      const enabled = this.builder.setDepthSliceCutaway(!this.builder.sliceCutawayEnabled);
      this.syncDepthSliceHUD();
      this.sound.playChime(enabled ? 1040 : 620, 0.07);
      this.ui.showStatusToast(
        enabled
          ? 'VOLUMETRIC DEPTH SLICE CUTAWAY ACTIVE // SHIFT+SCROLL OR [ ] TO SCRUB'
          : 'CONTINUOUS VOLUMETRIC STRATA MODE RESTORED'
      );
    });

    document.querySelectorAll('.palette-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const pal = e.currentTarget.getAttribute('data-palette') || 'cyan';
        this.builder.setPalette(pal);
        this.diagnosticPlates.setPalette(this.builder.getPaletteColors(pal));
        document.querySelectorAll('.palette-btn').forEach((b) => {
          if (b === e.currentTarget) {
            b.classList.add('border-2', 'border-white', 'scale-110');
            b.classList.remove('border', 'border-white/20', 'opacity-70');
          } else {
            b.classList.remove('border-2', 'border-white', 'scale-110');
            b.classList.add('border', 'border-white/20', 'opacity-70');
          }
        });
        this.sound.playChime(760, 0.06);
        this.ui.showStatusToast(`HOLOGRAM PALETTE // ${pal.toUpperCase().replace('_', ' ')}`);
      });
    });

    document.getElementById('btn-volumetric-relief')?.addEventListener('click', () => {
      this.toggleVolumetricRelief();
    });

    document.getElementById('btn-isolate')?.addEventListener('click', () => {
      this.isolateMode = !this.isolateMode;
      const dot = document.getElementById('isolate-dot');
      if (dot) {
        dot.className = this.isolateMode
          ? 'w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_8px_#ffaa00]'
          : 'w-2 h-2 rounded-full bg-slate-500';
      }
      this.sound.playChime(this.isolateMode ? 1120 : 680, 0.07);
      this.builder.updateVisualStates(this.selectedPartIndex, this.hoveredPartIndex, this.isolateMode);
      this.ui.showStatusToast(this.isolateMode ? 'SUB-PART ISOLATION ACTIVE' : 'FULL ASSEMBLY RESTORED');
    });

    document.getElementById('btn-autorotate')?.addEventListener('click', () => {
      this.autoRotate = !this.autoRotate;
      const dot = document.getElementById('autorotate-dot');
      if (dot) {
        dot.className = this.autoRotate
          ? 'w-2 h-2 rounded-full bg-cyan-300 shadow-[0_0_6px_#00f3ff]'
          : 'w-2 h-2 rounded-full bg-slate-500';
      }
      this.sound.playChime(880, 0.06);
    });

    document.getElementById('clear-selection-btn')?.addEventListener('click', () => {
      this.isolateMode = false;
      const dot = document.getElementById('isolate-dot');
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-slate-500';
      this.selectPart(-1, true);
    });

    document.getElementById('toggle-audio-btn')?.addEventListener('click', () => {
      this.sound.enabled = !this.sound.enabled;
      document.getElementById('toggle-audio-btn')?.classList.toggle('opacity-40', !this.sound.enabled);
      this.ui.showStatusToast(this.sound.enabled ? 'ACOUSTIC TELEMETRY ENABLED' : 'ACOUSTIC TELEMETRY MUTED');
    });

    document.getElementById('toggle-tracking-btn')?.addEventListener('click', async () => {
      const active = await this.gestures.toggle();
      this.ui.showStatusToast(active ? '21-PT SKELETAL HAND TRACKER ACTIVE' : 'GESTURE TRACKING OFFLINE');
    });

    document.getElementById('btn-pyramid')?.addEventListener('click', () => {
      this.scene.pyramidMode = true;
      this.builder.setRenderBudget(true);
      document.getElementById('main-hud')?.classList.add('hidden');
      document.getElementById('floating-callout')?.classList.add('hidden');
      document.getElementById('leader-line')?.classList.add('hidden');
      document.getElementById('leader-reticle')?.classList.add('hidden');
      document.getElementById('pyramid-hud')?.classList.remove('hidden');
      this.scene.floorGroup.visible = false;
      this.ui.showStatusToast('PEPPER’S GHOST 4-WAY OPTICAL PYRAMID ACTIVE (DRAW-CALL BUDGET OPTIMIZED)');
    });

    const togglePyramidParallax = () => {
      const isSymmetric = this.scene.pyramidParallaxMode === 'symmetric';
      this.scene.pyramidParallaxMode = isSymmetric ? 'disabled' : 'symmetric';
      const label = document.getElementById('pyramid-parallax-label');
      if (label) {
        label.textContent = isSymmetric ? 'PARALLAX: LOCKED' : 'PARALLAX: SYMMETRIC';
      }
      this.ui.showStatusToast(
        isSymmetric ? 'PYRAMID PARALLAX LOCKED (0° DISPLACEMENT)' : 'PYRAMID PARALLAX DYNAMIC (SO(2) SYMMETRIC)'
      );
    };

    document.getElementById('toggle-pyramid-parallax-btn')?.addEventListener('click', togglePyramidParallax);

    document.getElementById('exit-pyramid-btn')?.addEventListener('click', () => {
      this.scene.pyramidMode = false;
      this.builder.setRenderBudget(false);
      document.getElementById('main-hud')?.classList.remove('hidden');
      document.getElementById('pyramid-hud')?.classList.add('hidden');
      this.scene.floorGroup.visible = true;
      if (this.selectedPartIndex !== -1) this.selectPart(this.selectedPartIndex, false);
    });

    document.getElementById('vlm-form')?.addEventListener('submit', (e) => {
      e.preventDefault();
      const input = document.getElementById('vlm-input');
      if (input) this.synthesize(input.value);
    });

    document.querySelectorAll('.quick-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        const prompt = chip.getAttribute('data-prompt');
        const input = document.getElementById('vlm-input');
        if (input) input.value = prompt;
        this.synthesize(prompt);
      });
    });

    document.getElementById('btn-webcam-snapshot')?.addEventListener('click', () => {
      this.captureWebcamAndSynthesize();
    });

    window.addEventListener('keydown', (e) => {
      if (document.activeElement === document.getElementById('vlm-input')) return;
      if (e.code === 'Space') {
        e.preventDefault();
        this.setExplosionScalar(this.targetExplosion > 0.5 ? 0 : 1, true);
      } else if (e.key === 's' || e.key === 'S') {
        document.getElementById('btn-depth-slice')?.click();
      } else if (e.key === '[' || e.key === ']') {
        const pct = this.builder.adjustDepthSliceOffset(e.key === ']' ? 0.05 : -0.05);
        this.syncDepthSliceHUD();
        this.ui.showStatusToast(`DEPTH SLICE PLANE // ${pct}% ELEVATION`);
      } else if (e.key === 'i' || e.key === 'I') {
        document.getElementById('btn-isolate')?.click();
      } else if ((e.key === 'p' || e.key === 'P') && this.scene.pyramidMode) {
        togglePyramidParallax();
      } else if (e.code === 'ArrowDown') {
        e.preventDefault();
        const len = this.builder.activePartObjects.length;
        if (len > 0) this.selectPart((this.selectedPartIndex + 1) % len, true);
      } else if (e.code === 'ArrowUp') {
        e.preventDefault();
        const len = this.builder.activePartObjects.length;
        if (len > 0) this.selectPart((this.selectedPartIndex - 1 + len) % len, true);
      }
    });
  }

  syncDepthSliceHUD() {
    const dot = document.getElementById('slice-dot');
    const btn = document.getElementById('btn-depth-slice');
    const active = Boolean(this.builder.sliceCutawayEnabled);
    if (dot) {
      dot.className = active
        ? 'w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_8px_#ffaa00]'
        : 'w-2 h-2 rounded-full bg-slate-500';
    }
    if (btn) {
      btn.className = active
        ? 'px-3 py-1.5 rounded bg-amber-500/20 hover:bg-amber-500/35 border border-amber-400 text-amber-200 font-mono-tech text-xs transition flex items-center gap-1.5 shadow-[0_0_12px_rgba(255,170,0,0.2)]'
        : 'px-3 py-1.5 rounded bg-cyan-950/80 hover:bg-cyan-800/60 border border-cyan-400/40 text-cyan-200 font-mono-tech text-xs transition flex items-center gap-1.5';
    }
  }

  captureWebcamAndSynthesize() {
    const video = document.getElementById('webcam-video');
    if (!video || !this.gestures.active || video.readyState < 2) {
      this.ui.showStatusToast('ENABLE WEBCAM TRACKING FIRST TO SNAPSHOT');
      return;
    }

    const vw = video.videoWidth || video.clientWidth || 640;
    const vh = video.videoHeight || video.clientHeight || 360;
    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = vw;
    snapCanvas.height = vh;
    const sctx = snapCanvas.getContext('2d');
    sctx.drawImage(video, 0, 0, vw, vh);
    const b64 = snapCanvas.toDataURL('image/jpeg', 0.88);

    const input = document.getElementById('vlm-input');
    const promptText = input && input.value ? input.value.trim() : '';
    this.synthesize(promptText, b64);
  }

  animate(now = performance.now()) {
    requestAnimationFrame((t) => this.animate(t));

    this.frameCount++;
    if (now - this.lastFrameTime >= 1000) {
      const fpsEl = document.getElementById('fps-counter');
      if (fpsEl) fpsEl.textContent = `${this.frameCount} FPS // WEBGL2`;
      this.frameCount = 0;
      this.lastFrameTime = now;
    }

    const delta = Math.min(0.1, Math.max(0.001, (now - (this.lastAnimTime || now)) * 0.001));
    this.lastAnimTime = now;

    const expLerp = 1.0 - Math.exp(-6.5 * delta);
    const rotLerp = 1.0 - Math.exp(-7.8 * delta);

    const prevExplosion = this.currentExplosion;
    this.currentExplosion += (this.targetExplosion - this.currentExplosion) * expLerp;
    if (Math.abs(this.currentExplosion - this.targetExplosion) < 1e-4) {
      this.currentExplosion = this.targetExplosion;
    }

    if (Math.abs(this.currentExplosion - prevExplosion) > 1e-5 || this._explosionDirty) {
      this._explosionDirty = false;
      this.builder.updateExplosion(this.currentExplosion);
      if (!this._userAdjustedZoom) {
        this.scene.fitToAssembly(this.scene.masterGroup, { updateZoom: true, immediate: false });
      }
    }

    if (!this._userAdjustedZoom && Number.isFinite(this.scene.targetCameraDistance)) {
      this.scene.cameraDistance += (this.scene.targetCameraDistance - this.scene.cameraDistance) * expLerp;
    }

    this.builder.update(now, delta);

    if (this.autoRotate && !this.isDragging) {
      this.targetRotationY += 0.228 * delta;
    }

    this.scene.masterGroup.rotation.y += (this.targetRotationY - this.scene.masterGroup.rotation.y) * rotLerp;
    this.scene.masterGroup.rotation.x += (this.targetRotationX - this.scene.masterGroup.rotation.x) * rotLerp;

    this.scene.render(delta);
    this.diagnosticPlates.update(now, this.builder.activePartObjects);

    if (this.activePreGenGate && !this.activePreGenGate.submitting) {
      const gateEl = typeof document !== 'undefined' ? document.getElementById('candidate-escape-hatch') : null;
      const dwellEl = this.gestures?.dwellDomEl || null;
      const isDwellInGate = Boolean(
        this.gestures?.active &&
        gateEl &&
        dwellEl &&
        typeof gateEl.contains === 'function' &&
        gateEl.contains(dwellEl)
      );
      if (isDwellInGate !== this._gateLastDwellInside) {
        this._gateLastDwellInside = isDwellInGate;
        this.ui.setGatePaused(isDwellInGate, 'dwell');
      }
    } else if (this._gateLastDwellInside) {
      this._gateLastDwellInside = false;
      this.ui.setGatePaused(false, 'dwell');
    }

    if (this.selectedPartIndex !== -1 && this.builder.activePartObjects[this.selectedPartIndex]) {
      const grp = this.builder.activePartObjects[this.selectedPartIndex].group;
      this.ui.updateLeaderLine(grp, this.scene.camera, this.scene.pyramidMode);
    } else {
      this.ui.updateLeaderLine(null, this.scene.camera, this.scene.pyramidMode);
    }
  }

  async start() {
    this.animate();
    this.ui.showStatusToast('HOLO-CAD ONLINE // CONNECTING TO LOCAL QWEN VLM...');

    const health = await this.vlm.checkHealth();
    const vlmStatusBadge = document.getElementById('vlm-status-badge');
    if (vlmStatusBadge) {
      if (health.status === 'ready' || health.status === 'healthy') {
        vlmStatusBadge.textContent = 'QWEN-VL ONLINE';
        vlmStatusBadge.className = 'font-mono-tech text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-400/30';
      } else {
        vlmStatusBadge.textContent = 'QWEN-VL LOADING';
        vlmStatusBadge.className = 'font-mono-tech text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-400/30';
      }
    }

    try {
      const blueprints = await this.vlm.fetchBlueprints();
      if (Array.isArray(blueprints) && blueprints.length > 0) {
        this.loadBlueprint(blueprints[0]);
        this.setExplosionScalar(0.20, false);
      }
    } catch (e) {
      console.warn('[!] Auto-load blueprint warning:', e);
    }

    this.ui.showStatusToast('DIAGNOSTIC TABLE READY // ENTER ANY SYSTEM TO DECONSTRUCT');
  }
}

function bootApp() {
  const app = new HoloCADApp();
  window.holoCadApp = app;
  app.start();
}

if (document.readyState === 'loading') {
  window.addEventListener('DOMContentLoaded', bootApp);
} else {
  bootApp();
}