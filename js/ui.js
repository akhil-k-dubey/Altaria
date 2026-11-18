/**
 * UIManager - Controls Stark HUD Panels, Floating Callout Cards, SVG Leader Lines & Telemetry
 * Purely robust DOM management with zero regex usage.
 */

export const DEFAULT_UI_CONFIG = Object.freeze({
  toastTimeoutMs: 3800,
  maxSummaryLines: 2,
  maxCandidateThumbnails: 6,
  leaderReticleHiddenClass: 'hidden',
  calloutHiddenClass: 'hidden'
});

export const OPERATING_MODE_LABELS = Object.freeze({
  direct_capture: 'CAMERA CAPTURE',
  learned_auto: 'AUTO LOCKED',
  soft_veto_overlap: 'SOFT VETO',
  human_gate_spotlight: '2s SPOTLIGHT',
  rescue_webcam_prompt: 'CAMERA RESCUE',
  autonomous_subparts: 'AUTONOMOUS'
});

export class UIManager {
  constructor(config = {}) {
    this.config = { ...DEFAULT_UI_CONFIG, ...config };
    this.toastTimeout = null;
    this.tempWorldVec = new THREE.Vector3();
    this.escapeHatchCollapsed = false;
    this.preGenGate = null;
    this._gateHoverPaused = false;
    this._gateDwellPaused = false;

    const collapseBtn = document.getElementById('btn-toggle-escape-hatch');
    if (collapseBtn) {
      collapseBtn.addEventListener('click', () => {
        this.escapeHatchCollapsed = !this.escapeHatchCollapsed;
        const thumbs = document.getElementById('escape-hatch-thumbs');
        if (thumbs) thumbs.classList.toggle('hidden', this.escapeHatchCollapsed);
        collapseBtn.textContent = this.escapeHatchCollapsed ? 'SHOW' : 'HIDE';
      });
    }

    const panelEl = document.getElementById('candidate-escape-hatch');
    if (panelEl && typeof panelEl.addEventListener === 'function') {
      panelEl.addEventListener('mouseenter', () => {
        if (this.preGenGate && !this.preGenGate.submitting) {
          this.setGatePaused(true, 'hover');
        }
      });
      panelEl.addEventListener('mouseleave', () => {
        if (this.preGenGate) {
          this.setGatePaused(false, 'hover');
        }
      });
    }

    const rejectBtn = document.getElementById('btn-gate-reject-webcam');
    if (rejectBtn && typeof rejectBtn.addEventListener === 'function') {
      rejectBtn.addEventListener('click', (e) => {
        if (e && typeof e.stopPropagation === 'function') e.stopPropagation();
        if (this.preGenGate && !this.preGenGate.submitting) {
          this.preGenGate.submitting = true;
          this._clearGateTicker();
          this.setEscapeHatchStatus('CAMERA RESCUE...', 'amber');
          const gateBar = document.getElementById('escape-hatch-gate-bar');
          if (gateBar) gateBar.classList.add('hidden');
          if (typeof this.preGenGate.onReject === 'function') {
            this.preGenGate.onReject();
          }
        }
      });
    }
  }

  formatOperatingModeLabel(mode) {
    const key = String(mode || '').trim();
    if (!key) return '';
    return OPERATING_MODE_LABELS[key] || key.split('_').join(' ').toUpperCase();
  }

  _clearGateTicker() {
    if (this.preGenGate && this.preGenGate.tickerId) {
      clearInterval(this.preGenGate.tickerId);
      this.preGenGate.tickerId = null;
    }
  }

  escapeHtml(str) {
    return String(str ?? '')
      .split('&').join('&amp;')
      .split('<').join('&lt;')
      .split('>').join('&gt;')
      .split('"').join('&quot;')
      .split("'").join('&#39;');
  }

  proxyImageUrl(url) {
    const u = String(url || '').trim();
    if (!u) return '';
    if (u.startsWith('http://') || u.startsWith('https://')) {
      return `/api/proxy-image?url=${encodeURIComponent(u)}`;
    }
    return u;
  }

  _bindImageFallback(imgEl, fallbackRawUrl) {
    if (!imgEl) return;
    imgEl.dataset.fallbackTried = '';
    const rawFallback = String(fallbackRawUrl || '').trim();
    if (!rawFallback) {
      imgEl.onerror = null;
      return;
    }
    imgEl.onerror = () => {
      if (imgEl.dataset.fallbackTried === '1') return;
      imgEl.dataset.fallbackTried = '1';
      const nextSrc = this.proxyImageUrl(rawFallback);
      if (nextSrc && imgEl.getAttribute('src') !== nextSrc) {
        imgEl.src = nextSrc;
      }
    };
  }

  showStatusToast(message) {
    const toast = document.getElementById('status-toast');
    if (!toast) return;
    toast.textContent = `J.A.R.V.I.S. // ${message}`;
    toast.classList.remove('opacity-0');
    toast.classList.add('opacity-100');
    if (this.toastTimeout) clearTimeout(this.toastTimeout);
    this.toastTimeout = setTimeout(() => {
      toast.classList.remove('opacity-100');
      toast.classList.add('opacity-0');
    }, this.config.toastTimeoutMs);
  }

  updateAssemblyHeader(blueprint) {
    const titleEl = document.getElementById('assembly-title');
    const subtitleEl = document.getElementById('assembly-subtitle');
    const overviewEl = document.getElementById('assembly-overview');
    const vectorsEl = document.getElementById('telemetry-vertices');

    if (titleEl) titleEl.textContent = blueprint.title || 'UNNAMED ASSEMBLY';
    const numParts = Array.isArray(blueprint.parts) ? blueprint.parts.length : 0;
    if (subtitleEl) subtitleEl.textContent = blueprint.subtitle || `${numParts} SUB-ASSEMBLIES LOADED`;

    if (overviewEl) {
      if (blueprint.wikiSummary && blueprint.wikiSummary.extract) {
        overviewEl.innerHTML = `
          <div class="flex items-center justify-between mb-1">
            <span class="font-mono-tech text-[9px] text-cyan-400 font-bold uppercase tracking-wider">WIKIPEDIA ARCHIVE</span>
            <span class="font-mono-tech text-[8px] text-cyan-300/70">VERIFIED REFERENCE</span>
          </div>
          <p class="text-[10.5px] leading-relaxed text-cyan-100/90">${this.escapeHtml(blueprint.wikiSummary.extract)}</p>
        `;
      } else {
        overviewEl.textContent = blueprint.overview || 'Deconstructed 3D engineering model.';
      }
    }
    if (vectorsEl) vectorsEl.textContent = `${numParts} ASSEMBLIES`;
  }

  setReasoningStream(token, fullText) {
    const ticker = document.getElementById('vlm-stream-ticker');
    const textEl = document.getElementById('vlm-stream-text');
    if (ticker) ticker.classList.remove('hidden');
    if (textEl) {
      const cleanText = fullText ? fullText.split('\n').join(' ') : (token || '');
      textEl.textContent = cleanText.slice(-60);
    }
  }

  hideReasoningStream() {
    const ticker = document.getElementById('vlm-stream-ticker');
    if (ticker) ticker.classList.add('hidden');
  }

  updateStreamingPartsCount(count) {
    const countEl = document.getElementById('vlm-stream-parts-count');
    if (countEl) countEl.textContent = `${count} PART${count === 1 ? '' : 'S'}`;
    const vectorsEl = document.getElementById('telemetry-vertices');
    if (vectorsEl) vectorsEl.textContent = `${count} ASSEMBLIES`;
  }

  appendPartCard(part, idx, isSelected, onSelect) {
    const partsListEl = document.getElementById('parts-list');
    if (!partsListEl) return;

    let card = document.getElementById(`part-card-${idx}`);
    const isNew = !card;
    if (isNew) {
      card = document.createElement('button');
      card.type = 'button';
      card.id = `part-card-${idx}`;
      partsListEl.appendChild(card);
    }

    card.className = isSelected
      ? 'w-full text-left p-2 rounded border border-amber-400 bg-amber-500/20 shadow-[0_0_12px_rgba(255,170,0,0.2)] transition flex items-center justify-between gap-2.5'
      : 'w-full text-left p-2 rounded border border-cyan-500/25 bg-cyan-950/35 hover:bg-cyan-900/45 transition flex items-center justify-between gap-2.5';

    let thumbHtml = '';
    if (part.referenceImage && (part.referenceImage.thumbUrl || part.referenceImage.url || part.referenceImage.originalUrl)) {
      const u = part.referenceImage.thumbUrl || part.referenceImage.url || part.referenceImage.originalUrl;
      const proxied = this.proxyImageUrl(u);
      thumbHtml = `<img src="${this.escapeHtml(proxied)}" class="w-8 h-8 rounded object-cover border border-cyan-400/50 shadow-[0_0_6px_rgba(0,243,255,0.2)] shrink-0" />`;
    }

    let badgeStatus = part.meshUrl ? 'SOLID MESH' : 'LATTICE';
    let badgeStyle = part.meshUrl
      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-400/40'
      : 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30';

    if (part.usedHarmonicFallback) {
      badgeStatus = 'HARMONIC';
      badgeStyle = 'bg-amber-500/20 text-amber-300 border-amber-400/40';
    } else if (part.verified3D) {
      badgeStatus = '3D VERIFIED';
      badgeStyle = 'bg-emerald-500/20 text-emerald-300 border-emerald-400/50 shadow-[0_0_8px_rgba(52,211,153,0.28)]';
    }

    const partNum = String(idx + 1).padStart(2, '0');
    const thumbFallback = part.referenceImage?.originalUrl || part.referenceImage?.url || '';

    card.innerHTML = `
      ${thumbHtml}
      <div class="min-w-0 flex-1">
        <div class="flex items-center gap-1.5">
          <span class="font-mono-tech text-[10px] text-amber-400 font-bold">${partNum}</span>
          <span class="font-display text-xs font-semibold text-cyan-100 truncate">${this.escapeHtml(part.name || 'SUB-SYSTEM')}</span>
        </div>
        <div class="font-mono-tech text-[9.5px] text-cyan-400/75 truncate mt-0.5">${this.escapeHtml(part.role || 'SUB-SYSTEM')}</div>
      </div>
      <div class="flex flex-col items-end gap-1 shrink-0">
        <span id="part-mesh-badge-${idx}" class="font-mono-tech text-[8.5px] px-1.5 py-0.5 rounded border ${badgeStyle}">
          ${badgeStatus}
        </span>
      </div>
    `;

    const cardImg = card.querySelector('img');
    if (cardImg && thumbFallback) {
      this._bindImageFallback(cardImg, thumbFallback);
    }

    if (isNew) {
      card.addEventListener('click', () => {
        if (typeof onSelect === 'function') onSelect(idx);
      });
    }

    this.updateStreamingPartsCount(partsListEl.children.length);
  }

  updatePartMeshStatus(idx, status, progress = 100) {
    const badge = document.getElementById(`part-mesh-badge-${idx}`);
    if (!badge) return;

    if (status === 'generating') {
      badge.textContent = `MESH ${progress}%`;
      badge.className = 'font-mono-tech text-[8.5px] px-1.5 py-0.5 rounded border bg-amber-500/20 text-amber-300 border-amber-500/40 animate-pulse';
    } else if (status === 'verified') {
      badge.textContent = '3D VERIFIED';
      badge.className = 'font-mono-tech text-[8.5px] px-1.5 py-0.5 rounded border bg-emerald-500/20 text-emerald-300 border-emerald-400/50 shadow-[0_0_8px_rgba(52,211,153,0.3)]';
    } else if (status === 'harmonic') {
      badge.textContent = 'HARMONIC';
      badge.className = 'font-mono-tech text-[8.5px] px-1.5 py-0.5 rounded border bg-amber-500/20 text-amber-300 border-amber-400/40';
    } else if (status === 'ready' || status === 'solid') {
      badge.textContent = 'SOLID MESH';
      badge.className = 'font-mono-tech text-[8.5px] px-1.5 py-0.5 rounded border bg-emerald-500/20 text-emerald-300 border-emerald-400/40 shadow-[0_0_8px_rgba(52,211,153,0.3)]';
    }
  }

  populatePartsList(parts, selectedIdx, onSelect) {
    const partsListEl = document.getElementById('parts-list');
    if (!partsListEl) return;
    partsListEl.innerHTML = '';
    parts.forEach((part, idx) => {
      this.appendPartCard(part, idx, idx === selectedIdx, onSelect);
    });
  }

  showFloatingCallout(part, index, onToggleVolumetric) {
    const callout = document.getElementById('floating-callout');
    const leaderLine = document.getElementById('leader-line');
    const leaderReticle = document.getElementById('leader-reticle');

    if (!part || index === -1) {
      if (callout) callout.classList.add('hidden', 'opacity-0');
      if (leaderLine) leaderLine.classList.add('hidden');
      if (leaderReticle) leaderReticle.classList.add('hidden');
      return;
    }

    const indexEl = document.getElementById('callout-index');
    const roleEl = document.getElementById('callout-role');
    const titleEl = document.getElementById('callout-title');
    const partNum = String(index + 1).padStart(2, '0');
    if (indexEl) indexEl.textContent = `PART // ${partNum}`;
    if (roleEl) roleEl.textContent = part.role || 'SUB-SYSTEM';
    if (titleEl) titleEl.textContent = part.name || 'Sub-Assembly';

    const refContainer = document.getElementById('callout-ref-container');
    const refImg = document.getElementById('callout-ref-img');
    const refCaption = document.getElementById('callout-ref-caption');
    const sourceLink = document.getElementById('callout-source-link');
    const verifyBadge = document.getElementById('callout-verify-badge');

    if (part.referenceImage && (part.referenceImage.thumbUrl || part.referenceImage.url || part.referenceImage.originalUrl)) {
      const u = part.referenceImage.thumbUrl || part.referenceImage.url || part.referenceImage.originalUrl;
      const fallbackRaw = part.referenceImage.originalUrl || part.referenceImage.url || '';
      const proxied = this.proxyImageUrl(u);
      if (refImg) {
        this._bindImageFallback(refImg, fallbackRaw !== u ? fallbackRaw : '');
        refImg.src = proxied;
      }
      if (refCaption) refCaption.textContent = part.referenceImage.caption || part.referenceImage.title || 'Vetted photographic scan';
      if (sourceLink) sourceLink.href = part.referenceImage.originalUrl || u;
      if (verifyBadge) {
        if (part.referenceImage.userPreferred) {
          verifyBadge.textContent = 'USER VOTED';
          verifyBadge.className = 'font-mono-tech text-[8px] px-1 py-0.2 rounded bg-amber-950 text-amber-300 border border-amber-400/50';
        } else if (part.verified3D) {
          verifyBadge.textContent = '3D VERIFIED';
          verifyBadge.className = 'font-mono-tech text-[8px] px-1 py-0.2 rounded bg-emerald-950 text-emerald-300 border border-emerald-400/50';
        } else if (part.usedHarmonicFallback) {
          verifyBadge.textContent = 'HARMONIC';
          verifyBadge.className = 'font-mono-tech text-[8px] px-1 py-0.2 rounded bg-amber-950 text-amber-300 border border-amber-500/40';
        } else {
          verifyBadge.textContent = 'FUNNEL S4';
          verifyBadge.className = 'font-mono-tech text-[8px] px-1 py-0.2 rounded bg-cyan-950 text-cyan-400 border border-cyan-500/30';
        }
      }
      if (refContainer) refContainer.classList.remove('hidden');
    } else {
      if (refImg) {
        this._bindImageFallback(refImg, '');
        refImg.removeAttribute('src');
      }
      if (refContainer) refContainer.classList.add('hidden');
    }

    const lines = Array.isArray(part.summary) ? part.summary : [String(part.summary || '')];
    const summaryEl = document.getElementById('callout-summary');
    const principleEl = document.getElementById('callout-principle');
    if (summaryEl) {
      summaryEl.innerHTML = lines
        .slice(0, this.config.maxSummaryLines)
        .map((line) => this.escapeHtml(line))
        .join('<br>');
    }
    if (principleEl) {
      principleEl.textContent = part.principle || 'SPATIAL CAD MODEL';
    }

    const volBtn = document.getElementById('btn-toggle-part-volumetric');
    if (volBtn) {
      volBtn.onclick = (e) => {
        e.stopPropagation();
        if (typeof onToggleVolumetric === 'function') onToggleVolumetric(index);
      };
    }

    if (callout) {
      callout.classList.remove('hidden');
      requestAnimationFrame(() => callout.classList.remove('opacity-0'));
    }
    if (leaderLine) leaderLine.classList.remove('hidden');
    if (leaderReticle) leaderReticle.classList.remove('hidden');
  }

  updateLeaderLine(selectedPartGroup, camera, isPyramidMode) {
    const callout = document.getElementById('floating-callout');
    const leaderLine = document.getElementById('leader-line');
    const leaderReticle = document.getElementById('leader-reticle');

    if (!selectedPartGroup || isPyramidMode || !camera) {
      if (callout) callout.classList.add('hidden');
      if (leaderLine) leaderLine.classList.add('hidden');
      if (leaderReticle) leaderReticle.classList.add('hidden');
      return;
    }

    selectedPartGroup.getWorldPosition(this.tempWorldVec);
    this.tempWorldVec.project(camera);

    // Hide callout/reticle cleanly when projected point lies outside clip depth [-1, 1] (behind camera)
    if (!Number.isFinite(this.tempWorldVec.x) || !Number.isFinite(this.tempWorldVec.y) ||
        this.tempWorldVec.z > 1.0 || this.tempWorldVec.z < -1.0) {
      if (callout) callout.classList.add('hidden');
      if (leaderLine) leaderLine.classList.add('hidden');
      if (leaderReticle) leaderReticle.classList.add('hidden');
      return;
    }

    if (callout) {
      callout.classList.remove('hidden');
      callout.classList.remove('opacity-0');
    }
    if (leaderLine) leaderLine.classList.remove('hidden');
    if (leaderReticle) leaderReticle.classList.remove('hidden');

    const screenX = (this.tempWorldVec.x * 0.5 + 0.5) * window.innerWidth;
    const screenY = (-this.tempWorldVec.y * 0.5 + 0.5) * window.innerHeight;

    if (leaderReticle) {
      leaderReticle.setAttribute('transform', `translate(${screenX.toFixed(1)}, ${screenY.toFixed(1)})`);
    }

    if (!callout) return;

    const winW = Math.max(1, window.innerWidth);
    const winH = Math.max(1, window.innerHeight);
    const cardWidth = callout.offsetWidth || Math.min(300, winW * 0.42);
    const cardHeight = callout.offsetHeight || Math.min(220, winH * 0.34);

    const sideGap = Math.max(24, Math.min(55, winW * 0.04));
    const padX = Math.min(20, winW * 0.02);
    let topPad = Math.min(82, winH * 0.10);
    let bottomLimit = winH - Math.min(92, winH * 0.12);
    let leftBound = padX;
    let rightBound = Math.max(leftBound + cardWidth, winW - padX);

    // Dynamically measure visible chrome bounds so floating callout avoids HUD overlap at any viewport size
    const headerEl = document.querySelector('#main-hud > header');
    const headerRect = headerEl && typeof headerEl.getBoundingClientRect === 'function' ? headerEl.getBoundingClientRect() : null;
    if (headerRect && headerRect.height > 4 && headerRect.bottom < winH * 0.45) {
      topPad = Math.max(topPad, headerRect.bottom + 10);
    }

    const footerEl = document.querySelector('#main-hud > footer');
    const footerRect = footerEl && typeof footerEl.getBoundingClientRect === 'function' ? footerEl.getBoundingClientRect() : null;
    if (footerRect && footerRect.height > 4 && footerRect.top > topPad + cardHeight) {
      bottomLimit = Math.min(bottomLimit, footerRect.top - 10);
    }

    const asides = typeof document.querySelectorAll === 'function' ? document.querySelectorAll('#main-hud aside') : [];
    let dynLeft = padX;
    let dynRight = winW - padX;
    for (const aside of asides) {
      if (!aside || typeof aside.getBoundingClientRect !== 'function') continue;
      const r = aside.getBoundingClientRect();
      if (r.width <= 4 || r.height <= 4) continue;
      if (r.left < winW * 0.35 && r.right < winW * 0.5) {
        dynLeft = Math.max(dynLeft, r.right + 10);
      } else if (r.right > winW * 0.65 && r.left > winW * 0.5) {
        dynRight = Math.min(dynRight, r.left - 10);
      }
    }
    if (dynRight - dynLeft >= cardWidth + 24) {
      leftBound = dynLeft;
      rightBound = dynRight;
    }

    const isRightSide = screenX > winW * 0.52;
    const rawX = isRightSide ? screenX + sideGap : screenX - cardWidth - sideGap;
    const rawY = screenY - cardHeight * 0.65;

    const maxX = Math.max(leftBound, rightBound - cardWidth);
    const maxY = Math.max(topPad, bottomLimit - cardHeight);
    let cardX = Math.max(leftBound, Math.min(maxX, rawX));
    let cardY = Math.max(topPad, Math.min(maxY, rawY));

    // Prevent overlap with visible #escape-hatch-dock (#candidate-escape-hatch or #webcam-fallback-banner)
    const hatchEl = document.getElementById('candidate-escape-hatch');
    const webcamEl = document.getElementById('webcam-fallback-banner');
    const dockVisible =
      (hatchEl && !hatchEl.classList?.contains('hidden')) ||
      (webcamEl && !webcamEl.classList?.contains('hidden'));
    const dockEl = document.getElementById('escape-hatch-dock');
    const dockRect = dockVisible && dockEl && typeof dockEl.getBoundingClientRect === 'function'
      ? dockEl.getBoundingClientRect()
      : null;

    if (dockRect && dockRect.width > 4 && dockRect.height > 4 && dockRect.bottom > dockRect.top) {
      const dockGap = 12;
      const obsL = dockRect.left - dockGap;
      const obsR = dockRect.right + dockGap;
      const obsT = dockRect.top - dockGap;
      const obsB = dockRect.bottom + dockGap;

      const overlaps = (x, y) =>
        x < obsR && (x + cardWidth) > obsL && y < obsB && (y + cardHeight) > obsT;

      if (overlaps(cardX, cardY)) {
        const candidates = [];
        const upY = obsT - cardHeight;
        if (upY >= topPad) {
          candidates.push([cardX, upY]);
        }
        const leftShiftX = obsL - cardWidth;
        if (leftShiftX >= leftBound) {
          candidates.push([leftShiftX, cardY]);
        }
        const rightShiftX = obsR;
        if (rightShiftX + cardWidth <= rightBound) {
          candidates.push([rightShiftX, cardY]);
        }
        if (candidates.length === 0) {
          candidates.push([cardX, Math.max(topPad, Math.min(maxY, upY))]);
        }
        let best = candidates[0];
        let bestDist = Infinity;
        for (const [cx, cy] of candidates) {
          if (overlaps(cx, cy) && candidates.some(([ox, oy]) => !overlaps(ox, oy))) continue;
          const d = Math.hypot(cx - cardX, cy - cardY);
          if (d < bestDist) {
            bestDist = d;
            best = [cx, cy];
          }
        }
        cardX = Math.max(leftBound, Math.min(maxX, best[0]));
        cardY = Math.max(topPad, Math.min(maxY, best[1]));
      }
    }

    callout.style.left = `${cardX.toFixed(1)}px`;
    callout.style.top = `${cardY.toFixed(1)}px`;

    const attachLeftEdge = (cardX + cardWidth * 0.5) >= screenX;
    const anchorX = attachLeftEdge ? cardX : cardX + cardWidth;
    const anchorY = cardY + Math.min(26, cardHeight * 0.2);
    const elbowX = (screenX + anchorX) * 0.5;

    if (leaderLine) {
      leaderLine.setAttribute(
        'points',
        `${screenX.toFixed(1)},${screenY.toFixed(1)} ${elbowX.toFixed(1)},${anchorY.toFixed(1)} ${anchorX.toFixed(1)},${anchorY.toFixed(1)}`
      );
    }
  }

  updateExplosionUI(clampedValue) {
    const pct = Math.round(clampedValue * 100);
    const slider = document.getElementById('explosion-slider');
    if (slider) slider.value = String(pct);

    const badge = document.getElementById('explosion-pct-badge');
    if (badge) {
      badge.textContent = pct === 0 ? '0% ASSEMBLED' : `${pct}% EXPLODED`;
    }

    const tel = document.getElementById('telemetry-explosion');
    if (tel) {
      tel.textContent = `${clampedValue.toFixed(2)} (${pct}%)`;
    }
  }

  setEscapeHatchStatus(statusText, tone = 'cyan') {
    const badge = document.getElementById('escape-hatch-verify-badge');
    const dot = document.getElementById('escape-hatch-dot');
    if (badge && statusText) {
      badge.textContent = statusText;
      if (tone === 'emerald') {
        badge.className = 'font-mono-tech text-[8.5px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-400/50';
      } else if (tone === 'amber') {
        badge.className = 'font-mono-tech text-[8.5px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-400/50 animate-pulse';
      } else {
        badge.className = 'font-mono-tech text-[8.5px] px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-400/35';
      }
    }
    if (dot) {
      dot.className =
        tone === 'emerald'
          ? 'w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0'
          : tone === 'amber'
            ? 'w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping shrink-0'
            : 'w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse shrink-0';
    }
  }

  showCandidateEscapeHatch(candidates, activeUrl = '', metaInfo = {}, onSelect = null) {
    const panel = document.getElementById('candidate-escape-hatch');
    const thumbsEl = document.getElementById('escape-hatch-thumbs');
    const labelEl = document.getElementById('escape-hatch-label');
    const targetEl = document.getElementById('escape-hatch-target');
    const modeBadgeEl = document.getElementById('escape-hatch-mode-badge');
    const rejectBtn = document.getElementById('btn-gate-reject-webcam');
    const gateBar = document.getElementById('escape-hatch-gate-bar');
    if (!panel || !thumbsEl) return;

    const topCands = Array.isArray(candidates) ? candidates.slice(0, this.config.maxCandidateThumbnails) : [];
    if (topCands.length === 0) {
      this.hideCandidateEscapeHatch();
      return;
    }

    if (labelEl) {
      labelEl.textContent = metaInfo.label || `FUNNEL TOP-${topCands.length} CANDIDATES`;
    }
    if (targetEl) {
      targetEl.textContent = metaInfo.targetLabel || 'PART 01';
    }
    if (modeBadgeEl) {
      if (metaInfo.modeBadge) {
        modeBadgeEl.textContent = metaInfo.modeBadge;
        modeBadgeEl.classList.remove('hidden');
      } else {
        modeBadgeEl.classList.add('hidden');
      }
    }
    if (rejectBtn) {
      rejectBtn.classList.toggle('hidden', !metaInfo.allowWebcamRescue);
    }
    if (gateBar && !metaInfo.showGateBar && !this.preGenGate) {
      gateBar.classList.add('hidden');
      gateBar.classList.remove('gate-paused');
    }
    if (metaInfo.verifyBadge) {
      this.setEscapeHatchStatus(metaInfo.verifyBadge, metaInfo.verifyTone || 'cyan');
    }

    const colCount = Math.max(1, Math.min(topCands.length, 4));
    thumbsEl.style.gridTemplateColumns = `repeat(${colCount}, minmax(0, 1fr))`;
    thumbsEl.classList.toggle('hidden', this.escapeHatchCollapsed);
    thumbsEl.innerHTML = '';

    topCands.forEach((cand, idx) => {
      const rawUrl = cand.thumbUrl || cand.originalUrl || '';
      const fallbackRaw = cand.originalUrl && cand.originalUrl !== rawUrl ? cand.originalUrl : '';
      const proxied = this.proxyImageUrl(rawUrl);
      const isSelected = Boolean(
        (activeUrl && (cand.thumbUrl === activeUrl || cand.originalUrl === activeUrl)) ||
        (!activeUrl && idx === 0)
      );
      const isPreGenBusy = metaInfo.preGenBusyIndex === idx;
      const isBusy = metaInfo.switchingIndex === idx || isPreGenBusy;

      let scoreLabel = `T${cand.sourceTier || 1}`;
      if (cand.userPreferred) {
        scoreLabel = '★ VOTED';
      } else if (typeof cand.judgeTotal === 'number') {
        const criteriaCount = cand.judgeScores && typeof cand.judgeScores === 'object'
          ? Math.max(1, Object.keys(cand.judgeScores).length)
          : 5;
        const maxJudgeTotal = criteriaCount * 10;
        scoreLabel = `VLM ${Math.round(cand.judgeTotal)}/${maxJudgeTotal}`;
      } else if (cand.clipScore && typeof cand.clipScore.photoProb === 'number') {
        scoreLabel = `CLIP ${Math.round(cand.clipScore.photoProb * 100)}%`;
      }

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.setAttribute('data-dwell-target', 'candidate');
      btn.setAttribute('data-dwell-index', String(idx));
      btn.setAttribute('data-dwell-label', `🤏 SELECT CANDIDATE #${idx + 1}`);
      btn.title = cand.judgeReason || cand.title || `Candidate #${idx + 1}`;
      btn.className = isSelected
        ? 'group relative rounded border-2 border-amber-400 bg-amber-500/15 p-1 text-left transition shadow-[0_0_12px_rgba(255,170,0,0.3)] overflow-hidden'
        : 'group relative rounded border border-cyan-500/35 bg-cyan-950/50 hover:border-cyan-300 hover:bg-cyan-900/45 p-1 text-left transition overflow-hidden';

      const titleSafe = this.escapeHtml(cand.title || cand.caption || `Candidate ${idx + 1}`);
      const busyLabel = isPreGenBusy ? 'LOCKED...' : '3D BUILD...';
      const busyOverlay = isBusy
        ? `<div class="absolute inset-0 bg-black/75 flex items-center justify-center z-10"><span class="font-mono-tech text-[8.5px] text-amber-300 font-bold animate-pulse">${busyLabel}</span></div>`
        : '';

      btn.innerHTML = `
        ${busyOverlay}
        <div class="relative w-full h-11 rounded-sm overflow-hidden bg-black/70 border border-cyan-500/20">
          <img src="${this.escapeHtml(proxied)}" alt="${titleSafe}" class="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition" />
          <span class="absolute top-0.5 left-0.5 font-mono-tech text-[8px] font-bold px-1 rounded bg-black/80 text-cyan-200 border border-cyan-400/40">#${idx + 1}</span>
          <span class="absolute bottom-0.5 right-0.5 font-mono-tech text-[7.5px] px-1 rounded bg-black/80 ${cand.userPreferred ? 'text-amber-300 border-amber-400/50' : 'text-cyan-300 border-cyan-500/40'} border">${this.escapeHtml(scoreLabel)}</span>
        </div>
        <div class="font-mono-tech text-[8px] text-cyan-100/85 truncate mt-1 leading-tight">${titleSafe}</div>
      `;

      const thumbImg = btn.querySelector('img');
      if (thumbImg && fallbackRaw) {
        this._bindImageFallback(thumbImg, fallbackRaw);
      }

      btn.addEventListener('click', (e) => {
        if (e && typeof e.stopPropagation === 'function') e.stopPropagation();
        if (typeof onSelect === 'function' && !isBusy) onSelect(cand, idx);
      });

      thumbsEl.appendChild(btn);
    });

    panel.classList.remove('hidden');
  }

  _renderGateCountdown() {
    const gateBar = document.getElementById('escape-hatch-gate-bar');
    const timerEl = document.getElementById('escape-hatch-gate-timer');
    const fillEl = document.getElementById('escape-hatch-gate-fill');
    if (!this.preGenGate || !gateBar) return;

    if (this.preGenGate.timeoutMs <= 0) {
      gateBar.classList.add('hidden');
      gateBar.classList.remove('gate-paused');
      return;
    }

    gateBar.classList.remove('hidden');
    gateBar.classList.toggle('gate-paused', Boolean(this.preGenGate.paused));

    const pct = Math.max(0, Math.min(100, (this.preGenGate.remainingMs / this.preGenGate.timeoutMs) * 100));
    if (fillEl) {
      fillEl.style.width = `${pct.toFixed(1)}%`;
    }
    if (timerEl) {
      const secs = Math.max(0, this.preGenGate.remainingMs / 1000).toFixed(1);
      if (this.preGenGate.paused) {
        timerEl.textContent = `PAUSED (${secs}s LEFT) · PICK CANDIDATE OR WEBCAM`;
      } else if (this.preGenGate.remainingMs <= 0) {
        timerEl.textContent = 'AUTO-LOCKING #1 FOR 3D SYNTHESIS...';
      } else {
        timerEl.textContent = `AUTO-LOCK #1 IN ${secs}s · HOVER OR POINT TO PAUSE`;
      }
    }
  }

  showPreGenCandidateGate(gatePayload = {}, callbacks = {}) {
    this._clearGateTicker();
    this._gateHoverPaused = false;
    this._gateDwellPaused = false;

    const cands = Array.isArray(gatePayload.candidates) ? gatePayload.candidates : [];
    if (cands.length === 0) {
      this.preGenGate = null;
      return;
    }

    this.escapeHatchCollapsed = false;
    const collapseBtn = document.getElementById('btn-toggle-escape-hatch');
    if (collapseBtn) collapseBtn.textContent = 'HIDE';

    const timeoutMs = Math.max(0, Number(gatePayload.timeoutMs) || 0);
    const mode = String(gatePayload.mode || 'human_gate_spotlight');
    const modeBadge = this.formatOperatingModeLabel(mode);
    const verifyBadge = mode === 'soft_veto_overlap' ? 'CONFIRM PICK' : '2s HUMAN GATE';
    const verifyTone = mode === 'soft_veto_overlap' ? 'amber' : 'cyan';
    const allowWebcamRescue = gatePayload.allowWebcamRescue !== false;
    const partLabel = gatePayload.partName
      ? `01 · ${gatePayload.partName}`
      : (gatePayload.query || 'PART 01');
    const headerLabel = `PRE-GEN SPOTLIGHT · TOP ${Math.min(cands.length, this.config.maxCandidateThumbnails)}`;
    const firstUrl = cands[0]?.thumbUrl || cands[0]?.originalUrl || '';

    this.preGenGate = {
      gateId: gatePayload.gateId || '',
      mode,
      query: gatePayload.query || '',
      partName: gatePayload.partName || '',
      candidates: cands,
      timeoutMs,
      remainingMs: timeoutMs,
      paused: false,
      submitting: false,
      lastTickMs: typeof performance !== 'undefined' ? performance.now() : Date.now(),
      tickerId: null,
      onSelect: typeof callbacks.onSelect === 'function' ? callbacks.onSelect : null,
      onPauseChange: typeof callbacks.onPauseChange === 'function' ? callbacks.onPauseChange : null,
      onReject: typeof callbacks.onReject === 'function' ? callbacks.onReject : null
    };

    this.showCandidateEscapeHatch(
      cands,
      firstUrl,
      {
        label: headerLabel,
        targetLabel: partLabel,
        modeBadge,
        verifyBadge,
        verifyTone,
        allowWebcamRescue,
        showGateBar: timeoutMs > 0
      },
      (cand, idx) => {
        if (!this.preGenGate || this.preGenGate.submitting) return;
        this.preGenGate.submitting = true;
        this._clearGateTicker();
        const gateBar = document.getElementById('escape-hatch-gate-bar');
        if (gateBar) {
          gateBar.classList.add('hidden');
          gateBar.classList.remove('gate-paused');
        }
        const selectedUrl = cand?.thumbUrl || cand?.originalUrl || firstUrl;
        this.showCandidateEscapeHatch(
          cands,
          selectedUrl,
          {
            label: headerLabel,
            targetLabel: partLabel,
            modeBadge,
            verifyBadge: `LOCKING #${idx + 1}...`,
            verifyTone: 'amber',
            allowWebcamRescue: false,
            preGenBusyIndex: idx,
            showGateBar: false
          },
          null
        );
        if (typeof this.preGenGate.onSelect === 'function') {
          this.preGenGate.onSelect(cand, idx);
        }
      }
    );

    this._renderGateCountdown();

    if (timeoutMs > 0 && typeof setInterval === 'function') {
      const intervalId = setInterval(() => {
        if (!this.preGenGate) {
          clearInterval(intervalId);
          return;
        }
        const nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
        const dt = Math.max(0, nowMs - (this.preGenGate.lastTickMs || nowMs));
        this.preGenGate.lastTickMs = nowMs;
        if (!this.preGenGate.paused && !this.preGenGate.submitting) {
          this.preGenGate.remainingMs = Math.max(0, this.preGenGate.remainingMs - dt);
        }
        this._renderGateCountdown();
      }, 50);
      if (intervalId && typeof intervalId.unref === 'function') {
        intervalId.unref();
      }
      this.preGenGate.tickerId = intervalId;
    }
  }

  setGatePaused(paused, source = 'hover') {
    if (source === 'dwell') {
      this._gateDwellPaused = Boolean(paused);
    } else {
      this._gateHoverPaused = Boolean(paused);
    }
    if (!this.preGenGate || this.preGenGate.submitting) return false;

    const nextPaused = Boolean(this._gateHoverPaused || this._gateDwellPaused);
    if (nextPaused === this.preGenGate.paused) {
      return this.preGenGate.paused;
    }

    const nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (!this.preGenGate.paused && nextPaused) {
      const elapsed = Math.max(0, nowMs - (this.preGenGate.lastTickMs || nowMs));
      this.preGenGate.remainingMs = Math.max(0, this.preGenGate.remainingMs - elapsed);
    }
    this.preGenGate.paused = nextPaused;
    this.preGenGate.lastTickMs = nowMs;
    this._renderGateCountdown();

    if (typeof this.preGenGate.onPauseChange === 'function') {
      this.preGenGate.onPauseChange(nextPaused, this.preGenGate.remainingMs);
    }
    return nextPaused;
  }

  resolvePreGenCandidateGate(resolvedPayload = {}) {
    this._clearGateTicker();
    const hadGate = Boolean(this.preGenGate);
    this.preGenGate = null;
    this._gateHoverPaused = false;
    this._gateDwellPaused = false;

    const gateBar = document.getElementById('escape-hatch-gate-bar');
    if (gateBar) {
      gateBar.classList.add('hidden');
      gateBar.classList.remove('gate-paused');
    }
    const rejectBtn = document.getElementById('btn-gate-reject-webcam');
    if (rejectBtn) {
      rejectBtn.classList.add('hidden');
    }

    const outcome = String(resolvedPayload?.outcome || '');
    if (!hadGate && !outcome) return;

    const modeBadgeEl = document.getElementById('escape-hatch-mode-badge');
    if (modeBadgeEl && resolvedPayload?.mode) {
      modeBadgeEl.textContent = this.formatOperatingModeLabel(resolvedPayload.mode);
      modeBadgeEl.classList.remove('hidden');
    }

    const selIdx = Number(resolvedPayload?.selectedIndex);
    if (outcome === 'select' && Number.isFinite(selIdx) && selIdx >= 0) {
      this.setEscapeHatchStatus(`LOCKED #${selIdx + 1}`, 'amber');
    } else if (outcome === 'timeout') {
      this.setEscapeHatchStatus('AUTO-LOCKED #1', 'emerald');
    } else if (outcome === 'reject' || outcome === 'rescue_webcam') {
      this.setEscapeHatchStatus('CAMERA RESCUE', 'amber');
    }
  }

  hideCandidateEscapeHatch() {
    this._clearGateTicker();
    this.preGenGate = null;
    this._gateHoverPaused = false;
    this._gateDwellPaused = false;
    const gateBar = document.getElementById('escape-hatch-gate-bar');
    if (gateBar) {
      gateBar.classList.add('hidden');
      gateBar.classList.remove('gate-paused');
    }
    const rejectBtn = document.getElementById('btn-gate-reject-webcam');
    if (rejectBtn) rejectBtn.classList.add('hidden');
    const panel = document.getElementById('candidate-escape-hatch');
    if (panel) panel.classList.add('hidden');
  }

  showWebcamFallbackBanner(info = {}, onCapture = null, onDismiss = null) {
    const banner = document.getElementById('webcam-fallback-banner');
    const titleEl = document.getElementById('webcam-fallback-title');
    const reasonEl = document.getElementById('webcam-fallback-reason');
    const captureBtn = document.getElementById('btn-webcam-fallback');
    const dismissBtn = document.getElementById('btn-dismiss-webcam-fallback');
    if (!banner) return;

    if (titleEl) {
      const att = Number(info.attempts);
      const maxAtt = Number(info.maxAttempts);
      if (Number.isFinite(att) && att > 0 && Number.isFinite(maxAtt) && maxAtt > 0) {
        titleEl.textContent = `3D VERIFICATION UNCERTAIN (${att}/${maxAtt} ATTEMPTS)`;
      } else if (Number.isFinite(att) && att > 0) {
        titleEl.textContent = `3D VERIFICATION UNCERTAIN (${att} ATTEMPT${att === 1 ? '' : 'S'})`;
      } else {
        titleEl.textContent = '3D VERIFICATION UNCERTAIN';
      }
    }

    if (reasonEl) {
      reasonEl.textContent = info.message || info.reason || 'Show a clear object or photo to your camera to reconstruct accurately.';
    }
    if (captureBtn) {
      captureBtn.textContent = info.webcamActive ? '📸 SNAP OBJECT NOW' : '📸 SHOW OBJECT TO CAMERA';
      captureBtn.onclick = (e) => {
        e.stopPropagation();
        if (typeof onCapture === 'function') onCapture();
      };
    }
    if (dismissBtn) {
      dismissBtn.onclick = (e) => {
        e.stopPropagation();
        this.hideWebcamFallbackBanner();
        if (typeof onDismiss === 'function') onDismiss();
      };
    }

    banner.classList.remove('hidden');
  }

  hideWebcamFallbackBanner() {
    const banner = document.getElementById('webcam-fallback-banner');
    if (banner) banner.classList.add('hidden');
  }
}