/**
 * StarkSoundFX - Web Audio Synthesizer for Stark HUD Acoustic Telemetry
 * Zero hardcoded inline frequencies: driven by configurable acoustic profiles.
 */

export const DEFAULT_AUDIO_CONFIG = Object.freeze({
  chime: Object.freeze({
    defaultFreq: 880,
    defaultDuration: 0.08,
    defaultType: 'sine',
    pitchMultiplier: 1.35,
    peakGain: 0.06,
    floorGain: 0.001
  }),
  select: Object.freeze({
    primaryFreq: 1040,
    primaryDuration: 0.07,
    primaryType: 'triangle',
    secondaryFreq: 1560,
    secondaryDuration: 0.09,
    secondaryType: 'sine',
    secondaryDelayMs: 45
  }),
  sweep: Object.freeze({
    lowFreq: 240,
    highFreq: 620,
    duration: 0.22,
    type: 'sine',
    peakGain: 0.05,
    floorGain: 0.001
  })
});

export class StarkSoundFX {
  constructor(config = {}) {
    this.ctx = null;
    this.enabled = true;
    this.config = {
      chime: { ...DEFAULT_AUDIO_CONFIG.chime, ...(config.chime || {}) },
      select: { ...DEFAULT_AUDIO_CONFIG.select, ...(config.select || {}) },
      sweep: { ...DEFAULT_AUDIO_CONFIG.sweep, ...(config.sweep || {}) }
    };
  }

  init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) this.ctx = new AudioCtx();
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  _playEnvelope({ startFreq, endFreq, duration, type, peakGain, floorGain }) {
    if (!this.enabled) return;
    try {
      this.init();
      if (!this.ctx) return;

      const now = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(Math.max(20, startFreq), now);
      osc.frequency.exponentialRampToValueAtTime(Math.max(20, endFreq), now + duration);

      gain.gain.setValueAtTime(peakGain, now);
      gain.gain.exponentialRampToValueAtTime(floorGain, now + duration);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.onended = () => {
        try {
          osc.disconnect();
          gain.disconnect();
        } catch (_) { }
      };

      osc.start(now);
      osc.stop(now + duration);
    } catch (_) {
      // Ignore browser autoplay policy blocks prior to user gesture
    }
  }

  playChime(
    freq = this.config.chime.defaultFreq,
    duration = this.config.chime.defaultDuration,
    type = this.config.chime.defaultType
  ) {
    const { pitchMultiplier, peakGain, floorGain } = this.config.chime;
    this._playEnvelope({
      startFreq: freq,
      endFreq: freq * pitchMultiplier,
      duration,
      type,
      peakGain,
      floorGain
    });
  }

  playSelect() {
    const s = this.config.select;
    this.playChime(s.primaryFreq, s.primaryDuration, s.primaryType);
    setTimeout(() => {
      this.playChime(s.secondaryFreq, s.secondaryDuration, s.secondaryType);
    }, s.secondaryDelayMs);
  }

  playExplodeSweep(expanding = true) {
    const sw = this.config.sweep;
    this._playEnvelope({
      startFreq: expanding ? sw.lowFreq : sw.highFreq,
      endFreq: expanding ? sw.highFreq : sw.lowFreq,
      duration: sw.duration,
      type: sw.type,
      peakGain: sw.peakGain,
      floorGain: sw.floorGain
    });
  }
}