/**
 * VLMClient - Communicates with the local Qwen VLM (Vision-Language Model) API
 * Strictly dynamic: requests all 3D assemblies from the local Qwen model.
 * Supports both SSE (Server-Sent Events) Progressive Streaming and fallback REST.
 */

export const DEFAULT_API_CONFIG = Object.freeze({
  baseUrl: '',
  endpoints: Object.freeze({
    health: '/api/health',
    blueprints: '/api/blueprints',
    synthesizeStream: '/api/synthesize-stream',
    synthesize: '/api/synthesize',
    selectCandidate: '/api/select-candidate',
    confirmCandidate: '/api/confirm-candidate',
    reference: '/api/reference',
    proxyImage: '/api/proxy-image'
  })
});

export class VLMClient {
  constructor(baseUrl = '', config = {}) {
    this.baseUrl = baseUrl || config.baseUrl || DEFAULT_API_CONFIG.baseUrl;
    this.endpoints = {
      ...DEFAULT_API_CONFIG.endpoints,
      ...(config.endpoints || {})
    };
  }

  _buildUrl(path, queryParams = null) {
    const base = `${this.baseUrl}${path}`;
    if (!queryParams || typeof queryParams !== 'object') {
      return base;
    }
    const searchParams = new URLSearchParams();
    for (const [key, val] of Object.entries(queryParams)) {
      if (val !== undefined && val !== null && String(val).length > 0) {
        searchParams.set(key, String(val));
      }
    }
    const qs = searchParams.toString();
    return qs ? `${base}?${qs}` : base;
  }

  async _extractErrorMessage(res) {
    let msg = `HTTP ${res.status}`;
    try {
      const errText = await res.text();
      if (errText) {
        const errObj = JSON.parse(errText);
        if (errObj && typeof errObj.error === 'string') {
          msg = errObj.detail ? `${errObj.error}: ${errObj.detail}` : errObj.error;
        }
      }
    } catch (_) {
      // Fallback to status code message if body is not JSON
    }
    return msg;
  }

  async _requestJson(path, options = {}) {
    const res = await fetch(this._buildUrl(path), options);
    if (!res.ok) {
      const msg = await this._extractErrorMessage(res);
      throw new Error(msg);
    }
    return await res.json();
  }

  _normalizeLineEndings(chunkText) {
    if (!chunkText || !chunkText.includes('\r')) return chunkText;
    return chunkText.split('\r\n').join('\n').split('\r').join('\n');
  }

  _parseSseBlock(block) {
    const trimmed = block.trim();
    if (!trimmed) return null;

    let eventType = 'message';
    const dataLines = [];
    const lines = trimmed.split('\n');

    for (const line of lines) {
      if (line.startsWith(':')) {
        continue;
      }
      if (line.startsWith('event:')) {
        eventType = line.slice(6).trim() || 'message';
      } else if (line.startsWith('data:')) {
        const rawData = line.slice(5);
        dataLines.push(rawData.startsWith(' ') ? rawData.slice(1) : rawData);
      }
    }

    if (dataLines.length === 0) return null;
    return {
      eventType,
      dataStr: dataLines.join('\n')
    };
  }

  async checkHealth() {
    try {
      return await this._requestJson(this.endpoints.health);
    } catch (e) {
      console.warn('VLM Health check warning:', e.message);
      return { status: 'offline', error: e.message };
    }
  }

  async fetchBlueprints() {
    try {
      const data = await this._requestJson(this.endpoints.blueprints);
      return Array.isArray(data) ? data : [];
    } catch (_) {
      return [];
    }
  }

  async listSavedBlueprints() {
    return this.fetchBlueprints();
  }

  /**
   * Progressively streams VLM tokens, materializes parts in real-time,
   * and receives continuous mesh synthesis and geometry relaxation progress.
   *
   * @param {string} conceptPrompt
   * @param {string|null} imageBase64
   * @param {function(string, object): void} onEvent (eventType, eventData)
   * @param {AbortSignal|null} signal
   */
  async synthesizeConceptStream(conceptPrompt, imageBase64 = null, onEvent = null, signal = null) {
    const payload = {
      prompt: conceptPrompt,
      image: imageBase64
    };

    const res = await fetch(this._buildUrl(this.endpoints.synthesizeStream), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload),
      signal
    });

    if (!res.ok) {
      const msg = await this._extractErrorMessage(res);
      throw new Error(msg);
    }

    if (!res.body || typeof res.body.getReader !== 'function') {
      throw new Error('ReadableStream not supported by this browser environment');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += this._normalizeLineEndings(decoder.decode(value, { stream: true }));
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() || '';

        for (const block of blocks) {
          const parsedFrame = this._parseSseBlock(block);
          if (!parsedFrame) continue;

          try {
            const dataObj = JSON.parse(parsedFrame.dataStr);
            if (typeof onEvent === 'function') {
              onEvent(parsedFrame.eventType, dataObj);
            }
            if (parsedFrame.eventType === 'complete') {
              await reader.cancel();
              return;
            }
          } catch (e) {
            console.warn('[!] Failed to parse SSE JSON payload:', e, parsedFrame.dataStr);
          }
        }
      }

      if (buffer.trim()) {
        const finalFrame = this._parseSseBlock(buffer);
        if (finalFrame && typeof onEvent === 'function') {
          try {
            onEvent(finalFrame.eventType, JSON.parse(finalFrame.dataStr));
          } catch (_) { }
        }
      }
    } finally {
      try {
        reader.releaseLock();
      } catch (_) { }
    }
  }

  /**
   * Synchronous fallback REST endpoint.
   */
  async synthesizeConcept(conceptPrompt, imageBase64 = null) {
    const data = await this._requestJson(this.endpoints.synthesize, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        prompt: conceptPrompt,
        image: imageBase64
      })
    });

    if (!data || !Array.isArray(data.parts)) {
      throw new Error('VLM returned invalid 3D assembly format');
    }

    return data;
  }

  async selectCandidate({ query, blueprintTitle, partIndex = 0, partId = '', partName = '', candidate }) {
    return await this._requestJson(this.endpoints.selectCandidate, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query,
        blueprintTitle,
        partIndex,
        partId,
        partName,
        candidate
      })
    });
  }

  async confirmCandidate({
    gateId,
    action = 'select',
    candidateIndex = null,
    candidate = null,
    candidateUrl = '',
    timeoutMs = null,
    reason = ''
  }) {
    const body = { gateId, action };
    if (candidateIndex !== null && candidateIndex !== undefined) {
      body.candidateIndex = candidateIndex;
    }
    if (candidate && typeof candidate === 'object') {
      body.candidate = candidate;
    }
    if (candidateUrl) {
      body.candidateUrl = candidateUrl;
    }
    if (timeoutMs !== null && timeoutMs !== undefined) {
      body.timeoutMs = timeoutMs;
    }
    if (reason) {
      body.reason = reason;
    }
    return await this._requestJson(this.endpoints.confirmCandidate, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });
  }

  async fetchTopicReference(query) {
    try {
      const url = this._buildUrl(this.endpoints.reference, { query });
      const res = await fetch(url);
      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      console.warn('Reference fetch warning:', e.message);
      return null;
    }
  }
}