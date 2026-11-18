#!/usr/bin/env python3
"""
M.A.R.K. Holo-CAD // Pre-Generation Human Gate Coordinator
============================================================
Zero-Torch, thread-safe registry managing short (default 2.0s) pre-synthesis
Human Gate spotlights between candidate vetting (Stage 4) and 3D GPU synthesis
(TripoSR). Supports instant click/pinch selection, hover/dwell pause & resume,
camera-rescue rejection, clean client-disconnect abort, and timeout fallback.
"""

import os
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

DEFAULT_GATE_TIMEOUT_MS = max(0, int(os.environ.get("HOLO_CAD_GATE_TIMEOUT_MS", "2000")))
GATE_MAX_WAIT_S = max(5.0, float(os.environ.get("HOLO_CAD_GATE_MAX_WAIT_S", "45.0")))
GATE_PRUNE_AGE_S = max(60.0, float(os.environ.get("HOLO_CAD_GATE_PRUNE_AGE_S", "180.0")))
_POLL_INTERVAL_S = 0.20


class CandidateGate:
    """Thread-safe synchronization primitive for a single pre-generation gate."""

    def __init__(
        self,
        query: str,
        candidates: List[dict],
        mode: str = "human_gate_spotlight",
        timeout_ms: int = DEFAULT_GATE_TIMEOUT_MS,
        default_index: int = 0,
        gate_id: Optional[str] = None,
    ):
        self.gate_id = gate_id or uuid.uuid4().hex[:16]
        self.query = str(query or "")
        self.candidates = list(candidates or [])
        self.mode = str(mode or "human_gate_spotlight")
        self.timeout_ms = max(0, int(timeout_ms))
        self.selected_index = (
            max(0, min(len(self.candidates) - 1, int(default_index)))
            if self.candidates
            else 0
        )

        self.paused = False
        self.resolved = False
        self.action = "pending"
        self.reject_reason = ""
        self.created_at = time.time()
        self.resolved_at: Optional[float] = None

        now_mono = time.monotonic()
        self._remaining_s = self.timeout_ms / 1000.0
        self._deadline_mono = now_mono + self._remaining_s
        self._hard_deadline_mono = now_mono + max(GATE_MAX_WAIT_S, self._remaining_s + 2.0)
        self._cond = threading.Condition()

    def snapshot(self) -> Dict:
        with self._cond:
            remaining_ms = 0
            if not self.resolved:
                if self.paused:
                    remaining_ms = max(0, int(round(self._remaining_s * 1000.0)))
                else:
                    remaining_ms = max(0, int(round((self._deadline_mono - time.monotonic()) * 1000.0)))
            selected_cand = (
                self.candidates[self.selected_index]
                if 0 <= self.selected_index < len(self.candidates)
                else None
            )
            return {
                "gateId": self.gate_id,
                "query": self.query,
                "mode": self.mode,
                "timeoutMs": self.timeout_ms,
                "remainingMs": remaining_ms,
                "paused": self.paused,
                "resolved": self.resolved,
                "action": self.action,
                "selectedIndex": self.selected_index,
                "selectedCandidate": selected_cand,
                "rejectReason": self.reject_reason,
                "resolvedAt": self.resolved_at,
            }

    def pause(self) -> Dict:
        with self._cond:
            if self.resolved:
                return self.snapshot()
            if not self.paused:
                self._remaining_s = max(0.0, self._deadline_mono - time.monotonic())
                self.paused = True
                self._cond.notify_all()
            return self.snapshot()

    def resume(self, timeout_ms: Optional[int] = None) -> Dict:
        with self._cond:
            if self.resolved:
                return self.snapshot()
            if timeout_ms is not None:
                self._remaining_s = max(0.0, int(timeout_ms) / 1000.0)
            self.paused = False
            self._deadline_mono = time.monotonic() + self._remaining_s
            self._cond.notify_all()
            return self.snapshot()

    def _find_candidate_index(
        self, candidate_index: Optional[int], candidate_url: Optional[str]
    ) -> Optional[int]:
        if candidate_index is not None:
            try:
                idx = int(candidate_index)
                if 0 <= idx < len(self.candidates):
                    return idx
            except (TypeError, ValueError):
                pass
        if candidate_url:
            target = str(candidate_url).strip()
            for idx, c in enumerate(self.candidates):
                if c.get("thumbUrl") == target or c.get("originalUrl") == target:
                    return idx
        return None

    def select(
        self,
        candidate_index: Optional[int] = None,
        candidate_url: Optional[str] = None,
    ) -> Dict:
        with self._cond:
            if self.resolved:
                return self.snapshot()
            matched_idx = self._find_candidate_index(candidate_index, candidate_url)
            if matched_idx is not None:
                self.selected_index = matched_idx
            self.paused = False
            self.resolved = True
            self.action = "select"
            self.resolved_at = time.time()
            self._cond.notify_all()
            return self.snapshot()

    def reject(self, reason: str = "user_rejected_to_camera") -> Dict:
        with self._cond:
            if self.resolved:
                return self.snapshot()
            self.paused = False
            self.resolved = True
            self.action = "reject"
            self.reject_reason = str(reason or "user_rejected_to_camera")[:180]
            self.resolved_at = time.time()
            self._cond.notify_all()
            return self.snapshot()

    def wait(self, is_disconnected_fn: Optional[Callable[[], bool]] = None) -> Dict:
        """
        Blocks caller thread until user confirms/rejects, timer expires,
        hard wait ceiling hits, or client disconnects.
        """
        with self._cond:
            while not self.resolved:
                if is_disconnected_fn is not None:
                    try:
                        if is_disconnected_fn():
                            self.paused = False
                            self.resolved = True
                            self.action = "disconnected"
                            self.resolved_at = time.time()
                            break
                    except Exception:
                        pass

                now_mono = time.monotonic()
                if now_mono >= self._hard_deadline_mono:
                    self.paused = False
                    self.resolved = True
                    self.action = "timeout"
                    self.resolved_at = time.time()
                    break

                if not self.paused:
                    if now_mono >= self._deadline_mono:
                        self.resolved = True
                        self.action = "timeout"
                        self.resolved_at = time.time()
                        break
                    step_s = min(_POLL_INTERVAL_S, max(0.005, self._deadline_mono - now_mono))
                else:
                    step_s = min(_POLL_INTERVAL_S, max(0.01, self._hard_deadline_mono - now_mono))

                self._cond.wait(timeout=step_s)

        return self.snapshot()


class GateRegistry:
    """Thread-safe container managing active CandidateGate instances."""

    def __init__(self):
        self._lock = threading.Lock()
        self._gates: Dict[str, CandidateGate] = {}

    def _prune_locked(self, now_ts: Optional[float] = None) -> None:
        cutoff = (now_ts if now_ts is not None else time.time()) - GATE_PRUNE_AGE_S
        stale_ids = [
            gid
            for gid, g in self._gates.items()
            if g.resolved and (g.resolved_at or g.created_at) < cutoff
        ]
        for gid in stale_ids:
            self._gates.pop(gid, None)

    def create_gate(
        self,
        query: str,
        candidates: List[dict],
        mode: str = "human_gate_spotlight",
        timeout_ms: int = DEFAULT_GATE_TIMEOUT_MS,
        default_index: int = 0,
        gate_id: Optional[str] = None,
    ) -> CandidateGate:
        gate = CandidateGate(
            query=query,
            candidates=candidates,
            mode=mode,
            timeout_ms=timeout_ms,
            default_index=default_index,
            gate_id=gate_id,
        )
        with self._lock:
            self._prune_locked(gate.created_at)
            self._gates[gate.gate_id] = gate
        return gate

    def get_gate(self, gate_id: str) -> Optional[CandidateGate]:
        if not gate_id:
            return None
        with self._lock:
            return self._gates.get(str(gate_id).strip())

    def remove_gate(self, gate_id: str) -> Optional[CandidateGate]:
        if not gate_id:
            return None
        with self._lock:
            return self._gates.pop(str(gate_id).strip(), None)

    def apply_action(
        self,
        gate_id: str,
        action: str,
        candidate_index: Optional[int] = None,
        candidate_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        reason: str = "",
    ) -> Optional[Dict]:
        gate = self.get_gate(gate_id)
        if gate is None:
            return None
        act = str(action or "").strip().lower()
        if act == "select":
            return gate.select(candidate_index=candidate_index, candidate_url=candidate_url)
        if act == "pause":
            return gate.pause()
        if act == "resume":
            return gate.resume(timeout_ms=timeout_ms)
        if act == "reject":
            return gate.reject(reason=reason or "user_rejected_to_camera")
        raise ValueError(f"Unsupported gate action: {action!r}")

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for g in self._gates.values() if not g.resolved)


_DEFAULT_REGISTRY = GateRegistry()


def create_gate(
    query: str,
    candidates: List[dict],
    mode: str = "human_gate_spotlight",
    timeout_ms: int = DEFAULT_GATE_TIMEOUT_MS,
    default_index: int = 0,
    gate_id: Optional[str] = None,
) -> CandidateGate:
    return _DEFAULT_REGISTRY.create_gate(
        query=query,
        candidates=candidates,
        mode=mode,
        timeout_ms=timeout_ms,
        default_index=default_index,
        gate_id=gate_id,
    )


def get_gate(gate_id: str) -> Optional[CandidateGate]:
    return _DEFAULT_REGISTRY.get_gate(gate_id)


def remove_gate(gate_id: str) -> Optional[CandidateGate]:
    return _DEFAULT_REGISTRY.remove_gate(gate_id)


def apply_gate_action(
    gate_id: str,
    action: str,
    candidate_index: Optional[int] = None,
    candidate_url: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    reason: str = "",
) -> Optional[Dict]:
    return _DEFAULT_REGISTRY.apply_action(
        gate_id=gate_id,
        action=action,
        candidate_index=candidate_index,
        candidate_url=candidate_url,
        timeout_ms=timeout_ms,
        reason=reason,
    )


def get_active_gate_count() -> int:
    return _DEFAULT_REGISTRY.active_count()
