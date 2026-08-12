"""
vision/shadow.py
----------------
Phase 0: run the identity engine ALONGSIDE the existing one, in production,
without being able to break it.

The contract this file signs
----------------------------
1. **It cannot slow the frame loop.** All work happens on a background thread
   behind a queue of depth 2. When the worker is behind, frames are dropped,
   not queued. ``submit()`` costs a bounded-size copy and a put_nowait.
2. **It cannot crash the engine.** Every entry point is wrapped. After
   ``MAX_CONSECUTIVE_ERRORS`` failures it disables itself permanently, prints
   once, and every later call becomes a no-op.
3. **It cannot change a single existing number.** It never writes to the
   tracker, the recorder or the replay buffer. Its output is namespaced under
   ``shadow_*`` keys, so a dashboard that does not know about it is unaffected.
4. **It self-throttles.** If its own processing time drifts above the budget
   it raises its frame stride automatically rather than falling behind.

What it produces
----------------
A JSONL log per camera per day, containing paired observations of the OLD
counters and the NEW ones. That paired series is the entire point of Phase 0:
the divergence between ``legacy_tam`` and ``shadow_tam`` is a direct
measurement of how much double-counting the store is living with today.

    data/shadow/<camera_id>/<YYYY-MM-DD>.jsonl

Kill switch:  COREWISE_SHADOW=0
Stride:       COREWISE_SHADOW_STRIDE=3
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MAX_CONSECUTIVE_ERRORS = 5
DEFAULT_QUEUE_DEPTH = 2
DEFAULT_STRIDE = 2

# If a shadow frame costs more than this, the stride grows. The shadow must
# never be the reason the machine is busy.
FRAME_BUDGET_SECONDS = 0.080
MAX_STRIDE = 30

LOG_INTERVAL_SECONDS = 10.0
SCORE_BUCKETS = [0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01]


def _enabled_by_env() -> bool:
    return os.environ.get("COREWISE_SHADOW", "1") not in ("0", "false", "False", "no")


def _live_by_env() -> bool:
    """Phase 2 in one environment variable.

    OFF (default) — Phase 0. The dashboard keeps showing the old counters and
    the identity numbers ride alongside them as shadow_* keys. Nothing that
    works today can break.

    ON — the identity numbers REPLACE tam / sam / som / inside in the payload.
    Same keys, same dashboard, same reports; they now count people instead of
    tracker ids. Reversible instantly by unsetting the variable, because no
    code changed and the old tracker is still running underneath.
    """
    return os.environ.get("COREWISE_IDENTITY_LIVE", "0") in ("1", "true", "True", "yes")


class ShadowRunner:
    """Runs IdentityPipeline off the hot path and records the divergence."""

    def __init__(
        self,
        camera_id: str,
        log_dir: Optional[Path] = None,
        stride: Optional[int] = None,
        enabled: Optional[bool] = None,
        queue_depth: int = DEFAULT_QUEUE_DEPTH,
        pipeline: Optional[Any] = None,
    ) -> None:
        self.camera_id = camera_id
        self.enabled = _enabled_by_env() if enabled is None else bool(enabled)
        self.stride = int(stride or os.environ.get("COREWISE_SHADOW_STRIDE", DEFAULT_STRIDE))
        self.log_dir = Path(log_dir) if log_dir else (
            Path(__file__).resolve().parents[1] / "data" / "shadow" / camera_id)
        self.live = _live_by_env()

        self._pipeline = pipeline
        self._queue: "queue.Queue[dict]" = queue.Queue(maxsize=queue_depth)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._frame_counter = 0
        self._errors = 0
        self._consecutive_errors = 0
        self._disabled_reason = ""
        self._durations: deque = deque(maxlen=200)
        self._dropped = 0
        self._processed = 0

        self._verdicts: Counter = Counter()
        self._score_hist: Counter = Counter()
        self._latest: Dict[str, Any] = {}
        self._legacy: Dict[str, Any] = {}
        self._last_log = 0.0
        self._log_path: Optional[Path] = None
        self._started_at = time.time()

        if self.enabled and self._pipeline is None:
            try:
                from .pipeline import IdentityPipeline
                self._pipeline = IdentityPipeline(camera_id=camera_id)
            except Exception as exc:
                self._disable(f"could not construct IdentityPipeline: {exc!r}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "ShadowRunner":
        if not self.enabled or self._thread is not None:
            return self
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._worker, daemon=True,
                                            name="corewise-shadow")
            self._thread.start()
            if self.live:
                print(f"[SHADOW] IDENTITY LIVE (camera={self.camera_id}) — "
                      "tam/sam/som on the dashboard now count PEOPLE. "
                      "Unset COREWISE_IDENTITY_LIVE to go back.")
            else:
                print(f"[SHADOW] Phase 0 active (camera={self.camera_id}, "
                      f"stride={self.stride}, log={self.log_dir})")
        except Exception as exc:
            self._disable(f"could not start worker: {exc!r}")
        return self

    def stop(self, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        try:
            self._queue.put_nowait({"_stop": True})
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        try:
            self._write_log(final=True)
            if self._pipeline is not None:
                self._pipeline.shutdown()
        except Exception:
            pass

    def _disable(self, reason: str) -> None:
        if not self.enabled:
            return
        self.enabled = False
        self._disabled_reason = reason
        print(f"[SHADOW] disabled — {reason}. The live engine is unaffected.")

    # ------------------------------------------------------------------
    # Calibration passthrough (safe no-ops when disabled)
    # ------------------------------------------------------------------

    def configure(self, **kwargs: Any) -> None:
        if not self.enabled or self._pipeline is None:
            return
        try:
            self._pipeline.configure(**kwargs)
        except Exception as exc:
            self._note_error(exc)

    def set_min_stay(self, enabled: bool, seconds: Optional[float]) -> None:
        if not self.enabled or self._pipeline is None:
            return
        try:
            self._pipeline.set_min_stay(enabled, seconds)
        except Exception as exc:
            self._note_error(exc)

    def reset_counters(self) -> None:
        if not self.enabled or self._pipeline is None:
            return
        try:
            self._pipeline.reset_counters()
        except Exception as exc:
            self._note_error(exc)

    # ------------------------------------------------------------------
    # Hot path — must stay cheap
    # ------------------------------------------------------------------

    def submit(
        self,
        frame,
        detections: Sequence[Dict[str, Any]],
        legacy_stats: Optional[Dict[str, Any]] = None,
        now: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> None:
        """Hand one frame to the shadow. Never blocks, never raises."""
        if not self.enabled or self._pipeline is None:
            return
        try:
            self._frame_counter += 1
            if legacy_stats:
                self._legacy = dict(legacy_stats)
                self._legacy["fps"] = fps
            if self._frame_counter % max(1, self.stride):
                return
            if self._queue.full():
                self._dropped += 1
                return
            # The copy happens only for frames that will actually be processed,
            # and only after the queue check — a full queue costs nothing.
            self._queue.put_nowait({
                "frame": frame.copy() if frame is not None else None,
                "detections": [dict(d) for d in detections],
                "now": now or time.time(),
                "fps": fps,
            })
        except Exception as exc:
            self._note_error(exc)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job.get("_stop"):
                break
            try:
                started = time.perf_counter()
                self._process(job)
                elapsed = time.perf_counter() - started
                self._durations.append(elapsed)
                self._processed += 1
                self._consecutive_errors = 0
                self._autothrottle(elapsed)
            except Exception as exc:
                self._note_error(exc)

    def _process(self, job: Dict[str, Any]) -> None:
        telemetry = self._pipeline.process(job["frame"], job["detections"], now=job["now"])
        decisions = getattr(self._pipeline, "last_decisions", {}) or {}

        for decision in decisions.values():
            match = getattr(decision, "match", None)
            if match is None:
                continue
            self._verdicts[match.verdict.value] += 1
            self._score_hist[_bucket(match.score)] += 1

        with self._lock:
            self._latest = telemetry

        if job["now"] - self._last_log >= LOG_INTERVAL_SECONDS:
            self._last_log = job["now"]
            self._write_log()

    def _autothrottle(self, elapsed: float) -> None:
        """Never be the reason the machine is busy."""
        if elapsed > FRAME_BUDGET_SECONDS and self.stride < MAX_STRIDE:
            self.stride = min(MAX_STRIDE, self.stride + 1)
            print(f"[SHADOW] frame cost {elapsed*1000:.0f}ms > budget — "
                  f"stride raised to {self.stride}")
        elif (elapsed < FRAME_BUDGET_SECONDS / 3 and self.stride > DEFAULT_STRIDE
              and len(self._durations) >= 50):
            self.stride -= 1

    def _note_error(self, exc: Exception) -> None:
        self._errors += 1
        self._consecutive_errors += 1
        if self._consecutive_errors == 1:
            print(f"[SHADOW] error (engine unaffected): {exc.__class__.__name__}: {exc}")
        if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            self._disable(f"{self._consecutive_errors} consecutive errors")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_log(self, final: bool = False) -> None:
        row = self.comparison()
        row["final"] = final
        path = self.log_dir / f"{time.strftime('%Y-%m-%d')}.jsonl"
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._log_path = path
        except Exception as exc:
            self._note_error(exc)

    def comparison(self) -> Dict[str, Any]:
        """One paired sample of old counters vs new ones."""
        with self._lock:
            shadow = dict(self._latest)
        legacy = dict(self._legacy)

        def diff(key: str) -> Optional[float]:
            a, b = legacy.get(key), shadow.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return round(b - a, 2)
            return None

        durations = list(self._durations)
        durations.sort()
        return {
            "t": round(time.time(), 2),
            "camera_id": self.camera_id,
            "uptime_s": round(time.time() - self._started_at, 1),
            "legacy": {k: legacy.get(k) for k in
                       ("tam", "sam", "som", "inside", "average_stay_time",
                        "conversion_rate", "fps")},
            "shadow": {k: shadow.get(k) for k in
                       ("tam", "sam", "som", "inside", "average_stay_time",
                        "conversion_rate", "visitors_today", "currently_inside",
                        "returning_visitors")},
            "delta": {k: diff(k) for k in ("tam", "sam", "som", "inside")},
            "verdicts": dict(self._verdicts),
            "score_histogram": dict(sorted(self._score_hist.items())),
            "cost": {
                "processed": self._processed,
                "dropped": self._dropped,
                "stride": self.stride,
                "p50_ms": round(durations[len(durations) // 2] * 1000, 1) if durations else None,
                "p95_ms": round(durations[int(len(durations) * 0.95)] * 1000, 1) if durations else None,
            },
            "reid": self._reid_diagnostics(),
            "errors": self._errors,
        }

    def _reid_diagnostics(self) -> Dict[str, Any]:
        try:
            return self._pipeline.diagnostics()["identity"]["reid"]
        except Exception:
            return {}

    def telemetry(self) -> Dict[str, Any]:
        """``shadow_*`` keys to merge into the existing payload.

        Namespaced so a dashboard that has never heard of Phase 0 renders
        exactly as it does today.
        """
        if not self.enabled:
            return {"shadow_active": False,
                    "shadow_disabled_reason": self._disabled_reason}
        with self._lock:
            shadow = dict(self._latest)
        durations = list(self._durations)

        promoted: Dict[str, Any] = {}
        if self.live and shadow.get("tam") is not None:
            # Promote the identity numbers into the canonical keys. Guarded on
            # having a real reading: an engine that has not seen a frame yet
            # must not blank the dashboard with zeros.
            for key in ("tam", "sam", "som", "inside", "average_stay_time",
                        "conversion_rate", "people_today"):
                if shadow.get(key) is not None:
                    promoted[key] = shadow[key]
            promoted["counted_by"] = "global_person"

        return {
            **promoted,
            "shadow_active": True,
            "shadow_live": self.live,
            "shadow_tam": shadow.get("tam"),
            "shadow_sam": shadow.get("sam"),
            "shadow_som": shadow.get("som"),
            "shadow_inside": shadow.get("inside"),
            "shadow_visitors_today": shadow.get("visitors_today"),
            "shadow_returning": shadow.get("returning_visitors"),
            "shadow_backend": shadow.get("identity_backend"),
            "shadow_stride": self.stride,
            "shadow_cost_ms": round(sum(durations) / len(durations) * 1000, 1) if durations else None,
            "shadow_dropped": self._dropped,
            "shadow_errors": self._errors,
        }


def _bucket(score: float) -> str:
    for low, high in zip(SCORE_BUCKETS, SCORE_BUCKETS[1:]):
        if low <= score < high:
            return f"{low:.2f}-{high:.2f}"
    return "other"