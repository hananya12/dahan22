"""
vision/identity/events.py
-------------------------
The Identity Engine does not increment counters. It emits facts.

Why this matters more than it looks
-----------------------------------
Today's engine holds ``tam_count``, ``som_count``, ``_tam_dedup_count`` as
mutable integers. That design has three costs the current codebase is
already paying:

  1. A metric can never be recomputed. If the SAM threshold changes at 14:00,
     the morning's numbers are wrong forever.
  2. A new metric (Returning Visitors, Journey, Heatmap) requires new
     plumbing through the engine, the websocket, the dashboard and history.
  3. Nothing can be audited. "Why is Entered 154?" has no answer.

With an event log, every metric in the spec — Visitors Today, Currently
Inside, Entered, Exited, Returning, Avg Visit, Longest Visit, Heatmap,
Journey — is a *fold over the same immutable stream*. New metrics cost a
function, not a migration. Reports can be rebuilt from history. And the
existing ``reports.py`` / ``replay_buffer.py`` event track become consumers
of this stream instead of parallel implementations of it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    IDENTITY_CREATED = "identity_created"
    IDENTITY_RECOVERED = "identity_recovered"   # returning visitor (warm hit)
    IDENTITY_MERGED = "identity_merged"         # two ids proven to be one
    IDENTITY_EXPIRED = "identity_expired"
    ENTERED = "entered"
    EXITED = "exited"
    ZONE_ENTERED = "zone_entered"
    ZONE_LEFT = "zone_left"
    DWELL_MILESTONE = "dwell_milestone"         # e.g. crossed the SAM threshold


@dataclass(frozen=True)
class IdentityEvent:
    type: EventType
    global_id: str
    camera_id: str
    timestamp: float = field(default_factory=time.time)
    track_id: Optional[int] = None
    zone: Optional[str] = None
    position: Optional[tuple] = None            # foot point, pixels
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d


class EventBus:
    """Fan-out to sinks, with a bounded in-memory tail for late subscribers.

    Deliberately synchronous and tiny: it runs inside the frame loop, so it
    must never block. A sink that raises is dropped from the frame, never
    propagated — a broken dashboard connection must not kill the engine.
    """

    def __init__(self, tail_size: int = 2000) -> None:
        self._sinks: List[Any] = []
        self._tail: List[IdentityEvent] = []
        self._tail_size = tail_size
        self._counts: Dict[EventType, int] = {}

    def subscribe(self, sink: Any) -> "EventBus":
        self._sinks.append(sink)
        return self

    def emit(self, event: IdentityEvent) -> None:
        self._tail.append(event)
        if len(self._tail) > self._tail_size:
            del self._tail[: len(self._tail) - self._tail_size]
        self._counts[event.type] = self._counts.get(event.type, 0) + 1

        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:
                continue

    def tail(self, since: float = 0.0) -> List[IdentityEvent]:
        return [e for e in self._tail if e.timestamp >= since]

    def count(self, event_type: EventType) -> int:
        return self._counts.get(event_type, 0)
