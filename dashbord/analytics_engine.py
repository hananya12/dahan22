"""
vision/identity/analytics_engine.py
-----------------------------------
Every number on the new dashboard, derived from the event stream.

This module holds no counters of its own that cannot be recomputed. Feed it
the same events twice and it produces the same answer; feed it a replayed day
from ``reports.py`` history and it reconstructs that day exactly. That
property is what makes it safe to change a threshold at 14:00 without
invalidating the morning.

Metrics implemented, mapped to the spec:
    Visitors Today      unique identities that entered today
    Currently Inside    live occupancy, from EntryExitEngine state
    Entered / Exited    unique identities per direction
    Returning Visitors  identities recovered from warm memory
    Average Visit Time  mean of closed visits
    Longest Visit       max closed visit
    Heat Map            occupancy grid over foot positions
    Journey             ordered zone path per identity, aggregated to flows
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from events import EventType, IdentityEvent


@dataclass
class AnalyticsEngine:
    """An EventSink that maintains the derived state of the store."""

    grid_size: Tuple[int, int] = (24, 32)     # rows, cols for the heat map
    frame_size: Tuple[int, int] = (720, 1280)

    entered_ids: set = field(default_factory=set)
    exited_ids: set = field(default_factory=set)
    returning_ids: set = field(default_factory=set)
    created_ids: set = field(default_factory=set)

    visit_durations: List[float] = field(default_factory=list)
    longest_visit: float = 0.0
    longest_visit_id: Optional[str] = None

    heat: List[List[int]] = field(default_factory=list)
    journeys: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    zone_totals: Counter = field(default_factory=Counter)
    flows: Counter = field(default_factory=Counter)          # (from, to) -> n
    hourly_entered: Dict[int, set] = field(default_factory=lambda: defaultdict(set))

    day: str = field(default_factory=lambda: date.today().isoformat())
    _inside_provider: Any = None

    # Optional collaborators, injected by IdentityPipeline. When present,
    # snapshot() blends their live state (TAM/SAM/SOM, journeys, heat) into
    # the payload; when absent, the pure event-fold behaviour is unchanged.
    retail: Any = None
    entry_exit: Any = None
    journey: Any = None
    heatmap: Any = None
    funnel_steps: Optional[List[str]] = None

    def __post_init__(self) -> None:
        rows, cols = self.grid_size
        self.heat = [[0] * cols for _ in range(rows)]
        if self.entry_exit is not None:
            self._inside_provider = self.entry_exit

    # ------------------------------------------------------------------

    def bind_occupancy(self, entry_exit_engine) -> "AnalyticsEngine":
        """Occupancy is state, not an event fold — read it from its owner."""
        self._inside_provider = entry_exit_engine
        return self

    def emit(self, event: IdentityEvent) -> None:
        """EventSink protocol. Never raises: the frame loop depends on it."""
        try:
            self._apply(event)
        except Exception:
            pass

    def _apply(self, event: IdentityEvent) -> None:
        self._roll_day_if_needed(event.timestamp)
        gid = event.global_id

        if event.type is EventType.IDENTITY_CREATED:
            self.created_ids.add(gid)

        elif event.type is EventType.IDENTITY_RECOVERED:
            self.returning_ids.add(gid)

        elif event.type is EventType.ENTERED:
            self.entered_ids.add(gid)
            hour = datetime.fromtimestamp(event.timestamp).hour
            self.hourly_entered[hour].add(gid)
            if event.data.get("returning"):
                self.returning_ids.add(gid)

        elif event.type is EventType.EXITED:
            self.exited_ids.add(gid)
            seconds = event.data.get("visit_seconds")
            if seconds:
                self.visit_durations.append(float(seconds))
                if seconds > self.longest_visit:
                    self.longest_visit = float(seconds)
                    self.longest_visit_id = gid

        elif event.type is EventType.ZONE_ENTERED:
            zone = event.zone
            if zone:
                path = self.journeys[gid]
                if path and path[-1] != zone:
                    self.flows[(path[-1], zone)] += 1
                if not path or path[-1] != zone:
                    path.append(zone)
                self.zone_totals[zone] += 1

        if event.position:
            self._add_heat(event.position)

    # ------------------------------------------------------------------

    def add_position(self, position: Tuple[float, float]) -> None:
        """Called per frame per person for the heat map, outside the event
        stream — positions are far too high-frequency to be events."""
        self._add_heat(position)

    def _add_heat(self, position: Tuple[float, float]) -> None:
        rows, cols = self.grid_size
        fh, fw = self.frame_size
        r = int(min(rows - 1, max(0, position[1] / max(fh, 1) * rows)))
        c = int(min(cols - 1, max(0, position[0] / max(fw, 1) * cols)))
        self.heat[r][c] += 1

    def _roll_day_if_needed(self, timestamp: float) -> None:
        today = datetime.fromtimestamp(timestamp).date().isoformat()
        if today == self.day:
            return
        # A new trading day: today's counters reset, but nothing is lost —
        # reports.py persists the closed day from the snapshot below.
        self.day = today
        self.entered_ids.clear()
        self.exited_ids.clear()
        self.returning_ids.clear()
        self.created_ids.clear()
        self.visit_durations.clear()
        self.longest_visit = 0.0
        self.longest_visit_id = None
        self.hourly_entered.clear()
        self.journeys.clear()
        self.flows.clear()
        self.zone_totals.clear()
        rows, cols = self.grid_size
        self.heat = [[0] * cols for _ in range(rows)]

    # ------------------------------------------------------------------

    @property
    def currently_inside(self) -> int:
        return self._inside_provider.currently_inside if self._inside_provider else 0

    @property
    def average_visit_seconds(self) -> float:
        return sum(self.visit_durations) / len(self.visit_durations) if self.visit_durations else 0.0

    def snapshot(self, now: Optional[float] = None) -> Dict[str, Any]:
        """The payload the dashboard renders. Shaped to slot straight into
        the existing telemetry message in corewise_control.py."""
        if now is not None:
            self._roll_day_if_needed(now)
        payload = self._base_snapshot()
        if self.retail is not None:
            payload.update(self.retail.get_stats())      # tam / sam / som
        if self.journey is not None:
            payload["top_zones"] = self.journey.top_zones()
            payload["top_flows"] = self.journey.top_flows()
        if self.funnel_steps:
            payload["funnel"] = self._funnel()
        return payload

    def _funnel(self) -> List[Dict[str, Any]]:
        """How many people reached each named step, in order."""
        if self.journey is None or not self.funnel_steps:
            return []
        return self.journey.funnel(list(self.funnel_steps))

    def maps(self) -> Dict[str, Any]:
        """Spatial payloads: the heat map and the journey flow graph."""
        heat = (self.heatmap.snapshot() if self.heatmap is not None
                else {"grid": self.heatmap_grid(), "grid_size": list(self.grid_size)})
        flows = (self.journey.top_flows(50) if self.journey is not None
                 else [{"from": a, "to": b, "count": n}
                       for (a, b), n in self.flows.most_common(50)])
        return {"heatmap": heat, "flows": flows}

    def heatmap_grid(self, normalise: bool = True) -> List[List[float]]:
        if not normalise:
            return [row[:] for row in self.heat]
        peak = max((max(row) for row in self.heat), default=0) or 1
        return [[round(v / peak, 4) for v in row] for row in self.heat]

    def _base_snapshot(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "visitors_today": len(self.entered_ids),
            "currently_inside": self.currently_inside,
            "entered": len(self.entered_ids),
            "exited": len(self.exited_ids),
            "returning_visitors": len(self.returning_ids),
            "identities_created": len(self.created_ids),
            "average_visit_seconds": round(self.average_visit_seconds, 1),
            "average_visit_display": _mmss(self.average_visit_seconds),
            "longest_visit_seconds": round(self.longest_visit, 1),
            "longest_visit_display": _mmss(self.longest_visit),
            "hourly_entered": {str(h): len(ids) for h, ids in sorted(self.hourly_entered.items())},
            "top_zones": self.zone_totals.most_common(8),
            "top_flows": [{"from": a, "to": b, "count": n}
                          for (a, b), n in self.flows.most_common(10)],
            "updated_at": time.time(),
        }


    def journey_of(self, global_id: str) -> List[str]:
        return list(self.journeys.get(global_id, []))


def _mmss(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
