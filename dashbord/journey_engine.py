"""
vision/identity/journey_engine.py
---------------------------------
The ordered path of zones each PERSON walked, with per-step dwell,
aggregated into flows and funnels.

An EventSink: it folds ZONE_ENTERED / ZONE_LEFT / IDENTITY_MERGED events off
the bus. Because it is keyed by global id, a person whose track flickered
ten times still has one clean journey — which is the whole reason journeys
became possible at all.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

from events import EventType, IdentityEvent


class JourneyEngine:
    def __init__(self) -> None:
        self.paths: Dict[str, List[str]] = defaultdict(list)
        self._flows: Counter = Counter()          # (from, to) -> n
        self.zone_totals: Counter = Counter()
        # (gid, zone) -> accumulated dwell seconds (from ZONE_LEFT events)
        self._dwell: Dict[tuple, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # EventSink protocol
    # ------------------------------------------------------------------

    def emit(self, event: IdentityEvent) -> None:
        if event.type is EventType.IDENTITY_MERGED:
            self._merge(event.global_id, event.data.get("absorbed"))
            return
        if not event.zone or event.zone.startswith("__"):
            return                                # reserved zones: plumbing
        gid = event.global_id
        if event.type is EventType.ZONE_ENTERED:
            path = self.paths[gid]
            if path and path[-1] == event.zone:
                return
            if path:
                self._flows[(path[-1], event.zone)] += 1
            path.append(event.zone)
            self.zone_totals[event.zone] += 1
        elif event.type is EventType.ZONE_LEFT:
            self._dwell[(gid, event.zone)] += float(
                event.data.get("dwell_seconds", 0.0))

    def _merge(self, keep: str, absorbed: str) -> None:
        """Two ids proven to be one person: their journeys become one path,
        in visit order (the keeper is always the OLDER identity)."""
        if not absorbed or absorbed == keep or absorbed not in self.paths:
            return
        tail = self.paths.pop(absorbed)
        path = self.paths[keep]
        if path and tail and path[-1] != tail[0]:
            self._flows[(path[-1], tail[0])] += 1
        for zone in tail:
            if not path or path[-1] != zone:
                path.append(zone)
        for (gid, zone), sec in [(k, v) for k, v in self._dwell.items()
                                 if k[0] == absorbed]:
            self._dwell[(keep, zone)] += sec
            del self._dwell[(gid, zone)]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def path_of(self, global_id: str) -> List[str]:
        return list(self.paths.get(global_id, []))

    # Back-compat alias
    journey_of = path_of

    def path_with_dwell(self, global_id: str) -> List[Dict[str, Any]]:
        return [{"zone": z, "dwell": round(self._dwell.get((global_id, z), 0.0), 2)}
                for z in self.paths.get(global_id, [])]

    def flows(self) -> List[Dict[str, Any]]:
        return [{"from": a, "to": b, "count": c}
                for (a, b), c in self._flows.most_common()]

    def top_flows(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.flows()[:n]

    def top_zones(self, n: int = 8):
        return self.zone_totals.most_common(n)

    def funnel(self, steps: List[str]) -> List[Dict[str, Any]]:
        """How many DISTINCT people reached each named step, in order."""
        return [{"step": step,
                 "people": sum(1 for path in self.paths.values() if step in path)}
                for step in steps]

    def reset(self) -> None:
        self.paths.clear()
        self._flows.clear()
        self.zone_totals.clear()
        self._dwell.clear()
