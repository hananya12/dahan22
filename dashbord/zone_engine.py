"""
vision/identity/zone_engine.py
------------------------------
Per-IDENTITY zone presence and dwell. Every method takes a PersonProfile,
never a track id — a tracker gap is not a zone exit, so a person who is
briefly lost keeps their open dwell interval and resumes it on reappearing.

The dwell clock itself lives ON the PersonProfile (``zone_dwell_banked`` /
``zone_entered_at``): this engine only decides WHEN it starts and stops.
Keeping the state on the person is what lets ``memory.merge()`` reconcile
dwell when two ids are proven to be one human.

Two reserved zone names carry the retail semantics:
    TAM_ZONE  — the store floor polygon (the "Total" zone)
    SOM_ZONE  — the entrance, when it is drawn as a polygon
Named zones ("Vegetables", "Checkout", ...) drive the Journey feature.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from config import ZoneConfig
from events import EventBus, EventType, IdentityEvent
from geometry import point_in_polygon
from person_profile import PersonProfile

Point = Tuple[float, float]

TAM_ZONE = "__tam__"
SOM_ZONE = "__som__"

# Reserved names are internal plumbing; they never appear in Journey output.
_RESERVED = {TAM_ZONE, SOM_ZONE}


class ZoneEngine:
    """Answers, per person: which zones are you in, and for how long."""

    def __init__(self, config: ZoneConfig, camera_id: str = "cam_1",
                 bus: Optional[EventBus] = None) -> None:
        self.config = config
        self.camera_id = camera_id
        self.bus = bus or EventBus()
        self._zones: Dict[str, List[Point]] = {}
        # Hysteresis counters for pending ENTRIES: (gid, zone) -> frames.
        self._pending: Dict[Tuple[str, str], int] = {}
        # Last frame each person was seen inside each zone, for the grace
        # period: (gid, zone) -> timestamp.
        self._last_inside: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------

    def configure(self, zones: Dict[str, Sequence[Sequence[float]]],
                  replace: bool = False) -> None:
        if replace:
            self._zones.clear()
        for name, pts in (zones or {}).items():
            poly = [(float(x), float(y)) for x, y in pts]
            if len(poly) >= 3:
                self._zones[name] = poly

    def public_zone_names(self) -> List[str]:
        return [z for z in self._zones if z not in _RESERVED]

    def has_zone(self, name: str) -> bool:
        return name in self._zones

    # ------------------------------------------------------------------

    def update(self, profile: PersonProfile, foot: Point, now: float) -> None:
        gid = profile.global_id
        for zone, poly in self._zones.items():
            key = (gid, zone)
            inside = point_in_polygon(foot[0], foot[1], poly)

            if inside:
                self._last_inside[key] = now
                if zone in profile.zone_entered_at:
                    continue                       # interval already open
                # Hysteresis on ENTRY only: one noisy frame on a boundary
                # must not open a visit (and emit a Journey step) by itself.
                frames = self._pending.get(key, 0) + 1
                if frames >= max(1, self.config.confirm_frames):
                    self._pending.pop(key, None)
                    profile.zone_entered_at[zone] = now
                    if zone not in _RESERVED:
                        profile.record_zone(zone, now)
                    self._emit(EventType.ZONE_ENTERED, profile, zone, foot, now)
                else:
                    self._pending[key] = frames
            else:
                self._pending.pop(key, None)
                if (zone in profile.zone_entered_at and
                        now - self._last_inside.get(key, now)
                        > self.config.presence_grace_seconds):
                    self._close(profile, zone, now, foot)

    def suspend(self, profile: PersonProfile, now: float) -> bool:
        """Called while the person is not visible. Banks nothing until the
        grace period elapses — a tracker gap is not an exit. Returns True
        once every open interval has been banked, so the person can leave
        the live watch list. Banking at ``last_inside`` (not ``now``) keeps
        the invisible seconds out of the dwell number."""
        gid = profile.global_id
        if not profile.zone_entered_at:
            return True
        done = True
        for zone in list(profile.zone_entered_at):
            last = self._last_inside.get((gid, zone), profile.last_seen)
            if now - last > self.config.presence_grace_seconds:
                self._close(profile, zone, last, None)
            else:
                done = False
        return done

    def release(self, profile: PersonProfile, now: float) -> None:
        """Identity expired: close everything it still holds open."""
        gid = profile.global_id
        for zone in list(profile.zone_entered_at):
            last = self._last_inside.get((gid, zone), now)
            self._close(profile, zone, min(last, now), None)
        for key in [k for k in self._pending if k[0] == gid]:
            self._pending.pop(key, None)
        for key in [k for k in self._last_inside if k[0] == gid]:
            self._last_inside.pop(key, None)

    # ------------------------------------------------------------------

    def dwell_seconds(self, global_id_or_profile, zone: str,
                      now: Optional[float] = None) -> float:
        """Convenience passthrough; the truth lives on the profile."""
        profile = global_id_or_profile
        if isinstance(profile, PersonProfile):
            return profile.zone_dwell(zone, now)
        return 0.0

    def is_inside(self, profile: PersonProfile, zone: str) -> bool:
        return zone in profile.zone_entered_at

    # ------------------------------------------------------------------

    def _close(self, profile: PersonProfile, zone: str, at: float,
               foot: Optional[Point]) -> None:
        entered = profile.zone_entered_at.pop(zone, None)
        if entered is None:
            return
        interval = max(0.0, at - entered)
        profile.zone_dwell_banked[zone] = (
            profile.zone_dwell_banked.get(zone, 0.0) + interval)
        self._emit(EventType.ZONE_LEFT, profile, zone, foot, at,
                   data={"dwell_seconds": round(interval, 2),
                         "total_dwell_seconds": round(
                             profile.zone_dwell_banked[zone], 2)})

    def _emit(self, etype: EventType, profile: PersonProfile, zone: str,
              foot: Optional[Point], now: float, data: Optional[dict] = None) -> None:
        self.bus.emit(IdentityEvent(
            type=etype, global_id=profile.global_id, camera_id=self.camera_id,
            timestamp=now, zone=zone, position=foot, data=data or {}))
