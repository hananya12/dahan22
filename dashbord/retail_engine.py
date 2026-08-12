"""
vision/identity/retail_engine.py
--------------------------------
TAM / SAM / SOM — counted over GLOBAL IDENTITIES, not tracker ids.

This is the module that fixes the headline bug: five people walking in
together are five committed identities, so TAM goes up by five. And a person
whose tracker id churns three times near the entrance is still ONE member of
the ``_tam_ids`` set, so no cooldown heuristic is needed at all.

The per-person flags live on ``profile.retail`` (so ``memory.merge`` can
reconcile them); the sets here are the fast aggregate the dashboard reads.
On IDENTITY_MERGED the sets are re-pointed at the surviving id, which is how
"two ids proven to be one person" pulls TAM back down by one.

Semantics (identical to tracking.py, but per person):
    TAM  — the person entered the store floor (the TAM polygon).
    SAM  — the person qualified as a "serious" visitor:
             * if an SOM *polygon* / SAM area exists: they entered it, or
             * dwell-based: they stayed inside TAM at least ``min_stay``
               seconds. Until the dashboard sends a value, dwell-SAM is OFF
               (single source of truth preserved from tracking.py).
    SOM  — the person crossed the entrance line in the entry direction
           (or entered the SOM polygon when it is an area).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from config import RetailConfig, ZoneConfig
from events import EventBus, EventType, IdentityEvent
from geometry import crossing_direction
from person_profile import PersonProfile
from zone_engine import SOM_ZONE, TAM_ZONE, ZoneEngine

Point = Tuple[float, float]


class RetailEngine:
    """Folds per-person positions into the three business counters."""

    def __init__(self, config: RetailConfig, zone_config: ZoneConfig,
                 zones: ZoneEngine, camera_id: str = "cam_1",
                 bus: Optional[EventBus] = None) -> None:
        self.config = config
        self.zone_config = zone_config
        self.zones = zones
        self.camera_id = camera_id
        self.bus = bus or EventBus()

        self._som_line: Optional[List[Point]] = None
        self._som_direction: int = +1
        self._last_pos: Dict[str, Point] = {}

        # Sets of global ids. len() of a set IS the deduplicated count.
        self._tam_ids: Set[str] = set()
        self._sam_ids: Set[str] = set()
        self._som_ids: Set[str] = set()

    # ------------------------------------------------------------------
    # Configuration (dashboard-driven, mirrors tracking.py contracts)
    # ------------------------------------------------------------------

    def set_som_line(self, points: Optional[Sequence[Sequence[float]]],
                     direction: int = +1) -> None:
        if points and len(points) >= 2:
            self._som_line = [(float(x), float(y)) for x, y in points]
            self._som_direction = +1 if direction >= 0 else -1
        else:
            self._som_line = None

    def set_min_stay(self, enabled: bool, seconds: Optional[float]) -> None:
        self.config.sam_min_stay_enabled = bool(enabled)
        self.config.sam_min_stay_seconds = (
            float(seconds) if seconds is not None else None)
        self.config.sam_configured = True

    # ------------------------------------------------------------------
    # EventSink: merges must reconcile the aggregate sets
    # ------------------------------------------------------------------

    def emit(self, event: IdentityEvent) -> None:
        if event.type is not EventType.IDENTITY_MERGED:
            return
        keep = event.global_id
        absorbed = event.data.get("absorbed")
        if not absorbed:
            return
        for id_set, flag in ((self._tam_ids, "counted_tam"),
                             (self._sam_ids, "counted_sam"),
                             (self._som_ids, "counted_som")):
            if absorbed in id_set:
                id_set.discard(absorbed)
                id_set.add(keep)
        self._last_pos.pop(absorbed, None)

    # ------------------------------------------------------------------
    # Per frame, per person
    # ------------------------------------------------------------------

    def update(self, profile: PersonProfile, foot: Point, now: float) -> None:
        gid = profile.global_id

        # --- TAM: first confirmed presence on the store floor -------------
        if not profile.retail.counted_tam and self.zones.is_inside(profile, TAM_ZONE):
            profile.retail.counted_tam = True
            self._tam_ids.add(gid)
            profile.open_visit(now)
            self._emit(EventType.ENTERED, profile, foot, now,
                       data={"returning": profile.is_returning})

        # --- SAM ----------------------------------------------------------
        if not profile.retail.counted_sam:
            if self.zones.has_zone(SOM_ZONE):
                # An SOM polygon / SAM area was drawn: SAM is entry into it.
                if self.zones.is_inside(profile, SOM_ZONE):
                    self._mark_sam(profile, foot, now, "entered the SAM area")
                else:
                    profile.retail.sam_reason = "has not entered the SAM area"
            elif not self.config.sam_configured:
                profile.retail.sam_reason = (
                    "withheld: dashboard has not configured SAM minimum stay")
            elif (self.config.sam_min_stay_enabled
                  and self.config.sam_min_stay_seconds is not None):
                if not profile.retail.counted_tam:
                    profile.retail.sam_reason = "not inside TAM yet"
                else:
                    dwell = profile.zone_dwell(TAM_ZONE, now)
                    if dwell >= self.config.sam_min_stay_seconds:
                        self._mark_sam(profile, foot, now,
                                       f"dwell {dwell:.1f}s >= threshold")
                    else:
                        profile.retail.sam_reason = (
                            f"dwell {dwell:.1f}s below "
                            f"{self.config.sam_min_stay_seconds}s threshold")
            elif profile.retail.counted_tam:
                # Minimum stay explicitly disabled: SAM on TAM entry.
                self._mark_sam(profile, foot, now, "min-stay disabled")

        # --- SOM: entrance line crossing ----------------------------------
        if not profile.retail.counted_som and self._som_line is not None:
            prev = self._last_pos.get(gid)
            if prev is not None:
                if crossing_direction(prev, foot, self._som_line) == self._som_direction:
                    profile.retail.counted_som = True
                    self._som_ids.add(gid)
                    self._emit(EventType.DWELL_MILESTONE, profile, foot, now,
                               data={"milestone": "som"})
        self._last_pos[gid] = foot

    def release(self, profile: PersonProfile, now: float) -> None:
        """Identity expired: close its visit and bank the duration."""
        self._last_pos.pop(profile.global_id, None)
        if profile.retail.counted_tam:
            visit = profile.close_visit(now)
            self._emit(EventType.EXITED, profile, None, now, data={
                "visit_seconds": round(visit.duration, 1) if visit else 0.0})

    # ------------------------------------------------------------------

    def _mark_sam(self, profile: PersonProfile, foot: Point, now: float,
                  how: str) -> None:
        profile.retail.counted_sam = True
        profile.retail.sam_reason = how
        self._sam_ids.add(profile.global_id)
        self._emit(EventType.DWELL_MILESTONE, profile, foot, now,
                   data={"milestone": "sam", "how": how})

    def _emit(self, etype: EventType, profile: PersonProfile,
              foot: Optional[Point], now: float, data: Optional[dict] = None) -> None:
        self.bus.emit(IdentityEvent(
            type=etype, global_id=profile.global_id, camera_id=self.camera_id,
            timestamp=now, position=foot, data=data or {}))

    # ------------------------------------------------------------------
    # Output — key-for-key compatible with PersonTracker.get_stats()
    # ------------------------------------------------------------------

    @property
    def tam_count(self) -> int:
        return len(self._tam_ids)

    @property
    def sam_count(self) -> int:
        return len(self._sam_ids)

    @property
    def som_count(self) -> int:
        return len(self._som_ids)

    def counted(self, profile: PersonProfile) -> Dict[str, bool]:
        return {"tam": profile.retail.counted_tam,
                "sam": profile.retail.counted_sam,
                "som": profile.retail.counted_som}

    def get_stats(self) -> Dict[str, float]:
        tam = self.tam_count
        return {
            "tam": tam,
            "sam": self.sam_count,
            "som": self.som_count,
            "conversion_rate": round(100.0 * self.som_count / tam, 1) if tam else 0.0,
            "som_shape": ("polygon" if self.zones.has_zone(SOM_ZONE)
                          else "polyline" if self._som_line else "none"),
        }

    def reset_counters(self) -> None:
        self._tam_ids.clear()
        self._sam_ids.clear()
        self._som_ids.clear()
        self._last_pos.clear()
