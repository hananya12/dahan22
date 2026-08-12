"""
vision/pipeline.py
------------------
One object that wires the whole identity-first system together, and speaks
the data format ``main.py`` already produces.

Every engine downstream of the tracker receives a ``PersonProfile``. The
only place a ``track_id`` exists at all is the binding table inside
``IdentityManager`` and the conversion below — after that, the system knows
only people.

Integration into the existing engine:

    from vision.pipeline import IdentityPipeline

    # once, replacing self.tracker:
    self.identity = IdentityPipeline(camera_id=rp.default_camera_id())

    # in _apply_calibration, with the SAME pixel points already computed:
    self.identity.configure(tam_polygon_px=self.tam_zone_np.tolist(),
                            som_points_px=self.som_points_px,
                            som_is_polygon=self.som_is_polygon,
                            named_zones_px={"Vegetables": [...], "Checkout": [...]})

    # in _apply_sam_min_stay:
    self.identity.set_min_stay(enabled, seconds)

    # once per frame, after self.detector.track_people(frame):
    stats = self.identity.process(frame, detections)

``detections`` is exactly what ``detection.PersonDetector.track_people``
returns today. ``stats`` is key-for-key compatible with
``PersonTracker.get_stats()``, plus the new identity metrics — so the
websocket protocol, ``app.py`` and ``reports.py`` need no changes.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from analytics_engine import AnalyticsEngine
from config import IdentityConfig
from contracts import TrackObservation
from entry_exit_engine import EntryExitEngine
from events import EventBus
from heatmap_engine import HeatmapEngine
from identity_manager import IdentityManager
from journey_engine import JourneyEngine
from memory_engine import MemoryEngine
from person_profile import PersonProfile
from reid_engine import ReIDEngine
from retail_engine import RetailEngine
from zone_engine import SOM_ZONE, TAM_ZONE, ZoneEngine

Point = Tuple[float, float]


class IdentityPipeline:
    """Detector output in, retail analytics out — with people in between."""

    def __init__(
        self,
        camera_id: str = "cam_1",
        config: Optional[IdentityConfig] = None,
        backend: Optional[Any] = None,
        storage_dir=None,
        funnel_steps: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = config or IdentityConfig(camera_id=camera_id)
        self.config.camera_id = camera_id

        self.bus = EventBus()

        # --- identity layer -------------------------------------------------
        self.reid = ReIDEngine(self.config.reid, camera_id=camera_id, backend=backend)
        self.memory = MemoryEngine(self.config.memory, camera_id, self.bus,
                                   storage_dir=storage_dir)
        self.identity = IdentityManager(self.config, bus=self.bus,
                                        reid=self.reid, memory=self.memory)

        # --- everything below speaks PersonProfile only ---------------------
        self.zones = ZoneEngine(self.config.zones, camera_id, self.bus)
        self.retail = RetailEngine(self.config.retail, self.config.zones,
                                   self.zones, camera_id, self.bus)
        self.entry_exit = EntryExitEngine(self.config.zones, camera_id, self.bus)
        self.journey = JourneyEngine()
        self.heatmap = HeatmapEngine()
        self.analytics = AnalyticsEngine(retail=self.retail, entry_exit=self.entry_exit,
                                         journey=self.journey, heatmap=self.heatmap,
                                         funnel_steps=funnel_steps)

        for sink in (self.retail, self.journey, self.heatmap, self.analytics):
            self.bus.subscribe(sink)

        # Closing an identity must close its zone visits and bank its dwell,
        # or a long-staying customer who expires simply vanishes from the
        # average instead of contributing their (large) sample to it.
        self.memory.on_release(self._on_identity_released)

        self._frame_index = 0
        self._last_save = time.time()
        # Last frame's identity decisions, exposed for observability (the
        # shadow runner records score distributions and the ambiguity rate
        # from here). Read-only by convention.
        self.last_decisions: Dict[int, Any] = {}
        self._seen_this_frame: set[str] = set()
        self._live_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def configure(
        self,
        tam_polygon_px: Optional[Sequence[Sequence[float]]] = None,
        som_points_px: Optional[Sequence[Sequence[float]]] = None,
        som_is_polygon: bool = False,
        som_direction: int = +1,
        named_zones_px: Optional[Dict[str, Sequence[Sequence[float]]]] = None,
        store_polygon_px: Optional[Sequence[Sequence[float]]] = None,
        entry_line_px: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """Apply calibration in PIXELS.

        ``named_zones_px`` is what turns Journey from a diagram into a
        feature: {"Vegetables": [...], "Drinks": [...], "Checkout": [...]}.
        Without it everything else still works; the journey is simply empty.
        """
        zones: Dict[str, Sequence[Sequence[float]]] = {}
        if tam_polygon_px:
            zones[TAM_ZONE] = tam_polygon_px
        if som_points_px and som_is_polygon:
            zones[SOM_ZONE] = som_points_px
        if named_zones_px:
            zones.update(named_zones_px)
        self.zones.configure(zones, replace=True)

        if som_points_px and not som_is_polygon:
            self.retail.set_som_line(som_points_px, som_direction)
        else:
            self.retail.set_som_line(None)

        # Store occupancy: the store outline if given, else the TAM zone,
        # else the entrance line. One of the three always exists once a
        # camera has been calibrated at all.
        self.entry_exit.configure(
            inside_polygon=store_polygon_px or tam_polygon_px,
            entry_line=entry_line_px or (som_points_px if not som_is_polygon else None),
            entry_direction=som_direction,
        )

    def set_min_stay(self, enabled: bool, seconds: Optional[float]) -> None:
        self.retail.set_min_stay(enabled, seconds)

    def set_funnel(self, steps: Sequence[str]) -> None:
        self.analytics.funnel_steps = list(steps)

    # ------------------------------------------------------------------
    # Per-frame
    # ------------------------------------------------------------------

    def process(
        self,
        frame: Optional[np.ndarray],
        detections: Sequence[Dict[str, Any]],
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run one frame. Returns the telemetry payload for the dashboard."""
        now = now or time.time()
        self._frame_index += 1

        h, w = (frame.shape[:2] if frame is not None
                else getattr(self, "frame_size", (720, 1280)))
        self.heatmap.frame_size = (h, w)

        observations = [
            TrackObservation(
                track_id=int(d["id"]),
                bbox=(float(d["box"][0]), float(d["box"][1]),
                      float(d["box"][2]), float(d["box"][3])),
                confidence=float(d["box"][4]) if len(d["box"]) > 4 else 1.0,
                frame_index=self._frame_index,
                timestamp=now,
                frame_width=w,
                frame_height=h,
            )
            for d in detections
        ]

        # ---- the ONLY place a tracker id is resolved to a person ----------
        decisions = self.identity.update(frame, observations, now=now)
        self.last_decisions = decisions

        self._seen_this_frame = set()
        for obs in observations:
            decision = decisions.get(obs.track_id)
            if decision is None or not decision.is_committed or not decision.global_id:
                continue                      # still in probation: not counted
            profile = self.memory.get(decision.global_id)
            if profile is None:
                continue
            foot: Point = ((obs.bbox[0] + obs.bbox[2]) / 2.0, obs.bbox[3])
            self._observe(profile, foot, now)

        # Anyone not seen this frame stops accruing dwell — but keeps their
        # open zone visits, because a tracker gap is not an exit. They stay on
        # the watch list until the grace period elapses and their open interval
        # is banked; dropping them immediately would mean a person who vanishes
        # for one frame never has their clock closed at all.
        for gid in list(self._live_ids - self._seen_this_frame):
            profile = self.memory.get(gid)
            if profile is None or self.zones.suspend(profile, now):
                self._live_ids.discard(gid)
        self._live_ids |= self._seen_this_frame

        if now - self._last_save >= 300.0:
            self.memory.save()
            self._last_save = now

        return self.telemetry(now)

    def _observe(self, profile: PersonProfile, foot: Point, now: float) -> None:
        """One person, one frame. Note that every call below takes a
        PersonProfile — no engine past this line has ever seen a track id."""
        self.heatmap.record(profile, foot, now)   # before touch(): needs the previous time
        self.zones.update(profile, foot, now)
        self.retail.update(profile, foot, now)
        self.entry_exit.update(profile, foot, now)
        profile.touch(now, position=foot)
        self._seen_this_frame.add(profile.global_id)

    def _on_identity_released(self, profile: PersonProfile, now: float) -> None:
        self.zones.release(profile, now)
        self.retail.release(profile, now)
        self.entry_exit.forget(profile.global_id)
        self._live_ids.discard(profile.global_id)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def label_for(self, track_id: int) -> str:
        """Short id for the on-frame overlay: 'P-4431'.

        Deliberately not the raw uuid, and deliberately never a name: the
        overlay must not display anything that identifies a real person.
        """
        gid = self.identity.global_id_for(track_id)
        return f"P-{gid[-4:]}" if gid else "..."

    def profile_for(self, track_id: int) -> Optional[PersonProfile]:
        return self.identity.profile_for(track_id)

    def telemetry(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Every derived number is computed against the FRAME's clock, not the
        wall clock. One injected time source keeps live metrics, replayed
        footage and unit tests producing identical numbers."""
        payload = self.analytics.snapshot(now)
        payload["identity_backend"] = self.reid.backend.name
        return payload

    def maps(self) -> Dict[str, Any]:
        return self.analytics.maps()

    def people(self) -> List[Dict[str, Any]]:
        """Every live person, explained. Powers a 'who is in the store right
        now' panel and answers 'why is this number what it is'."""
        return [p.summary() for p in self.memory.hot_candidates()]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.diagnostics(),
            "zones": self.zones.public_zone_names(),
            "entry_exit": self.entry_exit.stats(),
            "retail": self.retail.get_stats(),
        }

    def reset_counters(self) -> None:
        """Dashboard's Reset Analytics: clears counts, keeps calibration."""
        self.retail.reset_counters()
        self.journey.reset()
        self.heatmap.reset()

    def shutdown(self) -> None:
        now = time.time()
        for profile in self.memory.hot_candidates():
            self._on_identity_released(profile, now)
        self.memory.save()