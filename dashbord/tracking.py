"""
tracking.py
-----------
Owns all Retail Analytics business logic for Corewise:

    * Person IDs (lifecycle: seen -> stale -> forgotten)
    * TAM  - Total Addressable Market: every person who entered the
             defined zone (Polygon, pointPolygonTest).
    * SAM  - Serviceable Available Market: a person who stayed inside
             the TAM zone for at least the MINIMUM STAY TIME configured
             ON THE DASHBOARD. Counted once.
    * SOM  - Serviceable Obtainable Market: a person who crossed the
             entrance line OR entered the entrance polygon. Counted once.
    * Inside counter, Average Stay Time, Conversion Rate.

This module has no OpenCV drawing code and no camera/model code in it -
main.py is responsible for detection + drawing, this module is
responsible only for turning detections into business metrics.

SAM minimum stay time (Task 1)
------------------------------
There is deliberately NO hardcoded dwell constant in this module any
more. The former ``SAM_DWELL_SECONDS = 5.0`` module constant has been
removed entirely. The threshold is per-instance state, owned by the
dashboard and pushed in via :meth:`PersonTracker.set_min_stay`.

Until the dashboard sends a value the tracker is *unconfigured* and will
not count SAM at all (it logs this once). That is intentional: it makes
the dashboard the single source of truth by construction, so it is
impossible for a stale engine-side default to silently win again.

Semantics, exactly as specified:
    enabled=False           -> every visitor inside TAM counts as SAM immediately
    enabled=True, seconds=0 -> counts immediately on entry
    enabled=True, seconds=1 -> counts after 1.0s of CONTINUOUS dwell
    enabled=True, seconds=3.5 -> counts after 3.5s of CONTINUOUS dwell

"Continuous" matters: the dwell clock restarts when a person leaves the
zone and comes back. Previously the clock was anchored to the person's
*first ever* entry and never reset, so someone who stepped out and back
in was counted using a stale, inflated dwell.

SOM shapes (Task 3)
-------------------
SOM is no longer limited to two points, and is no longer approximated by
a single horizontal y-coordinate. Two shape kinds are supported:

    * POLYLINE (2+ points) - a directional entrance line. A person is
      counted when the segment between their previous foot position and
      their current foot position genuinely intersects any segment of
      the polyline. Works for vertical, diagonal and multi-segment
      entrances, which the old ``line_y`` midpoint test got wrong in
      both directions (false positives AND false negatives).
    * POLYGON (3+ points, closed) - an entrance area. A person is
      counted the first time they transition from outside to inside.

Arbitrary vertex counts are supported: 3, 4, 5, 6, 8, 10 or more.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

STALE_TRACK_TIMEOUT_SECONDS = 3.0

# Set COREWISE_SAM_DEBUG=0 to silence the per-decision SAM logs.
import os as _os
SAM_DEBUG = _os.environ.get("COREWISE_SAM_DEBUG", "1") not in ("0", "false", "False")


# ---------------------------------------------------------------------------
# Geometry helpers (module-level so they are unit-testable on their own)
# ---------------------------------------------------------------------------

def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    """Cross product of AB x AC. >0 = counter-clockwise, <0 = clockwise, 0 = collinear."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> bool:
    """True if collinear point P lies within the bounding box of segment AB."""
    return (
        min(ax, bx) - 1e-9 <= px <= max(ax, bx) + 1e-9
        and min(ay, by) - 1e-9 <= py <= max(ay, by) + 1e-9
    )


def segments_intersect(
    p1: Tuple[float, float], p2: Tuple[float, float],
    q1: Tuple[float, float], q2: Tuple[float, float],
) -> bool:
    """True if segment P1P2 intersects segment Q1Q2 (including touching).

    This is the real crossing test that replaces the old
    ``last_foot_y < line_y <= foot_y`` horizontal-band approximation.
    """
    (p1x, p1y), (p2x, p2y) = p1, p2
    (q1x, q1y), (q2x, q2y) = q1, q2

    d1 = _orientation(q1x, q1y, q2x, q2y, p1x, p1y)
    d2 = _orientation(q1x, q1y, q2x, q2y, p2x, p2y)
    d3 = _orientation(p1x, p1y, p2x, p2y, q1x, q1y)
    d4 = _orientation(p1x, p1y, p2x, p2y, q2x, q2y)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear / touching cases.
    if abs(d1) < 1e-12 and _on_segment(q1x, q1y, q2x, q2y, p1x, p1y):
        return True
    if abs(d2) < 1e-12 and _on_segment(q1x, q1y, q2x, q2y, p2x, p2y):
        return True
    if abs(d3) < 1e-12 and _on_segment(p1x, p1y, p2x, p2y, q1x, q1y):
        return True
    if abs(d4) < 1e-12 and _on_segment(p1x, p1y, p2x, p2y, q2x, q2y):
        return True
    return False


def _to_int_points(points: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    return [(int(round(p[0])), int(round(p[1]))) for p in points]


# ---------------------------------------------------------------------------


@dataclass
class PersonTrack:
    """Per-person bookkeeping used to derive TAM/SAM/SOM/Inside/stay time."""

    person_id: int
    entered_tam: bool = False
    counted_sam: bool = False
    counted_som: bool = False

    # Start of the CURRENT continuous visit inside the TAM zone.
    # None whenever the person is outside the zone. This is what makes the
    # SAM dwell clock restart correctly on re-entry.
    tam_entry_time: Optional[float] = None

    last_seen: float = field(default_factory=time.time)
    last_foot_x: Optional[int] = None
    last_foot_y: Optional[int] = None
    currently_inside_zone: bool = False

    # True while the person is inside the SOM polygon (polygon mode only),
    # so an outside->inside transition can be detected exactly once.
    inside_som_shape: bool = False

    # Total dwell already banked from PREVIOUS completed visits, so average
    # stay time still reflects a person who entered, left and came back.
    banked_dwell: float = 0.0


class PersonTracker:
    """Turns per-frame person detections into Retail Analytics metrics."""

    def __init__(
        self,
        tam_zone_points: Sequence[Sequence[float]],
        line_start: Optional[Tuple[int, int]] = None,
        line_end: Optional[Tuple[int, int]] = None,
        som_points: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """
        Args:
            tam_zone_points: polygon (3+ points) defining the TAM zone.
            line_start / line_end: LEGACY 2-point entrance line. Kept so
                existing callers keep working unchanged.
            som_points: preferred - the full SOM shape with ANY number of
                points (2 = polyline, 3+ = polygon). When given, it takes
                precedence over line_start/line_end.
        """
        self.tam_zone = np.array(_to_int_points(tam_zone_points), dtype=np.int32)

        # --- SOM shape -----------------------------------------------------
        self.som_points: List[Tuple[int, int]] = []
        self.som_is_polygon: bool = False
        self.som_polygon: Optional[np.ndarray] = None

        if som_points:
            self.set_som_shape(som_points)
        elif line_start is not None and line_end is not None:
            self.set_som_shape([line_start, line_end])

        # Legacy attributes kept for backward compatibility with any caller
        # or overlay code that still reads them. They are NO LONGER used for
        # the crossing decision.
        self.line_start = tuple(self.som_points[0]) if self.som_points else None
        self.line_end = tuple(self.som_points[-1]) if self.som_points else None

        self.tracks: Dict[int, PersonTrack] = {}

        self.tam_count = 0
        self.sam_count = 0
        self.som_count = 0
        self.people_today = 0

        self._completed_stay_durations: List[float] = []

        # --- SAM minimum stay time: dashboard-owned, no default constant ---
        self._sam_min_stay_enabled: bool = True
        self._sam_min_stay_seconds: Optional[float] = None   # None = not yet configured
        self._sam_unconfigured_warned: bool = False

    # ------------------------------------------------------------------
    # SOM shape configuration (Task 3)
    # ------------------------------------------------------------------

    def set_som_shape(
        self,
        points: Sequence[Sequence[float]],
        closed: Optional[bool] = None,
    ) -> None:
        """Define the SOM entrance shape from ANY number of points.

        2 points  -> directional entrance polyline
        3+ points -> closed entrance polygon (unless closed=False, which
                     keeps it a multi-segment polyline)
        """
        pts = _to_int_points(points or [])
        if len(pts) < 2:
            self.som_points = []
            self.som_is_polygon = False
            self.som_polygon = None
            return

        # Drop a duplicated closing vertex if the caller sent one.
        if len(pts) > 2 and pts[0] == pts[-1]:
            pts = pts[:-1]

        self.som_points = pts
        if closed is None:
            closed = len(pts) >= 3
        self.som_is_polygon = bool(closed) and len(pts) >= 3
        self.som_polygon = np.array(pts, dtype=np.int32) if self.som_is_polygon else None

        self.line_start = tuple(pts[0])
        self.line_end = tuple(pts[-1])

    @property
    def som_point_count(self) -> int:
        return len(self.som_points)

    def is_inside_som_shape(self, x: int, y: int) -> bool:
        """True if (x, y) is inside the SOM polygon. False in polyline mode."""
        if not self.som_is_polygon or self.som_polygon is None:
            return False
        return cv2.pointPolygonTest(self.som_polygon, (float(x), float(y)), False) >= 0

    def _crossed_som_polyline(
        self, prev_x: int, prev_y: int, cur_x: int, cur_y: int
    ) -> bool:
        """True if the person's movement segment crosses any polyline segment."""
        if len(self.som_points) < 2:
            return False
        move_a = (float(prev_x), float(prev_y))
        move_b = (float(cur_x), float(cur_y))
        for i in range(len(self.som_points) - 1):
            seg_a = (float(self.som_points[i][0]), float(self.som_points[i][1]))
            seg_b = (float(self.som_points[i + 1][0]), float(self.som_points[i + 1][1]))
            if segments_intersect(move_a, move_b, seg_a, seg_b):
                return True
        return False

    # ------------------------------------------------------------------
    # SAM minimum stay time (Task 1) - dashboard is the ONLY source
    # ------------------------------------------------------------------

    def set_min_stay(self, enabled: bool, seconds: Optional[float]) -> None:
        """Push the dashboard's live SAM minimum-stay setting into the tracker.

        Called by the engine on EVERY control tick, so changing the value in
        the dashboard takes effect on the very next frame with no restart and
        no recalibration.
        """
        try:
            secs = None if seconds is None else max(0.0, float(seconds))
        except (TypeError, ValueError):
            secs = None

        changed = (
            enabled != self._sam_min_stay_enabled
            or secs != self._sam_min_stay_seconds
        )
        self._sam_min_stay_enabled = bool(enabled)
        self._sam_min_stay_seconds = secs
        if changed and SAM_DEBUG:
            print(
                f"[SAM] Threshold updated from dashboard -> "
                f"enabled={self._sam_min_stay_enabled} seconds={self._sam_min_stay_seconds}"
            )

    @property
    def sam_min_stay_enabled(self) -> bool:
        return self._sam_min_stay_enabled

    @sam_min_stay_enabled.setter
    def sam_min_stay_enabled(self, value: bool) -> None:
        # Kept so the engine's legacy attribute-assignment path still works.
        self.set_min_stay(bool(value), self._sam_min_stay_seconds)

    @property
    def sam_min_stay_seconds(self) -> Optional[float]:
        return self._sam_min_stay_seconds

    @sam_min_stay_seconds.setter
    def sam_min_stay_seconds(self, value: Optional[float]) -> None:
        self.set_min_stay(self._sam_min_stay_enabled, value)

    @property
    def sam_is_configured(self) -> bool:
        """True once the dashboard has supplied a threshold."""
        return (not self._sam_min_stay_enabled) or (self._sam_min_stay_seconds is not None)

    # ------------------------------------------------------------------

    def is_inside_zone(self, x: int, y: int) -> bool:
        """True if point (x, y) is inside the TAM polygon."""
        return cv2.pointPolygonTest(self.tam_zone, (float(x), float(y)), False) >= 0

    def update(self, person_id: int, foot_x: int, foot_y: int) -> PersonTrack:
        """Update tracking state for one detected person in the current frame."""
        track = self.tracks.get(person_id)
        now = time.time()

        if track is None:
            track = PersonTrack(
                person_id=person_id, last_foot_x=foot_x, last_foot_y=foot_y
            )
            self.tracks[person_id] = track
            self.people_today += 1

        track.last_seen = now
        inside_now = self.is_inside_zone(foot_x, foot_y)
        was_inside = track.currently_inside_zone

        # --- TAM (counted once per person) ---
        if inside_now and not track.entered_tam:
            track.entered_tam = True
            self.tam_count += 1

        # --- Continuous dwell clock ---
        if inside_now and not was_inside:
            # Entered (or re-entered) the zone: restart the dwell clock.
            track.tam_entry_time = now
        elif not inside_now and was_inside:
            # Left the zone: bank the completed visit, stop the clock.
            if track.tam_entry_time is not None:
                completed = now - track.tam_entry_time
                track.banked_dwell += completed
                self._completed_stay_durations.append(completed)
            track.tam_entry_time = None

        # --- SAM (dashboard-configured minimum stay) ---
        if inside_now and not track.counted_sam:
            self._evaluate_sam(track, now)

        # --- SOM (entrance crossing / entrance area) ---
        self._evaluate_som(track, foot_x, foot_y)

        track.last_foot_x = foot_x
        track.last_foot_y = foot_y
        track.currently_inside_zone = inside_now

        return track

    def _evaluate_sam(self, track: PersonTrack, now: float) -> None:
        """Decide whether this person becomes SAM, using ONLY the live
        dashboard threshold. Emits a full decision log line."""
        enabled = self._sam_min_stay_enabled
        threshold = self._sam_min_stay_seconds
        dwell = (now - track.tam_entry_time) if track.tam_entry_time is not None else 0.0

        if not enabled:
            decision = "COUNT (minimum stay disabled -> immediate)"
            counted = True
        elif threshold is None:
            if not self._sam_unconfigured_warned:
                self._sam_unconfigured_warned = True
                print(
                    "[SAM] WARNING: no minimum stay time received from the dashboard yet "
                    "- SAM will not be counted until the dashboard sends one. "
                    "(There is deliberately no engine-side default.)"
                )
            decision = "WAIT (threshold not yet received from dashboard)"
            counted = False
        elif dwell >= threshold:
            decision = f"COUNT (dwell {dwell:.2f}s >= threshold {threshold:.2f}s)"
            counted = True
        else:
            decision = f"WAIT (dwell {dwell:.2f}s < threshold {threshold:.2f}s)"
            counted = False

        if counted:
            track.counted_sam = True
            self.sam_count += 1

        if SAM_DEBUG and (counted or int(dwell * 2) != int((dwell - 0.05) * 2)):
            # Log every decision that counts, and roughly twice a second while waiting.
            print(
                f"[SAM] id={track.person_id} "
                f"dashboard(enabled={enabled}, seconds={threshold}) "
                f"tracker(enabled={self._sam_min_stay_enabled}, "
                f"seconds={self._sam_min_stay_seconds}) "
                f"dwell={dwell:.2f}s "
                f"state={'INSIDE_TAM' if track.tam_entry_time else 'OUTSIDE_TAM'} "
                f"counted_sam={track.counted_sam} -> {decision}"
            )

    def _evaluate_som(self, track: PersonTrack, foot_x: int, foot_y: int) -> None:
        """Count SOM once per person, using the real SOM shape geometry."""
        if track.counted_som or not self.som_points:
            # Still maintain polygon membership so re-entry logic stays sane.
            if self.som_is_polygon:
                track.inside_som_shape = self.is_inside_som_shape(foot_x, foot_y)
            return

        if self.som_is_polygon:
            inside_now = self.is_inside_som_shape(foot_x, foot_y)
            if inside_now and not track.inside_som_shape:
                track.counted_som = True
                self.som_count += 1
                if SAM_DEBUG:
                    print(
                        f"[SOM] id={track.person_id} entered SOM polygon "
                        f"({len(self.som_points)} vertices) -> SOM={self.som_count}"
                    )
            track.inside_som_shape = inside_now
            return

        # Polyline mode: needs a previous position to form a movement segment.
        if track.last_foot_x is None or track.last_foot_y is None:
            return
        if self._crossed_som_polyline(track.last_foot_x, track.last_foot_y, foot_x, foot_y):
            track.counted_som = True
            self.som_count += 1
            if SAM_DEBUG:
                print(
                    f"[SOM] id={track.person_id} crossed SOM polyline "
                    f"({len(self.som_points)} points) -> SOM={self.som_count}"
                )

    def dwell_seconds(self, person_id: int) -> float:
        """Seconds a given person has been CONTINUOUSLY inside the TAM zone."""
        track = self.tracks.get(person_id)
        if track is None or track.tam_entry_time is None:
            return 0.0
        return time.time() - track.tam_entry_time

    def expire_stale_tracks(self, active_ids: List[int]) -> None:
        """Drop tracks not seen this frame for too long, banking their stay time."""
        now = time.time()
        active_set = set(active_ids)
        stale_ids = []

        for pid, track in self.tracks.items():
            if pid in active_set:
                continue
            if now - track.last_seen >= STALE_TRACK_TIMEOUT_SECONDS:
                if track.tam_entry_time is not None:
                    self._completed_stay_durations.append(
                        max(0.0, track.last_seen - track.tam_entry_time)
                    )
                stale_ids.append(pid)

        for pid in stale_ids:
            del self.tracks[pid]

    # ------------------------------------------------------------------

    @property
    def inside_count(self) -> int:
        """People currently tracked and standing inside the TAM zone."""
        return sum(1 for t in self.tracks.values() if t.currently_inside_zone)

    @property
    def average_stay_time(self) -> float:
        """Average seconds spent inside the TAM zone, across completed + active visits."""
        durations = list(self._completed_stay_durations)
        now = time.time()
        for track in self.tracks.values():
            if track.tam_entry_time is not None:
                durations.append(now - track.tam_entry_time)

        if not durations:
            return 0.0
        return sum(durations) / len(durations)

    @property
    def conversion_rate(self) -> float:
        """Percentage of TAM visitors who went on to become SOM (entered store)."""
        if self.tam_count == 0:
            return 0.0
        return (self.som_count / self.tam_count) * 100.0

    def get_stats(self) -> Dict[str, float]:
        """Full metrics snapshot, ready to be sent to the dashboard."""
        return {
            "tam": self.tam_count,
            "sam": self.sam_count,
            "som": self.som_count,
            "inside": self.inside_count,
            "average_stay_time": round(self.average_stay_time, 1),
            "conversion_rate": round(self.conversion_rate, 1),
            "people_today": self.people_today,
            # Surfaced so the dashboard can prove which threshold the engine
            # is actually applying right now.
            "sam_min_stay_enabled": self._sam_min_stay_enabled,
            "sam_min_stay_seconds": self._sam_min_stay_seconds,
            "som_shape": "polygon" if self.som_is_polygon else "polyline",
            "som_points": len(self.som_points),
        }

    def reset_counters(self) -> None:
        """Reset every count and tracked person while keeping the current
        TAM zone / SOM shape calibration intact - used by the dashboard's
        Settings > Analytics > "Reset Analytics" action, which must not
        force a recalibration."""
        self.tracks.clear()
        self.tam_count = 0
        self.sam_count = 0
        self.som_count = 0
        self.people_today = 0
        self._completed_stay_durations.clear()