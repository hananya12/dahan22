"""
vision/identity/entry_exit_engine.py
------------------------------------
Entered / Exited / Currently Inside — computed per GLOBAL ID, not per track.

Why the current engine cannot produce these numbers
---------------------------------------------------
``tracking.py`` counts SOM per ``person_id``, which is a ByteTrack id. When
ByteTrack drops and re-acquires a shopper standing near the entrance, that
shopper produces a second SOM. ``main.py`` already fights this with
``_tam_dedup_count`` + a recount cooldown — a heuristic that trades one error
for another (it also suppresses two *different* people entering within the
cooldown window). Once identities exist, that whole mechanism is deleted:
counting a set of global ids needs no cooldown, because the set already knows
who it has seen.

Hysteresis
----------
Real boundary crossings are noisy. A person standing in the doorway on a
windy day produces dozens of inside/outside flips per minute, each one a
+1/-1 on the store's headline number. Every production people-counter
debounces this; the naive implementation is the single most common reason a
counting dashboard shows impossible numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from geometry import Point, crossing_direction, point_in_polygon
from config import ZoneConfig
from contracts import Side
from events import EventBus, EventType, IdentityEvent
from person_profile import PersonProfile


@dataclass
class _CrossingState:
    side: Side = Side.UNKNOWN
    candidate_side: Side = Side.UNKNOWN
    candidate_frames: int = 0
    last_transition: float = 0.0
    last_position: Optional[Point] = None


class EntryExitEngine:
    """Decides when a known person entered or left the store.

    Supports both calibration styles the dashboard already produces:
      * inside_polygon — the store floor as a closed area (3+ points)
      * entry_line     — an oriented polyline; direction decides in vs out

    When both are given the polygon wins for state and the line is used to
    validate the crossing, which rejects the common false positive of a
    person appearing *inside* the polygon by detection pop-in rather than by
    walking through the door.
    """

    def __init__(
        self,
        config: ZoneConfig,
        camera_id: str,
        bus: Optional[EventBus] = None,
        inside_polygon: Optional[Sequence[Point]] = None,
        entry_line: Optional[Sequence[Point]] = None,
        entry_direction: int = +1,
    ) -> None:
        self.config = config
        self.camera_id = camera_id
        self.bus = bus
        self.inside_polygon = list(inside_polygon) if inside_polygon else None
        self.entry_line = list(entry_line) if entry_line else None
        self.entry_direction = entry_direction

        self._states: Dict[str, _CrossingState] = {}
        self.entered_ids: set[str] = set()
        self.exited_ids: set[str] = set()
        self.entered_count = 0
        self.exited_count = 0

    # ------------------------------------------------------------------

    def configure(self, inside_polygon=None, entry_line=None, entry_direction=None) -> None:
        """Re-calibrate live from the dashboard without losing identities."""
        if inside_polygon is not None:
            self.inside_polygon = list(inside_polygon) or None
        if entry_line is not None:
            self.entry_line = list(entry_line) or None
        if entry_direction is not None:
            self.entry_direction = entry_direction

    def is_configured(self) -> bool:
        return bool(self.inside_polygon) or bool(self.entry_line)

    # ------------------------------------------------------------------

    def update(self, profile: PersonProfile, position: Point,
               now: Optional[float] = None) -> Optional[EventType]:
        """Feed one position for one identity. Returns an event if the person
        just entered or exited (after debouncing), else None."""
        if not self.is_configured():
            return None

        now = now or time.time()
        state = self._states.setdefault(profile.global_id, _CrossingState())
        previous_position = state.last_position
        state.last_position = position

        observed = self._observe_side(position, previous_position, state)
        if observed is Side.UNKNOWN:
            return None

        # First ever observation: adopt the side silently. Someone who is
        # already inside when the engine starts must NOT be counted as an
        # entry — that is how a restart fabricates a hundred visitors.
        if state.side is Side.UNKNOWN:
            state.side = observed
            profile.side = observed
            if observed is Side.INSIDE:
                profile.open_visit(now)
            return None

        if observed == state.side:
            state.candidate_frames = 0
            state.candidate_side = Side.UNKNOWN
            return None

        # --- hysteresis ---------------------------------------------------
        if observed == state.candidate_side:
            state.candidate_frames += 1
        else:
            state.candidate_side = observed
            state.candidate_frames = 1

        if state.candidate_frames < self.config.confirm_frames:
            return None
        if now - state.last_transition < self.config.transition_cooldown_seconds:
            return None

        # --- committed transition ----------------------------------------
        state.side = observed
        state.candidate_frames = 0
        state.candidate_side = Side.UNKNOWN
        state.last_transition = now
        profile.side = observed

        if observed is Side.INSIDE:
            profile.open_visit(now)
            self.entered_ids.add(profile.global_id)
            self.entered_count += 1
            self._emit(EventType.ENTERED, profile, position, now,
                       {"visit_number": profile.visit_count,
                        "returning": profile.is_returning})
            return EventType.ENTERED

        visit = profile.close_visit(now)
        self.exited_ids.add(profile.global_id)
        self.exited_count += 1
        self._emit(EventType.EXITED, profile, position, now,
                   {"visit_seconds": round(visit.duration, 1) if visit else None})
        return EventType.EXITED

    # ------------------------------------------------------------------

    def _observe_side(self, position: Point, previous: Optional[Point],
                      state: _CrossingState) -> Side:
        if self.inside_polygon:
            return Side.INSIDE if point_in_polygon(position[0], position[1],
                                                   self.inside_polygon) else Side.OUTSIDE

        # Line-only mode: the side is only known once a crossing is observed,
        # which is why an oriented line needs a previous position.
        if previous is None:
            return Side.UNKNOWN
        direction = crossing_direction(previous, position, self.entry_line)
        if direction == 0:
            return state.side
        return Side.INSIDE if direction == self.entry_direction else Side.OUTSIDE

    def _emit(self, event_type: EventType, profile: PersonProfile,
              position: Point, now: float, data: dict) -> None:
        if self.bus is None:
            return
        self.bus.emit(IdentityEvent(
            type=event_type, global_id=profile.global_id, camera_id=self.camera_id,
            timestamp=now, position=position, data=data,
        ))

    # ------------------------------------------------------------------

    def forget(self, global_id: str) -> None:
        self._states.pop(global_id, None)

    @property
    def currently_inside(self) -> int:
        """Live occupancy.

        Derived from per-identity state rather than ``entered - exited``,
        because that subtraction drifts permanently: anyone missed on the way
        out inflates occupancy forever. Recomputing from state is
        self-healing — a person whose identity expires simply stops counting.
        """
        return sum(1 for s in self._states.values() if s.side is Side.INSIDE)

    def stats(self) -> Dict[str, int]:
        return {
            "entered": len(self.entered_ids),
            "exited": len(self.exited_ids),
            "entered_events": self.entered_count,
            "exited_events": self.exited_count,
            "currently_inside": self.currently_inside,
            "tracked_identities": len(self._states),
        }
