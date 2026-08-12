"""
vision/identity/person_profile.py
---------------------------------
``PersonProfile`` — everything the system knows about one real human being.

The important design decision here is the **gallery**: an identity is not
represented by one embedding but by a small set of diverse, high-quality
views. A single embedding is a photograph of a person at one instant; a
person walking through a store presents their front, side and back, under
three different light levels, at four different scales. Matching a rear view
against a stored front view scores ~0.4 and looks exactly like a different
person.

Matching against a gallery of diverse views is what turns re-identification
from a demo into a product.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from contracts import Embedding, Side


@dataclass
class Visit:
    """One continuous presence inside the store."""
    entered_at: float
    exited_at: Optional[float] = None

    @property
    def duration(self) -> float:
        return (self.exited_at or time.time()) - self.entered_at

    @property
    def is_open(self) -> bool:
        return self.exited_at is None


@dataclass
class RetailState:
    """The three business flags of one person, plus WHY they are what they
    are. ``sam_reason`` exists so "why is SAM zero" is answerable from a
    live profile instead of from a debugger."""
    counted_tam: bool = False
    counted_sam: bool = False
    counted_som: bool = False
    sam_reason: str = "not evaluated yet"

    def absorb(self, other: "RetailState") -> None:
        self.counted_tam = self.counted_tam or other.counted_tam
        self.counted_sam = self.counted_sam or other.counted_sam
        self.counted_som = self.counted_som or other.counted_som
        if other.counted_sam and not self.counted_sam:
            self.sam_reason = other.sam_reason


@dataclass
class PersonProfile:
    """A globally-identified person. The 'GlobalPerson' of the spec."""

    global_id: str
    camera_id: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    gallery: List[Embedding] = field(default_factory=list)

    side: Side = Side.UNKNOWN
    current_zone: Optional[str] = None
    zone_path: List[Tuple[str, float]] = field(default_factory=list)  # journey
    visits: List[Visit] = field(default_factory=list)

    cameras_seen: List[str] = field(default_factory=list)
    track_ids_seen: List[int] = field(default_factory=list)
    last_position: Optional[Tuple[float, float]] = None
    # Corewise Intelligence Layer

    appearance: Dict = field(default_factory=dict)

    pose: Dict = field(default_factory=dict)

    behavior: Dict = field(default_factory=dict)

    # --- per-person retail + dwell state (owned by Zone/Retail engines) ----
    # Living ON the profile is what makes memory.merge() able to reconcile
    # TAM/SAM/dwell in one place instead of chasing state across engines.
    retail: "RetailState" = None            # created in __post_init__
    zone_dwell_banked: Dict[str, float] = field(default_factory=dict)
    zone_entered_at: Dict[str, float] = field(default_factory=dict)
    # Set when this identity was pulled back out of warm memory, i.e. the
    # person left and came back later. Drives the Returning Visitors metric.
    is_returning: bool = False
    recovered_at: Optional[float] = None

    def __post_init__(self) -> None:
        if self.retail is None:
            self.retail = RetailState()

    # ------------------------------------------------------------- dwell
    def zone_dwell(self, zone: str, now: Optional[float] = None) -> float:
        """Total seconds this person has spent in ``zone`` — banked closed
        intervals plus the currently open one. Survives tracker loss by
        construction: the clock lives on the PERSON, not on any track."""
        now = now if now is not None else time.time()
        banked = self.zone_dwell_banked.get(zone, 0.0)
        entered = self.zone_entered_at.get(zone)
        if entered is not None:
            banked += max(0.0, now - entered)
        return banked

    # ---------------------------------------------------------------- gallery

    def add_embedding(self, emb: Embedding, max_size: int, min_improvement: float) -> bool:
        """Add a view to the gallery, keeping it small AND diverse.

        Replacement policy, in order:
          1. Room left      -> just append.
          2. New crop is very similar to an existing one (a near-duplicate,
             e.g. the person standing still) -> only replace that one, and
             only if it is meaningfully better quality. This is what stops a
             stationary shopper from flushing every other viewpoint out of
             their own gallery.
          3. Otherwise      -> evict the lowest-quality entry, if the newcomer
             beats it.
        """
        if len(self.gallery) < max_size:
            self.gallery.append(emb)
            return True

        sims = [emb.similarity(g) for g in self.gallery]
        most_similar = int(np.argmax(sims))

        if sims[most_similar] > 0.92:  # near-duplicate viewpoint
            if emb.quality > self.gallery[most_similar].quality + min_improvement:
                self.gallery[most_similar] = emb
                return True
            return False

        weakest = min(range(len(self.gallery)), key=lambda i: self.gallery[i].quality)
        if emb.quality > self.gallery[weakest].quality + min_improvement:
            self.gallery[weakest] = emb
            return True
        return False

    def similarity_to(self, query: Embedding) -> float:
        """Best match across every stored viewpoint.

        Max (not mean) is correct: a person seen from behind SHOULD match
        their stored rear view strongly, and it is irrelevant that it matches
        their stored front view weakly. Averaging would punish exactly the
        diversity the gallery exists to provide.
        """
        if not self.gallery:
            return 0.0
        return max(query.similarity(g) for g in self.gallery)

    @property
    def best_quality(self) -> float:
        return max((g.quality for g in self.gallery), default=0.0)

    # ------------------------------------------------------------ lifecycle

    def touch(self, timestamp: float, track_id: Optional[int] = None,
              position: Optional[Tuple[float, float]] = None) -> None:
        self.last_seen = timestamp
        if position is not None:
            self.last_position = position
        if track_id is not None and track_id not in self.track_ids_seen:
            self.track_ids_seen.append(track_id)

    def open_visit(self, timestamp: float) -> Visit:
        if self.visits and self.visits[-1].is_open:
            return self.visits[-1]
        visit = Visit(entered_at=timestamp)
        self.visits.append(visit)
        return visit

    def close_visit(self, timestamp: float) -> Optional[Visit]:
        if self.visits and self.visits[-1].is_open:
            self.visits[-1].exited_at = timestamp
            return self.visits[-1]
        return None

    def record_zone(self, zone: Optional[str], timestamp: float) -> bool:
        """Append to the journey when the zone actually changes."""
        if zone == self.current_zone:
            return False
        self.current_zone = zone
        if zone is not None:
            self.zone_path.append((zone, timestamp))
        return True

    # -------------------------------------------------------------- metrics

    @property
    def total_dwell(self) -> float:
        return sum(v.duration for v in self.visits)

    @property
    def visit_count(self) -> int:
        return len(self.visits)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.first_seen

    def summary(self) -> Dict:
        return {
            "global_id": self.global_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "side": self.side.value,
            "visits": self.visit_count,
            "total_dwell": round(self.total_dwell, 1),
            "gallery": len(self.gallery),
            "returning": self.is_returning,
            "journey": [z for z, _ in self.zone_path],
            "cameras": list(self.cameras_seen),
        }

    def to_persistable(self) -> Dict:
        """Serialisable form for the warm tier.

        NOTE: embeddings only — never crops, never faces. See the privacy
        section of ARCHITECTURE.md; this is the boundary that keeps the
        product on the right side of biometric-data rules.
        """
        return {
            "global_id": self.global_id,
            "camera_id": self.camera_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "visits": [(v.entered_at, v.exited_at) for v in self.visits],
            "gallery": [
                {"v": g.vector.tolist(), "q": g.quality, "t": g.timestamp, "c": g.camera_id}
                for g in self.gallery
            ],
        }
