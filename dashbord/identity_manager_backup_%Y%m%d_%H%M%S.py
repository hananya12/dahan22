"""
vision/identity/identity_manager.py
-----------------------------------
The orchestrator. Turns a stream of tracker observations into stable global
identities.

Its whole job is to answer one question per track: *who is this?* — and to
answer it as rarely as possible.

Two rules make the difference between a system that works and one that
oscillates:

  1. STICKY BINDINGS. Once a track is committed to a global id, the binding
     is not revisited. The tracker (BoT-SORT/ByteTrack) is *better* than any
     ReID model at frame-to-frame association; it uses motion and IoU, which
     appearance models do not have. ReID exists to bridge the gaps the
     tracker cannot cross — not to second-guess it. Re-matching every frame
     costs CPU and buys identity flicker.

  2. PROBATION. A brand-new track_id is not counted immediately. It gathers
     evidence over a few embeddable frames and only then commits — either to
     an existing identity or to a new one. This is why a person walking back
     into frame does not increment Visitors Today before the system has
     decided who they are.

Everything that changes a business number leaves through the EventBus, so
counting, analytics, replay metadata and the dashboard all read the same
stream of facts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import IdentityConfig
from contracts import Embedding, IdentityDecision, MatchResult, TrackObservation, Verdict
from events import EventBus
from matching_engine import MatchingEngine
from memory_engine import MemoryEngine
from person_profile import PersonProfile
from reid_engine import ReIDEngine


@dataclass
class TrackBinding:
    """The live link between a tracker id and a global identity."""
    track_id: int
    global_id: Optional[str] = None
    committed: bool = False
    votes: List[MatchResult] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    embeddings_seen: int = 0


class IdentityManager:
    """Tracker ids in, global ids out."""

    def __init__(
        self,
        config: Optional[IdentityConfig] = None,
        bus: Optional[EventBus] = None,
        reid: Optional[ReIDEngine] = None,
        memory: Optional[MemoryEngine] = None,
    ) -> None:
        self.config = config or IdentityConfig()
        self.bus = bus or EventBus()
        self.reid = reid or ReIDEngine(self.config.reid, camera_id=self.config.camera_id)
        self.memory = memory or MemoryEngine(self.config.memory, self.config.camera_id, self.bus)
        self.matcher = MatchingEngine(self.config.matching)

        self._bindings: Dict[int, TrackBinding] = {}
        # Lets the ReID scheduler spend idle budget only where it helps.
        self.reid.gallery_has_room = self._gallery_has_room
        self._last_sweep = 0.0
        self._frame_index = 0

    # ------------------------------------------------------------------
    # Main entry point — called once per frame from main.py
    # ------------------------------------------------------------------

    def update(
        self,
        frame: Optional[np.ndarray],
        observations: Sequence[TrackObservation],
        now: Optional[float] = None,
    ) -> Dict[int, IdentityDecision]:
        """Resolve every visible track to a global identity."""
        now = now or time.time()
        self._frame_index += 1

        active_ids = {obs.track_id for obs in observations}
        self._retire_missing_tracks(active_ids, now)

        # 1. Which tracks still need an answer? Those get the ReID budget.
        pending = [obs for obs in observations
                   if not self._binding(obs.track_id, now).committed]

        embeddings: Dict[int, Embedding] = {}
        # Whether pixels are required is the ReID engine's decision, not this
        # one's: an offline replay drives the same logic with recorded vectors.
        if observations:
            selected = self.reid.select(
                observations,
                priority_track_ids=[o.track_id for o in pending],
                frame_index=self._frame_index,
            )
            embeddings = self.reid.embed(frame, selected, frame_index=self._frame_index)

        # 2. Resolve identities.
        decisions: Dict[int, IdentityDecision] = {}
        for obs in observations:
            decisions[obs.track_id] = self._resolve(obs, embeddings.get(obs.track_id), now)

        # 3. Housekeeping, at most once a second.
        if now - self._last_sweep >= 1.0:
            self.memory.sweep(now)
            self._last_sweep = now

        return decisions

    # ------------------------------------------------------------------

    def _resolve(self, obs: TrackObservation, emb: Optional[Embedding],
                 now: float) -> IdentityDecision:
        binding = self._binding(obs.track_id, now)
        binding.last_seen = now

        # --- already committed: sticky, no re-matching ---------------------
        if binding.committed and binding.global_id:
            profile = self.memory.get(binding.global_id)
            if profile is not None:
                _remember_tracker_id(profile, obs.track_id)
                if emb is not None:
                    profile.add_embedding(emb, self.config.memory.gallery_size,
                                          self.config.memory.min_quality_improvement)
                    binding.embeddings_seen += 1
                return IdentityDecision(obs.track_id, binding.global_id, True, False,
                                        profile.is_returning)
            # The identity was swept while the track was alive (very long
            # occlusion). Drop the binding and let probation run again.
            binding.committed = False
            binding.global_id = None

        # --- probation: gather evidence ------------------------------------
        if emb is None:
            # Nothing new to learn this frame — stay uncommitted, stay uncounted.
            return IdentityDecision(obs.track_id, binding.global_id, False, False, False)

        binding.embeddings_seen += 1
        claimed = {b.global_id for b in self._bindings.values()
                   if b.global_id and b.track_id != obs.track_id and b.committed}

        foot = ((obs.bbox[0] + obs.bbox[2]) / 2.0, obs.bbox[3])
        vote = self.matcher.match(emb, self.memory.hot_candidates(), exclude_ids=claimed,
                                  position=foot, now=now, frame_width=obs.frame_width)

        # Hot memory drew a blank — ask warm memory: is this a returning visitor?
        recovered_from_warm = False
        if vote.verdict is Verdict.NEW:
            # Warm memory is minutes or hours old, so motion says nothing
            # useful about it — appearance has to carry that decision alone.
            warm_vote = self.matcher.match(emb, self.memory.warm_candidates(),
                                           exclude_ids=claimed)
            if warm_vote.verdict is Verdict.MATCH:
                vote = warm_vote
                recovered_from_warm = True

        binding.votes.append(vote)

        if len(binding.votes) < self.config.matching.probation_frames:
            return IdentityDecision(obs.track_id, None, False, False, False, match=vote)

        # --- commit --------------------------------------------------------
        final = self.matcher.evaluate_probation(binding.votes)
        is_new = False
        is_returning = False

        if final.verdict is Verdict.MATCH and final.global_id:
            profile = self.memory.get(final.global_id)
            if profile is None:
                profile = self.memory.create(now, emb)
                is_new = True
            elif recovered_from_warm or profile.global_id in {
                    p.global_id for p in self.memory.warm_candidates()}:
                profile = self.memory.promote(final.global_id, now) or profile
                is_returning = True
        else:
            profile = self.memory.create(now, emb)
            is_new = True

        _remember_tracker_id(profile, obs.track_id)
        profile.add_embedding(emb, self.config.memory.gallery_size,
                              self.config.memory.min_quality_improvement)
        if self.config.camera_id not in profile.cameras_seen:
            profile.cameras_seen.append(self.config.camera_id)

        binding.global_id = profile.global_id
        binding.committed = True
        binding.votes.clear()

        if self.config.debug:
            print(f"[IDENTITY] track={obs.track_id} -> {profile.global_id} "
                  f"({'new' if is_new else 'matched'}"
                  f"{', returning' if is_returning else ''}) {final.explain()}")

        return IdentityDecision(obs.track_id, profile.global_id, True, is_new,
                                is_returning or profile.is_returning, match=final)

    # ------------------------------------------------------------------

    def _gallery_has_room(self, tracker_id: int) -> bool:
        binding = self._bindings.get(tracker_id)
        if binding is None or not binding.global_id:
            return True
        profile = self.memory.get(binding.global_id)
        return profile is None or len(profile.gallery) < self.config.memory.gallery_size

    def _binding(self, track_id: int, now: float) -> TrackBinding:
        binding = self._bindings.get(track_id)
        if binding is None:
            binding = TrackBinding(track_id=track_id, first_seen=now, last_seen=now)
            self._bindings[track_id] = binding
        return binding

    def _retire_missing_tracks(self, active: set, now: float) -> None:
        """A tracker id that vanished is forgotten immediately — but its
        IDENTITY is not. That is the entire point of the architecture: the
        track is ephemeral, the person is not."""
        for track_id in [t for t in self._bindings if t not in active]:
            self._bindings.pop(track_id, None)
            self.reid.forget_track(track_id)

    def global_id_for(self, track_id: int) -> Optional[str]:
        binding = self._bindings.get(track_id)
        return binding.global_id if binding and binding.committed else None

    def profile_for(self, track_id: int) -> Optional[PersonProfile]:
        gid = self.global_id_for(track_id)
        return self.memory.get(gid) if gid else None

    def diagnostics(self) -> Dict[str, object]:
        return {
            "frame": self._frame_index,
            "live_tracks": len(self._bindings),
            "committed_tracks": sum(1 for b in self._bindings.values() if b.committed),
            "memory": self.memory.stats,
            "reid": self.reid.diagnostics(),
        }


def _remember_tracker_id(profile: PersonProfile, tracker_id: int) -> None:
    """Diagnostics only. The length of this list is a direct measure of how
    much tracker churn the identity layer absorbed on this person's behalf —
    it is never read by any metric."""
    if tracker_id not in profile.track_ids_seen:
        profile.track_ids_seen.append(tracker_id)