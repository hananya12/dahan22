"""
vision/identity/memory_engine.py
--------------------------------
Where identities live, and for how long.

The contradiction in the original spec
--------------------------------------
The Bible says two things that cannot both be true:

    TTL: 10 minutes -> Delete
    Dashboard metric: Returning Visitors

A returning visitor is, by definition, someone who comes back *after* the
TTL. If identities are deleted at 10 minutes, Returning Visitors is
permanently 0 and Average Visit Time silently truncates every visit longer
than the TTL.

Two tiers resolve it, and each tier has a different job:

    HOT  (RAM, ~10 min)   bridges occlusions and walk-out-and-back.
                          Matched on every new track. Small and fast.
    WARM (disk, ~12 h)    answers "have we seen this person today?".
                          Matched only when HOT produced nothing, so it costs
                          nothing on the hot path.

Eviction from hot is a DEMOTION, never a deletion. Deletion happens once, at
the warm TTL, and it is a hard delete of the embeddings — which is also the
data-retention guarantee the store's privacy policy will need to state.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from config import MemoryConfig
from contracts import Embedding
from events import EventBus, EventType, IdentityEvent
from person_profile import PersonProfile


class MemoryEngine:
    """Two-tier identity store with TTL, capacity limits and persistence."""

    def __init__(
        self,
        config: MemoryConfig,
        camera_id: str,
        bus: Optional[EventBus] = None,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.camera_id = camera_id
        self.bus = bus
        self._hot: Dict[str, PersonProfile] = {}
        self._warm: Dict[str, PersonProfile] = {}
        self._lock = threading.RLock()
        self._storage_dir = storage_dir
        self._created_total = 0
        # Release callbacks: fired when an identity leaves the HOT tier, so
        # downstream engines (zones, retail) can close its open intervals.
        self._release_callbacks: List = []

    def on_release(self, callback) -> None:
        """Register callback(profile, now) fired on demotion from hot."""
        self._release_callbacks.append(callback)

    # ------------------------------------------------------------ creation

    def create(self, timestamp: float, first_embedding: Optional[Embedding] = None) -> PersonProfile:
        """Mint a brand-new identity.

        The id is a random UUID prefix, not a counter. A counter leaks how
        many people the store has seen and, worse, is not stable across a
        multi-camera cluster where two engines allocate concurrently.
        """
        gid = f"p_{uuid.uuid4().hex[:12]}"
        profile = PersonProfile(global_id=gid, camera_id=self.camera_id,
                                first_seen=timestamp, last_seen=timestamp)
        profile.cameras_seen.append(self.camera_id)
        if first_embedding is not None:
            profile.add_embedding(first_embedding, self.config.gallery_size,
                                  self.config.min_quality_improvement)
        with self._lock:
            self._hot[gid] = profile
            self._created_total += 1
        self._emit(EventType.IDENTITY_CREATED, profile, timestamp)
        return profile

    # ------------------------------------------------------------- lookup

    def get(self, global_id: str) -> Optional[PersonProfile]:
        with self._lock:
            return self._hot.get(global_id) or self._warm.get(global_id)

    def hot_candidates(self) -> List[PersonProfile]:
        with self._lock:
            return list(self._hot.values())

    def warm_candidates(self) -> List[PersonProfile]:
        with self._lock:
            return list(self._warm.values())

    def promote(self, global_id: str, timestamp: float) -> Optional[PersonProfile]:
        """Bring an identity back from warm to hot — a returning visitor."""
        with self._lock:
            profile = self._warm.pop(global_id, None)
            if profile is None:
                return self._hot.get(global_id)
            profile.is_returning = True
            profile.recovered_at = timestamp
            profile.last_seen = timestamp
            self._hot[global_id] = profile
        self._emit(EventType.IDENTITY_RECOVERED, profile, timestamp,
                   data={"away_seconds": round(timestamp - profile.last_seen, 1)})
        return profile

    # ---------------------------------------------------------- maintenance

    def sweep(self, now: Optional[float] = None) -> Dict[str, int]:
        """Demote stale hot identities, delete expired warm ones.

        Cheap enough to call every frame; call it once per second in practice.
        """
        now = now or time.time()
        demoted: List[PersonProfile] = []
        deleted = 0

        with self._lock:
            for gid, profile in list(self._hot.items()):
                if now - profile.last_seen >= self.config.hot_ttl_seconds:
                    profile.close_visit(profile.last_seen)
                    self._hot.pop(gid, None)
                    self._warm[gid] = profile
                    demoted.append(profile)

            # Capacity guard: if still over budget, demote the least recently
            # seen. A hard cap beats an OOM at 18:00 on a Friday.
            if len(self._hot) > self.config.max_hot_identities:
                overflow = sorted(self._hot.values(), key=lambda p: p.last_seen)
                for profile in overflow[: len(self._hot) - self.config.max_hot_identities]:
                    self._hot.pop(profile.global_id, None)
                    self._warm[profile.global_id] = profile
                    demoted.append(profile)

            for gid, profile in list(self._warm.items()):
                if now - profile.last_seen >= self.config.warm_ttl_seconds:
                    del self._warm[gid]
                    deleted += 1

            if len(self._warm) > self.config.max_warm_identities:
                overflow = sorted(self._warm.values(), key=lambda p: p.last_seen)
                for profile in overflow[: len(self._warm) - self.config.max_warm_identities]:
                    del self._warm[profile.global_id]
                    deleted += 1

        for profile in demoted:
            for cb in self._release_callbacks:
                try:
                    cb(profile, now)
                except Exception:
                    pass
            self._emit(EventType.IDENTITY_EXPIRED, profile, now,
                       data={"tier": "demoted_to_warm",
                             "total_dwell": round(profile.total_dwell, 1)})

        return {"demoted": len(demoted), "deleted": deleted,
                "hot": len(self._hot), "warm": len(self._warm)}

    # ------------------------------------------------------------ merging

    def merge(self, keep_id: str, absorb_id: str, timestamp: float) -> Optional[PersonProfile]:
        """Prove two ids are one person and fold one into the other.

        Used when late evidence resolves an earlier split (e.g. a person is
        confidently re-matched after their probation created a new id). The
        surviving identity is the OLDER one, so first_seen — and therefore
        visit duration — stays truthful.
        """
        with self._lock:
            keep = self.get(keep_id)
            absorb = self.get(absorb_id)
            if keep is None or absorb is None or keep_id == absorb_id:
                return keep
            if absorb.first_seen < keep.first_seen:
                keep, absorb = absorb, keep

            for emb in absorb.gallery:
                keep.add_embedding(emb, self.config.gallery_size,
                                   self.config.min_quality_improvement)
            keep.visits.extend(absorb.visits)
            keep.visits.sort(key=lambda v: v.entered_at)
            keep.zone_path.extend(absorb.zone_path)
            keep.zone_path.sort(key=lambda pair: pair[1])
            keep.track_ids_seen.extend(absorb.track_ids_seen)
            for cam in absorb.cameras_seen:
                if cam not in keep.cameras_seen:
                    keep.cameras_seen.append(cam)
            keep.last_seen = max(keep.last_seen, absorb.last_seen)
            # Fold per-zone dwell and retail flags — the merge must leave
            # every engine's numbers consistent, not just the gallery.
            for zone, banked in absorb.zone_dwell_banked.items():
                keep.zone_dwell_banked[zone] = keep.zone_dwell_banked.get(zone, 0.0) + banked
            for zone, entered in absorb.zone_entered_at.items():
                keep.zone_entered_at.setdefault(zone, entered)
            keep.retail.absorb(absorb.retail)

            self._hot.pop(absorb.global_id, None)
            self._warm.pop(absorb.global_id, None)

        self._emit(EventType.IDENTITY_MERGED, keep, timestamp,
                   data={"absorbed": absorb.global_id})
        return keep

    # -------------------------------------------------------- persistence

    def save(self) -> Optional[Path]:
        """Persist the warm tier so a restart does not lose today's visitors."""
        if self._storage_dir is None:
            return None
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._storage_dir / f"identities_{self.camera_id}.json"
        with self._lock:
            payload = {
                "camera_id": self.camera_id,
                "saved_at": time.time(),
                "profiles": [p.to_persistable() for p in
                             list(self._warm.values()) + list(self._hot.values())],
            }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
        return path

    def load(self) -> int:
        """Restore the warm tier from disk (only entries within warm TTL)."""
        if self._storage_dir is None:
            return 0
        path = self._storage_dir / f"identities_{self.camera_id}.json"
        if not path.is_file():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0

        import numpy as np
        now = time.time()
        restored = 0
        for rec in payload.get("profiles", []):
            if now - rec.get("last_seen", 0) >= self.config.warm_ttl_seconds:
                continue
            profile = PersonProfile(
                global_id=rec["global_id"], camera_id=rec.get("camera_id", self.camera_id),
                first_seen=rec.get("first_seen", now), last_seen=rec.get("last_seen", now),
            )
            for g in rec.get("gallery", []):
                profile.gallery.append(Embedding(
                    vector=np.asarray(g["v"], dtype="float32"),
                    quality=g.get("q", 0.5), timestamp=g.get("t", now),
                    camera_id=g.get("c", self.camera_id),
                ))
            with self._lock:
                self._warm[profile.global_id] = profile
            restored += 1
        return restored

    # ------------------------------------------------------------- helpers

    def _emit(self, event_type: EventType, profile: PersonProfile,
              timestamp: float, data: Optional[dict] = None) -> None:
        if self.bus is None:
            return
        self.bus.emit(IdentityEvent(
            type=event_type, global_id=profile.global_id, camera_id=self.camera_id,
            timestamp=timestamp, position=profile.last_position, data=data or {},
        ))

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"hot": len(self._hot), "warm": len(self._warm),
                    "created_total": self._created_total}

    def __iter__(self) -> Iterator[PersonProfile]:
        with self._lock:
            return iter(list(self._hot.values()) + list(self._warm.values()))
