"""
vision/identity/config.py
-------------------------
Every tunable of the Identity Engine, in one place, with the reasoning
behind each default written next to it.

Rule for this whole package: **no magic numbers outside this file.** If a
constant appears in an algorithm, it lives here. That is what makes it
possible to tune the system per store (a narrow boutique and a supermarket
entrance need different thresholds) without editing algorithm code, and to
A/B two configs against the same recorded footage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatchingConfig:
    """Thresholds that decide "same person" vs "new person"."""

    # Cosine similarity above which a query is accepted as an existing
    # identity. NOTE: 0.96 (the number in the original spec) is unreachable
    # with any real appearance model — genuine re-entries of the same person
    # under a different pose typically land at 0.60-0.85 with OSNet. A 0.96
    # gate would simply never match, and every returning visitor would be
    # counted as new. 0.72 is a realistic starting point; tune it against
    # recorded footage using tools/tune_thresholds.py.
    match_threshold: float = 0.72

    # Below this, the query is definitely a new person.
    new_threshold: float = 0.55

    # Between new_threshold and match_threshold the answer is "don't know".
    # We refuse to guess: the track goes into probation and is re-evaluated
    # with better crops instead of being force-merged or force-split.
    # This band is the single most important safety property of the engine.

    # The best candidate must beat the runner-up by this margin. Without it,
    # two similarly-dressed shoppers (very common: staff uniforms, winter
    # coats) get merged on a coin-flip.
    min_margin_over_runner_up: float = 0.05

    # A track must accumulate this many consistent frames of evidence before
    # its global id is committed and counted. Prevents a single lucky/unlucky
    # crop from creating or merging an identity.
    probation_frames: int = 3

    # --- spatio-temporal prior --------------------------------------------
    # Appearance is not the only evidence available, and on a weak feature
    # extractor it is not even the strongest. A person who vanishes and
    # reappears two seconds later, one metre from where they were, is almost
    # certainly the same person — no clothing model required. Physics rules
    # out the alternatives that appearance cannot.
    #
    # This was added after real footage: colour-histogram embeddings could not
    # bridge a 3.5-second detection gap, but the position could.
    use_motion_prior: bool = True

    # A walking person covers at most this fraction of the frame width per
    # second. Anything further apart than reach * tolerance is not the same
    # human being, whatever the embedding says.
    max_speed_frac_per_s: float = 0.55
    motion_tolerance: float = 1.5

    # Maximum similarity credit for being exactly where they were. Small on
    # purpose: it should tip a borderline decision, never make one.
    motion_bonus: float = 0.10

    # --- solo re-association ----------------------------------------------
    # The narrowest, safest override of appearance there is: when the scene
    # contains exactly ONE plausible person, a track that reappears near
    # where that person was, seconds later, IS that person — whatever the
    # embedding says. There is nobody else it could be.
    #
    # This is not a trick, it is the same reasoning ByteTrack already applies
    # inside its 30-frame buffer, extended to the seconds-long gaps that a
    # detector produces when someone is half out of frame.
    #
    # It fires only when ALL of these hold, so a crowded scene never reaches
    # it and two shoppers can never be merged by it:
    #     * exactly one identity is within physical reach
    #     * that identity was seen within solo_max_gap_seconds
    #     * no OTHER identity has been seen at all in that window
    #     * the reappearance is within solo_max_distance_frac of the frame
    # DEFAULT OFF, and the reason is measured, not theoretical. Enabling it on
    # a real clip did not recover the split it was written for, and it DID
    # merge two different people in the regression suite: one person left, a
    # different person appeared 0.6s later 283px away, and the rule declared
    # them the same human. In a shop that sequence happens every few minutes.
    #
    # It is kept because on some installations it is genuinely safe — a single
    # narrow corridor, a back entrance used by one member of staff. Turn it on
    # only where you can argue that two people cannot swap places, and only
    # after checking it against your own footage.
    solo_reassociation: bool = False
    solo_max_gap_seconds: float = 6.0
    solo_max_distance_frac: float = 0.35

    # Once a track is committed to a global id, it STAYS bound to it and is
    # not re-matched every frame. Re-matching a stable track is pure cost and
    # pure risk: the tracker is already better than ReID at short-range
    # association. ReID exists to bridge gaps, not to second-guess the tracker.
    sticky_binding: bool = True


@dataclass
class MemoryConfig:
    """How long an identity survives, and where."""

    # HOT tier: in-RAM, matched against on every new track. This is the
    # "5-10 minutes" from the spec — it exists to bridge occlusions and
    # walk-out-of-frame-and-back.
    hot_ttl_seconds: float = 600.0

    # WARM tier: an identity evicted from hot is not deleted, it is demoted.
    # It is still matchable, which is the ONLY way "Returning Visitors" can
    # work — a customer who comes back after 40 minutes is by definition
    # outside the hot TTL. The spec's "TTL 10min -> Delete" and its
    # "Returning Visitors" metric were in direct contradiction; two tiers
    # resolve it.
    warm_ttl_seconds: float = 12 * 3600.0

    # Hard cap on hot identities, so a busy Saturday cannot exhaust RAM or
    # turn matching into an O(n) scan of thousands of profiles.
    max_hot_identities: int = 400
    max_warm_identities: int = 5000

    # Embeddings kept per identity. More views = more robust matching, but
    # the gallery must stay small or matching cost explodes. 8 diverse crops
    # is the sweet spot in the ReID literature and in practice.
    gallery_size: int = 8

    # A gallery entry is only replaced by a strictly better-quality crop.
    # Otherwise a long dwell fills the gallery with near-identical frames of
    # someone standing still, and the identity loses view diversity.
    min_quality_improvement: float = 0.02


@dataclass
class ReIDConfig:
    """When and how appearance embeddings are computed."""

    backend: str = os.environ.get("COREWISE_REID_BACKEND", "auto")

    # Crops per frame budget. The whole reason this exists: embedding every
    # person on every frame is what turns a 25 FPS engine into a 6 FPS one.
    # With this budget the cost is bounded no matter how crowded the store is.
    max_crops_per_frame: int = 4

    # A committed track is re-embedded at most this often (frames), only to
    # refresh its gallery with new viewpoints.
    refresh_every_n_frames: int = 30

    # Floor on how often one track may be re-embedded when spending spare
    # budget. Two crops a few frames apart are the same viewpoint, so they
    # would cost inference and add nothing to gallery diversity.
    min_frames_between_crops: int = 8

    # ---- quality gate: crops that fail these are never embedded ----------
    # A tiny, blurry or half-out-of-frame crop produces a garbage embedding,
    # and a garbage embedding is worse than no embedding: it pollutes the
    # gallery permanently and causes false merges forever after.
    min_bbox_height_px: int = 90
    min_bbox_width_px: int = 32
    min_aspect_ratio: float = 1.2          # h/w — a person is taller than wide
    max_aspect_ratio: float = 5.0
    min_detection_confidence: float = 0.55
    border_margin_px: int = 8              # touching the frame edge = truncated body
    min_sharpness: float = 25.0            # variance of Laplacian

    embedding_dim: int = 512

    # Which part of the box becomes the embedding.
    #   "full"       whole box. Correct when people are always fully visible.
    #   "upper_body" head-to-hips ALWAYS, including when the legs are visible.
    #                Consistency is the point: an embedding of a full body and
    #                an embedding of a torso are not comparable, so mixing the
    #                two silently destroys matching. One convention applied
    #                everywhere costs a little trouser colour and buys
    #                re-identification that survives people walking close.
    #   "auto"       measure the footage: if bottom-truncation is common,
    #                switch to upper_body for everything and say so once.
    crop_mode: str = os.environ.get("COREWISE_CROP_MODE", "auto")
    upper_body_fraction: float = 0.62
    auto_sample_size: int = 60
    auto_truncation_ratio: float = 0.25


@dataclass
class ZoneConfig:
    """Entry/exit debouncing."""

    # Frames a person must be consistently on the new side before the
    # transition is accepted. Without hysteresis, someone standing on the
    # threshold chatting generates dozens of Entered/Exited pairs and
    # destroys every metric on the dashboard.
    confirm_frames: int = 3

    # After an accepted transition, ignore further transitions for this long.
    transition_cooldown_seconds: float = 2.0

    # A person not observed for this long has their open dwell interval
    # banked. The VISIT stays open — a tracker gap is not a zone exit — so
    # re-appearing inside the zone resumes the same dwell instead of
    # restarting it. This single number is why identity-based SAM is more
    # accurate than track-based SAM.
    presence_grace_seconds: float = 3.0


@dataclass
class RetailConfig:
    """TAM / SAM / SOM semantics.

    The SAM threshold is deliberately NOT defaulted. tracking.py made the
    dashboard the single source of truth by construction: until a value
    arrives, SAM is not counted at all, so a stale engine-side default can
    never silently win. That property is preserved here exactly.
    """

    sam_min_stay_enabled: bool = False
    sam_min_stay_seconds: Optional[float] = None
    sam_configured: bool = False

    # Which zone name carries TAM, and which carries the entrance.
    tam_zone: str = "__tam__"
    som_zone: str = "__som__"


@dataclass
class IdentityConfig:
    """Root config object passed to the IdentityManager."""

    camera_id: str = field(default_factory=lambda: os.environ.get("CAMERA_ID", "cam_1"))
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    retail: RetailConfig = field(default_factory=RetailConfig)

    # Cross-camera identity is resolved centrally (corewise_server.py), not
    # per-engine. When False the engine allocates local ids only and the
    # server owns global id assignment.
    allow_local_global_ids: bool = True

    # Print per-track commit decisions ("track=7 -> person_ab12 (new)").
    debug: bool = False

    # PRIVACY: raw person crops are never persisted, only embeddings, and
    # only for the retention window above. Flip this on solely for offline
    # threshold tuning on footage you own, never in production.
    persist_crops_for_debug: bool = False

    debug: bool = os.environ.get("COREWISE_IDENTITY_DEBUG", "0") not in ("0", "false", "False")