"""
vision/identity/contracts.py
----------------------------
The seams of the Identity Engine: the data that crosses module boundaries,
and the interfaces modules depend on.

Every module in this package depends on THIS file, never on each other's
implementations. That is what makes it possible to swap OSNet for a
transformer ReID model, or the in-RAM memory for Redis, without touching the
Identity Manager — and to run the entire pipeline in a unit test with a fake
embedding backend and no camera, no GPU and no YOLO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


class Side(str, Enum):
    """Which side of the store boundary a person is on."""
    UNKNOWN = "unknown"
    OUTSIDE = "outside"
    INSIDE = "inside"


class Verdict(str, Enum):
    """The Matching Engine's three possible answers. The third one is the
    reason this system can be trusted: it is allowed to say 'I don't know'."""
    MATCH = "match"
    NEW = "new"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class TrackObservation:
    """One person, one frame, straight out of the detector + tracker.

    This is the ONLY input type of the Identity Engine. It deliberately
    carries no image data beyond the crop, so the engine can be driven from
    live frames, from recorded footage, or from a synthetic test fixture.
    """

    track_id: int
    bbox: BBox
    confidence: float
    frame_index: int
    timestamp: float
    frame_width: int
    frame_height: int
    crop: Optional[np.ndarray] = None  # BGR, only when the quality gate passed


@dataclass
class Embedding:
    """An L2-normalised appearance vector plus how much we trust it."""

    vector: np.ndarray
    quality: float          # 0..1, from the quality gate
    timestamp: float
    camera_id: str

    def __post_init__(self) -> None:
        v = np.asarray(self.vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        # Pre-normalising here means cosine similarity is a plain dot product
        # everywhere else — matching stays O(d) with no per-comparison sqrt.
        self.vector = v / norm if norm > 1e-6 else v

    def similarity(self, other: "Embedding") -> float:
        return float(np.dot(self.vector, other.vector))


@dataclass
class MatchResult:
    """Why the Matching Engine decided what it decided.

    The explanation fields are not decoration: when a store manager says
    'it counted my cashier eleven times', this is the only thing that can
    answer why.
    """

    verdict: Verdict
    global_id: Optional[str] = None
    score: float = 0.0
    runner_up_score: float = 0.0
    runner_up_id: Optional[str] = None
    candidates_considered: int = 0

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score

    def explain(self) -> str:
        return (
            f"{self.verdict.value} id={self.global_id} score={self.score:.3f} "
            f"runner_up={self.runner_up_id}:{self.runner_up_score:.3f} "
            f"margin={self.margin:.3f} candidates={self.candidates_considered}"
        )


@dataclass
class IdentityDecision:
    """What the Identity Manager concluded for one track, this frame."""

    track_id: int
    global_id: Optional[str]
    is_committed: bool          # False while in probation — do not count yet
    is_new_identity: bool
    is_returning: bool          # matched an identity recovered from warm memory
    match: Optional[MatchResult] = None


class EmbeddingBackend(Protocol):
    """Anything that turns person crops into appearance vectors.

    Batched by contract, because per-crop inference calls are where the
    frame budget dies.
    """

    name: str
    dim: int

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalised vectors."""
        ...


class EventSink(Protocol):
    """Anything that consumes identity events: the websocket client, the
    analytics engine, the replay metadata track, a future SQL writer."""

    def emit(self, event: "IdentityEvent") -> None:  # noqa: F821
        ...
