"""
vision/identity/reid_engine.py
------------------------------
Turns person crops into appearance embeddings — and, just as importantly,
decides WHICH crops are worth embedding at all.

The performance reality
-----------------------
The naive design in the spec ("Crop Person -> ReID" for every person on every
frame) does not survive contact with the hardware this runs on. On a laptop
CPU, OSNet-x0.25 costs roughly 6-12 ms per crop. Eight shoppers in frame is
~80 ms of ReID *on top of* YOLO, and a 25 FPS engine becomes a 7 FPS engine —
at which point the tracker starts losing people, which creates more identity
work, which slows it further.

Two mechanisms keep the cost bounded and the accuracy high:

  1. QUALITY GATE — a tiny, blurry, or truncated crop yields a garbage
     vector, and a garbage vector in a gallery causes false merges *forever
     after*. Rejecting bad crops is not an optimisation, it is the main
     accuracy lever. Most frames of most people are not worth embedding.
  2. FRAME BUDGET — at most N crops per frame, spent on the tracks that need
     it most (new/probationary first, then stalest gallery). A committed,
     well-known track is refreshed rarely. Cost becomes O(budget), not
     O(people).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import ReIDConfig
from contracts import BBox, Embedding, TrackObservation


# Placeholder handed to pixel-free backends so batch lengths stay aligned.
_NO_PIXELS = np.zeros((1, 1, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    passed: bool
    score: float
    reason: str = ""


def assess_crop(obs: TrackObservation, cfg: ReIDConfig, crop: Optional[np.ndarray]) -> QualityReport:
    """Decide whether this observation deserves an embedding, and how much to
    trust it if so. Returns a 0..1 quality score used for gallery ranking."""
    x1, y1, x2, y2 = obs.bbox
    w, h = x2 - x1, y2 - y1

    if obs.confidence < cfg.min_detection_confidence:
        return QualityReport(False, 0.0, f"low_confidence({obs.confidence:.2f})")
    if h < cfg.min_bbox_height_px or w < cfg.min_bbox_width_px:
        return QualityReport(False, 0.0, f"too_small({int(w)}x{int(h)})")

    # Not all truncation is equal — a distinction learned from real footage.
    #
    # The original rule rejected a crop touching ANY frame edge. On a store
    # camera that is right. On a camera where people walk close, EVERY person
    # is cut off at the bottom, so the gate rejected 98% of crops and the
    # engine had almost no embeddings to re-identify anyone with.
    #
    #   left / right  -> the body is cut ACROSS its width. Half a torso is
    #                    genuinely unusable, and the missing half is a
    #                    different half from frame to frame: reject.
    #   top / bottom  -> the head or the legs are missing. The TORSO — which
    #                    carries nearly all of the clothing signal a ReID
    #                    model relies on — is intact. Usable, provided crops
    #                    are normalised consistently (see crop_mode).
    #
    # This distinction came from real footage: on a camera people walk close
    # to, 94% of detections touched the TOP edge (heads above frame). The
    # original rule rejected all of them, the engine got 6 embeddings out of
    # 394 detections, and re-identification had nothing to work with.
    m = cfg.border_margin_px
    if x1 <= m or x2 >= obs.frame_width - m:
        return QualityReport(False, 0.0, "truncated_at_side")

    vertical_cut = y1 <= m or y2 >= obs.frame_height - m
    bottom_cut = vertical_cut
    aspect = h / max(w, 1.0)
    if not bottom_cut:
        if not (cfg.min_aspect_ratio <= aspect <= cfg.max_aspect_ratio):
            # Wrong aspect = two overlapping people in one box, or a partial body.
            return QualityReport(False, 0.0, f"bad_aspect({aspect:.2f})")
    elif aspect > cfg.max_aspect_ratio:
        return QualityReport(False, 0.0, f"bad_aspect({aspect:.2f})")

    sharpness = 1.0
    if crop is not None and crop.size:
        try:
            import cv2
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if lap_var < cfg.min_sharpness:
                return QualityReport(False, 0.0, f"blurry({lap_var:.0f})")
            sharpness = min(1.0, lap_var / 400.0)
        except Exception:
            pass

    # Composite score: bigger, more confident, sharper crops rank higher and
    # therefore win gallery slots.
    size_score = min(1.0, h / 320.0)
    score = 0.45 * size_score + 0.30 * float(obs.confidence) + 0.25 * sharpness
    return QualityReport(True, round(min(1.0, score), 4))


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class ColorHistogramBackend:
    """Dependency-free baseline: HSV histograms of the torso and legs.

    Honest about what it is — this is *weak* ReID. It handles "same person
    walked behind a shelf and came back 8 seconds later" acceptably and fails
    on anything harder, especially across cameras with different white
    balance. It exists so the pipeline runs end-to-end on day one, and so
    every other module can be developed and tested before a torch dependency
    is introduced. Ship the real backend before trusting cross-camera ids.
    """

    name = "color_histogram"

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        import cv2
        out = np.zeros((len(crops), self.dim), dtype=np.float32)
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            h = crop.shape[0]
            # Split vertically: torso and legs carry the clothing signal;
            # the head region is mostly noise at retail resolutions.
            bands = [crop[int(h * 0.15):int(h * 0.55)], crop[int(h * 0.55):int(h * 0.95)]]
            feats = []
            for band in bands:
                if band.size == 0:
                    feats.append(np.zeros(64, dtype=np.float32))
                    continue
                hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
                hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
                feats.append(hist[:64])
            vec = np.concatenate(feats)[: self.dim]
            if vec.shape[0] < self.dim:
                vec = np.pad(vec, (0, self.dim - vec.shape[0]))
            out[i] = vec
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-6)


class OSNetBackend:
    """Production backend: OSNet via torchreid, batched, half precision on GPU.

    Loaded lazily so importing this package never drags in torch. Raises on
    construction if torchreid is unavailable, so ReIDEngine can fall back
    loudly rather than silently degrading.
    """

    name = "osnet"

    def __init__(self, model_name: str = "osnet_x0_25", device: Optional[str] = None) -> None:
        import torch  # noqa: F401
        import torchreid

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=True
        )
        self.model.eval().to(self.device)
        self.dim = 512
        self._input_size = (256, 128)  # h, w — the ReID standard
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        import cv2
        torch = self._torch
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)

        batch = np.zeros((len(crops), 3, *self._input_size), dtype=np.float32)
        for i, crop in enumerate(crops):
            img = cv2.resize(crop, (self._input_size[1], self._input_size[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - self._mean) / self._std
            batch[i] = img.transpose(2, 0, 1)

        with torch.no_grad():
            tensor = torch.from_numpy(batch).to(self.device)
            feats = self.model(tensor).cpu().numpy().astype(np.float32)

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-6)


def build_backend(cfg: ReIDConfig):
    """Pick a backend, announcing loudly what was chosen and why.

    Silent fallback is forbidden here: an operator must never be told
    "identity is running" while it is actually running on colour histograms.
    """
    wanted = (cfg.backend or "auto").lower()
    if wanted in ("auto", "osnet"):
        try:
            backend = OSNetBackend()
            print(f"[REID] backend=osnet device={backend.device} dim={backend.dim}")
            return backend
        except Exception as exc:
            if wanted == "osnet":
                raise RuntimeError(
                    "COREWISE_REID_BACKEND=osnet was requested but torchreid could not "
                    f"be loaded: {exc}. Install with: pip install torch torchreid"
                ) from exc
            print(f"[REID] WARNING: OSNet unavailable ({exc.__class__.__name__}). "
                  "Falling back to the colour-histogram baseline — cross-camera "
                  "identity will be unreliable until torchreid is installed.")
    backend = ColorHistogramBackend()
    print(f"[REID] backend=color_histogram dim={backend.dim} (baseline)")
    return backend


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ReIDEngine:
    """Quality gate + frame budget + batched inference."""

    def __init__(self, config: ReIDConfig, camera_id: str = "cam_1", backend=None) -> None:
        self.config = config
        self.camera_id = camera_id
        self.backend = backend or build_backend(config)
        self._last_embedded_frame: Dict[int, int] = {}
        self._upper_body_mode = (config.crop_mode == "upper_body")
        self._auto_decided = config.crop_mode != "auto"
        self._auto_seen = 0
        self._auto_truncated = 0
        # Set by IdentityManager: "does this track's identity still have room
        # for another viewpoint?" Keeps the spare budget from re-embedding a
        # person whose gallery is already full and diverse.
        self.gallery_has_room = None
        self.stats = {"gated_out": 0, "embedded": 0, "budget_skipped": 0}
        self._gate_reasons: Dict[str, int] = {}

    # ------------------------------------------------------------------

    def crop_person(self, frame: np.ndarray, bbox: BBox,
                    upper_body: Optional[bool] = None) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        if upper_body is None:
            upper_body = self._upper_body_mode
        if upper_body:
            # Central torso band. Taking the MIDDLE of the visible box gives
            # roughly the same body region whether it is the head or the legs
            # that fell outside the frame — which is what makes two crops of
            # the same person comparable at all.
            height = y2 - y1
            band = max(1, int(height * self.config.upper_body_fraction))
            centre = y1 + height // 2
            y1 = max(0, centre - band // 2)
            y2 = min(frame.shape[0], y1 + band)
        return frame[y1:y2, x1:x2]

    def _observe_truncation(self, obs: TrackObservation) -> None:
        """Decide the crop convention from the footage itself.

        Runs only while crop_mode is "auto", over the first N observations.
        Announcing the decision matters: an operator must be able to see WHY
        the engine is looking at torsos.
        """
        if self.config.crop_mode != "auto" or self._auto_decided:
            return
        self._auto_seen += 1
        if (obs.bbox[3] >= obs.frame_height - self.config.border_margin_px
                or obs.bbox[1] <= self.config.border_margin_px):
            self._auto_truncated += 1
        if self._auto_seen < self.config.auto_sample_size:
            return
        ratio = self._auto_truncated / self._auto_seen
        self._auto_decided = True
        if ratio >= self.config.auto_truncation_ratio:
            self._upper_body_mode = True
            print(f"[REID] {ratio*100:.0f}% of detections are cut off at the top or "
                  "bottom of the frame — switching to torso crops so every embedding "
                  "is comparable. (People walk close to this camera.)")
        else:
            print(f"[REID] full-body crops ({ratio*100:.0f}% bottom-truncated).")

    def select(
        self,
        observations: Sequence[TrackObservation],
        priority_track_ids: Sequence[int] = (),
        frame_index: int = 0,
    ) -> List[TrackObservation]:
        """Choose which observations get embedded this frame.

        Priority order:
          1. tracks with no identity yet (new / in probation) — these are the
             only ones whose answer actually changes a count;
          2. committed tracks whose gallery refresh is due;
          3. SPARE BUDGET: anything else, to build galleries out.

        Step 3 was added after real footage. The budget exists to bound cost
        when the store is busy — but with one or two people in frame it went
        almost entirely unspent, and identities ended up with two or three
        views each. A gallery that thin decides a match on one unlucky crop:
        the same person scored 0.99 on their best pair of views and 0.46 on
        their worst, and with three views in the gallery it was luck which one
        the comparison landed on.

        Spending an idle budget is free accuracy. When the store IS busy,
        steps 1 and 2 fill the budget on their own and this changes nothing.
        """
        priority = set(priority_track_ids)
        urgent, routine, spare = [], [], []

        for obs in observations:
            if obs.track_id in priority:
                urgent.append(obs)
                continue
            last = self._last_embedded_frame.get(obs.track_id)
            if last is None or frame_index - last >= self.config.refresh_every_n_frames:
                routine.append(obs)
            elif frame_index - last >= self.config.min_frames_between_crops:
                # Not due, but the budget is there. Only worth spending on a
                # gallery that still has room for another viewpoint.
                spare.append(obs)

        routine.sort(key=lambda o: self._last_embedded_frame.get(o.track_id, -10_000))
        spare.sort(key=lambda o: self._last_embedded_frame.get(o.track_id, -10_000))

        budget = self.config.max_crops_per_frame
        chosen = (urgent + routine)[:budget]
        if len(chosen) < budget and self.gallery_has_room is not None:
            for obs in spare:
                if len(chosen) >= budget:
                    break
                if self.gallery_has_room(obs.track_id):
                    chosen.append(obs)

        skipped = len(urgent) + len(routine) - len(chosen)
        if skipped > 0:
            self.stats["budget_skipped"] += skipped
        return chosen

    def embed(
        self,
        frame: np.ndarray,
        observations: Sequence[TrackObservation],
        frame_index: int = 0,
    ) -> Dict[int, Embedding]:
        """Embed the selected observations that survive the quality gate."""
        crops: List[np.ndarray] = []
        keep: List[Tuple[TrackObservation, float]] = []
        # A backend may declare that it works without pixels — the offline
        # tuner serves embeddings recorded during an earlier pass, so a
        # threshold sweep never re-runs detection or inference.
        needs_pixels = getattr(self.backend, "needs_pixels", True)

        for obs in observations:
            self._observe_truncation(obs)
            crop = obs.crop
            if crop is None and frame is not None:
                crop = self.crop_person(frame, obs.bbox)
            report = assess_crop(obs, self.config, crop)
            if not report.passed or (crop is None and needs_pixels):
                self.stats["gated_out"] += 1
                reason = report.reason or "no_pixels"
                self._gate_reasons[reason] = self._gate_reasons.get(reason, 0) + 1
                continue
            crops.append(crop if crop is not None else _NO_PIXELS)
            keep.append((obs, report.score))

        if not crops:
            return {}

        if getattr(self.backend, "wants_keys", False):
            # Opt-in: the backend receives a stable (frame, track) key per crop.
            # Used by the offline tuner to serve PRE-COMPUTED embeddings, so a
            # threshold sweep costs seconds instead of another full YOLO pass.
            vectors = self.backend.embed_batch(
                crops, keys=[(o.frame_index, o.track_id) for o, _ in keep])
        else:
            vectors = self.backend.embed_batch(crops)
        self.stats["embedded"] += len(crops)

        out: Dict[int, Embedding] = {}
        for (obs, quality), vector in zip(keep, vectors):
            out[obs.track_id] = Embedding(
                vector=vector, quality=quality,
                timestamp=obs.timestamp, camera_id=self.camera_id,
            )
            self._last_embedded_frame[obs.track_id] = frame_index
        return out

    def forget_track(self, track_id: int) -> None:
        self._last_embedded_frame.pop(track_id, None)

    def diagnostics(self) -> Dict[str, object]:
        """What the quality gate is actually rejecting — the first thing to
        look at when identity accuracy is poor. A gate rejecting 95% of crops
        means the camera is too far away or the thresholds are wrong, and no
        amount of threshold tuning downstream will fix it."""
        return {**self.stats, "backend": self.backend.name,
                "crop_mode": "upper_body" if self._upper_body_mode else "full",
                "gate_reasons": dict(sorted(self._gate_reasons.items(),
                                            key=lambda kv: -kv[1])[:8])}