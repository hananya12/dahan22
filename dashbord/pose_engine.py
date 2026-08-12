"""
pose_engine.py
--------------
Advanced human skeleton / pose estimation — an OPTIONAL module.

The contract (from the product spec):

  * OFF (default): the pose model is NEVER loaded. No inference runs, no
    GPU/CPU is spent, and the rest of the system behaves exactly as if this
    file did not exist. Turning it off after use RELEASES the model.
  * ON: a YOLO pose model (17 COCO keypoints) is lazily loaded, run on the
    live frame, and each skeleton is matched to a tracked person's bounding
    box — from where main.py attaches it to that person's GLOBAL identity
    (PersonProfile.pose), never to a bare bounding box.

The dashboard flips this at runtime via the ``pose_enabled`` control field;
``set_enabled`` is safe to call every frame (it only acts on transitions).

Model: Ultralytics YOLO Pose (``yolov8n-pose.pt`` by default). Ultralytics
is already a project dependency for detection, so enabling skeletons adds no
new installs; the checkpoint downloads on first use. Swap ``model_path`` for
a larger checkpoint (yolov8s/m-pose) when accuracy matters more than FPS.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# COCO-17 keypoint order produced by YOLO pose models.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton edges for drawing (indices into KEYPOINT_NAMES).
SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 5), (0, 6),                           # head to shoulders
]

MIN_KP_CONF = 0.35


class PoseEngine:
    """Lazy-loading, toggleable skeleton estimator."""

    def __init__(self, model_path: str = "yolov8n-pose.pt",
                 device: Optional[str] = None,
                 every_n_frames: int = 1) -> None:
        self.model_path = model_path
        self.device = device
        self.every_n_frames = max(1, int(every_n_frames))
        self.enabled = False
        self._model = None            # None while disabled — that IS the contract
        self._load_error: Optional[str] = None
        self._frame_i = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_enabled(self, enabled: bool, model_path: Optional[str] = None) -> None:
        """Idempotent per-frame toggle. Loads on OFF->ON, releases on ON->OFF."""
        if model_path and model_path != self.model_path:
            self.model_path = model_path
            if self._model is not None:           # hot-swap: drop, reload below
                self._model = None
        enabled = bool(enabled)
        if enabled == self.enabled and (self._model is not None or not enabled):
            return
        self.enabled = enabled
        if not enabled:
            self._model = None                    # release weights / VRAM
            print("[POSE] Skeleton detection OFF — model released.")
            return
        self._load()

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            t0 = time.time()
            self._model = YOLO(self.model_path)
            self._load_error = None
            print(f"[POSE] Skeleton detection ON — loaded {self.model_path} "
                  f"in {time.time() - t0:.1f}s.")
        except Exception as exc:
            self._load_error = str(exc)
            self.enabled = False
            self._model = None
            print(f"[POSE] Could not load pose model '{self.model_path}': {exc}. "
                  "Skeleton detection stays OFF; the rest of the system is unaffected.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray,
                persons: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Run pose on the frame and assign each skeleton to a tracked person.

        ``persons`` is detection.py's output: [{"id": track_id, "box": (x1,
        y1, x2, y2, conf)}, ...]. Returns {track_id: pose_payload}. The
        caller attaches the payload to the person's GLOBAL id — this module
        deliberately knows nothing about identities.
        """
        if not self.enabled or self._model is None or frame is None or not persons:
            return {}
        self._frame_i += 1
        if self._frame_i % self.every_n_frames:
            return {}

        try:
            results = self._model.predict(frame, verbose=False, conf=0.25,
                                          device=self.device)
        except Exception as exc:
            print(f"[POSE] inference failed: {exc}")
            return {}
        if not results or results[0].keypoints is None:
            return {}

        r = results[0]
        kps = r.keypoints
        xy = kps.xy.cpu().numpy() if hasattr(kps.xy, "cpu") else np.asarray(kps.xy)
        conf = (kps.conf.cpu().numpy() if kps.conf is not None and hasattr(kps.conf, "cpu")
                else (np.asarray(kps.conf) if kps.conf is not None
                      else np.ones(xy.shape[:2], dtype=np.float32)))
        pose_boxes = (r.boxes.xyxy.cpu().numpy()
                      if r.boxes is not None and len(r.boxes) else None)

        out: Dict[int, Dict[str, Any]] = {}
        used: set = set()
        for person in persons:
            x1, y1, x2, y2 = (float(v) for v in person["box"][:4])
            best_i, best_iou = -1, 0.15            # minimum overlap to accept
            for i in range(xy.shape[0]):
                if i in used:
                    continue
                pb = (pose_boxes[i] if pose_boxes is not None
                      else self._bbox_of(xy[i], conf[i]))
                iou = self._iou((x1, y1, x2, y2), pb)
                if iou > best_iou:
                    best_i, best_iou = i, iou
            if best_i < 0:
                continue
            used.add(best_i)
            out[int(person["id"])] = self._payload(xy[best_i], conf[best_i])
        return out

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _payload(self, xy: np.ndarray, conf: np.ndarray) -> Dict[str, Any]:
        keypoints = []
        named: Dict[str, Tuple[float, float]] = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            x, y, c = float(xy[i][0]), float(xy[i][1]), float(conf[i])
            keypoints.append([round(x, 1), round(y, 1), round(c, 3)])
            if c >= MIN_KP_CONF:
                named[name] = (x, y)
        visible = len(named)
        return {
            "skeleton_detected": visible >= 4,
            "keypoints_count": visible,
            "keypoints": keypoints,
            "keypoint_names": KEYPOINT_NAMES,
            "posture_features": self._posture(named),
            "detected_at": time.time(),
        }

    def _posture(self, kp: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Coarse, explainable posture features derived from the skeleton."""
        feats: Dict[str, Any] = {}

        ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
        lh, rh = kp.get("left_hip"), kp.get("right_hip")

        # Body orientation: shoulder width vs torso height. Shoulders seen
        # nearly edge-on (narrow) => the person faces the camera sideways.
        if ls and rs:
            sw = abs(ls[0] - rs[0])
            mid_sh = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            if lh and rh:
                mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
                torso = math.hypot(mid_sh[0] - mid_hip[0], mid_sh[1] - mid_hip[1])
                if torso > 1:
                    ratio = sw / torso
                    feats["orientation"] = ("frontal" if ratio > 0.55
                                            else "profile" if ratio < 0.3
                                            else "angled")
                    # Torso lean off vertical, degrees.
                    lean = math.degrees(math.atan2(mid_sh[0] - mid_hip[0],
                                                   mid_hip[1] - mid_sh[1]))
                    feats["torso_lean_deg"] = round(lean, 1)
                    feats["leaning"] = abs(lean) > 20.0

        # Standing / crouching: knee angle (hip-knee-ankle), averaged.
        angles = []
        for side in ("left", "right"):
            h, k, a = kp.get(f"{side}_hip"), kp.get(f"{side}_knee"), kp.get(f"{side}_ankle")
            if h and k and a:
                angles.append(self._angle(h, k, a))
        if angles:
            knee = sum(angles) / len(angles)
            feats["knee_angle_deg"] = round(knee, 1)
            feats["stance"] = ("standing" if knee > 150
                               else "crouching" if knee < 110 else "bent")

        # Reaching: a wrist above its shoulder (image y grows downward).
        for side in ("left", "right"):
            w, s = kp.get(f"{side}_wrist"), kp.get(f"{side}_shoulder")
            if w and s and w[1] < s[1]:
                feats["arm_raised"] = True
                break
        feats.setdefault("arm_raised", False)
        return feats

    # ------------------------------------------------------------------
    # Drawing (used by main.py when labels are shown)
    # ------------------------------------------------------------------

    @staticmethod
    def draw(frame: np.ndarray, pose: Dict[str, Any],
             color=(90, 220, 255)) -> None:
        import cv2
        pts = pose.get("keypoints") or []
        for a, b in SKELETON_EDGES:
            if a < len(pts) and b < len(pts):
                xa, ya, ca = pts[a]
                xb, yb, cb = pts[b]
                if ca >= MIN_KP_CONF and cb >= MIN_KP_CONF:
                    cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                             color, 2, lineType=cv2.LINE_AA)
        for x, y, c in pts:
            if c >= MIN_KP_CONF:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 160), -1,
                           lineType=cv2.LINE_AA)

    # ------------------------------------------------------------------

    def analyze(self, crop) -> Dict[str, Any]:
        """Back-compat with the original stub API (single-crop analysis)."""
        if not self.enabled or self._model is None or crop is None:
            return {"skeleton_detected": False, "keypoints_count": 0, "keypoints": []}
        res = self.process(crop, [{"id": 0, "box": (0, 0, crop.shape[1],
                                                    crop.shape[0], 1.0)}])
        return res.get(0, {"skeleton_detected": False, "keypoints_count": 0,
                           "keypoints": []})

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.enabled,
                "model": self.model_path if self.enabled else None,
                "loaded": self._model is not None,
                "error": self._load_error}

    # ------------------------------------------------------------------

    @staticmethod
    def _angle(a, b, c) -> float:
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        return math.degrees(math.acos(cosang))

    @staticmethod
    def _bbox_of(xy: np.ndarray, conf: np.ndarray):
        good = xy[conf >= MIN_KP_CONF]
        if not len(good):
            return (0.0, 0.0, 0.0, 0.0)
        return (float(good[:, 0].min()), float(good[:, 1].min()),
                float(good[:, 0].max()), float(good[:, 1].max()))

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
        return inter / union if union > 0 else 0.0
