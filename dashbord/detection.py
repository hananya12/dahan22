"""
detection.py
-------------
YOLO Person Detection + Tracking for Corewise.

Adds persistent IDs using ByteTrack. Confidence threshold and model can
be changed live (e.g. from a dashboard command) without restarting the
engine.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0


class PersonDetector:
    """Wraps a YOLOv8 model with ByteTrack for live person tracking."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5) -> None:
        self._model_cache: Dict[str, YOLO] = {}
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> YOLO:
        if model_path not in self._model_cache:
            try:
                self._model_cache[model_path] = YOLO(model_path)
            except Exception as exc:
                raise RuntimeError(f"Failed loading YOLO model '{model_path}': {exc}") from exc
        return self._model_cache[model_path]

    def set_confidence(self, confidence: float) -> None:
        """Update the minimum detection confidence used from now on."""
        self.confidence_threshold = max(0.0, min(1.0, confidence))

    def set_model(self, model_path: str) -> None:
        """Hot-swap the YOLO model used for detection (e.g. yolov8n/s/m)."""
        if model_path == self.model_path:
            return
        self.model = self._load_model(model_path)
        self.model_path = model_path

    def track_people(self, frame: Optional[np.ndarray]) -> List[dict]:
        """Run detection + ByteTrack on one frame, returning person boxes with IDs."""
        if frame is None:
            return []

        results = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS_ID],
            verbose=False,
            tracker="bytetrack.yaml",
        )[0]

        people: List[dict] = []

        if results.boxes.id is None:
            return people

        ids = results.boxes.id.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for track_id, box, conf in zip(ids, boxes, confs):
            if conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)

            people.append(
                {
                    "id": int(track_id),
                    "box": (x1, y1, x2, y2, float(conf)),
                }
            )

        return people
