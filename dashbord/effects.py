"""
effects.py
----------
Live image effects for Corewise, applied on the raw camera frame each
loop iteration in main.py:

    * Face Blur    - blurs ONLY detected faces, never the full body.
    * Night Vision - a professional green CCTV look (contrast boost +
                      green channel grading + vignette + scanlines),
                      not a flat green color filter.

Future Image Effects (heatmap overlays, low-light enhancement, etc.)
should be added here as additional `Effect` subclasses so main.py never
needs to change to support them.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


def resolve_cascade(cascade_filename: str) -> Optional[str]:
    """Locate a Haar cascade XML, wherever it actually lives.

    ROOT CAUSE of "Face Blur does nothing": the cascade was loaded ONLY from
    a hardcoded path next to this file. That file is not shipped by pip and
    is usually absent, so ``CascadeClassifier`` silently produced an empty
    classifier, ``available`` became False and ``apply()`` returned the frame
    untouched - a silent no-op with no error anywhere.

    Search order:
      1. next to this file (lets a project ship its own tuned cascade)
      2. ``cv2.data.haarcascades`` - always present with the opencv wheel
    Returns the resolved path, or None if it genuinely cannot be found.
    """
    local = os.path.join(os.path.dirname(__file__), cascade_filename)
    if os.path.isfile(local):
        return local
    try:
        bundled = os.path.join(cv2.data.haarcascades, cascade_filename)
        if os.path.isfile(bundled):
            return bundled
    except AttributeError:
        # Very old cv2 builds without cv2.data
        pass
    return None


class FaceBlurEffect:
    """Detects faces with Haar cascades and blurs only the face region."""

    def __init__(self, cascade_filename: str = "haarcascade_frontalface_default.xml") -> None:
        self.cascade = cv2.CascadeClassifier()
        resolved = resolve_cascade(cascade_filename)
        if resolved:
            self.cascade = cv2.CascadeClassifier(resolved)

        # Profile cascade as well: in a supermarket most heads are turned, and
        # a frontal-only cascade misses them entirely. Loaded best-effort.
        self.profile_cascade = None
        profile_path = resolve_cascade("haarcascade_profileface.xml")
        if profile_path:
            candidate = cv2.CascadeClassifier(profile_path)
            if not candidate.empty():
                self.profile_cascade = candidate

        self.available = not self.cascade.empty()
        self.cascade_path = resolved
        if self.available:
            print(f"[EFFECTS] Face cascade loaded from: {resolved}"
                  + (" (+ profile cascade)" if self.profile_cascade is not None else ""))
        else:
            print("[EFFECTS] WARNING: face cascade could not be loaded from the project "
                  "folder OR from cv2.data.haarcascades - face blur will be disabled. "
                  "Reinstall opencv-contrib-python to restore it.")

    def _detect_faces_in_box(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Run face detection only inside a person's bounding box (cheaper + more accurate)."""
        x1, y1, x2, y2 = box
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

        if x2 <= x1 or y2 <= y1:
            return []

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        gray = cv2.equalizeHist(gray)

        found = list(
            self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
            )
        )
        if self.profile_cascade is not None:
            found += list(
                self.profile_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
                )
            )
            # Mirrored pass catches the other-facing profile.
            flipped = cv2.flip(gray, 1)
            w_roi = gray.shape[1]
            for (fx, fy, fw, fh) in self.profile_cascade.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
            ):
                found.append((w_roi - fx - fw, fy, fw, fh))

        return [(x1 + fx, y1 + fy, fw, fh) for (fx, fy, fw, fh) in found]

    def _detect_faces_full_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Whole-frame face pass.

        Needed because searching only inside YOLO person boxes means a face
        YOLO missed (edge of frame, heavy occlusion, detection paused) was
        never blurred - which violates "every detected face is blurred".
        """
        h, w = frame.shape[:2]
        return self._detect_faces_in_box(frame, (0, 0, w, h))

    def apply(
        self,
        frame: np.ndarray,
        person_boxes: List[Tuple[int, int, int, int]],
        full_frame_pass: bool = True,
    ) -> np.ndarray:
        """Blur every detected face.

        Runs inside each person box (cheap + accurate) AND, when
        *full_frame_pass* is on, over the whole frame so faces outside any
        person box are still covered.
        """
        if not self.available:
            return frame

        regions: List[Tuple[int, int, int, int]] = []
        for box in person_boxes:
            regions.extend(self._detect_faces_in_box(frame, box))
        if full_frame_pass:
            regions.extend(self._detect_faces_full_frame(frame))

        for (fx, fy, fw, fh) in regions:
            # Pad generously so hairline/jaw/ears are covered too.
            pad_x = int(fw * 0.25)
            pad_y = int(fh * 0.30)
            x1 = max(fx - pad_x, 0)
            y1 = max(fy - pad_y, 0)
            x2 = min(fx + fw + pad_x, frame.shape[1])
            y2 = min(fy + fh + pad_y, frame.shape[0])

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # Kernel scales with face size so small/far faces are still
            # fully anonymised rather than merely softened.
            sigma = max(12.0, (x2 - x1) / 4.0)
            blurred = cv2.GaussianBlur(face_roi, (0, 0), sigmaX=sigma, sigmaY=sigma)
            frame[y1:y2, x1:x2] = blurred

        return frame


class NightVisionEffect:
    """Professional green CCTV-style night vision, not a flat green tint."""

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._vignette_cache: dict[Tuple[int, int], np.ndarray] = {}
        self._noise_seed = 0

    def _get_vignette(self, height: int, width: int) -> np.ndarray:
        key = (height, width)
        if key in self._vignette_cache:
            return self._vignette_cache[key]

        kernel_x = cv2.getGaussianKernel(width, width * 0.55)
        kernel_y = cv2.getGaussianKernel(height, height * 0.55)
        mask = kernel_y @ kernel_x.T
        mask = mask / mask.max()
        self._vignette_cache[key] = mask.astype(np.float32)
        return self._vignette_cache[key]

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return a CCTV-style green night-vision version of the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)

        # Slight film grain, subtle so it doesn't hide detections.
        self._noise_seed += 1
        rng = np.random.default_rng(self._noise_seed % 997)
        noise = rng.normal(0, 6, enhanced.shape).astype(np.float32)
        enhanced = np.clip(enhanced.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Vignette (darker corners, like an IR camera lens).
        vignette = self._get_vignette(*enhanced.shape[:2])
        enhanced = (enhanced.astype(np.float32) * (0.55 + 0.45 * vignette)).astype(np.uint8)

        # Map through a green-dominant curve instead of a flat tint.
        zeros = np.zeros_like(enhanced)
        green_channel = cv2.equalizeHist(enhanced)
        night = cv2.merge([zeros, green_channel, zeros])

        # Faint horizontal scanlines every 3 pixels.
        night[::3, :, 1] = (night[::3, :, 1] * 0.85).astype(np.uint8)

        return night


def apply_effects(
    frame: np.ndarray,
    blur_faces: bool,
    night_vision: bool,
    person_boxes: List[Tuple[int, int, int, int]],
    face_blur_effect: FaceBlurEffect,
    night_vision_effect: NightVisionEffect,
) -> np.ndarray:
    """Apply the requested combination of effects, in a stable order."""
    output = frame

    if blur_faces:
        output = face_blur_effect.apply(output, person_boxes)

    if night_vision:
        output = night_vision_effect.apply(output)

    return output