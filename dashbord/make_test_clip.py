"""
tools/make_test_clip.py
-----------------------
Generate a synthetic clip with known ground truth, so the whole Phase 0
toolchain can be verified end to end before any real footage exists.

    python tools/make_test_clip.py --out /tmp/clip

Writes ``clip.mp4``, ``detections.jsonl`` (with deliberate tracker-id churn)
and ``truth.csv`` (track_id → real person), which is exactly the input
``tune_thresholds.py --ground-truth`` expects.

The scenario is built out of the four things that break identity systems in
real stores:

    person 1  walks out of frame at 10s and returns at 16s WITH A NEW TRACK ID
    person 2  is present throughout, and is briefly occluded by person 3
    person 3  wears clothing similar to person 2 — the uniformed-staff case
    lighting  dims across the clip, so embeddings drift

This is not a substitute for real footage. It is a way to know the tooling is
correct before you go and collect some.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

W, H, FPS = 1280, 720, 20
BODY_H = 260


def person_patch(rng, shirt, trousers, height: int) -> np.ndarray:
    """A crudely textured body. Texture matters: a flat colour block fails the
    sharpness check in the quality gate, exactly as a real out-of-focus person
    would."""
    width = int(height / 2.6)
    patch = np.zeros((height, width, 3), dtype=np.uint8)
    split = int(height * 0.55)
    patch[:split] = shirt
    patch[split:] = trousers
    head = int(height * 0.16)
    patch[:head] = (90, 110, 140)
    noise = rng.normal(0, 16, patch.shape)
    return np.clip(patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def build(out_dir: Path, seconds: int = 30) -> None:
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)

    people = {
        1: {"shirt": (40, 40, 200), "trousers": (60, 60, 90)},    # red shirt
        2: {"shirt": (60, 180, 70), "trousers": (40, 40, 45)},    # green shirt
        3: {"shirt": (70, 190, 85), "trousers": (45, 45, 50)},    # similar to 2
    }

    def position(person: int, t: float):
        if person == 1:
            if 10.0 <= t < 16.0:
                return None                       # out of frame
            x = 300 + 250 * np.sin(t * 0.5)
            return x, 480 + 40 * np.sin(t * 0.9)
        if person == 2:
            return 700 + 180 * np.sin(t * 0.35 + 1.0), 520
        if 6.0 <= t < 12.0:                       # person 3 walks past person 2
            return 640 + 60 * (t - 6.0), 500
        return None

    # Tracker churn: person 1 gets a new id after re-entering; person 2's id is
    # dropped and reissued mid-clip, exactly as ByteTrack does after occlusion.
    def track_id(person: int, t: float) -> int:
        if person == 1:
            return 1 if t < 10.0 else 41
        if person == 2:
            return 2 if t < 13.0 else 52
        return 3

    background = rng.integers(70, 95, (H, W, 3), dtype=np.uint8)
    background = cv2.GaussianBlur(background, (0, 0), 3)

    writer = cv2.VideoWriter(str(out_dir / "clip.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rows: List[dict] = []
    truth: Dict[int, str] = {}

    for frame_index in range(seconds * FPS):
        t = frame_index / FPS
        # Lighting dims over the clip, so the same person's embedding drifts.
        gain = 1.0 - 0.35 * (t / seconds)
        frame = np.clip(background.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        for person, style in people.items():
            pos = position(person, t)
            if pos is None:
                continue
            cx, foot = pos
            scale = 0.85 + 0.3 * (foot - 400) / 300.0
            height = int(BODY_H * scale)
            patch = person_patch(rng, style["shirt"], style["trousers"], height)
            patch = np.clip(patch.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            ph, pw = patch.shape[:2]
            x1, y1 = int(cx - pw / 2), int(foot - ph)
            x2, y2 = x1 + pw, y1 + ph
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                continue
            frame[y1:y2, x1:x2] = patch

            tid = track_id(person, t)
            truth[tid] = f"person_{person}"
            rows.append({"frame": frame_index, "id": tid,
                         "box": [float(x1), float(y1), float(x2), float(y2), 0.92]})

        writer.write(frame)
    writer.release()

    with (out_dir / "detections.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    with (out_dir / "truth.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track_id", "person"])
        for tid, label in sorted(truth.items()):
            w.writerow([tid, label])

    print(f"clip      {out_dir/'clip.mp4'}  ({seconds}s, {seconds*FPS} frames)")
    print(f"detections{out_dir/'detections.jsonl'}  ({len(rows)} boxes)")
    print(f"truth     {out_dir/'truth.csv'}  "
          f"({len(truth)} tracker ids -> {len(set(truth.values()))} real people)")
    print("\nGround truth: 3 people, 5 tracker ids. A correct engine reports 3.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic validation clip.")
    ap.add_argument("--out", default="/tmp/corewise_clip")
    ap.add_argument("--seconds", type=int, default=30)
    args = ap.parse_args()
    build(Path(args.out), args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
