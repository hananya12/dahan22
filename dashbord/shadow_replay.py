"""
tools/shadow_replay.py
----------------------
Run the identity engine over REAL footage, offline, and report what happened.

This is the Phase 0 workhorse. Live shadow mode tells you the divergence on
today's traffic; this tells you *why*, on footage you can watch frame by
frame, as many times as you like, with different settings.

    # a video file
    python tools/shadow_replay.py --video store_morning.mp4

    # footage Corewise already recorded (uses recording_paths.py)
    python tools/shadow_replay.py --camera cam_1 --date 2026-07-27

    # dump embeddings so thresholds can be swept without re-running YOLO
    python tools/shadow_replay.py --video clip.mp4 --dump runs/morning

    # a specific window, at speed
    python tools/shadow_replay.py --video clip.mp4 --start 120 --end 400 --stride 2

Outputs, per run:
    events.jsonl     every identity event, in order
    summary.json     counts, cost, gate rejections, score distribution
    people.json      one row per GlobalPerson: dwell, journey, tracker ids absorbed
    embeddings.npz   (with --dump) vectors keyed by (frame, track) for the tuner
    observations.jsonl (with --dump) the detections that produced them

The report at the end is designed around the four questions that matter on
real footage: did returning people get re-identified, did occlusions break
anything, did the quality gate reject most of the frame, and what did it cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.identity import EventType  # noqa: E402
from vision.pipeline import IdentityPipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

class YoloDetector:
    """The project's own detector, unchanged, so offline and live agree."""

    name = "yolo"

    def __init__(self, model: str = "yolov8n.pt", confidence: float = 0.4) -> None:
        project_root = _find_project_root()
        if project_root and str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from detection import PersonDetector          # noqa
        self._detector = PersonDetector(model, confidence)

    def __call__(self, frame: np.ndarray, frame_index: int) -> List[dict]:
        return self._detector.track_people(frame)


class JsonlDetector:
    """Replays detections recorded earlier — no YOLO, no GPU, deterministic.

    This is what makes a threshold sweep cheap: detection is by far the most
    expensive stage and it does not change when identity settings change.
    """

    name = "jsonl"

    def __init__(self, path: Path) -> None:
        self._by_frame: Dict[int, List[dict]] = {}
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                self._by_frame.setdefault(int(row["frame"]), []).append(
                    {"id": int(row["id"]), "box": tuple(row["box"])})

    def __call__(self, frame: np.ndarray, frame_index: int) -> List[dict]:
        return self._by_frame.get(frame_index, [])


def _find_project_root() -> Optional[Path]:
    """Locate the folder holding main.py / detection.py, if we are inside it."""
    for candidate in (ROOT, ROOT.parent, Path.cwd()):
        if (candidate / "detection.py").is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

def frames_from_video(path: Path, start: float, end: Optional[float],
                      stride: int) -> Iterator[Tuple[int, float, np.ndarray]]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    index = int(start * fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = index / fps
            if end is not None and timestamp > end:
                break
            if index % stride == 0:
                yield index, timestamp, frame
            index += 1
    finally:
        cap.release()


def recordings_for(camera_id: str, day: str) -> List[Path]:
    """Every clip Corewise already recorded for a camera on a date."""
    project_root = _find_project_root()
    if project_root and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        import recording_paths as rp
    except ImportError:
        raise SystemExit("recording_paths.py not found — run this from the project root, "
                         "or pass --video instead of --camera/--date")
    folder = rp.day_folder(camera_id, day)
    if not folder.is_dir():
        raise SystemExit(f"no recordings at {folder}")
    return sorted(p for p in folder.glob("*.mp4") if rp.is_valid_recording(p))


# ---------------------------------------------------------------------------
# Embedding capture
# ---------------------------------------------------------------------------

class RecordingBackend:
    """Wraps a real backend and records every vector it produces.

    Keyed by (frame_index, track_id) so tools/tune_thresholds.py can replay
    the identity logic against different settings without a second YOLO pass.
    """

    wants_keys = True

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = f"recording({inner.name})"
        self.dim = getattr(inner, "dim", 512)
        self.keys: List[Tuple[int, int]] = []
        self.vectors: List[np.ndarray] = []

    def embed_batch(self, crops, keys=None):
        vectors = self.inner.embed_batch(crops)
        if keys:
            for key, vector in zip(keys, vectors):
                self.keys.append(key)
                self.vectors.append(np.asarray(vector, dtype=np.float32))
        return vectors

    def save(self, path: Path) -> int:
        if not self.vectors:
            return 0
        np.savez_compressed(
            path,
            keys=np.asarray(self.keys, dtype=np.int64),
            vectors=np.stack(self.vectors).astype(np.float32),
        )
        return len(self.vectors)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out or (ROOT / "data" / "shadow_runs" /
                                time.strftime("%Y%m%d-%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.detections:
        detector = JsonlDetector(Path(args.detections))
    else:
        detector = YoloDetector(args.model, args.confidence)

    pipeline = IdentityPipeline(camera_id=args.camera or "offline",
                                funnel_steps=args.funnel or None)
    recorder: Optional[RecordingBackend] = None
    if args.dump:
        recorder = RecordingBackend(pipeline.reid.backend)
        pipeline.reid.backend = recorder
        # Dump-everything mode: a permissive budget now means a threshold
        # sweep later can use ANY subset of these crops.
        pipeline.config.reid.max_crops_per_frame = 32
        pipeline.config.reid.refresh_every_n_frames = 1

    zones = json.loads(Path(args.zones).read_text(encoding="utf-8")) if args.zones else {}
    pipeline.configure(
        tam_polygon_px=zones.get("tam"),
        som_points_px=zones.get("som"),
        som_is_polygon=bool(zones.get("som_is_polygon")),
        named_zones_px={k: v for k, v in zones.items()
                        if k not in ("tam", "som", "som_is_polygon", "store")},
        store_polygon_px=zones.get("store"),
    )
    if args.min_stay is not None:
        pipeline.set_min_stay(True, args.min_stay)

    events_path = out_dir / "events.jsonl"
    events_file = events_path.open("w", encoding="utf-8")
    obs_file = (out_dir / "observations.jsonl").open("w", encoding="utf-8") if args.dump else None

    class FileSink:
        def emit(self, event) -> None:
            events_file.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    pipeline.bus.subscribe(FileSink())

    sources: List[Path] = []
    if args.video:
        sources = [Path(args.video)]
    else:
        sources = recordings_for(args.camera or "cam_1", args.date)

    print(f"[replay] {len(sources)} source(s), detector={detector.name}, out={out_dir}")

    frame_total = 0
    detection_total = 0
    wall_start = time.perf_counter()
    detect_seconds = 0.0
    identity_seconds = 0.0
    base_time = time.time()
    telemetry: Dict[str, Any] = {}
    offset = 0.0

    for source in sources:
        print(f"[replay] {source.name}")
        last_ts = 0.0
        for frame_index, timestamp, frame in frames_from_video(
                source, args.start, args.end, args.stride):
            t0 = time.perf_counter()
            detections = detector(frame, frame_index)
            t1 = time.perf_counter()
            now = base_time + offset + timestamp
            telemetry = pipeline.process(frame, detections, now=now)
            t2 = time.perf_counter()

            detect_seconds += t1 - t0
            identity_seconds += t2 - t1
            frame_total += 1
            detection_total += len(detections)
            last_ts = timestamp

            if obs_file is not None:
                for d in detections:
                    obs_file.write(json.dumps(
                        {"frame": frame_index, "id": int(d["id"]),
                         "box": [float(v) for v in d["box"]], "t": now,
                         "w": int(frame.shape[1]), "h": int(frame.shape[0])}) + "\n")

            if args.progress and frame_total % args.progress == 0:
                print(f"   {frame_total:6d} frames  t={timestamp:7.1f}s  "
                      f"people={telemetry.get('tam', 0)}  "
                      f"inside={telemetry.get('inside', 0)}", flush=True)
        offset += last_ts + 1.0

    pipeline.shutdown()
    events_file.close()
    if obs_file is not None:
        obs_file.close()
    if recorder is not None:
        n = recorder.save(out_dir / "embeddings.npz")
        print(f"[replay] dumped {n} embeddings for offline tuning")

    summary = build_summary(pipeline, telemetry, events_path, frame_total,
                            detection_total, detect_seconds, identity_seconds,
                            time.perf_counter() - wall_start)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "people.json").write_text(
        json.dumps(pipeline.people(), ensure_ascii=False, indent=2), encoding="utf-8")

    report(summary, out_dir)
    return 0


def build_summary(pipeline, telemetry, events_path, frames, detections,
                  detect_s, identity_s, wall_s) -> Dict[str, Any]:
    kinds: Counter = Counter()
    with events_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                kinds[json.loads(line)["type"]] += 1

    people = pipeline.people()
    churn = [p["tracker_ids_absorbed"] for p in people]
    diag = pipeline.diagnostics()

    return {
        "frames": frames,
        "detections": detections,
        "avg_people_per_frame": round(detections / frames, 2) if frames else 0.0,
        "telemetry": telemetry,
        "events": dict(kinds),
        "identities": {
            "total": len(people),
            "returning": sum(1 for p in people if p.get("returning")),
            "tracker_ids_absorbed_total": sum(churn),
            "max_tracker_ids_for_one_person": max(churn) if churn else 0,
            "people_with_id_churn": sum(1 for c in churn if c > 1),
        },
        "reid": diag["identity"]["reid"],
        "cost": {
            "wall_seconds": round(wall_s, 1),
            "detect_ms_per_frame": round(detect_s / frames * 1000, 2) if frames else None,
            "identity_ms_per_frame": round(identity_s / frames * 1000, 2) if frames else None,
            "identity_share": round(identity_s / (detect_s + identity_s) * 100, 1)
                              if (detect_s + identity_s) else None,
        },
        "zones": diag["zones"],
    }


def report(summary: Dict[str, Any], out_dir: Path) -> None:
    tel = summary["telemetry"]
    ident = summary["identities"]
    reid = summary["reid"]
    cost = summary["cost"]

    print("\n" + "=" * 70)
    print("SHADOW REPLAY REPORT")
    print("=" * 70)
    print(f"frames {summary['frames']}   detections {summary['detections']}   "
          f"avg {summary['avg_people_per_frame']} people/frame")

    print("\nIDENTITY")
    print(f"   distinct people          {ident['total']}")
    print(f"   returning visitors       {ident['returning']}")
    print(f"   people whose tracker id changed  {ident['people_with_id_churn']} "
          f"(worst: {ident['max_tracker_ids_for_one_person']} ids for one person)")
    if ident["total"]:
        churn = ident["tracker_ids_absorbed_total"] / ident["total"]
        print(f"   tracker ids per person   {churn:.2f}")
        print(f"   >> counting by track id would have reported ~"
              f"{ident['tracker_ids_absorbed_total']} visitors instead of {ident['total']}")

    print("\nRETAIL")
    for key in ("tam", "sam", "som", "inside", "average_stay_time", "conversion_rate"):
        print(f"   {key:20} {tel.get(key)}")

    print("\nQUALITY GATE")
    total = reid.get("embedded", 0) + reid.get("gated_out", 0)
    if total:
        pct = reid["gated_out"] / total * 100
        print(f"   embedded {reid['embedded']}   rejected {reid['gated_out']} ({pct:.0f}%)")
        for reason, n in (reid.get("gate_reasons") or {}).items():
            print(f"      {reason:28} {n}")
        if pct > 90:
            print("   >> the gate is rejecting almost everything: the camera is probably")
            print("      too far away, or people are cut off at the frame edge.")
            print("      No threshold tuning will fix that — reframe the camera first.")

    print("\nCOST")
    print(f"   detection {cost['detect_ms_per_frame']} ms/frame   "
          f"identity {cost['identity_ms_per_frame']} ms/frame "
          f"({cost['identity_share']}% of the work)")

    print(f"\nwritten to {out_dir}")
    print("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the identity engine over recorded footage.")
    src = ap.add_argument_group("source")
    src.add_argument("--video", help="path to a video file")
    src.add_argument("--camera", help="camera id (uses Corewise's own recordings)")
    src.add_argument("--date", help="YYYY-MM-DD, with --camera")
    src.add_argument("--start", type=float, default=0.0, help="seconds into the clip")
    src.add_argument("--end", type=float, default=None)
    src.add_argument("--stride", type=int, default=1, help="process every Nth frame")

    det = ap.add_argument_group("detection")
    det.add_argument("--model", default="yolov8n.pt")
    det.add_argument("--confidence", type=float, default=0.4)
    det.add_argument("--detections", help="replay detections from a JSONL instead of YOLO")

    cfg = ap.add_argument_group("configuration")
    cfg.add_argument("--zones", help="JSON with tam / som / store / named zone polygons in pixels")
    cfg.add_argument("--min-stay", type=float, default=None, help="SAM threshold, seconds")
    cfg.add_argument("--funnel", nargs="*", help="zone names, in order")

    out = ap.add_argument_group("output")
    out.add_argument("--out", help="output folder")
    out.add_argument("--dump", action="store_true",
                     help="save embeddings + observations for tools/tune_thresholds.py")
    out.add_argument("--progress", type=int, default=250)

    args = ap.parse_args()
    if not args.video and not (args.camera and args.date):
        ap.error("give --video, or --camera together with --date")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())