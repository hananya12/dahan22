"""
tools/tune_thresholds.py
------------------------
Pick the matching thresholds from DATA, not from a blog post.

``match_threshold = 0.72`` is a starting point taken from the ReID
literature. The right value for one store, with its camera angle, its
lighting and its clientele, is an empirical question — and the only honest
way to answer it is to sweep the parameter over that store's own footage.

This tool replays the embeddings dumped by ``shadow_replay.py --dump``, so a
full sweep costs seconds instead of another YOLO pass per configuration:

    python tools/shadow_replay.py --video clip.mp4 --dump --out runs/morning
    python tools/tune_thresholds.py runs/morning

With ground truth (a CSV of ``track_id,person_label`` — ten minutes of manual
labelling on one clip) it also reports the two error rates that matter:

    python tools/tune_thresholds.py runs/morning --ground-truth labels.csv

Without ground truth it still reports the identity-count curve, whose elbow
is informative on its own: below the elbow identities collapse into each
other (false merges), above it they shatter (false splits).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.identity import IdentityConfig, TrackObservation  # noqa: E402
from vision.pipeline import IdentityPipeline  # noqa: E402


class PrecomputedBackend:
    """Serves embeddings recorded during the dump run, keyed by (frame, track)."""

    wants_keys = True
    needs_pixels = False
    name = "precomputed"

    def __init__(self, path: Path) -> None:
        data = np.load(path)
        keys = data["keys"]
        vectors = data["vectors"]
        self.dim = int(vectors.shape[1])
        self._table: Dict[Tuple[int, int], np.ndarray] = {
            (int(k[0]), int(k[1])): v for k, v in zip(keys, vectors)
        }
        self.misses = 0

    def embed_batch(self, crops, keys=None):
        out = np.zeros((len(crops), self.dim), dtype=np.float32)
        for i, key in enumerate(keys or []):
            vector = self._table.get((int(key[0]), int(key[1])))
            if vector is None:
                self.misses += 1
                continue
            out[i] = vector
        return out


def load_observations(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def replay(run_dir: Path, observations: Sequence[dict],
           config: IdentityConfig) -> Tuple[Dict[str, Any], Dict[int, str]]:
    """Replay one configuration. Returns telemetry and track -> identity map."""
    backend = PrecomputedBackend(run_dir / "embeddings.npz")
    pipeline = IdentityPipeline(camera_id="tune", config=config, backend=backend)
    # The dump was permissive; a sweep must be allowed to use all of it.
    pipeline.config.reid.max_crops_per_frame = 32
    pipeline.config.reid.refresh_every_n_frames = 1

    by_frame: Dict[int, List[dict]] = {}
    for row in observations:
        by_frame.setdefault(int(row["frame"]), []).append(row)

    assignment: Dict[int, str] = {}
    telemetry: Dict[str, Any] = {}
    for frame_index in sorted(by_frame):
        rows = by_frame[frame_index]
        now = rows[0]["t"]
        detections = [{"id": int(r["id"]), "box": tuple(r["box"])} for r in rows]
        # The pipeline's own frame counter must line up with the dumped keys.
        pipeline._frame_index = frame_index - 1
        pipeline.frame_size = (int(rows[0].get("h", 720)), int(rows[0].get("w", 1280)))
        telemetry = pipeline.process(None, detections, now=now)
        for track_id in (int(r["id"]) for r in rows):
            gid = pipeline.identity.global_id_for(track_id)
            if gid:
                assignment[track_id] = gid

    telemetry["_backend_misses"] = backend.misses
    telemetry["_identities"] = pipeline.memory.stats["created_total"]
    return telemetry, assignment


def score_against_truth(assignment: Dict[int, str],
                        truth: Dict[int, str]) -> Dict[str, float]:
    """Pairwise clustering metrics over tracks with known labels.

    Reported as the two errors that are NOT interchangeable:
      * merge_rate — pairs of DIFFERENT people given one identity. Silent,
        permanent, corrupts galleries. This is the number to minimise.
      * split_rate — fragments of ONE person given different identities.
        Visible and self-correcting; inflates visitor counts.
    """
    tracks = [t for t in assignment if t in truth]
    same_person = different_person = merged = split = 0
    for a, b in combinations(tracks, 2):
        same_truth = truth[a] == truth[b]
        same_pred = assignment[a] == assignment[b]
        if same_truth:
            same_person += 1
            if not same_pred:
                split += 1
        else:
            different_person += 1
            if same_pred:
                merged += 1
    return {
        "labelled_tracks": len(tracks),
        "same_person_pairs": same_person,
        "different_person_pairs": different_person,
        "split_rate": round(split / same_person * 100, 1) if same_person else 0.0,
        "merge_rate": round(merged / different_person * 100, 1) if different_person else 0.0,
        "true_people": len(set(truth[t] for t in tracks)),
        "predicted_people": len(set(assignment[t] for t in tracks)),
    }


def load_truth(path: Path) -> Dict[int, str]:
    truth: Dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or row[0].strip().lower() in ("track_id", "track"):
                continue
            try:
                truth[int(row[0])] = row[1].strip()
            except ValueError:
                continue
    return truth


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep identity thresholds over recorded footage.")
    ap.add_argument("run_dir", help="folder produced by shadow_replay.py --dump")
    ap.add_argument("--match", nargs="*", type=float,
                    default=[0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85])
    ap.add_argument("--margin", nargs="*", type=float, default=[0.05])
    ap.add_argument("--probation", nargs="*", type=int, default=[3])
    ap.add_argument("--ground-truth", help="CSV: track_id,person_label")
    ap.add_argument("--json", help="write full results here")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    for required in ("embeddings.npz", "observations.jsonl"):
        if not (run_dir / required).is_file():
            raise SystemExit(f"{run_dir/required} missing — re-run shadow_replay.py with --dump")

    observations = load_observations(run_dir / "observations.jsonl")
    truth = load_truth(Path(args.ground_truth)) if args.ground_truth else {}
    print(f"[tune] {len(observations)} observations"
          + (f", {len(truth)} labelled tracks" if truth else ", no ground truth"))

    results: List[Dict[str, Any]] = []
    for match in args.match:
        for margin in args.margin:
            for probation in args.probation:
                config = IdentityConfig(camera_id="tune")
                config.matching.match_threshold = match
                config.matching.new_threshold = min(match - 0.05, config.matching.new_threshold)
                config.matching.min_margin_over_runner_up = margin
                config.matching.probation_frames = probation

                telemetry, assignment = replay(run_dir, observations, config)
                row: Dict[str, Any] = {
                    "match_threshold": match,
                    "margin": margin,
                    "probation": probation,
                    "identities": telemetry["_identities"],
                    "tam": telemetry.get("tam"),
                    "backend_misses": telemetry["_backend_misses"],
                }
                if truth:
                    row.update(score_against_truth(assignment, truth))
                results.append(row)

    _print_table(results, bool(truth))
    _recommend(results, bool(truth))

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwritten to {args.json}")
    return 0


def _print_table(results: List[Dict[str, Any]], with_truth: bool) -> None:
    print("\n" + "=" * 74)
    if with_truth:
        print(f"{'match':>7} {'margin':>7} {'prob':>5} {'ids':>5} {'true':>5} "
              f"{'merge%':>8} {'split%':>8}")
        print("-" * 74)
        for r in results:
            print(f"{r['match_threshold']:>7.2f} {r['margin']:>7.2f} {r['probation']:>5} "
                  f"{r['identities']:>5} {r.get('true_people', 0):>5} "
                  f"{r.get('merge_rate', 0):>8.1f} {r.get('split_rate', 0):>8.1f}")
    else:
        print(f"{'match':>7} {'margin':>7} {'prob':>5} {'identities':>11} {'TAM':>6}")
        print("-" * 74)
        for r in results:
            print(f"{r['match_threshold']:>7.2f} {r['margin']:>7.2f} {r['probation']:>5} "
                  f"{r['identities']:>11} {str(r['tam']):>6}")
    print("=" * 74)


def _recommend(results: List[Dict[str, Any]], with_truth: bool) -> None:
    if not results:
        return
    if with_truth:
        # Merge errors are permanent and silent; split errors are visible and
        # recoverable. So: minimise merges first, break ties on splits.
        best = min(results, key=lambda r: (r.get("merge_rate", 100), r.get("split_rate", 100)))
        print(f"\nRECOMMENDED  match={best['match_threshold']} margin={best['margin']} "
              f"probation={best['probation']}")
        print(f"   merge {best.get('merge_rate')}%  split {best.get('split_rate')}%  "
              f"({best['identities']} identities vs {best.get('true_people')} real people)")
        print("   chosen by minimising FALSE MERGES first: a merge is silent and")
        print("   permanent, a split is visible and self-correcting.")

        merges = {r.get("merge_rate") for r in results}
        splits = {r.get("split_rate") for r in results}
        if len(merges) == 1 and len(splits) == 1 and len(results) > 2:
            print("\n   NOTE: every threshold produced the same result. The threshold is")
            print("   NOT the lever here — the embeddings are either far apart or far")
            print("   too close, and moving the cut point changes nothing. Fix the")
            print("   FEATURES instead: install torchreid (OSNet) if you are still on")
            print("   the colour-histogram baseline, improve lighting consistency, or")
            print("   move the camera closer so crops carry more detail.")
        elif best.get("split_rate", 0) > 30:
            print("\n   Splits are high even at the best setting: the same person is not")
            print("   recognisable across their gap. Look at the gate reject rate and at")
            print("   how much the lighting changes between entry and re-entry.")
        return

    counts = [(r["match_threshold"], r["identities"]) for r in results]
    print("\nNo ground truth, so read the curve rather than a single number:")
    print("   identities rise with the threshold — below the elbow people are")
    print("   being merged together, above it one person shatters into several.")
    jumps = [(b[1] - a[1], a[0], b[0]) for a, b in zip(counts, counts[1:])]
    if jumps:
        biggest = max(jumps)
        print(f"   sharpest rise between {biggest[1]} and {biggest[2]} "
              f"(+{biggest[0]} identities) — the elbow is around there.")
    print("\n   Label 10 minutes of one clip (track_id,person) and re-run with")
    print("   --ground-truth to turn this into an actual answer.")


if __name__ == "__main__":
    raise SystemExit(main())
