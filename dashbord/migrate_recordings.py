"""
migrate_recordings.py
---------------------
Convert legacy recordings that browsers cannot play (OpenCV ``mp4v`` /
MPEG-4 Part 2) into H.264, so they play in Replay like everything recorded
after the codec fix.

Run from the PROJECT ROOT:

    python migrate_recordings.py                # show what would happen
    python migrate_recordings.py --convert      # convert them (keeps originals)
    python migrate_recordings.py --convert --delete-originals
    python migrate_recordings.py --delete       # just delete the unplayable ones

Nothing is touched without an explicit flag: a bare run is a dry run.

Converted clips keep their real position on the Replay timeline. Legacy
files named ``<start_ms>_<seconds>.mp4`` carry their start time in the
filename, so the timeline is reconstructed exactly rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import recording_paths as rp          # noqa: E402
import replay_buffer as rbuf          # noqa: E402


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def find_unplayable() -> list[tuple[str, dict]]:
    """Every clip, across every camera, that a browser cannot decode."""
    out: list[tuple[str, dict]] = []
    root = rp.RECORDINGS_DIR
    if not root.is_dir():
        return out
    for cam_dir in sorted(root.iterdir()):
        if not cam_dir.is_dir():
            continue
        cam = cam_dir.name
        for seg in rbuf.list_segments(cam):
            if not seg["browser_playable"]:
                out.append((cam, seg))
    return out


def probe(ffmpeg: str, path: Path) -> tuple[str, int, int]:
    """(codec, width, height) via ffprobe, best effort."""
    probe_bin = ffmpeg.replace("ffmpeg", "ffprobe")
    try:
        r = subprocess.run(
            [probe_bin, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        parts = r.stdout.strip().split(",")
        if len(parts) >= 3:
            return parts[0], int(parts[1]), int(parts[2])
    except Exception:
        pass
    return "unknown", 0, 0


def convert_one(ffmpeg: str, cam: str, seg: dict, delete_original: bool) -> bool:
    """Transcode one clip to H.264 and write its metadata sidecar."""
    src: Path = seg["path"]
    start = seg["start_epoch"]
    end = seg["end_epoch"]
    cam_name = seg.get("camera_name") or cam

    new_name = rbuf.segment_filename(cam_name, start, end)
    dst = src.parent / new_name
    if dst.exists() and dst != src:
        stem, suf = dst.stem, dst.suffix
        i = 2
        while (src.parent / f"{stem}({i}){suf}").exists():
            i += 1
        dst = src.parent / f"{stem}({i}){suf}"

    tmp = src.parent / f".convert_{int(time.time()*1000)}.mp4"
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not tmp.is_file() or tmp.stat().st_size < rbuf.MIN_VALID_BYTES:
            print(f"      FAILED: {r.stderr.strip()[:160]}")
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dst)
    except Exception as exc:
        print(f"      FAILED: {exc}")
        tmp.unlink(missing_ok=True)
        return False

    codec, w, h = probe(ffmpeg, dst)
    sidecar = {
        "camera_id": cam,
        "camera_name": cam_name,
        "file": dst.name,
        "start_epoch": start,
        "end_epoch": end,
        "duration": round(end - start, 2),
        "playback_seconds": round(seg.get("playback_seconds") or (end - start), 3),
        "width": w or seg.get("width"),
        "height": h or seg.get("height"),
        "fps": rbuf.BUFFER_FPS,
        "codec": codec,
        "browser_playable": codec == "h264",
        "size_bytes": dst.stat().st_size,
        "converted_from": src.name,
    }
    dst.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

    if delete_original and src != dst:
        src.unlink(missing_ok=True)
        src.with_suffix(".json").unlink(missing_ok=True)
    return True


def rebuild_index(cam: str) -> None:
    """Refresh live/index.json so the player sees the converted clips."""
    try:
        segs = []
        for seg in rbuf.list_segments(cam):
            segs.append({
                "file": seg["name"],
                "start": round(seg["start_epoch"], 3),
                "end": round(seg["end_epoch"], 3),
                "duration": round(seg["duration"], 2),
                "pb": round(seg.get("playback_seconds") or seg["duration"], 3),
                "playable": bool(seg["browser_playable"]),
                "w": seg["width"], "h": seg["height"],
                "size": seg["size_bytes"], "codec": seg["codec"],
            })
        kfs = rbuf.list_keyframes(cam)
        evs = [{"t": round(e["t"], 2), "k": e["kind"]} for e in rbuf.list_events(cam)][-400:]
        lows = [s["start"] for s in segs] + ([kfs[0]] if kfs else [])
        highs = [s["end"] for s in segs] + ([kfs[-1]] if kfs else [])
        idx = {
            "camera_id": cam,
            "camera_name": cam,
            "updated": time.time(),
            "start": min(lows) if lows else time.time(),
            "end": max(highs) if highs else time.time(),
            "latest_kf": int(kfs[-1] * 1000) if kfs else None,
            "kf_count": len(kfs),
            "kf_first": kfs[0] if kfs else None,
            "kf_last": kfs[-1] if kfs else None,
            "kf_step": rbuf.KEYFRAME_INTERVAL_SECONDS,
            "kf_list": [int(t * 1000) for t in kfs[-1800:]],
            "segments": segs,
            "events": evs,
        }
        d = rbuf.live_dir(cam)
        tmp = d / ".index.tmp"
        tmp.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
        tmp.replace(d / "index.json")
        print(f"   index rebuilt for {cam}: {len(segs)} clips "
              f"({sum(1 for s in segs if s['playable'])} playable)")
    except Exception as exc:
        print(f"   index rebuild failed for {cam}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert legacy mp4v recordings to H.264.")
    ap.add_argument("--convert", action="store_true", help="transcode to H.264")
    ap.add_argument("--delete", action="store_true", help="delete unplayable clips instead")
    ap.add_argument("--delete-originals", action="store_true",
                    help="with --convert, remove the mp4v file after a successful convert")
    args = ap.parse_args()

    print("=" * 66)
    print("COREWISE RECORDING MIGRATION")
    print("=" * 66)

    ffmpeg = rbuf.resolve_ffmpeg()
    print(f"ffmpeg: {ffmpeg or 'NOT FOUND'}")
    if not ffmpeg and args.convert:
        print("\nCannot convert without ffmpeg. Install it with either:")
        print("   pip install imageio-ffmpeg        (no admin rights needed)")
        print("   winget install Gyan.FFmpeg        (system-wide)")
        return 1

    items = find_unplayable()
    if not items:
        print("\nNo unplayable clips found - every recording is browser-playable.")
        for cam_dir in sorted(rp.RECORDINGS_DIR.iterdir()) if rp.RECORDINGS_DIR.is_dir() else []:
            if cam_dir.is_dir():
                rebuild_index(cam_dir.name)
        return 0

    total = sum(s["size_bytes"] for _, s in items)
    print(f"\nFound {len(items)} unplayable clip(s), {human(total)} total:")
    by_cam: dict[str, int] = {}
    for cam, seg in items:
        by_cam[cam] = by_cam.get(cam, 0) + 1
    for cam, n in by_cam.items():
        print(f"   {cam}: {n} clips")
    print(f"   example: {items[0][1]['name']}  ({items[0][1]['codec']})")

    if not args.convert and not args.delete:
        print("\nDRY RUN - nothing changed. Re-run with one of:")
        print("   python migrate_recordings.py --convert")
        print("   python migrate_recordings.py --convert --delete-originals")
        print("   python migrate_recordings.py --delete")
        return 0

    if args.delete:
        removed = 0
        for cam, seg in items:
            try:
                seg["path"].unlink(missing_ok=True)
                seg["path"].with_suffix(".json").unlink(missing_ok=True)
                removed += 1
            except OSError as exc:
                print(f"   could not delete {seg['name']}: {exc}")
        print(f"\nDeleted {removed} clip(s), freed ~{human(total)}.")
        for cam in by_cam:
            rebuild_index(cam)
        return 0

    ok = fail = 0
    t0 = time.time()
    for i, (cam, seg) in enumerate(items, 1):
        print(f"   [{i}/{len(items)}] {seg['name'][:52]} ...", flush=True)
        if convert_one(ffmpeg, cam, seg, args.delete_originals):
            ok += 1
        else:
            fail += 1

    print(f"\nConverted {ok}, failed {fail}, in {time.time()-t0:.0f}s.")
    for cam in by_cam:
        rebuild_index(cam)
    print("\nRestart the dashboard (or just reload Replay) to see them play.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())