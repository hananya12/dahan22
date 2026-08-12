"""
replay_buffer.py
----------------
NVR-style continuous recording store for Corewise Replay.

This is the SINGLE Replay storage implementation - the dashboard reads
everything it shows through the reader API at the bottom of this file.

Why the previous version showed a black screen when scrubbing back
-----------------------------------------------------------------
Segments were written with OpenCV's ``mp4v`` FourCC, i.e. MPEG-4 Part 2.
No browser can decode that in an HTML5 <video> element, so the file loaded
and then rendered black. LIVE looked fine only because it is a JPEG image,
not a video. Segments are now encoded as **H.264 (avc1)** through ffmpeg,
which every browser decodes natively, with ``+faststart`` so seeking works
before the whole file has downloaded.

Storage layout
--------------
    static/recordings/<camera_id>/live/
        <CameraName>_<YYYY-MM-DD>_<HH-MM>_to_<HH-MM>.mp4   finalized clip
        <same name>.json                                    clip metadata
        frames/<epoch_ms>.jpg                               1 fps key frames
        meta/<epoch_ms>.json                                AI metadata per key frame
        events-<YYYY-MM-DD>.jsonl                           append-only event log

Three tracks, one timeline:

  * **Video track** - minute-aligned H.264 clips. Smooth playback, real
    seeking, browser-native.
  * **Key-frame track** - one clean JPEG per second. Gives *instant*
    scrubbing with zero decode cost, and guarantees Replay is never black
    even for the not-yet-finalized current minute.
  * **Metadata track** - per-key-frame AI state (boxes, track ids,
    confidence, zone membership) so the Replay AI overlay can be toggled on
    and off instead of being burned into the pixels.

Recorded frames are CLEAN (privacy effects like face blur are applied, but
bounding boxes / zone outlines / the stats bar are not). The overlay is
reconstructed client-side from the metadata track, which is what makes
Task 11's optional overlay possible.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import recording_paths as rp

# --- Tunables -------------------------------------------------------------
BUFFER_FPS = 10                        # frames per second written to each clip
FRAME_INTERVAL_SECONDS = 1.0 / BUFFER_FPS
# Clip length. Kept SHORT on purpose: a clip only becomes playable once it is
# finalized, so with 60 s clips the most recent minute had no video at all and
# pressing Play fell back to the key-frame track (a slideshow). At 20 s the
# newest footage becomes real, seekable video within seconds.
SEGMENT_SECONDS = 20.0
KEYFRAME_INTERVAL_SECONDS = 1.0        # one JPEG + metadata record per second
KEYFRAME_JPEG_QUALITY = 72
RETENTION_SECONDS = 24 * 60 * 60       # keep 24h of history per camera
MIN_VALID_BYTES = 2048

_SAFE_NAME = re.compile(r"[^A-Za-z0-9\u0590-\u05FF._-]+")

# Event kinds surfaced on the timeline / event list.
EV_PERSON = "person"
EV_TAM = "tam"
EV_SAM = "sam"
EV_SOM = "som"
EV_REC_START = "rec_start"
EV_REC_STOP = "rec_stop"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def live_dir(camera_id: str) -> Path:
    """Folder holding a camera's recording history."""
    return rp.camera_recordings_root(camera_id) / "live"


def frames_dir(camera_id: str) -> Path:
    return live_dir(camera_id) / "frames"


def meta_dir(camera_id: str) -> Path:
    return live_dir(camera_id) / "meta"


def events_path(camera_id: str, day: Optional[str] = None) -> Path:
    day = day or datetime.now().strftime("%Y-%m-%d")
    return live_dir(camera_id) / f"events-{day}.jsonl"


def safe_camera_name(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", (name or "Camera").strip())
    return cleaned.strip("_") or "Camera"


def segment_filename(camera_name: str, start_epoch: float, end_epoch: float) -> str:
    """``<CameraName>_<YYYY-MM-DD>_<HH-MM>_to_<HH-MM>.mp4`` (Task 4)."""
    s = datetime.fromtimestamp(start_epoch)
    e = datetime.fromtimestamp(end_epoch)
    return (
        f"{safe_camera_name(camera_name)}_{s.strftime('%Y-%m-%d')}_"
        f"{s.strftime('%H-%M-%S')}_to_{e.strftime('%H-%M-%S')}.mp4"
    )


# Legacy epoch-based names from the previous build are still parsed so old
# recordings keep appearing in Replay (backward compatibility requirement).
def parse_segment_name(path: Path) -> Optional[Tuple[int, int]]:
    """Return (start_ms, seconds) for a LEGACY ``<start_ms>_<seconds>.mp4``."""
    try:
        start_ms_str, secs_str = path.stem.split("_", 1)
        return int(start_ms_str), int(secs_str)
    except Exception:
        return None


def resolve_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg binary that can produce browser-playable H.264.

    Order:
      1. ``ffmpeg`` on PATH (a normal system install)
      2. the static binary shipped by the ``imageio-ffmpeg`` wheel

    (2) matters on Windows, where users rarely have ffmpeg on PATH. Without a
    working encoder the recorder silently fell back to OpenCV's ``mp4v``, which
    NO browser can decode - producing clips that show up in Replay as a black
    screen / key-frame slideshow. ``pip install imageio-ffmpeg`` is enough to
    fix that without installing anything system-wide.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        pass
    return None


# Cached result of the startup encoder probe.
_ENCODER: Optional[dict] = None


def _produces_h264(path: Path) -> bool:
    """True if *path* is a real H.264 file (browsers decode 'avc1', not 'mp4v')."""
    try:
        if not path.is_file() or path.stat().st_size < 512:
            return False
        raw = path.read_bytes()
        return b"avc1" in raw and b"mp4v" not in raw[:2048]
    except OSError:
        return False


def probe_encoder(force: bool = False) -> dict:
    """Find an encoder that ACTUALLY produces browser-playable H.264.

    Every candidate is tested by encoding a throwaway clip and inspecting the
    bytes - assuming an encoder works is exactly how 189 unplayable clips got
    written while the log claimed new recordings were H.264.

    Order: ffmpeg binary (PATH or the imageio-ffmpeg wheel) -> OpenCV 'avc1'
    -> OpenCV 'H264' -> OpenCV 'mp4v' (last resort, NOT browser-playable).

    Returns {"kind": "ffmpeg"|"opencv", "exe": str|None, "fourcc": str|None,
             "h264": bool, "label": str}.
    """
    global _ENCODER
    if _ENCODER is not None and not force:
        return _ENCODER

    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="cw_enc_"))
    result = {"kind": "opencv", "exe": None, "fourcc": "mp4v",
              "h264": False, "label": "OpenCV mp4v (NOT browser-playable)"}
    try:
        exe = resolve_ffmpeg()
        if exe:
            out = tmpdir / "ff.mp4"
            try:
                subprocess.run(
                    [exe, "-y", "-loglevel", "error", "-f", "lavfi",
                     "-i", "testsrc=size=64x64:rate=10:duration=1",
                     "-c:v", "libx264", "-preset", "ultrafast",
                     "-pix_fmt", "yuv420p", str(out)],
                    capture_output=True, timeout=60)
                if _produces_h264(out):
                    result = {"kind": "ffmpeg", "exe": exe, "fourcc": None,
                              "h264": True, "label": f"ffmpeg libx264 ({exe})"}
                    return result
            except Exception:
                pass

        # OpenCV's bundled FFmpeg can sometimes emit H.264 directly.
        frames = [np.full((64, 64, 3), i * 8 % 255, dtype=np.uint8) for i in range(12)]
        for fourcc in ("avc1", "H264", "h264"):
            out = tmpdir / f"cv_{fourcc}.mp4"
            try:
                w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), 10, (64, 64))
                if w.isOpened():
                    for f in frames:
                        w.write(f)
                w.release()
                if _produces_h264(out):
                    result = {"kind": "opencv", "exe": None, "fourcc": fourcc,
                              "h264": True, "label": f"OpenCV {fourcc} (bundled FFmpeg)"}
                    return result
            except Exception:
                continue
        return result
    finally:
        _ENCODER = result
        shutil.rmtree(tmpdir, ignore_errors=True)


def encoder_status() -> dict:
    """Encoder info for the dashboard, without re-probing."""
    return probe_encoder()


def _h264_available() -> bool:
    return probe_encoder()["h264"]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Continuous recorder for ONE camera. Owned by a single writer thread."""

    def __init__(self, camera_id: str, camera_name: Optional[str] = None) -> None:
        self.camera_id = camera_id
        self.camera_name = safe_camera_name(camera_name or camera_id)
        self._dir = live_dir(camera_id)
        self._frames = frames_dir(camera_id)
        self._meta = meta_dir(camera_id)
        for d in (self._dir, self._frames, self._meta):
            d.mkdir(parents=True, exist_ok=True)

        enc = probe_encoder()
        self._encoder = enc
        self._ffmpeg = enc["exe"]
        self._use_ffmpeg = enc["kind"] == "ffmpeg"
        self._cv_fourcc = enc["fourcc"] or "mp4v"
        if enc["h264"]:
            print(f"[REPLAY] H.264 encoder verified: {enc['label']}")
        else:
            print(
                "[REPLAY] WARNING: ffmpeg was not found on PATH. Falling back to "
                "OpenCV's mp4v encoder, which BROWSERS CANNOT PLAY - recorded "
                "playback in Replay will fall back to the key-frame track.\n"
                "         FIX: pip install imageio-ffmpeg   (no system install needed)"
            )

        self._proc: Optional[subprocess.Popen] = None
        self._cv_writer: Optional[cv2.VideoWriter] = None
        self._staging: Optional[Path] = None
        self._seg_start = 0.0
        self._frame_w = 0
        self._frame_h = 0
        self._frames_in_seg = 0
        self._last_keyframe = 0.0
        self._last_sweep = 0.0
        self._started_at = time.time()
        self.log_event(EV_REC_START, {"camera": self.camera_name})

    # -- public API --------------------------------------------------------

    def write(self, frame: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
        """Append one CLEAN frame (privacy effects applied, no annotations).

        *meta* is the AI state for this instant and is stored on the metadata
        track so the Replay overlay can be toggled. Never raises - recording
        must not be able to crash the detection loop.
        """
        try:
            fh, fw = frame.shape[:2]
            now = time.time()
            if (self._proc is None and self._cv_writer is None
                    or fw != self._frame_w or fh != self._frame_h
                    or (now - self._seg_start) >= SEGMENT_SECONDS):
                self._roll(fw, fh, now)

            if self._proc is not None and self._proc.stdin is not None:
                try:
                    self._proc.stdin.write(frame.tobytes())
                    self._frames_in_seg += 1
                except (BrokenPipeError, OSError):
                    self._proc = None
            elif self._cv_writer is not None:
                self._cv_writer.write(frame)
                self._frames_in_seg += 1

            if now - self._last_keyframe >= KEYFRAME_INTERVAL_SECONDS:
                self._last_keyframe = now
                self._write_keyframe(frame, now, meta or {})
                self._write_index()

            if now - self._last_sweep > 60:
                self._last_sweep = now
                self._sweep_old()
        except Exception:
            pass

    def _write_index(self) -> None:
        """Write live/index.json - the manifest the Replay player polls.

        This is what lets the player refresh itself (new clips, new events,
        newest live frame) WITHOUT Streamlit re-running and re-mounting the
        iframe. Re-mounting was destroying the <video> element several times a
        second, which is why playback could never actually start.
        """
        try:
            segs = []
            for seg in list_segments(self.camera_id):
                segs.append({
                    "file": seg["name"],
                    "start": round(seg["start_epoch"], 3),
                    "end": round(seg["end_epoch"], 3),
                    "duration": round(seg["duration"], 2),
                    "pb": round(seg.get("playback_seconds") or seg["duration"], 3),
                    "playable": bool(seg["browser_playable"]),
                    "w": seg["width"], "h": seg["height"],
                    "size": seg["size_bytes"],
                    "codec": seg["codec"],
                })
            kfs = list_keyframes(self.camera_id)
            evs = [{"t": round(e["t"], 2), "k": e["kind"]}
                   for e in list_events(self.camera_id)][-400:]
            lo = min([s["start"] for s in segs] + ([kfs[0]] if kfs else []) or [time.time()])
            hi = max([s["end"] for s in segs] + ([kfs[-1]] if kfs else []) or [time.time()])
            idx = {
                "camera_id": self.camera_id,
                "camera_name": self.camera_name,
                "updated": time.time(),
                "start": lo,
                "end": hi,
                "latest_kf": int(kfs[-1] * 1000) if kfs else None,
                "kf_count": len(kfs),
                # Key frames are evenly spaced; sending first/last/step instead
                # of 1378 timestamps keeps the manifest tiny.
                "kf_first": kfs[0] if kfs else None,
                "kf_last": kfs[-1] if kfs else None,
                "kf_step": KEYFRAME_INTERVAL_SECONDS,
                # INTEGER MILLISECONDS - these ARE the key-frame filenames.
                # They were previously rounded to 2 decimal SECONDS, which threw
                # away the millisecond part, so the player built
                # "frames/1785064828450.jpg" for a file actually named
                # "frames/1785064828453.jpg". Every REPLAY seek 404'd and the
                # stage went black, while LIVE kept working because latest_kf
                # was already exact. Never round these.
                "kf_list": [int(t * 1000) for t in kfs[-1800:]],
                "segments": segs,
                "events": evs,
            }
            tmp = self._dir / ".index.tmp"
            tmp.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._dir / "index.json")
        except Exception:
            pass

    def log_event(self, kind: str, data: Optional[Dict[str, Any]] = None,
                  when: Optional[float] = None) -> None:
        """Append one event to the day's log (timeline markers + event list)."""
        try:
            rec = {"t": when or time.time(), "kind": kind, "data": data or {}}
            path = events_path(self.camera_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def release(self) -> None:
        """Finalize the current clip (called on shutdown)."""
        self.log_event(EV_REC_STOP, {"camera": self.camera_name})
        self._finalize()
        self._write_index()

    # -- internals ---------------------------------------------------------

    def _write_keyframe(self, frame: np.ndarray, now: float, meta: Dict[str, Any]) -> None:
        ms = int(now * 1000)
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, KEYFRAME_JPEG_QUALITY])
        if not ok:
            return
        tmp = self._frames / f".{ms}.tmp"
        tmp.write_bytes(buf.tobytes())
        tmp.replace(self._frames / f"{ms}.jpg")

        payload = dict(meta)
        payload["t"] = now
        payload["w"] = int(frame.shape[1])
        payload["h"] = int(frame.shape[0])
        mtmp = self._meta / f".{ms}.tmp"
        mtmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        mtmp.replace(self._meta / f"{ms}.json")

    def _roll(self, fw: int, fh: int, now: float) -> None:
        self._finalize()
        self._frame_w, self._frame_h = fw, fh
        self._seg_start = now
        self._frames_in_seg = 0
        self._staging = self._dir / f".writing_{int(now * 1000)}.mp4"

        if self._use_ffmpeg:
            self._proc = self._spawn_ffmpeg(self._staging, fw, fh)
            if self._proc is None:
                self._use_ffmpeg = False
        if not self._use_ffmpeg:
            fourcc = cv2.VideoWriter_fourcc(*self._cv_fourcc)
            w = cv2.VideoWriter(str(self._staging), fourcc, BUFFER_FPS, (fw, fh))
            self._cv_writer = w if w.isOpened() else None

    def _spawn_ffmpeg(self, staging: Path, fw: int, fh: int) -> Optional[subprocess.Popen]:
        cmd = [
            self._ffmpeg or "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{fw}x{fh}", "-r", str(BUFFER_FPS), "-i", "-",
            "-an",
            # H.264 baseline-friendly settings: decodes in every browser.
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-g", str(BUFFER_FPS),
            "-movflags", "+faststart",
            str(staging),
        ]
        try:
            return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except OSError as exc:
            print(f"[REPLAY] Could not start ffmpeg ({exc}); falling back to mp4v.")
            return None

    def _finalize(self) -> None:
        proc, writer, staging = self._proc, self._cv_writer, self._staging
        started, frames = self._seg_start, self._frames_in_seg
        # WALL-CLOCK end of this segment. Deriving the end from frames/fps left
        # a ~1s hole in the timeline between consecutive clips whenever the
        # engine captured slightly slower than BUFFER_FPS.
        wall_end = time.time()
        self._proc = self._cv_writer = self._staging = None
        if proc is None and writer is None:
            return

        def _finish() -> None:
            try:
                if proc is not None:
                    if proc.stdin is not None:
                        try:
                            proc.stdin.close()
                        except OSError:
                            pass
                    proc.wait(timeout=20)
                if writer is not None:
                    writer.release()
            except Exception:
                try:
                    if proc is not None:
                        proc.kill()
                except Exception:
                    pass

            if staging is None or not staging.exists():
                return
            try:
                if staging.stat().st_size < MIN_VALID_BYTES or frames <= 0:
                    staging.unlink(missing_ok=True)
                    return
                ended = max(wall_end, started + 0.5)
                playback = max(0.5, frames / float(BUFFER_FPS))
                name = segment_filename(self.camera_name, started, ended)
                final = self._dir / name
                if final.exists():                       # never duplicate
                    stem, suf = final.stem, final.suffix
                    i = 2
                    while (self._dir / f"{stem}({i}){suf}").exists():
                        i += 1
                    final = self._dir / f"{stem}({i}){suf}"
                staging.replace(final)

                sidecar = {
                    "camera_id": self.camera_id,
                    "camera_name": self.camera_name,
                    "file": final.name,
                    "start_epoch": started,
                    "end_epoch": ended,
                    "duration": round(ended - started, 2),
                    # Length of the FILE when played back. Differs from the
                    # wall-clock duration when capture rate != BUFFER_FPS, so
                    # the player scales its seek offset by the ratio.
                    "playback_seconds": round(playback, 3),
                    "width": self._frame_w,
                    "height": self._frame_h,
                    "fps": BUFFER_FPS,
                    "codec": ("h264" if (proc is not None or self._encoder["h264"])
                              else "mp4v"),
                    # Verified against the real file, not assumed.
                    "browser_playable": _produces_h264(final),
                    "size_bytes": final.stat().st_size,
                }
                final.with_suffix(".json").write_text(
                    json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
                # Refresh the manifest FROM INSIDE the finalize thread. Writing
                # it in release() raced this thread and published an index that
                # did not yet list the clip that had just been finalized.
                self._write_index()
            except Exception:
                try:
                    staging.unlink(missing_ok=True)
                except Exception:
                    pass

        threading.Thread(target=_finish, daemon=True).start()

    def _sweep_old(self) -> None:
        cutoff = time.time() - RETENTION_SECONDS
        try:
            for seg in list_segments(self.camera_id):
                if seg["end_epoch"] < cutoff:
                    Path(seg["path"]).unlink(missing_ok=True)
                    Path(seg["path"]).with_suffix(".json").unlink(missing_ok=True)
            for d, suf in ((self._frames, ".jpg"), (self._meta, ".json")):
                for f in d.glob(f"*{suf}"):
                    try:
                        if int(f.stem) / 1000.0 < cutoff:
                            f.unlink(missing_ok=True)
                    except (ValueError, OSError):
                        continue
            for f in self._dir.glob(".writing_*.mp4"):
                if time.time() - f.stat().st_mtime > SEGMENT_SECONDS * 3:
                    f.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Reader API (used by the dashboard)
# ---------------------------------------------------------------------------

def list_segments(camera_id: str) -> List[dict]:
    """Finalized clips for a camera, oldest -> newest, with real clock times.

    Understands both the new named format (via its .json sidecar) and the
    legacy ``<start_ms>_<seconds>.mp4`` names, so existing recordings keep
    working.
    """
    d = live_dir(camera_id)
    if not d.is_dir():
        return []
    out: List[dict] = []
    for f in d.glob("*.mp4"):
        if f.name.startswith(".writing_"):
            continue
        try:
            if f.stat().st_size < MIN_VALID_BYTES:
                continue
        except OSError:
            continue

        sidecar = f.with_suffix(".json")
        if sidecar.is_file():
            try:
                m = json.loads(sidecar.read_text(encoding="utf-8"))
                out.append({
                    "path": f, "name": f.name,
                    "camera_name": m.get("camera_name", camera_id),
                    "start_epoch": float(m["start_epoch"]),
                    "end_epoch": float(m["end_epoch"]),
                    "duration": float(m.get("duration", 0)),
                    "width": m.get("width"), "height": m.get("height"),
                    "codec": m.get("codec", "h264"),
                    "playback_seconds": float(
                        m.get("playback_seconds", m.get("duration", 0)) or 0),
                    "browser_playable": bool(m.get("browser_playable", True)),
                    "size_bytes": m.get("size_bytes", f.stat().st_size),
                    # legacy keys kept so older callers keep working
                    "start_ms": int(float(m["start_epoch"]) * 1000),
                    "seconds": int(float(m.get("duration", 0))),
                })
                continue
            except Exception:
                pass

        legacy = parse_segment_name(f)
        if legacy:
            start_ms, secs = legacy
            out.append({
                "path": f, "name": f.name, "camera_name": camera_id,
                "start_epoch": start_ms / 1000.0,
                "end_epoch": start_ms / 1000.0 + secs,
                "duration": float(secs), "width": None, "height": None,
                "codec": "mp4v", "browser_playable": False,
                "playback_seconds": float(secs),
                "size_bytes": f.stat().st_size,
                "start_ms": start_ms, "seconds": secs,
            })
    out.sort(key=lambda s: s["start_epoch"])
    return out


def list_keyframes(camera_id: str, since: Optional[float] = None,
                   until: Optional[float] = None) -> List[float]:
    """Epoch timestamps of every stored key frame, ascending."""
    d = frames_dir(camera_id)
    if not d.is_dir():
        return []
    out = []
    for f in d.glob("*.jpg"):
        try:
            t = int(f.stem) / 1000.0
        except ValueError:
            continue
        if since is not None and t < since:
            continue
        if until is not None and t > until:
            continue
        out.append(t)
    out.sort()
    return out


def keyframe_path(camera_id: str, epoch: float) -> Optional[Path]:
    """Key frame nearest to *epoch* (this is what makes scrubbing instant)."""
    frames = list_keyframes(camera_id)
    if not frames:
        return None
    nearest = min(frames, key=lambda t: abs(t - epoch))
    p = frames_dir(camera_id) / f"{int(nearest * 1000)}.jpg"
    return p if p.is_file() else None


def keyframe_meta(camera_id: str, epoch: float) -> Dict[str, Any]:
    frames = list_keyframes(camera_id)
    if not frames:
        return {}
    nearest = min(frames, key=lambda t: abs(t - epoch))
    p = meta_dir(camera_id) / f"{int(nearest * 1000)}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def timeline_bounds(camera_id: str) -> Optional[Tuple[float, float]]:
    """(oldest, newest) epoch covered by ANY track - the real clock range."""
    lows, highs = [], []
    segs = list_segments(camera_id)
    if segs:
        lows.append(segs[0]["start_epoch"])
        highs.append(segs[-1]["end_epoch"])
    kf = list_keyframes(camera_id)
    if kf:
        lows.append(kf[0])
        highs.append(kf[-1])
    if not lows:
        return None
    return min(lows), max(highs)


def segment_at(camera_id: str, epoch: float) -> Optional[dict]:
    """Clip containing *epoch*, plus the offset in seconds to seek to."""
    for seg in list_segments(camera_id):
        if seg["start_epoch"] <= epoch <= seg["end_epoch"]:
            s = dict(seg)
            s["seek"] = max(0.0, epoch - seg["start_epoch"])
            return s
    return None


def list_events(camera_id: str, since: Optional[float] = None,
                until: Optional[float] = None, kinds: Optional[List[str]] = None,
                limit: int = 2000) -> List[dict]:
    """Events across every day log, ascending."""
    d = live_dir(camera_id)
    if not d.is_dir():
        return []
    out: List[dict] = []
    for f in sorted(d.glob("events-*.jsonl")):
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = float(rec.get("t", 0))
                if since is not None and t < since:
                    continue
                if until is not None and t > until:
                    continue
                if kinds and rec.get("kind") not in kinds:
                    continue
                out.append(rec)
        except OSError:
            continue
    out.sort(key=lambda r: r["t"])
    return out[-limit:]


def has_live_footage(camera_id: str, fresh_within_seconds: float = 20.0) -> bool:
    """True if this camera produced footage very recently."""
    kf = list_keyframes(camera_id)
    if kf:
        return (time.time() - kf[-1]) <= fresh_within_seconds
    segs = list_segments(camera_id)
    if not segs:
        return False
    return (time.time() - segs[-1]["end_epoch"]) <= fresh_within_seconds


def total_available_seconds(camera_id: str) -> float:
    b = timeline_bounds(camera_id)
    return (b[1] - b[0]) if b else 0.0


def latest_segment(camera_id: str) -> Optional[dict]:
    segs = list_segments(camera_id)
    return segs[-1] if segs else None


def storage_usage_bytes(camera_id: str) -> int:
    total = 0
    d = live_dir(camera_id)
    if not d.is_dir():
        return 0
    for f in d.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def sweep_orphans() -> int:
    """Remove orphaned staging files across all cameras on startup."""
    removed = 0
    root = rp.RECORDINGS_DIR
    if not root.is_dir():
        return 0
    for pattern in (".writing_*.mp4", "*.tmp"):
        for f in root.rglob(pattern):
            try:
                f.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    return removed