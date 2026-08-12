"""
main.py
-------
Corewise AI Engine.

Responsibilities:
    * Connect to the Iriun Webcam (camera.py)
    * Read frames in real time
    * Run YOLOv8 detection + ByteTrack tracking (detection.py)
    * Run face detection for the blur effect (effects.py)
    * Compute TAM / SAM / SOM / Inside / Average Stay / Conversion
      (tracking.py)
    * Draw bounding boxes, the TAM zone, the entrance line and the
      on-screen overlay
    * Apply live effects (face blur / night vision) driven by dashboard
      commands, with no restart required
    * Stream telemetry to the Corewise Server over WebSocket
      (corewise_control.py) so the dashboard updates in real time

Calibration note
----------------
The old file-based calibration (line_config.json / zones.json) has been
removed. Zone and entrance-line data now arrive at runtime via the
dashboard command channel (CorewiseClient.get_state() -> "calibration").
No config files need to exist on disk for the engine to start.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import subprocess
import threading

import cv2
import numpy as np
import os
from datetime import datetime as _dt
from pathlib import Path

# Optional system monitoring
try:
    import psutil
except Exception:
    psutil = None

import recording_paths as rp
import recordings_index as rindex
import replay_buffer as rbuf_events
from replay_buffer import ReplayBuffer, sweep_orphans, FRAME_INTERVAL_SECONDS
from camera import CameraManager, CameraNotFoundError
from corewise_control import CorewiseClient
from detection import PersonDetector
from effects import FaceBlurEffect, NightVisionEffect, apply_effects
from face_recognition import FaceIdentifier
from tracking import PersonTracker
from pipeline import IdentityPipeline
from pose_engine import PoseEngine

TELEMETRY_INTERVAL_SECONDS = 0.2
FACE_ID_EVERY_N_FRAMES = 15
CAMERA_RETRY_INTERVAL_SECONDS = 2.0

# NOTE (Task 1): there is deliberately NO SAM dwell constant in this file.
# The minimum stay time is owned exclusively by the dashboard and arrives on
# the control channel as "sam_min_stay_seconds". If the dashboard has not sent
# one yet the engine forwards None and the tracker declines to count SAM,
# rather than silently falling back to an engine-side number (which is exactly
# how the old hardcoded 5.0 kept overriding the dashboard).

# Persons who leave the TAM zone cannot be re-counted until this cooldown expires.
TAM_RECOUNT_COOLDOWN_SECONDS = 20.0

# NOTE (Task 5): the former TAM_SPIKE_GUARD_MULTIPLIER and TAM_MAX_COUNT = 6
# demo limitations have been REMOVED. They silently truncated the primary
# business metric at six people per session and rewrote the count downward
# whenever fewer than three people were visible. TAM is now reported as
# actually measured; genuine duplicate suppression is handled solely by
# TAM_RECOUNT_COOLDOWN_SECONDS below, which is real de-duplication logic and
# not an artificial ceiling.

# Recording settings. The on-disk storage contract (root folder, camera id,
# file layout) lives in recording_paths.py — shared with the dashboard —
# so the writer (here) and the reader (app.py) can never drift apart.
RECORDING_FPS = 10             # fps written to disk
RECORDING_FRAME_INTERVAL = 3  # write every Nth processed frame

# How often (seconds) the engine re-reads the dashboard's Replay settings
# (recording mode / video quality) from disk. Cheap, and means toggling
# "motion" recording or changing video quality in the dashboard takes
# effect on the engine without a restart.
REPLAY_SETTINGS_POLL_SECONDS = 3.0

_REPLAY_SETTINGS_DIR = Path(__file__).resolve().parent / "data" / "replay_settings"


def load_replay_settings() -> dict:
    """Read the current Replay recording settings the dashboard saved.

    app.py persists these per-store to data/replay_settings/<store_id>.json
    (recording_mode: continuous|motion, video_quality: low|medium|high).
    A single engine serves one store today, so the newest settings file is
    the authoritative one; if none exists yet we return the safe defaults
    that match the previous always-on continuous behavior.
    """
    default = {"recording_mode": "continuous", "video_quality": "medium"}
    try:
        files = sorted(
            _REPLAY_SETTINGS_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return default
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                default.update(
                    {
                        "recording_mode": data.get("recording_mode", "continuous"),
                        "video_quality": data.get("video_quality", "medium"),
                    }
                )
                return default
        except Exception:
            continue
    return default


# --- Visual style -----------------------------------------------------------
# Colours are BGR. Kept muted and consistent so the live feed reads as a clean
# CCTV picture rather than a debug canvas.
COLOR_TAM = (80, 190, 240)     # soft amber
COLOR_SOM = (120, 220, 130)    # soft green
COLOR_BOX = (150, 230, 150)
COLOR_TEXT = (240, 240, 240)


def draw_zone(
    frame: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int],
    closed: bool,
    fill: bool = True,
) -> None:
    """Draw one calibration zone cleanly.

    Thin ANTI-ALIASED outline plus a very light translucent fill. No vertex
    markers - those are an editing aid and belong in the dashboard's
    calibration canvas, not burned into the live feed.
    """
    if points is None or len(points) < 2:
        return
    if fill and closed and len(points) >= 3:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    cv2.polylines(frame, [points], closed, color, 1, lineType=cv2.LINE_AA)


def draw_box(frame: np.ndarray, box: Tuple[int, int, int, int, float], label: str, color: Tuple[int, int, int]) -> None:
    """Draw a single bounding box with a text label above it."""
    x1, y1, x2, y2, _conf = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, lineType=cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, stats: Dict[str, Any]) -> None:
    """Compact single-line stats strip in the bottom-left corner.

    Replaces the old 220x150 solid black box, which covered a large part of
    the picture. This is a slim, semi-transparent bar that stays readable
    without hiding the scene.
    """
    h, w = frame.shape[:2]
    pad = 10
    bar_h = 30
    x0, y0 = pad, h - bar_h - pad
    text = (
        f"COREWISE   TAM {stats.get('tam', 0)}   "
        f"SAM {stats.get('sam', 0)}   SOM {stats.get('som', 0)}"
    )
    (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    bar_w = min(tw + 24, w - 2 * pad)

    strip = frame[y0:y0 + bar_h, x0:x0 + bar_w]
    if strip.size:
        dark = np.zeros_like(strip)
        cv2.addWeighted(dark, 0.55, strip, 0.45, 0, strip)

    cv2.putText(frame, text, (x0 + 12, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, COLOR_TEXT, 1, lineType=cv2.LINE_AA)


def encode_frame_jpeg_b64(frame: np.ndarray, quality: int = 60) -> str:
    """Encode a frame as a base64 JPEG string for the dashboard live view."""
    import base64

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("ascii")


def _pct_to_px(pts: List[List[float]], frame_w: int, frame_h: int) -> List[List[int]]:
    """Convert percentage-based points (0-100) to pixel coordinates."""
    return [
        [int(p[0] / 100 * frame_w), int(p[1] / 100 * frame_h)]
        for p in pts
    ]


def _remux_faststart(src: Path, dst: Path) -> bool:
    """Fast (no re-encode) remux that moves the MP4 index (moov atom) to
    the front of the file so browsers can seek before the whole file has
    downloaded. Used to fix up segments recorded with the OpenCV fallback
    encoder when ffmpeg is available but wasn't used for live encoding."""
    try:
        result = subprocess.run(
            [rp.resolve_ffmpeg() or "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-c", "copy", "-movflags", "+faststart", str(dst)],
            timeout=30,
        )
        return result.returncode == 0 and rp.is_valid_recording(dst)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[REC] Faststart remux failed: {exc}")
        return False


class RecordingWriter:
    """Writes annotated frames to hourly MP4 segments.

    Storage contract: see recording_paths.py (shared with the dashboard).
    A new segment starts automatically at each new hour or resolution
    change.

    Reliability rules — these directly target the previously reported
    "Replay sometimes doesn't work" symptoms:

      * Frames are always written to a hidden ``<hour>.part.mp4`` staging
        file. The final ``<hour>.mp4`` only appears once the segment is
        fully finalized, so the Replay UI can never list or try to play a
        half-written / corrupt file — even if the engine is killed mid
        recording.
      * When ``ffmpeg`` is on PATH, frames are piped straight into it and
        encoded as H.264 with ``-movflags +faststart`` — the format every
        modern browser can decode and seek in natively. Finalizing a
        completed segment (closing the pipe, waiting for ffmpeg to exit,
        renaming into place) happens on a background thread so an hourly
        rollover never stalls the live detection loop.
      * If ``ffmpeg`` is not available, this transparently falls back to
        OpenCV's ``mp4v`` writer (recording still works end-to-end) and
        remuxes to faststart in the background if ffmpeg becomes
        available later, printing one clear warning either way.
    """

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._ffmpeg_exe = rp.resolve_ffmpeg()
        self._use_ffmpeg = self._ffmpeg_exe is not None
        if not self._use_ffmpeg:
            print(
                "[REC] WARNING: no ffmpeg binary found (PATH or imageio-ffmpeg). "
                "Falling back to "
                "OpenCV's mp4v encoder — recordings will still be saved, but "
                "browser seek/playback compatibility is not guaranteed. Install "
                "ffmpeg and restart the engine for production-quality Replay."
            )
        self._proc: Optional[subprocess.Popen] = None
        self._cv_writer: Optional[cv2.VideoWriter] = None
        self._staging_path: Optional[Path] = None
        self._final_path: Optional[Path] = None
        self._current_hour: Optional[str] = None
        self._frame_w = 0
        self._frame_h = 0

    def _hour_key(self) -> str:
        return _dt.now().strftime("%Y-%m-%d/%H")

    # CRF (quality) presets for the ffmpeg encoder — lower CRF = higher
    # quality + bigger files. Configurable from Settings > Replay > Video
    # Quality; defaults to "medium" (unchanged behavior) until a store
    # picks something else.
    _QUALITY_CRF = {"low": 30, "medium": 23, "high": 18}

    def configure(self, recording_mode: str = "continuous", video_quality: str = "medium") -> None:
        """Apply the latest Replay settings from the dashboard. Safe to call
        every frame — cheap, and only takes effect on the *next* segment for
        the encoder settings (mid-segment quality changes would require
        restarting the encoder, which isn't worth the hourly-segment
        interruption); ``recording_mode`` takes effect immediately."""
        self.recording_mode = recording_mode if recording_mode in ("continuous", "motion") else "continuous"
        self.video_quality = video_quality if video_quality in self._QUALITY_CRF else "medium"

    def write(self, frame: np.ndarray, has_detections: bool = True) -> None:
        if getattr(self, "recording_mode", "continuous") == "motion" and not has_detections:
            # Motion-recording mode: skip frames with nobody in view. The
            # segment file is still rolled over on the hour even if nothing
            # was ever written to it (see _close's is_valid_recording guard,
            # which quietly discards a segment that ended up empty).
            return

        h_key = self._hour_key()
        fh, fw = frame.shape[:2]
        if h_key != self._current_hour or fw != self._frame_w or fh != self._frame_h:
            self._roll_segment(h_key, fw, fh)

        if self._use_ffmpeg and self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                print(f"[REC] ffmpeg pipe broke ({exc}); this segment stops recording.")
                self._proc = None
        elif self._cv_writer is not None:
            self._cv_writer.write(frame)

    def _roll_segment(self, h_key: str, fw: int, fh: int) -> None:
        self._close(finalize_async=True)
        self._current_hour = h_key
        self._frame_w, self._frame_h = fw, fh
        date_str, hour_str = h_key.split("/")

        final_path = rp.segment_path(self.camera_id, date_str, hour_str)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path = rp.staging_path_for(final_path)
        self._final_path = final_path
        self._staging_path = staging_path

        if self._use_ffmpeg:
            self._proc = self._spawn_ffmpeg(staging_path, fw, fh)
            if self._proc is None:
                self._use_ffmpeg = False  # spawn failed — degrade for good this run
        if not self._use_ffmpeg:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._cv_writer = cv2.VideoWriter(str(staging_path), fourcc, RECORDING_FPS, (fw, fh))

        print(f"[REC] Recording segment -> {final_path}")

    def _spawn_ffmpeg(self, staging_path: Path, fw: int, fh: int) -> Optional["subprocess.Popen"]:
        # Video quality (from Settings > Replay) maps to an x264 CRF — lower
        # CRF = higher quality + larger files. Defaults to "medium" until a
        # store picks otherwise, preserving the previous behavior.
        crf = self._QUALITY_CRF.get(getattr(self, "video_quality", "medium"), 23)
        cmd = [
            self._ffmpeg_exe or "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{fw}x{fh}", "-r", str(RECORDING_FPS),
            "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(staging_path),
        ]
        try:
            return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            print(f"[REC] Could not start ffmpeg ({exc}); falling back to OpenCV encoder.")
            return None

    def _close(self, finalize_async: bool) -> None:
        proc, cv_writer = self._proc, self._cv_writer
        staging, final = self._staging_path, self._final_path
        self._proc = None
        self._cv_writer = None
        self._staging_path = None
        self._final_path = None

        if proc is None and cv_writer is None:
            return

        def _finish() -> None:
            if proc is not None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                    proc.wait(timeout=15)
                except Exception as exc:
                    print(f"[REC] ffmpeg finalize error: {exc}")
                    proc.kill()
            if cv_writer is not None:
                cv_writer.release()

            if staging is None or final is None:
                return
            if not rp.is_valid_recording(staging):
                # Nothing usable was ever written (e.g. rollover happened
                # before a single frame arrived) — discard instead of
                # leaving a broken file the Replay UI would have to hide.
                staging.unlink(missing_ok=True)
                return
            if cv_writer is not None and rp.ffmpeg_available():
                # Recorded with the OpenCV fallback but ffmpeg exists —
                # do a fast, no-re-encode remux so the file still gets a
                # faststart moov atom and is guaranteed seekable.
                remuxed = staging.with_name(staging.stem + ".remux.mp4")
                if _remux_faststart(staging, remuxed):
                    staging.unlink(missing_ok=True)
                    # If a final already exists, prefer the larger/better file.
                    if final.exists() and rp.is_valid_recording(final):
                        try:
                            if remuxed.stat().st_size > final.stat().st_size:
                                remuxed.replace(final)
                                print(f"[REC] Finalized (replaced) {final}")
                                return
                            else:
                                remuxed.unlink(missing_ok=True)
                                print(f"[REC] Finalized: existing final kept {final}")
                                return
                        except Exception:
                            remuxed.replace(final)
                            print(f"[REC] Finalized {final}")
                            return
                    remuxed.replace(final)
                    print(f"[REC] Finalized {final}")
                    try:
                        # update recordings index for fast Replay listing
                        rindex.add_segment(self.camera_id, final.parent.name, final.stem, final)
                    except Exception:
                        pass
                    return
                remuxed.unlink(missing_ok=True)

            # If a final already exists and is valid, keep the best (larger)
            # file to avoid producing spurious duplicate-hour files.
            try:
                if final.exists() and rp.is_valid_recording(final):
                    if staging.stat().st_size > final.stat().st_size:
                        staging.replace(final)
                        print(f"[REC] Finalized (replaced) {final}")
                        try:
                            rindex.add_segment(self.camera_id, final.parent.name, final.stem, final)
                        except Exception:
                            pass
                    else:
                        staging.unlink(missing_ok=True)
                        print(f"[REC] Final exists and is valid; kept existing {final}")
                else:
                    staging.replace(final)
                    print(f"[REC] Finalized {final}")
                    try:
                        rindex.add_segment(self.camera_id, final.parent.name, final.stem, final)
                    except Exception:
                        pass
            except Exception as exc:
                try:
                    staging.replace(final)
                except Exception:
                    print(f"[REC] Could not finalize {final}: {exc}")
                else:
                    print(f"[REC] Finalized {final}")
                    try:
                        rindex.add_segment(self.camera_id, final.parent.name, final.stem, final)
                    except Exception:
                        pass

        if finalize_async:
            threading.Thread(target=_finish, daemon=True).start()
        else:
            _finish()

    def release(self) -> None:
        """Called on engine shutdown — finalize synchronously (the process
        is exiting anyway) so the very last segment is never lost to a
        background thread being killed along with the process."""
        self._close(finalize_async=False)




class CorewiseEngine:
    """Orchestrates the camera, detection, tracking, effects and telemetry loop.

    Calibration (TAM zone + SOM entrance line) is no longer loaded from disk.
    It arrives via the dashboard command channel and is applied the first time
    a valid "calibration" key appears in the control state. Re-calibrating from
    the dashboard takes effect on the next control tick without restarting the
    engine.
    """

    def __init__(self) -> None:
        # ── Calibration state (populated from dashboard command, not files) ──
        self._calib_key: Optional[str] = None          # JSON fingerprint, detects changes
        self.tam_zone_np: Optional[np.ndarray] = None  # pixel polygon for drawing
        self.line_start: Optional[Tuple[int, int]] = None
        self.line_end: Optional[Tuple[int, int]] = None
        self.som_points_px: List[List[int]] = []   # full N-point SOM shape
        self.sam_zone_px: List[List[int]] = []
        self.som_is_polygon: bool = False
        # Last SAM setting received from the dashboard (None = never received).
        self._sam_min_stay_enabled: bool = True
        self._sam_min_stay_seconds = None
        self.analytics_mode: str = "tam_sam_som"       # or "tam_som"

        self.detector = PersonDetector()

        # Tracker is created lazily on first valid calibration. Until then
        # tracking is skipped and the live frame is still streamed.
        self.tracker: Optional[PersonTracker] = None

        # ── Person Identity system (Re-ID) ──────────────────────────────
        # Turns ByteTrack's per-frame track ids into stable GLOBAL person
        # identities. Created together with the tracker on first calibration
        # and reconfigured with the same pixel points. This is what makes
        # five people walking in together count as FIVE visitors, and one
        # person with a churned track id count as ONE.
        self.identity: Optional[IdentityPipeline] = None

        # ── Optional human skeleton / pose estimation ───────────────────
        # OFF by default: no model is loaded and no inference runs until the
        # dashboard sends pose_enabled=True. Skeletons are attached to the
        # GLOBAL person identity (PersonProfile.pose), never to a bare box.
        self.pose = PoseEngine()
        self._pose_results: Dict[int, Dict[str, Any]] = {}
        self._identity_stats: Dict[str, Any] = {}

        self.face_blur_effect = FaceBlurEffect()
        self.night_vision_effect = NightVisionEffect()

        try:
            self.face_identifier: Optional[FaceIdentifier] = FaceIdentifier()
        except Exception as exc:
            print(f"[MAIN] Face identification disabled: {exc}")
            self.face_identifier = None

        self.client = CorewiseClient(role="engine")
        self.camera = CameraManager()

        self._frame_count = 0
        self._last_telemetry_sent = 0.0
        self._fps_window_start = time.time()
        self._fps_frame_count = 0
        self._current_fps = 0.0

        # TAM deduplication
        self._tam_counted_ids: Set[int] = set()
        self._tam_active_ids: Set[int] = set()
        self._tam_exit_times: Dict[int, float] = {}
        self._tam_dedup_count: int = 0

        # Recording
        self._recorder: Optional[RecordingWriter] = None
        # TASK 2: rolling short-segment live buffer. This makes footage
        # playable within seconds instead of after an hourly rollover,
        # which is why Replay used to show nothing (and therefore black)
        # for up to 60 minutes after the engine started.
        self._replay_buffer: Optional[ReplayBuffer] = None
        self._last_buffer_write: float = 0.0

        # TASK 6: the CameraPipeline multi-camera scaffolding that used to be
        # referenced here was UNREACHABLE - it was only ever constructed by a
        # second, shadowed run() definition that Python discarded in favour of
        # the single-camera loop below. It has been removed along with its
        # bookkeeping. The system-health monitor it also owned has been kept
        # and is now started from the loop that actually runs (see run()).
        self._system_health_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def connect_camera(self) -> bool:
        """Try to (re)connect to the camera. Returns True on success.

        Legacy single-camera support: attempts to connect the default CameraManager
        used by older deployments. New multi-camera setups prefer the per-pipeline
        CameraPipeline instances managed by the identity watcher.
        """
        try:
            self.camera.connect()
            return True
        except CameraNotFoundError as exc:
            print(f"[MAIN] Camera connection failed: {exc}")
            return False






    def _start_system_health_monitor(self) -> None:
        if self._system_health_thread is not None:
            return
        self._system_health_thread = threading.Thread(target=self._system_health_loop, daemon=True)
        self._system_health_thread.start()

    def _system_health_loop(self) -> None:
        while True:
            data = {"cpu_percent": None, "ram_percent": None, "disk_percent": None, "gpu": None}
            try:
                if psutil is not None:
                    data["cpu_percent"] = psutil.cpu_percent(interval=0.5)
                    mem = psutil.virtual_memory()
                    data["ram_percent"] = mem.percent
                    disk = psutil.disk_usage(str(rp.STATIC_DIR))
                    data["disk_percent"] = disk.percent
                # GPU: try nvidia-smi
                try:
                    proc = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
                    if proc.returncode == 0:
                        lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
                        gpus = []
                        for ln in lines:
                            parts = [p.strip() for p in ln.split(",")]
                            if len(parts) >= 2:
                                gpus.append({"util": float(parts[0]), "mem_used": float(parts[1])})
                        data["gpu"] = gpus
                except Exception:
                    data["gpu"] = None
            except Exception:
                pass
            try:
                self.client.send("system_health", data)
            except Exception:
                pass
            time.sleep(5.0)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _apply_calibration(self, control: Dict[str, Any], frame: np.ndarray) -> None:
        """Parse the calibration payload from the dashboard command and rebuild
        the tracker whenever the payload changes (fingerprinted by JSON dump).

        Coordinates are stored as percentages (0-100) in the dashboard and
        converted here to pixels using the actual frame dimensions so the
        calibration stays valid across resolution changes.
        """
        calib = control.get("calibration")
        self.analytics_mode = control.get("analytics_mode", "tam_sam_som")

        # SAM minimum stay time — never hardcoded. Read fresh from the
        # dashboard on every tick so a settings change takes effect
        # immediately, without needing a new calibration payload.
        sam_min_stay_enabled = bool(control.get("sam_min_stay_enabled", True))
        raw_seconds = control.get("sam_min_stay_seconds", None)
        if raw_seconds is None:
            sam_min_stay_seconds = None      # not configured by the dashboard yet
        else:
            try:
                sam_min_stay_seconds = max(0.0, float(raw_seconds))
            except (TypeError, ValueError):
                sam_min_stay_seconds = None
        # Pushed on EVERY control tick (not only when calibration changes), so
        # editing the value in the dashboard takes effect on the next frame.
        self._sam_min_stay_enabled = sam_min_stay_enabled
        self._sam_min_stay_seconds = sam_min_stay_seconds
        if self.tracker is not None:
            self._apply_sam_min_stay(sam_min_stay_enabled, sam_min_stay_seconds)

        if not isinstance(calib, dict):
            if self._calib_key is not None:
                self._clear_calibration()
                self._calib_key = None
            return  # no calibration sent yet — engine stays paused for tracking

        key = json.dumps(calib, sort_keys=True)
        if key == self._calib_key:
            return  # nothing changed

        frame_h, frame_w = frame.shape[:2]
        tam_pct = calib.get("tam_area", [])
        som_pct = calib.get("som_line", [])
        sam_pct = calib.get("sam_area", [])

        # SOM accepts ANY number of points: 2 = entrance polyline,
        # 3/4/5/6/8/10+ = entrance polygon. TAM still needs 3+.
        if len(tam_pct) < 3 or len(som_pct) < 2:
            print("[MAIN] Calibration received but incomplete/empty — clearing zone.")
            self._clear_calibration()
            self._calib_key = key
            return

        tam_px = _pct_to_px(tam_pct, frame_w, frame_h)
        som_px = _pct_to_px(som_pct, frame_w, frame_h)

        # In TAM+SOM mode the SAM zone is the same as TAM (computed by dwell).
        if self.analytics_mode == "tam_som" or len(sam_pct) < 3:
            sam_px = tam_px
        else:
            sam_px = _pct_to_px(sam_pct, frame_w, frame_h)

        self.tam_zone_np = np.array(tam_px, dtype=np.int32)
        self.sam_zone_px = sam_px          # kept for overlay + future SAM-zone mode
        # TASK 3: the full SOM shape is handed to the tracker intact. It used
        # to be collapsed to a single first->last segment, silently discarding
        # every intermediate vertex of a 3+ point shape.
        self.som_points_px = som_px
        self.som_is_polygon = len(som_px) >= 3
        self.line_start = tuple(som_px[0])
        self.line_end = tuple(som_px[-1])

        try:
            self.tracker = PersonTracker(tam_px, som_points=som_px)
            self._reset_tam_dedup()
            self._apply_sam_min_stay(sam_min_stay_enabled, sam_min_stay_seconds)
            print(
                f"[MAIN] Calibration applied — TAM {len(tam_px)} pts, "
                f"SOM {len(som_px)} pts, "
                f"mode={self.analytics_mode}"
            )
        except Exception as exc:
            print(f"[MAIN] Failed to create PersonTracker with new calibration: {exc}")
            self.tracker = None

        # Identity pipeline: created once, RE-configured on every calibration
        # change with the exact same pixel points the tracker received, so
        # both layers always agree on where the store floor is. Identities
        # survive a re-calibration — redrawing a zone must not forget who is
        # in the store.
        try:
            if self.identity is None:
                self.identity = IdentityPipeline(camera_id=rp.default_camera_id())
            self.identity.configure(
                tam_polygon_px=tam_px,
                som_points_px=som_px,
                som_is_polygon=self.som_is_polygon,
            )
            self.identity.set_min_stay(sam_min_stay_enabled, sam_min_stay_seconds)
            print("[MAIN] Identity pipeline configured "
                  f"(backend={self.identity.reid.backend.name}).")
        except Exception as exc:
            print(f"[MAIN] Identity pipeline unavailable: {exc}")
            self.identity = None

        self._calib_key = key

    def _clear_calibration(self) -> None:
        """Wipe the drawn TAM zone / SOM line and stop tracking.

        Called whenever the dashboard reports no calibration, or an
        incomplete one — most importantly right after the user presses
        "Delete Calibration" in Settings. Without this, the engine kept
        drawing the last-known TAM rectangle and SOM line forever, since
        nothing ever told it the calibration was gone (only a genuinely NEW,
        complete calibration payload used to update the shapes).
        """
        self.tam_zone_np = None
        self.line_start = None
        self.line_end = None
        self.som_points_px = []
        self.sam_zone_px = []
        self.som_is_polygon = False
        self.tracker = None
        self._reset_tam_dedup()
        # Identities are kept: deleting a drawn zone is a calibration act,
        # not an amnesty. Only the zone geometry is dropped.
        if self.identity is not None:
            try:
                self.identity.configure(tam_polygon_px=None, som_points_px=None)
            except Exception:
                pass

    def _apply_sam_min_stay(self, enabled: bool, seconds) -> None:
        """Push the dashboard's live SAM minimum-stay setting into the tracker.

        tracking.PersonTracker now exposes set_min_stay(); the old attribute
        shim (which wrote to attributes the tracker ignored - the Task 1 root
        cause) is gone. ``seconds`` may be None, meaning "the dashboard has not
        told us yet", which the tracker treats as "do not count SAM".
        """
        if self.tracker is not None:
            self.tracker.set_min_stay(enabled, seconds)
        if self.identity is not None:
            self.identity.set_min_stay(enabled, seconds)

    # ------------------------------------------------------------------
    # Dashboard commands
    # ------------------------------------------------------------------

    def apply_dashboard_commands(self) -> Dict[str, Any]:
        """Read the latest control state and apply it to detector/effects."""
        control = self.client.get_state()

        confidence = control.get("confidence")
        if isinstance(confidence, (int, float)):
            self.detector.set_confidence(float(confidence))

        model = control.get("model")
        if isinstance(model, str) and model:
            try:
                self.detector.set_model(model)
            except RuntimeError as exc:
                print(f"[MAIN] Could not switch model: {exc}")

        # Human Skeleton toggle. set_enabled is idempotent: it only acts on
        # OFF->ON (load the pose model) and ON->OFF (release it), so reading
        # it on every control tick costs nothing.
        self.pose.set_enabled(bool(control.get("pose_enabled", False)),
                              model_path=control.get("pose_model") or None)

        return control


    # ------------------------------------------------------------------
    # FPS + Telemetry
    # ------------------------------------------------------------------

    def update_fps(self) -> None:
        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_window_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_window_start = time.time()

    def maybe_send_telemetry(self, camera_status: str, frame: "Optional[np.ndarray]" = None) -> None:
        now = time.time()
        if now - self._last_telemetry_sent < TELEMETRY_INTERVAL_SECONDS:
            return
        self._last_telemetry_sent = now

        stats = self.tracker.get_stats() if self.tracker is not None else {"tam": 0, "sam": 0, "som": 0, "inside": 0, "average_stay_time": 0, "conversion_rate": 0, "people_today": 0}
        stats = dict(stats)
        stats["tam"] = self._tam_dedup_count

        # Identity layer overrides the headline counts when it is running:
        # its numbers are per PERSON, so a family of five is five and a
        # churned track id is still one. The tracker's numbers remain the
        # fallback while no calibration / identity pipeline exists.
        identity_payload: Dict[str, Any] = {}
        if self.identity is not None:
            try:
                idt = self.identity.telemetry()
                stats["tam"] = idt.get("tam", stats["tam"])
                stats["sam"] = idt.get("sam", stats["sam"])
                stats["som"] = idt.get("som", stats["som"])
                stats["people_today"] = idt.get("visitors_today",
                                                stats.get("people_today", 0))
                stats["inside"] = idt.get("currently_inside",
                                          stats.get("inside", 0))
                if "conversion_rate" in idt:
                    stats["conversion_rate"] = idt["conversion_rate"]
                identity_payload = {
                    "identity_enabled": True,
                    "identity_backend": idt.get("identity_backend"),
                    "visitors_today": idt.get("visitors_today", 0),
                    "currently_inside": idt.get("currently_inside", 0),
                    "returning_visitors": idt.get("returning_visitors", 0),
                    "average_visit_seconds": idt.get("average_visit_seconds", 0),
                    "longest_visit_seconds": idt.get("longest_visit_seconds", 0),
                }
            except Exception as exc:
                print(f"[MAIN] identity telemetry failed: {exc}")
        payload: Dict[str, Any] = {
            **stats,
            "fps": round(self._current_fps, 1),
            "camera_status": camera_status,
            "current_model": self.detector.model_path,
            "current_confidence": self.detector.confidence_threshold,
            "calibration_ready": self.tracker is not None,
            # Echoed back so the dashboard can PROVE which threshold the engine
            # is applying right now (Task 1 verification aid).
            "sam_min_stay_enabled": self._sam_min_stay_enabled,
            "sam_min_stay_seconds": self._sam_min_stay_seconds,
            "som_points": len(self.som_points_px),
            "som_shape": "polygon" if self.som_is_polygon else "polyline",
            # Pose module status — echoed so the dashboard can PROVE whether
            # the skeleton model is actually loaded right now.
            "pose_enabled": self.pose.enabled,
            "pose_model_loaded": self.pose.status()["loaded"],
            **identity_payload,
        }
        if frame is not None:
            small = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))
            payload["live_frame_jpeg_b64"] = encode_frame_jpeg_b64(small)

        self.client.send("telemetry", payload)

    # ------------------------------------------------------------------
    # Replay metadata + events
    # ------------------------------------------------------------------

    def _build_ai_meta(self, detections, ids_in_tam, extra: Dict[str, Any]) -> Dict[str, Any]:
        """AI state for this instant, stored on the Replay metadata track.

        Kept deliberately small (one record per second) so a full day of
        history stays a few megabytes rather than gigabytes.
        """
        people = []
        for person in detections:
            pid = person["id"]
            x1, y1, x2, y2, conf = person["box"]
            track = self.tracker.tracks.get(pid) if self.tracker is not None else None
            record = {
                "id": int(pid),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "conf": round(float(conf), 3),
                "in_tam": pid in ids_in_tam,
                "sam": bool(getattr(track, "counted_sam", False)),
                "som": bool(getattr(track, "counted_som", False)),
                "dwell": round(self.tracker.dwell_seconds(pid), 1) if self.tracker else 0.0,
            }
            # Global person identity — the id that survives track churn and
            # re-entries. Also promote the per-PERSON retail flags: they are
            # the truthful ones (track flags reset when a track id churns).
            if self.identity is not None:
                gid = self.identity.identity.global_id_for(pid)
                if gid:
                    record["gid"] = gid
                    record["label"] = self.identity.label_for(pid)
                    profile = self.identity.memory.get(gid)
                    if profile is not None:
                        record["sam"] = bool(profile.retail.counted_sam)
                        record["som"] = bool(profile.retail.counted_som)
                        record["returning"] = bool(profile.is_returning)
            # Skeleton (only present when the pose module is enabled).
            _pose = self._pose_results.get(pid)
            if _pose is not None:
                record["pose"] = {
                    "keypoints": _pose.get("keypoints", []),
                    "posture_features": _pose.get("posture_features", {}),
                }
            people.append(record)
        meta: Dict[str, Any] = {"people": people}
        meta.update(extra or {})
        if self.tam_zone_np is not None:
            meta["tam_zone"] = self.tam_zone_np.tolist()
        if self.som_points_px:
            meta["som_zone"] = [[int(a), int(b)] for a, b in self.som_points_px]
            meta["som_closed"] = bool(self.som_is_polygon)
        return meta

    def _emit_replay_events(self, stats_before: Dict[str, int], stats_now: Dict[str, Any],
                            new_ids: Set[int]) -> None:
        """Append TAM / SAM / SOM / person events to the Replay event log."""
        if self._replay_buffer is None:
            return
        buf = self._replay_buffer
        for pid in new_ids:
            buf.log_event(rbuf_events.EV_PERSON, {"id": int(pid)})
        for key, kind in (("tam", rbuf_events.EV_TAM),
                          ("sam", rbuf_events.EV_SAM),
                          ("som", rbuf_events.EV_SOM)):
            before = int(stats_before.get(key, 0))
            now = int(stats_now.get(key, 0) or 0)
            if now > before:
                buf.log_event(kind, {key: now, "delta": now - before})

    # ------------------------------------------------------------------
    # TAM deduplication
    # ------------------------------------------------------------------

    def _update_tam_dedup(self, active_ids: List[int], ids_in_tam: Set[int]) -> int:
        """Compute deduplicated TAM count, guarded against ByteTrack ID churn."""
        now = time.time()
        expired = [p for p, t in self._tam_exit_times.items()
                   if now - t > TAM_RECOUNT_COOLDOWN_SECONDS]
        for pid in expired:
            del self._tam_exit_times[pid]
            self._tam_counted_ids.discard(pid)

        for pid in ids_in_tam:
            if pid not in self._tam_counted_ids:
                self._tam_counted_ids.add(pid)
                self._tam_dedup_count += 1

        left_tam = self._tam_active_ids - ids_in_tam
        for pid in left_tam:
            self._tam_exit_times[pid] = now
        self._tam_active_ids = ids_in_tam.copy()

        # TASK 5: the spike guard and the TAM_MAX_COUNT hard cap that used to
        # sit here have been removed. They rewrote the count downward whenever
        # fewer than three people were visible, and capped the store's primary
        # business metric at six visitors per session. TAM is now the real
        # de-duplicated count, with duplicate suppression handled solely by
        # TAM_RECOUNT_COOLDOWN_SECONDS above.
        return self._tam_dedup_count

    def _reset_tam_dedup(self) -> None:
        """Reset counts when calibration changes."""
        self._tam_counted_ids.clear()
        self._tam_active_ids.clear()
        self._tam_exit_times.clear()
        self._tam_dedup_count = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("[COREWISE] Starting AI Engine...")
        print("[COREWISE] Calibration will be applied automatically once the dashboard sends it.")
        self.client.start()
        # TASK 6: previously started only from the dead run(), so the dashboard's
        # System Health page never received engine metrics. Now started here.
        self._start_system_health_monitor()

        camera_connected = self.connect_camera()
        if camera_connected and self._recorder is None:
            # This id MUST match the id the dashboard assigned to this
            # camera in the Setup Wizard (data/camera_identity.json) —
            # recording_paths.default_camera_id() is the single source of
            # truth for both sides. Previously this fell back to a
            # hardcoded "cam0" that no dashboard camera ever used, so
            # every recording was invisible to Replay.
            self._recorder = RecordingWriter(rp.default_camera_id())
            self._replay_buffer = ReplayBuffer(
                rp.default_camera_id(),
                camera_name=os.environ.get("CAMERA_NAME", rp.default_camera_id()),
            )
            try:
                rs = load_replay_settings()
                self._recorder.configure(rs["recording_mode"], rs["video_quality"])
            except Exception:
                self._recorder.configure()
        self._last_replay_cfg = time.time()
        last_camera_attempt = time.time()

        try:
            while True:
                control = self.apply_dashboard_commands()

                if not camera_connected:
                    self.maybe_send_telemetry(camera_status="offline")
                    if time.time() - last_camera_attempt >= CAMERA_RETRY_INTERVAL_SECONDS:
                        last_camera_attempt = time.time()
                        camera_connected = self.connect_camera()
                    time.sleep(0.1)
                    continue

                if not control.get("camera_running", True):
                    self.maybe_send_telemetry(camera_status="paused")
                    time.sleep(0.1)
                    continue

                frame = self.camera.read_frame()
                if frame is None:
                    camera_connected = False
                    continue

                self.update_fps()
                self._frame_count += 1

                # Apply calibration from dashboard (no-op if nothing changed).
                self._apply_calibration(control, frame)

                # Display switches (dashboard-controlled, purely cosmetic - they
                # never affect detection, counting or what gets recorded).
                # CLEAN BY DEFAULT. Nothing is painted on the picture unless the
                # dashboard explicitly turns it on, so a fresh start - or a start
                # with no dashboard connected at all - gives a completely clean
                # camera feed.
                show_zones = bool(control.get("show_zones", False))
                show_labels = bool(control.get("show_labels", False))
                show_overlay = bool(control.get("show_overlay", False))

                person_boxes: List[Tuple[int, int, int, int]] = []

                if not control.get("detection_paused", False) and self.tracker is not None:
                    detections = self.detector.track_people(frame)
                else:
                    detections = []

                # ── Identity + Pose run on the ORIGINAL pixels ─────────────
                # Both must see the frame BEFORE privacy effects: appearance
                # embeddings and skeletons extracted from a blurred frame are
                # garbage. Nothing from either is drawn here — drawing stays
                # a display-only decision further down.
                identity_stats: Dict[str, Any] = {}
                if self.identity is not None and detections:
                    try:
                        identity_stats = self.identity.process(frame, detections)
                    except Exception as exc:
                        print(f"[MAIN] identity.process failed: {exc}")
                elif self.identity is not None:
                    try:
                        identity_stats = self.identity.process(frame, [])
                    except Exception:
                        identity_stats = {}
                self._identity_stats = identity_stats

                # Skeletons: only when the dashboard enabled them. The result
                # is attached to the GLOBAL person id, so the skeleton belongs
                # to the human, not to a transient bounding box.
                self._pose_results = {}
                if self.pose.enabled and detections:
                    try:
                        self._pose_results = self.pose.process(frame, detections)
                        if self.identity is not None:
                            for _tid, _pose in self._pose_results.items():
                                _prof = self.identity.profile_for(_tid)
                                if _prof is not None:
                                    _prof.pose = {"pose_enabled": True, **_pose}
                    except Exception as exc:
                        print(f"[MAIN] pose.process failed: {exc}")

                # Privacy effects are applied BEFORE anything is drawn, so the
                # frame that gets recorded is blurred/night-visioned but free of
                # bounding boxes, zone outlines and the stats bar. That clean
                # frame is what Replay stores; the AI overlay is reconstructed
                # client-side from the metadata track, which is what makes the
                # Replay overlay toggleable instead of burned into the pixels.
                for _p in detections:
                    _x1, _y1, _x2, _y2, _c = _p["box"]
                    person_boxes.append((_x1, _y1, _x2, _y2))

                frame = apply_effects(
                    frame,
                    blur_faces=bool(control.get("blur_faces", False)),
                    night_vision=bool(control.get("night_vision", False)),
                    person_boxes=person_boxes,
                    face_blur_effect=self.face_blur_effect,
                    night_vision_effect=self.night_vision_effect,
                )
                clean_frame = frame.copy()

                # Draw calibration zones (only when calibration is set).
                if show_zones:
                    if self.tam_zone_np is not None:
                        draw_zone(frame, self.tam_zone_np, COLOR_TAM, True)
                    if self.som_points_px:
                        # Renders ANY vertex count: closed polygon for a 3+ point
                        # SOM area, open polyline for a 2-point entrance.
                        som_np = np.array(self.som_points_px, dtype=np.int32)
                        draw_zone(frame, som_np, COLOR_SOM, self.som_is_polygon)

                if self.tracker is None:
                    # No calibration yet. The notice is itself an overlay, so it
                    # only appears when overlays are enabled - otherwise the feed
                    # stays completely clean.
                    if show_overlay:
                        cv2.putText(
                            frame,
                            "Awaiting calibration from dashboard...",
                            (20, 34),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (200, 200, 200),
                            1,
                            lineType=cv2.LINE_AA,
                        )
                    self.maybe_send_telemetry(camera_status="online", frame=frame)
                    cv2.imshow("Corewise AI", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                som_before = self.tracker.som_count
                _stats_before = {
                    "tam": self._tam_dedup_count,
                    "sam": self.tracker.sam_count,
                    "som": self.tracker.som_count,
                }
                _known_ids = set(self.tracker.tracks.keys())
                frame_meta: Dict[str, Any] = {}
                active_ids = []
                ids_in_tam: Set[int] = set()
                for person in detections:
                    pid = person["id"]
                    x1, y1, x2, y2, conf = person["box"]
                    active_ids.append(pid)

                    foot_x = int((x1 + x2) / 2)
                    foot_y = y2

                    if show_labels:
                        cv2.circle(frame, (foot_x, foot_y), 3, (90, 220, 255), -1,
                                   lineType=cv2.LINE_AA)

                    track = self.tracker.update(pid, foot_x, foot_y)

                    # The on-frame label is the PERSON, not the track: 'P-4431'
                    # stays the same when ByteTrack churns ids, and '...' shows
                    # while an identity is still in probation.
                    if self.identity is not None:
                        label = self.identity.label_for(pid)
                    else:
                        label = f"ID {pid}"
                    if (
                        self.face_identifier is not None
                        and self._frame_count % FACE_ID_EVERY_N_FRAMES == 0
                    ):
                        name, _confidence = self.face_identifier.identify(frame, (x1, y1, x2, y2))
                        if name != "Unknown":
                            label = f"{name}"

                    if track.currently_inside_zone:
                        ids_in_tam.add(pid)

                    if show_labels:
                        if track.currently_inside_zone:
                            seconds = self.tracker.dwell_seconds(pid)
                            label = f"{label} · {seconds:.1f}s"
                        draw_box(frame, (x1, y1, x2, y2, conf), label, COLOR_BOX)
                        _pose = self._pose_results.get(pid)
                        if _pose is not None:
                            PoseEngine.draw(frame, _pose)

                self.tracker.expire_stale_tracks(active_ids)
                self._update_tam_dedup(active_ids, ids_in_tam)

                _stats_now = dict(self.tracker.get_stats())
                _stats_now["tam"] = self._tam_dedup_count
                frame_meta = {
                    "tam": _stats_now.get("tam", 0),
                    "sam": _stats_now.get("sam", 0),
                    "som": _stats_now.get("som", 0),
                    "inside": _stats_now.get("inside", 0),
                    "fps": round(self._current_fps, 1),
                }
                self._emit_replay_events(
                    _stats_before, _stats_now,
                    set(active_ids) - _known_ids,
                )

                if show_overlay:
                    draw_overlay(frame, self.tracker.get_stats())

                # TASK 2: feed the rolling live buffer on a TIME-based cadence
                # (not every Nth frame), so recorded playback speed matches real
                # time regardless of how fast the camera is actually running.
                # Recording continues while the user watches Replay - this writer
                # is completely independent of the dashboard.
                if self._replay_buffer is not None:
                    _now_buf = time.time()
                    if _now_buf - self._last_buffer_write >= FRAME_INTERVAL_SECONDS:
                        self._last_buffer_write = _now_buf
                        self._replay_buffer.write(clean_frame, self._build_ai_meta(
                            detections, ids_in_tam, frame_meta))

                if self._recorder is not None and self._frame_count % RECORDING_FRAME_INTERVAL == 0:
                    now_cfg = time.time()
                    if now_cfg - getattr(self, "_last_replay_cfg", 0.0) >= REPLAY_SETTINGS_POLL_SECONDS:
                        self._last_replay_cfg = now_cfg
                        rs = load_replay_settings()
                        self._recorder.configure(rs["recording_mode"], rs["video_quality"])
                    self._recorder.write(frame, has_detections=len(active_ids) > 0)

                if self.tracker.som_count > som_before:
                    self.client.send(
                        "event",
                        {"event": "person_entered", "timestamp": time.time(), "som": self.tracker.som_count},
                    )

                self.maybe_send_telemetry(camera_status="online", frame=frame)

                cv2.imshow("Corewise AI", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.camera.release()
            if self._recorder is not None:
                self._recorder.release()
            if self._replay_buffer is not None:
                self._replay_buffer.release()
            cv2.destroyAllWindows()


def main() -> None:
    removed = rp.sweep_stale_staging_files()
    enc = rbuf_events.probe_encoder()
    if enc["h264"]:
        print(f"[REC] Replay encoder verified: {enc['label']}")
    else:
        print("=" * 70)
        print("[REC] NO H.264 ENCODER FOUND.")
        print("      Every clip recorded in this session will be mp4v, which NO")
        print("      browser can play - Replay will show key frames, not video.")
        print("      FIX:  pip install imageio-ffmpeg")
        print("      (install it for THIS interpreter: " + __import__("sys").executable + ")")
        print("=" * 70)
    orphans = sweep_orphans()
    if orphans:
        print(f"[REC] Cleaned up {orphans} orphaned live-buffer segment(s).")
    if removed:
        print(f"[REC] Cleaned up {removed} abandoned recording segment(s) from a previous run.")
    engine = CorewiseEngine()
    engine.run()


if __name__ == "__main__":
    main()