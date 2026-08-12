"""
recording_paths.py
-------------------
Single source of truth for the Replay System's on-disk storage contract.

Both the AI engine (main.py — writer side) and the dashboard (app.py —
reader side) import this module so the two sides can never drift apart on
the recordings root folder or the camera id used to namespace them. That
drift ("cam0" written by the engine vs. "cam_1" expected by the dashboard)
was the #1 root cause of "no recordings ever appear in Replay".

Storage contract:

    static/recordings/<camera_id>/<YYYY-MM-DD>/<HH>.mp4

Recordings are stored directly under ./static (Streamlit's own static
folder) instead of a top-level recordings/ folder that then gets
symlinked into static/. That symlink trick required admin/Developer-Mode
privileges on Windows and does not survive Replit's filesystem, so it was
a second, independent root cause of Replay randomly not working. Writing
straight into static/ removes the symlink entirely — there is exactly one
folder, one contract, one thing that can go wrong.

To actually be served over HTTP, `.streamlit/config.toml` must have:

    [server]
    enableStaticServing = true

(this repo now ships that file — see .streamlit/config.toml).
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import date
from pathlib import Path
from typing import Optional, Union

# ``static/`` next to whichever of app.py / main.py imports this module —
# both live at the project root, so this resolves to the same folder for
# both processes.
STATIC_DIR = Path(__file__).resolve().parent / "static"
RECORDINGS_DIR = STATIC_DIR / "recordings"

# A finished, playable MP4 (even a very short one) is always well above
# this size. Anything smaller is what's left behind when a segment is
# interrupted before it could be finalized (crash / kill / power loss) —
# treated as corrupt everywhere in the Replay System so a broken file can
# never be listed, played, or crash a duration probe.
MIN_VALID_RECORDING_BYTES = 4096

DayLike = Union[str, date]


def _day_str(day: DayLike) -> str:
    return day.strftime("%Y-%m-%d") if isinstance(day, date) else day


def default_camera_id() -> str:
    """The camera id this AI engine process represents.

    Configurable via the ``CAMERA_ID`` environment variable so that,
    when this store grows to multiple cameras, each engine process can be
    launched with its own stable id (``CAMERA_ID=cam_2 python main.py``)
    that matches the id assigned to that physical camera in the
    dashboard's Setup Wizard (data/camera_identity.json).

    Defaults to "cam_1" — the id already assigned to this store's single
    camera today (see data/camera_identity.json and the dashboard's
    single-camera telemetry fallback in app.py's get_camera_manager) —
    so a fresh checkout keeps working without any extra configuration.
    """
    return os.environ.get("CAMERA_ID", "cam_1")


def camera_recordings_root(camera_id: str) -> Path:
    """Folder holding every date subfolder for one camera."""
    return RECORDINGS_DIR / camera_id


def day_folder(camera_id: str, day: DayLike) -> Path:
    return camera_recordings_root(camera_id) / _day_str(day)


def segment_path(camera_id: str, day: DayLike, hour: str) -> Path:
    """Final, playable path for one hourly segment."""
    return day_folder(camera_id, day) / f"{hour}.mp4"


def staging_path_for(final_path: Path) -> Path:
    """Path a segment is written to while still in progress / being
    finalized. Never listed or played by the dashboard — only renamed (or
    discarded, if it never became valid) into ``final_path`` once
    complete. This is what guarantees the Replay UI can never see a
    half-written or corrupt file, even if the engine process is killed
    mid-recording."""
    return final_path.with_name(final_path.stem + ".part.mp4")


def is_valid_recording(path: Path) -> bool:
    """True if *path* looks like a complete, playable recording.

    Used everywhere (listing, search, duration probing, playback) so a
    corrupt/incomplete file is silently skipped instead of surfacing as a
    confusing error or an unplayable entry in the Replay UI."""
    try:
        return path.is_file() and path.stat().st_size >= MIN_VALID_RECORDING_BYTES
    except OSError:
        return False


def resolve_ffmpeg() -> Optional[str]:
    """Path to an ffmpeg binary, or None.

    Checks PATH first, then the static binary shipped by the imageio-ffmpeg
    wheel. Previously this only looked at PATH, so on a machine where ffmpeg
    came from the wheel the hourly recorder still announced "ffmpeg was not
    found" and silently fell back to mp4v - contradicting the Replay recorder,
    which had already found and verified the very same binary.
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


def ffmpeg_available() -> bool:
    return resolve_ffmpeg() is not None


# A staging file actively being written to gets its mtime touched on
# basically every frame (RECORDING_FRAME_INTERVAL is a handful of frames
# at RECORDING_FPS). Anything older than this was abandoned by an engine
# process that never got to finalize it (killed / crashed / power loss)
# rather than one that's still recording — safe to treat as garbage.
STALE_STAGING_AGE_SECONDS = 15 * 60


def sweep_stale_staging_files(older_than_seconds: float = STALE_STAGING_AGE_SECONDS) -> int:
    """Delete abandoned ``*.part.mp4`` / ``*.remux.mp4`` staging files left
    behind by a previous engine process that was killed before it could
    finalize its current segment (normal shutdown always finalizes
    synchronously — see RecordingWriter.release() in main.py — so this
    only ever cleans up after an unclean exit).

    Meant to be called once, on engine startup, before any camera's
    RecordingWriter is created. Returns the number of files removed.
    Safe to call even if RECORDINGS_DIR doesn't exist yet.
    """
    if not RECORDINGS_DIR.is_dir():
        return 0
    now = time.time()
    removed = 0
    for pattern in ("*.part.mp4", "*.remux.mp4"):
        for stale in RECORDINGS_DIR.rglob(pattern):
            try:
                if now - stale.stat().st_mtime >= older_than_seconds:
                    stale.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
    return removed