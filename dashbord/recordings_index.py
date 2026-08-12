"""
recordings_index.py
-------------------
Lightweight index for fast Replay listing across many cameras/dates.

This is intentionally simple and file-backed (JSON) to avoid adding a
dependency. The index stores minimal metadata for each finalized hourly
segment so the dashboard's Replay page can list quickly without scanning
entire recording folders on every request.

Index format (data/recordings_index.json):
{
  "cam_1": {
     "2026-07-21": {
         "13": {"path": "static/recordings/cam_1/2026-07-21/13.mp4", "size": 12345, "mtime": 1690000000.0}
     }
  }
}

This module provides add_segment(), remove_segment(), get_segments_for_date()
and a full rebuild() which scans the recordings tree and populates the index.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import recording_paths as rp

INDEX_PATH = Path(__file__).resolve().parent / "data" / "recordings_index.json"
_lock = threading.Lock()


def _load_index() -> Dict[str, Any]:
    if not INDEX_PATH.exists():
        return {}
    try:
        with INDEX_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_index(idx: Dict[str, Any]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=2, ensure_ascii=False)
    tmp.replace(INDEX_PATH)


def add_segment(camera_id: str, day: str, hour: str, path: Path) -> None:
    """Add or update an indexed segment entry.

    camera_id: e.g. 'cam_1'
    day: 'YYYY-MM-DD'
    hour: 'HH' (24h)
    path: final file path
    """
    if not rp.is_valid_recording(path):
        return
    stat = path.stat()
    entry = {"path": str(path.as_posix()), "size": stat.st_size, "mtime": stat.st_mtime}
    with _lock:
        idx = _load_index()
        cam = idx.setdefault(camera_id, {})
        day_map = cam.setdefault(day, {})
        day_map[hour] = entry
        _save_index(idx)


def remove_segment(camera_id: str, day: str, hour: str) -> None:
    with _lock:
        idx = _load_index()
        if camera_id in idx and day in idx[camera_id] and hour in idx[camera_id][day]:
            del idx[camera_id][day][hour]
            _save_index(idx)


def get_segments_for_date(camera_id: str, day: str) -> Dict[str, Dict[str, Any]]:
    idx = _load_index()
    return idx.get(camera_id, {}).get(day, {})


def rebuild_index() -> None:
    """Scan the recordings directory and rebuild the entire index from disk."""
    root = rp.RECORDINGS_DIR
    new_idx: Dict[str, Any] = {}
    if not root.exists():
        _save_index(new_idx)
        return
    for cam_dir in root.iterdir():
        if not cam_dir.is_dir():
            continue
        cam = cam_dir.name
        for day_dir in cam_dir.iterdir():
            if not day_dir.is_dir():
                continue
            day = day_dir.name
            for file in day_dir.iterdir():
                if not file.is_file() or not file.suffix == ".mp4":
                    continue
                hour = file.stem
                if not rp.is_valid_recording(file):
                    continue
                new_idx.setdefault(cam, {}).setdefault(day, {})[hour] = {
                    "path": str(file.as_posix()),
                    "size": file.stat().st_size,
                    "mtime": file.stat().st_mtime,
                }
    with _lock:
        _save_index(new_idx)
