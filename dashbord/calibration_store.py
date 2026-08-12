"""
calibration_store.py
--------------------
Single source of truth for camera TAM/SAM/SOM calibration.

This intentionally replaces the old ``camera_identity.json`` as the place
where zone calibration lives. The engine no longer uses any identity
registry; it reads calibration from here, and the dashboard's Camera
Settings calibration process writes here. Nothing recreates an identity
file.

Storage: ``data/calibration.json`` mapping ``camera_id -> {tam_area,
sam_area, som_area, som_line}`` with points as percentages (0-100).

If the file does not exist, cameras simply run with no zones (clean live
feed, no TAM/SAM/SOM counting) until the user calibrates — exactly the
"clean by default" behavior we want.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

_CALIB_FILE = Path(__file__).resolve().parent / "data" / "calibration.json"

_EMPTY = {"tam_area": [], "sam_area": [], "som_area": [], "som_line": []}


def _load_all() -> Dict[str, Dict[str, Any]]:
    try:
        with _CALIB_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_all(data: Dict[str, Dict[str, Any]]) -> None:
    try:
        _CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CALIB_FILE.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        tmp.replace(_CALIB_FILE)
    except Exception:
        pass


def get_calibration(camera_id: str) -> Dict[str, Any]:
    """Return the saved calibration for a camera (or empty lists)."""
    rec = _load_all().get(camera_id)
    if not isinstance(rec, dict):
        return dict(_EMPTY)
    return {
        "tam_area": rec.get("tam_area", []) or [],
        "sam_area": rec.get("sam_area", []) or [],
        "som_area": rec.get("som_area", []) or [],
        "som_line": rec.get("som_line", []) or [],
    }


def set_calibration(
    camera_id: str,
    tam_area: Optional[list] = None,
    sam_area: Optional[list] = None,
    som_area: Optional[list] = None,
    som_line: Optional[list] = None,
) -> None:
    """Persist a camera's calibration. Any omitted field becomes empty."""
    data = _load_all()
    data[camera_id] = {
        "tam_area": tam_area or [],
        "sam_area": sam_area or [],
        "som_area": som_area or [],
        "som_line": som_line or [],
    }
    _save_all(data)


def clear_calibration(camera_id: str) -> None:
    """Reset one camera's calibration to empty (clean state)."""
    data = _load_all()
    data[camera_id] = dict(_EMPTY)
    _save_all(data)


def clear_all() -> int:
    """Reset every camera's calibration. Returns how many were cleared."""
    data = _load_all()
    n = len(data)
    for cam_id in list(data.keys()):
        data[cam_id] = dict(_EMPTY)
    _save_all(data)
    return n


def is_complete(camera_id: str) -> bool:
    """True if a camera has a usable calibration (TAM >=3 pts and a SOM area
    >=3 pts or a SOM line >=2 pts)."""
    c = get_calibration(camera_id)
    tam_ok = len(c["tam_area"]) >= 3
    som_ok = len(c["som_area"]) >= 3 or len(c["som_line"]) >= 2
    return tam_ok and som_ok
