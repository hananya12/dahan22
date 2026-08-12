"""
app.py
------
Corewise Dashboard (Streamlit) - SINGLE FILE BUILD.
 
Live-camera-only investor-grade dashboard. No image/video upload. All
live numbers (TAM/SAM/SOM/FPS/live frame/etc.) arrive from the Corewise
Server over WebSocket via `corewise_control.py` - there is no JSON/TXT/
CSV file involved in keeping this dashboard in sync with the AI engine.
 
Weather and location are the only two values fetched directly by the
dashboard itself (best-effort, cached, fail silently) since they are
about the store's environment, not about the AI engine.
 
Streamlit itself only re-renders on script rerun, so a lightweight
auto-refresh (streamlit_autorefresh) is used purely to trigger a
redraw of already-arrived, in-memory data - it never touches disk and
never blocks on the network, so pushes from the engine are reflected
within one refresh tick (default: every 250ms).
 
-----------------------------------------------------------------------
THIS FILE IS A SINGLE-FILE BUILD.
Everything - store selection, auth, session state, camera placeholder
architecture, and the replay system - lives in this one app.py so the
whole project is one file. Sections are clearly marked below. The only
things that live outside this file are:
  - corewise_control.py   (your existing WebSocket client - untouched)
  - assets/logos/*.png    (binary logo images - can't be Python code)
-----------------------------------------------------------------------
"""
 
from __future__ import annotations
 
import base64
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
 
import io

import altair as alt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from streamlit_autorefresh import st_autorefresh
from streamlit_image_coordinates import streamlit_image_coordinates
 
from corewise_control import CorewiseClient
import recording_paths as rp
import recordings_index as rindex
import replay_buffer as rbuf
import reports as reports_mod
from store_search_index import StoreSearchIndex, SearchRecord












def _ffprobe_duration(path: Path) -> Optional[float]:
    try:
        res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            return float(res.stdout.strip())
    except Exception:
        pass
    return None





 
# =========================================================================
# STORE MANAGER  (store registry + per-store workspace architecture)
# =========================================================================
ASSETS_DIR = Path(__file__).parent / "assets" / "logos"
 
 
@dataclass(frozen=True)
class Store:
    """Static definition of a store available in the store selector."""
 
    store_id: str          # stable internal id, never shown in UI
    display_name: str      # human-readable name (used only in admin/logs, not the selector)
    logo_path: Path        # transparent PNG logo used as the selectable card
    hebrew_name: str = ""  # Hebrew display/search name (optional)
    branch: str = ""       # optional branch label, shown in search suggestions
    city: str = ""         # optional city label, shown in search suggestions
 
 
@dataclass
class StoreWorkspace:
    """
    Per-store runtime context.
 
    Every store is independent: once cameras / analytics / replay /
    employees / reports / settings / camera groups / permissions are
    implemented, each will be looked up through this object so that no
    store can ever see or affect another store's data.
    """
 
    store: Store
    cameras: Dict[str, Any] = field(default_factory=dict)          # placeholder
    analytics_config: Dict[str, Any] = field(default_factory=dict)  # placeholder
    replay_config: Dict[str, Any] = field(default_factory=dict)     # placeholder
    employees: Dict[str, Any] = field(default_factory=dict)         # placeholder
    reports: Dict[str, Any] = field(default_factory=dict)           # placeholder
    settings: Dict[str, Any] = field(default_factory=dict)          # placeholder
    camera_groups: Dict[str, Any] = field(default_factory=dict)     # placeholder
    permissions: Dict[str, Any] = field(default_factory=dict)       # placeholder
 
    @property
    def store_id(self) -> str:
        return self.store.store_id
 
 
# Exact selector order requested for the store carousel.
STORES: list[Store] = [
    Store("super_pharm", "Super-Pharm", ASSETS_DIR / "super_pharm.png", hebrew_name="סופר פארם"),
    Store("super_dahan", "Super Dahan", ASSETS_DIR / "super_dahan.png", hebrew_name="סופר דהן"),
    Store("mania_jeans", "Mania Jeans", ASSETS_DIR / "mania_jeans.png", hebrew_name="מאניה ג'ינס"),
    Store("renuar", "Renuar", ASSETS_DIR / "renuar.png", hebrew_name="רנואר"),
    Store("mega_sport", "Mega Sport", ASSETS_DIR / "mega_sport.png", hebrew_name="מגה ספורט"),
    Store("golda", "Golda", ASSETS_DIR / "golda.png", hebrew_name="גולדה"),
    # --- Newly onboarded stores (Feature 3: entrance logos) ---
    Store("sushimoto", "SushiMoto", ASSETS_DIR / "sushimoto.png", hebrew_name="סושימוטו"),
    Store("idan_2000", "Idan 2000", ASSETS_DIR / "idan_2000.png", hebrew_name="עידן 2000"),
    Store("laline", "Laline", ASSETS_DIR / "laline.png", hebrew_name="ללין"),
    Store("steimatzky", "Steimatzky", ASSETS_DIR / "steimatzky.png", hebrew_name="סטימצקי"),
    Store("stock_big", "Stock BIG", ASSETS_DIR / "stock_big.png", hebrew_name="סטוק ביג"),
    Store("shuka_bair", "Shuka Ba'ir", ASSETS_DIR / "shuka_bair.png", hebrew_name="שוקה בעיר"),
    Store("pizza_hut", "Pizza Hut", ASSETS_DIR / "pizza_hut.png", hebrew_name="פיצה האט"),
]
 
_STORES_BY_ID: Dict[str, Store] = {s.store_id: s for s in STORES}
 
 
def get_store(store_id: str) -> Optional[Store]:
    return _STORES_BY_ID.get(store_id)
 
 
def get_all_stores() -> list[Store]:
    return list(STORES)


@st.cache_resource(show_spinner=False)
def get_store_search_index() -> StoreSearchIndex:
    """Build the autocomplete index ONCE and reuse it across reruns/sessions.

    Cached as a resource so it isn't rebuilt on every keystroke — queries
    walk the prebuilt index, keeping search fast from 10 to 10,000 stores.
    Logos are embedded as data URIs so suggestions render a thumbnail with
    no extra file reads at query time.
    """
    records = []
    for store in STORES:
        try:
            logo_uri = _logo_data_uri(str(store.logo_path), store.display_name)
        except Exception:
            logo_uri = ""
        aliases = tuple(a for a in (store.hebrew_name, store.store_id) if a)
        records.append(
            SearchRecord(
                store_id=store.store_id,
                name=store.hebrew_name or store.display_name,
                logo_uri=logo_uri,
                branch=store.branch,
                city=store.city,
                aliases=(store.display_name, *aliases),
            )
        )
    return StoreSearchIndex(records)
 
 
def build_workspace(store_id: str) -> Optional[StoreWorkspace]:
    """Create a fresh, isolated workspace context for a given store."""
    store = get_store(store_id)
    if store is None:
        return None
    return StoreWorkspace(store=store)
 
# =========================================================================
# STORE CONFIG  (single source of truth for all store login credentials)
#
# ALL store credentials live in data/stores_config.json - nowhere else.
# No password or access code may ever be hardcoded in this file (or any
# other .py file) again. This section only *reads and validates* that
# file; it never contains a raw password itself.
#
# --- Where to add a new supermarket -------------------------------------
#   1. Add a Store(...) entry to the STORES list below (id, display name,
#      logo) so it shows up in the store-selector carousel, same as today.
#   2. Add a matching object to data/stores_config.json, keyed by the same
#      store_id, with store_name / username / password_hash / enabled /
#      settings. A store with no entry in stores_config.json (or with
#      "enabled": false) can never log in - this is intentional
#      fail-closed behavior, not a bug.
#
# --- Where to change a password ------------------------------------------
#   Passwords are never stored in plaintext - only their SHA-256 hash is
#   kept in stores_config.json. To set/change a store's password:
#     1. Generate the hash:
#          python3 -c "import hashlib; print(hashlib.sha256(b'NewPassword').hexdigest())"
#     2. Paste the result into that store's "password_hash" field in
#        data/stores_config.json.
#     3. Set "enabled": true once a real password has been set (stores
#        still on the "changeme" placeholder hash are kept disabled so a
#        guessable default password can never be used to log in).
#
# --- How the login system loads the data ---------------------------------
#   _load_stores_config() reads data/stores_config.json fresh on every
#   call (so an admin can edit passwords/enable a store without
#   restarting the app) and fails safe: any read/parse error (missing
#   file, bad JSON, stray BOM, etc.) is treated as "no stores configured"
#   rather than crashing the dashboard or falling back to any hardcoded
#   password. verify_password() then checks, in order: the store has a
#   config entry -> the entry is enabled -> the entry has a password hash
#   -> the (whitespace-trimmed) password attempt hashes to that value.
# =========================================================================
def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# Central store configuration file (credentials + per-store settings).
# Never hardcode passwords/access codes anywhere else in this project.
STORES_CONFIG_FILE = Path(__file__).parent / "data" / "stores_config.json"


def _load_stores_config() -> dict:
    """Read data/stores_config.json. Fails safe (returns {}) on any error.

    encoding="utf-8-sig" tolerates a UTF-8 byte-order-mark, which some
    editors (e.g. Windows Notepad/PowerShell) silently add to saved JSON
    files and which previously made the *entire* credentials file fail
    to parse - locking every store out regardless of password.
    """
    try:
        if STORES_CONFIG_FILE.exists():
            with open(STORES_CONFIG_FILE, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def get_store_credentials(store_id: str) -> dict | None:
    """Look up one store's config entry, or None if it isn't configured."""
    config = _load_stores_config()
    entry = config.get(store_id)
    return entry if isinstance(entry, dict) else None


def verify_password(store_id: str, attempt: str) -> bool:
    """Validate a plaintext password attempt against stores_config.json.

    - Missing store (no entry, or unknown store_id) -> rejected.
    - Store present but "enabled": false -> rejected.
    - Store present but no password_hash set -> rejected.
    - Accidental leading/trailing spaces in the typed password are
      ignored, since they are a common cashier typing mistake and not a
      meaningful part of the password.
    - Otherwise compared as SHA-256(attempt) == stored password_hash.
    """
    entry = get_store_credentials(store_id)
    if not entry:
        return False
    if not entry.get("enabled", False):
        return False
    expected = entry.get("password_hash")
    if not expected:
        return False
    cleaned_attempt = (attempt or "").strip()
    if not cleaned_attempt:
        return False
    return _hash(cleaned_attempt) == expected
 
# =========================================================================
# AUTHENTICATION MANAGER  (local password check + brute-force guard)
# =========================================================================
MAX_ATTEMPTS_BEFORE_COOLDOWN = 5
COOLDOWN_SECONDS = 30
 
 
@dataclass
class AuthResult:
    success: bool
    message: str = ""
 
 
class AuthenticationManager:
    """Stateless password check + simple in-memory brute-force guard."""
 
    def __init__(self) -> None:
        self._failed_attempts: dict[str, list[float]] = {}
 
    def _recent_failures(self, store_id: str) -> list[float]:
        now = time.time()
        attempts = self._failed_attempts.get(store_id, [])
        attempts = [t for t in attempts if now - t < COOLDOWN_SECONDS]
        self._failed_attempts[store_id] = attempts
        return attempts
 
    def is_locked_out(self, store_id: str) -> bool:
        return len(self._recent_failures(store_id)) >= MAX_ATTEMPTS_BEFORE_COOLDOWN
 
    def attempt_login(self, store_id: str, password: str) -> AuthResult:
        if self.is_locked_out(store_id):
            return AuthResult(False, f"יותר מדי ניסיונות. נסו שוב בעוד {COOLDOWN_SECONDS} שניות.")
 
        if verify_password(store_id, password):
            self._failed_attempts[store_id] = []
            return AuthResult(True, "ברוכים הבאים.")
 
        self._failed_attempts.setdefault(store_id, []).append(time.time())
        return AuthResult(False, "סיסמה שגויה. נסו שוב.")
 
# =========================================================================
# SESSION MANAGER  (store selection / auth / active-page state)
# =========================================================================
_SELECTED_STORE_KEY = "cw_selected_store_id"
_AUTHENTICATED_STORES_KEY = "cw_authenticated_store_ids"
_ACTIVE_PAGE_KEY = "cw_active_page"
_LOGIN_ERROR_KEY = "cw_login_error"
 
 
def session_init() -> None:
    if _SELECTED_STORE_KEY not in st.session_state:
        st.session_state[_SELECTED_STORE_KEY] = None
    if _AUTHENTICATED_STORES_KEY not in st.session_state:
        st.session_state[_AUTHENTICATED_STORES_KEY] = set()
    if _ACTIVE_PAGE_KEY not in st.session_state:
        st.session_state[_ACTIVE_PAGE_KEY] = "Dashboard"
    if _LOGIN_ERROR_KEY not in st.session_state:
        st.session_state[_LOGIN_ERROR_KEY] = None
 
 
def session_get_selected_store_id() -> Optional[str]:
    return st.session_state.get(_SELECTED_STORE_KEY)
 
 
def session_set_selected_store_id(store_id: Optional[str]) -> None:
    st.session_state[_SELECTED_STORE_KEY] = store_id
 
 
def session_is_authenticated(store_id: str) -> bool:
    return store_id in st.session_state.get(_AUTHENTICATED_STORES_KEY, set())
 
 
def session_mark_authenticated(store_id: str) -> None:
    st.session_state.setdefault(_AUTHENTICATED_STORES_KEY, set()).add(store_id)
 
 
def session_get_login_error() -> Optional[str]:
    return st.session_state.get(_LOGIN_ERROR_KEY)
 
 
def session_set_login_error(message: Optional[str]) -> None:
    st.session_state[_LOGIN_ERROR_KEY] = message
 
 
def session_get_active_page() -> str:
    return st.session_state.get(_ACTIVE_PAGE_KEY, "Dashboard")
 
 
def session_set_active_page(page: str) -> None:
    st.session_state[_ACTIVE_PAGE_KEY] = page
 
 
def session_sign_out_current_store() -> None:
    """Clear the active selection so the store picker reappears."""
    st.session_state[_SELECTED_STORE_KEY] = None
    st.session_state[_ACTIVE_PAGE_KEY] = "Dashboard"
 
# =========================================================================
# CAMERA MANAGER  (real cameras are discovered from the live engine's
# telemetry; nothing here is fabricated - if the engine reports no cameras,
# none are shown)
# =========================================================================
@dataclass(frozen=True)
class CameraDescriptor:
    """Description of a single camera reported by the engine for this store."""
 
    camera_id: str
    label: str
    group: str = "Unassigned"
    status: str = "unknown"
 
 
@dataclass
class CameraManager:
    """Per-store camera registry, hydrated from live telemetry."""
 
    store_id: str
    cameras: list[CameraDescriptor] = field(default_factory=list)
 
    def list_cameras(self) -> list[CameraDescriptor]:
        return list(self.cameras)
 
    def list_groups(self) -> list[str]:
        return sorted({c.group for c in self.cameras}) or ["Unassigned"]
 
    def register_camera(self, camera: CameraDescriptor) -> None:
        self.cameras.append(camera)
 
    def get_camera(self, camera_id: str) -> Optional[CameraDescriptor]:
        return next((c for c in self.cameras if c.camera_id == camera_id), None)
 
 
def get_camera_manager(store_id: str, telemetry: Optional[Dict[str, Any]] = None) -> CameraManager:
    """Build the camera registry for a store from the live engine telemetry.
 
    The engine can advertise cameras in two ways, and only real, reported
    cameras are ever registered:
 
      * ``telemetry["cameras"]`` — a list of ``{"id", "label", "status",
        "group"}`` dicts (multi-camera engines). This is the preferred shape
        and is what drives per-camera independent settings when a store has
        more than one camera.
      * ``telemetry["camera_status"]`` — a single status string (the current
        single-camera engines). When present and not ``offline``, the store's
        one live camera is registered.
 
    If neither is available (e.g. the cameras are offline / not yet
    connected), the registry is intentionally left empty so the UI shows an
    honest "no cameras" state instead of inventing hardware.
    """
    manager = CameraManager(store_id=store_id)
    telemetry = telemetry or {}
 
    cameras = telemetry.get("cameras")
    if isinstance(cameras, list) and cameras:
        for idx, cam in enumerate(cameras):
            if not isinstance(cam, dict):
                continue
            manager.register_camera(
                CameraDescriptor(
                    camera_id=str(cam.get("id", f"cam_{idx + 1}")),
                    label=str(cam.get("label") or f"מצלמה {idx + 1}"),
                    group=str(cam.get("group") or "Unassigned"),
                    status=str(cam.get("status") or "unknown"),
                )
            )
        return manager
 
    status = telemetry.get("camera_status")
    if status and status != "offline":
        manager.register_camera(
            CameraDescriptor(camera_id="cam_1", label="מצלמה 1", status=str(status))
        )
    return manager
 
 
# =========================================================================
# CAMERA FEATURE CONTROLS  (global feature catalog + per-camera settings)
#
# Features exist globally for the whole system (Face Blur, Night Vision,
# Replay, and any future feature added to CAMERA_FEATURES below). Each
# camera, however, keeps its OWN independent ON/OFF value for every feature,
# persisted to disk per store so a camera always remembers its own
# configuration and changing one camera never affects another.
# =========================================================================
@dataclass(frozen=True)
class CameraFeature:
    key: str
    label_he: str
    icon: str
    default: bool
    help_he: str = ""
 
 
# The single source of truth for which features exist system-wide. To add a
# future camera feature, append one entry here — the per-camera settings UI
# and persistence pick it up automatically.
CAMERA_FEATURES: list[CameraFeature] = [
    CameraFeature("face_blur", "טשטוש פנים", "🙈", False, "מטשטש פנים של אנשים לשמירה על פרטיות."),
    CameraFeature("night_vision", "ראיית לילה", "🌙", False, "משפר את התמונה בתנאי תאורה חלשה."),
    CameraFeature("replay", "צפייה חוזרת (Replay)", "⏺", True, "מאפשר הקלטה וצפייה חוזרת עבור המצלמה."),
]
 
_CAMERA_FEATURES_BY_KEY: Dict[str, CameraFeature] = {f.key: f for f in CAMERA_FEATURES}
 
CAMERA_SETTINGS_DIR = Path(__file__).parent / "data" / "camera_settings"
 
 
def _camera_settings_file(store_id: str) -> Path:
    CAMERA_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    return CAMERA_SETTINGS_DIR / f"{store_id}.json"
 
 
def _load_camera_settings(store_id: str) -> Dict[str, Dict[str, bool]]:
    path = _camera_settings_file(store_id)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
 
 
def _save_camera_settings(store_id: str, data: Dict[str, Dict[str, bool]]) -> None:
    """Best-effort persistence — never allowed to break the dashboard."""
    try:
        path = _camera_settings_file(store_id)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)
    except Exception:
        pass
 
 
def get_camera_feature_settings(store_id: str, camera_id: str) -> Dict[str, bool]:
    """The full ON/OFF map for one camera, backfilled with feature defaults."""
    stored = _load_camera_settings(store_id).get(camera_id, {})
    return {
        feature.key: bool(stored.get(feature.key, feature.default))
        for feature in CAMERA_FEATURES
    }
 
 
def set_camera_feature(store_id: str, camera_id: str, feature_key: str, value: bool) -> None:
    """Persist a single feature toggle for a single camera (isolated per camera)."""
    if feature_key not in _CAMERA_FEATURES_BY_KEY:
        return
    data = _load_camera_settings(store_id)
    camera_conf = data.setdefault(camera_id, {})
    camera_conf[feature_key] = bool(value)
    _save_camera_settings(store_id, data)

# =========================================================================
# SAM MINIMUM STAY TIME  (per-store Settings, never hardcoded)
#
# Only relevant in "TAM + SOM" analytics mode, where there is no separate
# SAM polygon: a visitor is counted as SAM once they've remained inside TAM
# longer than this many seconds. The dashboard is the only place this value
# is set — it is persisted per store, exactly like camera feature settings,
# and sent to the engine as part of the normal control payload. When OFF, a
# visitor becomes SAM immediately (no dwell-time requirement).
# =========================================================================
SAM_SETTINGS_DIR = Path(__file__).parent / "data" / "sam_settings"
DEFAULT_SAM_MIN_STAY_SECONDS = 5  # only used the very first time a store is configured


def _sam_settings_file(store_id: str) -> Path:
    SAM_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    return SAM_SETTINGS_DIR / f"{store_id}.json"


def _load_sam_settings(store_id: str) -> Dict[str, Any]:
    path = _sam_settings_file(store_id)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_sam_settings(store_id: str, data: Dict[str, Any]) -> None:
    """Best-effort persistence — never allowed to break the dashboard."""
    try:
        path = _sam_settings_file(store_id)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)
    except Exception:
        pass


def get_sam_min_stay_settings(store_id: str) -> Dict[str, Any]:
    """{'enabled': bool, 'seconds': float} for this store, backfilled with
    sane defaults for a store that has never touched this setting.

    'seconds' is a free positive decimal (0, 0.5, 1.2, 2.75, 30, 120, ...) —
    there is no hardcoded step or ceiling. When 'enabled' is False, the
    engine should treat every visitor entering the TAM/SAM area as SAM
    immediately, regardless of whatever value 'seconds' holds."""
    stored = _load_sam_settings(store_id)
    try:
        seconds = float(stored.get("seconds", DEFAULT_SAM_MIN_STAY_SECONDS))
    except (TypeError, ValueError):
        seconds = float(DEFAULT_SAM_MIN_STAY_SECONDS)
    return {
        "enabled": bool(stored.get("enabled", True)),
        "seconds": max(0.0, seconds),
    }


def save_sam_min_stay_settings(store_id: str, enabled: bool, seconds: float) -> None:
    """Persist this store's SAM minimum-stay-time setting. Called only from
    the dashboard's Settings UI — there is no config file for the user to
    hand-edit. 'seconds' accepts any non-negative decimal value."""
    try:
        seconds_val = max(0.0, float(seconds))
    except (TypeError, ValueError):
        seconds_val = float(DEFAULT_SAM_MIN_STAY_SECONDS)
    _save_sam_settings(store_id, {"enabled": bool(enabled), "seconds": seconds_val})

# =========================================================================
# HUMAN SKELETON (POSE) SETTINGS -- per store, persisted like SAM settings.
#
# The skeleton module is OPTIONAL by contract: when disabled the engine
# never loads the pose model and spends zero GPU/CPU on it. The dashboard is
# the single source of truth; the flag rides on every control command as
# "pose_enabled" and the engine echoes back "pose_model_loaded" in telemetry
# so the UI can PROVE the model's real state.
# =========================================================================
POSE_SETTINGS_DIR = Path(__file__).parent / "data" / "pose_settings"
POSE_MODEL_OPTIONS = {
    "מהיר (YOLOv8n-Pose)": "yolov8n-pose.pt",
    "מאוזן (YOLOv8s-Pose)": "yolov8s-pose.pt",
    "מדויק (YOLOv8m-Pose)": "yolov8m-pose.pt",
}


def get_pose_settings(store_id: str) -> Dict[str, Any]:
    """{'enabled': bool, 'model': str}. Default: DISABLED (per the spec —
    the pose model must never load unless the user opted in)."""
    path = POSE_SETTINGS_DIR / f"{store_id}.json"
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                data = loaded if isinstance(loaded, dict) else {}
        except Exception:
            data = {}
    model = data.get("model", "yolov8n-pose.pt")
    if model not in POSE_MODEL_OPTIONS.values():
        model = "yolov8n-pose.pt"
    return {"enabled": bool(data.get("enabled", False)), "model": model}


def save_pose_settings(store_id: str, enabled: bool, model: str) -> None:
    """Best-effort persistence — never allowed to break the dashboard."""
    try:
        POSE_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = POSE_SETTINGS_DIR / f"{store_id}.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"enabled": bool(enabled), "model": str(model)}, f)
        tmp.replace(path)
    except Exception:
        pass

# =========================================================================
# CAMERA IDENTITY REGISTRY  (permanent camera identity + per-camera
# calibration -- replaces manual TAM/SAM/SOM config files)
#
# Every camera the engine ever reports gets, at most, one entry here: which
# store it belongs to, its display name, its type, and its TAM/SAM/SOM
# calibration. A camera with no entry is "unassigned" and triggers the Setup
# Wizard the next time it's seen; once saved, it is permanently remembered
# and never asks again. This is a single global registry (not per-store)
# because a brand-new camera isn't known to belong to any store yet -- the
# wizard is what assigns it to one, and from then on it's only ever shown
# inside that store's workspace (see `filter_cameras_by_store`).
# =========================================================================
CAMERA_TYPES: list[str] = ["Entrance", "Exit", "Street", "Other"]
CAMERA_TYPE_LABELS_HE: Dict[str, str] = {
    "Entrance": "כניסה", "Exit": "יציאה", "Street": "רחוב", "Other": "אחר",
}

CAMERA_IDENTITY_FILE = Path(__file__).parent / "data" / "camera_identity.json"


@dataclass
class CameraCalibration:
    """A camera's own TAM/SAM/SOM zones, as points in percent-of-frame
    coordinates (0-100) so they stay valid regardless of the actual frame
    resolution. Independent per camera -- saving one camera's calibration
    never touches another's."""

    tam_area: list[list[float]] = field(default_factory=list)   # polygon, >=3 points
    sam_area: list[list[float]] = field(default_factory=list)   # polygon, >=3 points
    # TASK 3: ANY number of points. 2 = entrance polyline, 3+ = entrance
    # polygon. The field name is kept for on-disk backward compatibility with
    # every existing data/camera_identity.json.
    som_line: list[list[float]] = field(default_factory=list)   # >=2 points (2=line, 3+=polygon)

    def is_complete(self, analytics_mode: str = "tam_sam_som") -> bool:
        """Returns True when the calibration has enough data for the given mode.

        In TAM+SOM mode SAM is derived from dwell time, so a separate SAM
        polygon is not required and is silently filled from TAM on save.
        """
        tam_ok = len(self.tam_area) >= 3
        # ROOT CAUSE (Task 3): this used to require EXACTLY 2 points. Drawing a
        # 3+ point SOM polygon made is_complete() False, so render_sidebar sent
        # calibration=None, so the engine ran _clear_calibration() and ALL
        # analytics silently stopped. Any shape with 2+ points is now valid.
        som_ok = len(self.som_line) >= 2
        if analytics_mode == "tam_som":
            return tam_ok and som_ok
        return tam_ok and som_ok and len(self.sam_area) >= 3


@dataclass
class CameraIdentity:
    camera_id: str
    store_id: str
    name: str
    camera_type: str
    description: str = ""
    location: str = ""
    calibration: CameraCalibration = field(default_factory=CameraCalibration)
    disconnected: bool = False
    created_at: str = ""


def _load_camera_identity_registry() -> Dict[str, Dict[str, Any]]:
    if not CAMERA_IDENTITY_FILE.exists():
        return {}
    try:
        with open(CAMERA_IDENTITY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_camera_identity_registry(data: Dict[str, Dict[str, Any]]) -> None:
    """Best-effort persistence -- never allowed to break the dashboard."""
    try:
        CAMERA_IDENTITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CAMERA_IDENTITY_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(CAMERA_IDENTITY_FILE)
    except Exception:
        pass


def get_camera_identity(camera_id: str) -> Optional[CameraIdentity]:
    """The saved identity for one camera, or None if it has never been set up."""
    raw = _load_camera_identity_registry().get(camera_id)
    if not raw:
        return None
    calib = raw.get("calibration") or {}
    return CameraIdentity(
        camera_id=camera_id,
        store_id=str(raw.get("store_id", "")),
        name=str(raw.get("name", "")),
        camera_type=str(raw.get("camera_type", "Other")),
        description=str(raw.get("description", "")),
        location=str(raw.get("location", "")),
        calibration=CameraCalibration(
            tam_area=list(calib.get("tam_area", [])),
            sam_area=list(calib.get("sam_area", [])),
            som_line=list(calib.get("som_line", [])),
        ),
        disconnected=bool(raw.get("disconnected", False)),
        created_at=str(raw.get("created_at", "")),
    )


def save_camera_identity(identity: CameraIdentity) -> None:
    """Permanently remember a camera's identity + calibration. Only ever
    called from the dashboard (Setup Wizard / Calibration editor) -- there is
    no config file for the user to hand-edit."""
    data = _load_camera_identity_registry()
    data[identity.camera_id] = {
        "store_id": identity.store_id,
        "name": identity.name,
        "camera_type": identity.camera_type,
        "description": identity.description,
        "location": identity.location,
        "calibration": {
            "tam_area": identity.calibration.tam_area,
            "sam_area": identity.calibration.sam_area,
            "som_line": identity.calibration.som_line,
        },
        "disconnected": bool(identity.disconnected),
        "created_at": identity.created_at or datetime.now().isoformat(timespec="seconds"),
    }
    _save_camera_identity_registry(data)


def is_camera_known(camera_id: str) -> bool:
    return camera_id in _load_camera_identity_registry()


def delete_camera_calibration(camera_id: str) -> None:
    """Wipe only the calibration data for a camera (keep its name/type/store).

    The camera card still appears in settings, but shows "no calibration"
    and prompts the user to recalibrate. The engine will pause analytics until
    new calibration is received from the dashboard.
    """
    data = _load_camera_identity_registry()
    if camera_id in data:
        data[camera_id]["calibration"] = {"tam_area": [], "sam_area": [], "som_line": []}
        _save_camera_identity_registry(data)


def disconnect_camera(camera_id: str) -> None:
    """Mark a camera as disconnected while preserving its identity and settings.

    The camera remains in the registry (name/type/location/calibration preserved)
    but is flagged as disconnected so the UI can show it as such and the
    admin can reconnect it later.
    """
    data = _load_camera_identity_registry()
    if camera_id in data:
        data[camera_id]["disconnected"] = True
        _save_camera_identity_registry(data)


def reconnect_camera(camera_id: str) -> None:
    """Clear the disconnected flag so the camera becomes active again."""
    data = _load_camera_identity_registry()
    if camera_id in data and data[camera_id].get("disconnected", False):
        data[camera_id]["disconnected"] = False
        _save_camera_identity_registry(data)


def delete_camera(store_id: str, camera_id: str) -> None:
    """Fully remove a camera from this store.

    This is a hard delete, distinct from delete_camera_calibration above:
    it removes the camera's assignment/identity record (name, type, store
    link), all of its calibration data (TAM/SAM/SOM), and its per-camera
    feature settings (face blur / night vision / replay toggles). Every
    other camera's identity, calibration and settings live under their own
    camera_id keys and are left completely untouched.

    Note: this only removes dashboard-side configuration. If the camera
    hardware is still connected and the engine keeps reporting it in
    telemetry, it will reappear as a fresh "unassigned" camera on the next
    refresh — that's expected, since camera discovery is driven by live
    telemetry, not by this registry.
    """
    identity_data = _load_camera_identity_registry()
    if camera_id in identity_data:
        del identity_data[camera_id]
        _save_camera_identity_registry(identity_data)

    settings_data = _load_camera_settings(store_id)
    if camera_id in settings_data:
        del settings_data[camera_id]
        _save_camera_settings(store_id, settings_data)


def filter_cameras_by_store(cameras: list["CameraDescriptor"], store_id: str) -> list["CameraDescriptor"]:
    """Only the cameras permanently assigned to this store -- so Store A can
    never see or affect Store B's camera, even though the underlying engine
    connection is shared."""
    out = []
    for cam in cameras:
        identity = get_camera_identity(cam.camera_id)
        if identity is not None and identity.store_id == store_id:
            out.append(cam)
    return out


def find_unassigned_camera(cameras: list["CameraDescriptor"]) -> Optional["CameraDescriptor"]:
    """The first camera the engine currently reports that has never been
    through the Setup Wizard. Used to trigger the wizard automatically."""
    for cam in cameras:
        if not is_camera_known(cam.camera_id):
            return cam
    return None

# =========================================================================
# RECORDING FILESYSTEM + METADATA HELPERS
# =========================================================================
# Storage contract lives in recording_paths.py — the SAME module main.py
# imports for its writer side, so the dashboard (reader) and the AI
# engine (writer) can never drift apart on the root folder or camera id.
# That drift ("cam0" written by the engine vs. "cam_1" expected here) used
# to be the reason recordings never appeared in Replay at all.
#
# Recordings live directly under ./static/recordings/<camera_id>/<date>/
# <hour>.mp4 — no symlink involved (see recording_paths.py docstring for
# why the old symlink-into-static approach was removed). They are served
# by Streamlit's own static file server, enabled via
# .streamlit/config.toml (enableStaticServing = true), which now ships
# with this repo. (recording_paths is already imported at the top of the file.)


@dataclass(frozen=True)
class RecordingEntry:
    """One playable recording file, with everything the Replay UI needs."""

    camera_id: str
    rec_date: date
    hour: str                  # "00".."23"
    path: Path
    size_bytes: int

    @property
    def start_dt(self) -> datetime:
        return datetime.combine(self.rec_date, dt_time(hour=int(self.hour)))

    @property
    def label(self) -> str:
        return f'{self.rec_date.strftime("%d/%m/%Y")} · {self.hour}:00'

    @property
    def static_url(self) -> Optional[str]:
        """Browser-reachable URL for this recording."""
        try:
            rel = self.path.relative_to(rp.STATIC_DIR).as_posix()
        except ValueError:
            return None
        return f"app/static/{rel}"


@st.cache_data(show_spinner=False, ttl=300)
def _probe_duration_seconds(path_str: str, _mtime: float, _size: int) -> Optional[float]:
    """Probe a video's duration with ffprobe. Cached on (path, mtime, size)
    so re-renders don't re-shell-out, and a re-recorded file at the same
    path busts the cache automatically. Returns None if ffprobe is
    unavailable or the file can't be parsed — callers must handle that
    (show "—", never crash the page)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path_str,
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def get_recording_duration(entry: RecordingEntry) -> Optional[float]:
    try:
        stat = entry.path.stat()
    except OSError:
        return None
    return _probe_duration_seconds(str(entry.path), stat.st_mtime, stat.st_size)


def list_recording_hours(camera_id: str, for_date: date) -> list[str]:
    """Return sorted list of hours (e.g. ['08', '09', '10']) that have a
    complete, playable recording file for *camera_id* on *for_date*.

    Files that never finished recording (engine crash / kill mid-segment)
    are filtered out here via ``rp.is_valid_recording`` — the Replay UI
    never has to show, or try to play, a broken file."""
    folder = rp.day_folder(camera_id, for_date)
    if not folder.is_dir():
        return []
    return sorted(
        p.stem for p in folder.glob("*.mp4")
        if p.stem.isdigit() and rp.is_valid_recording(p)
    )


def get_recording_path(camera_id: str, for_date: date, hour: str) -> Optional[Path]:
    """Full path to a recording file, or None if it doesn't exist / isn't
    a complete, playable recording."""
    p = rp.segment_path(camera_id, for_date, hour)
    return p if rp.is_valid_recording(p) else None


def list_recording_dates(camera_id: str) -> list[date]:
    """All dates that have at least one *playable* recording for this
    camera, newest first. Used to power search and to avoid showing a
    date-picker that silently lands on an empty day."""
    root = rp.camera_recordings_root(camera_id)
    if not root.is_dir():
        return []
    out: list[date] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        try:
            day = datetime.strptime(p.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        if list_recording_hours(camera_id, day):
            out.append(day)
    return sorted(out, reverse=True)


def list_all_recordings(camera_id: str) -> list[RecordingEntry]:
    """Every recording that exists for one camera — used for search.
    A camera's recordings are looked up strictly under recordings/<camera_id>/,
    so Camera A can never surface Camera B's files."""
    entries: list[RecordingEntry] = []
    for d in list_recording_dates(camera_id):
        for hour in list_recording_hours(camera_id, d):
            path = get_recording_path(camera_id, d, hour)
            if path is None:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            entries.append(RecordingEntry(camera_id, d, hour, path, size))
    return sorted(entries, key=lambda e: e.start_dt, reverse=True)


def list_recordings_for_date(camera_id: str, for_date: date) -> list[RecordingEntry]:
    entries: list[RecordingEntry] = []
    for hour in list_recording_hours(camera_id, for_date):
        path = get_recording_path(camera_id, for_date, hour)
        if path is None:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        entries.append(RecordingEntry(camera_id, for_date, hour, path, size))
    return sorted(entries, key=lambda e: e.start_dt, reverse=True)


def search_recordings(camera_id: str, query: str) -> list[RecordingEntry]:
    """Simple, forgiving search across every recording of one camera —
    matches against the displayed date and time text, e.g. '14', '14:00',
    '05/07', '05/07/2026'."""
    query = query.strip().lower()
    if not query:
        return list_all_recordings(camera_id)
    out = []
    for e in list_all_recordings(camera_id):
        haystack = f"{e.label} {e.rec_date.isoformat()} {e.hour}".lower()
        if query in haystack:
            out.append(e)
    return out


def delete_recording(entry: RecordingEntry) -> bool:
    """Delete one recording file and prune the date folder if it's now
    empty. Returns True on success."""
    try:
        entry.path.unlink(missing_ok=True)
        parent = entry.path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
        return True
    except OSError:
        return False


def get_camera_id_for_label(cameras: list, label: str) -> Optional[str]:
    """Reverse-lookup: find the camera_id for a given display label."""
    for cam in cameras:
        if cam.label == label:
            return cam.camera_id
    return None


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _format_size(num_bytes: int) -> str:
    mb = num_bytes / (1024 * 1024)
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.1f} MB"


# =========================================================================
# REPLAY SYSTEM  (real playback engine + recording browser)
# =========================================================================
PLAYBACK_SPEEDS = [0.5, 1.0, 2.0, 4.0]


def _inject_replay_theme() -> None:
    st.markdown(
        """
        <style>
            .cw-replay-header {
                font-size: 1.6rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                color: #e6edf3;
                margin-bottom: 0;
            }
            .cw-replay-subheader {
                color: #7d8590;
                font-size: 0.85rem;
                letter-spacing: 0.03em;
                margin-top: -4px;
                margin-bottom: 18px;
            }
            .cw-replay-panel {
                background: #0d1117;
                border: 1px solid #1c2128;
                border-radius: 14px;
                padding: 16px 18px;
                margin-bottom: 14px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            }
            .cw-replay-panel h4 {
                margin: 0 0 10px 0;
                font-size: 0.82rem;
                letter-spacing: 0.09em;
                text-transform: uppercase;
                color: #7d8590;
            }
            .cw-cam-badge {
                display: inline-block;
                background: rgba(0,0,0,0.55);
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 0.72rem;
                letter-spacing: 0.05em;
                color: #7d8590;
                margin-bottom: 10px;
            }
            .cw-empty-state {
                color: #4a5260;
                font-size: 0.82rem;
                text-align: center;
                padding: 18px 8px;
                border: 1px dashed #1c2128;
                border-radius: 10px;
            }
            .cw-live-badge {
    display:inline-block; background:#c8383c; color:#fff; font-size:11px;
    letter-spacing:.06em; padding:4px 12px; border-radius:999px;
    margin-bottom:8px; font-weight:700;
}
.cw-rec-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 8px 10px;
                border-radius: 8px;
                border: 1px solid #1c2128;
                margin-bottom: 6px;
                font-size: 0.82rem;
                color: #c9d1d9;
            }
            .cw-rec-row.selected {
                border-color: #3ddc97;
                background: rgba(61,220,151,0.06);
            }
            .cw-rec-meta { color: #7d8590; font-size: 0.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _replay_static_url(path) -> Optional[str]:
    """Browser-reachable URL for a file inside Streamlit's static folder."""
    try:
        rel = Path(path).resolve().relative_to(rp.STATIC_DIR.resolve()).as_posix()
    except (ValueError, OSError):
        return None
    return f"app/static/{rel}"


_EVENT_STYLE = {
    rbuf.EV_PERSON:    ("👤", "#8b9bb4", "אדם זוהה"),
    rbuf.EV_TAM:       ("🟠", "#e3b341", "TAM"),
    rbuf.EV_SAM:       ("🔵", "#3b82f6", "SAM"),
    rbuf.EV_SOM:       ("🟢", "#3ddc97", "SOM"),
    rbuf.EV_REC_START: ("⏺", "#c8383c", "התחלת הקלטה"),
    rbuf.EV_REC_STOP:  ("⏹", "#6e7681", "סיום הקלטה"),
}


def _replay_meta_lookup(camera_id: str, epoch: float) -> Dict[str, Any]:
    """AI metadata nearest a timestamp (used for the server-side overlay path)."""
    return rbuf.keyframe_meta(camera_id, epoch)


def _render_nvr_player(camera_id: str, camera_label: str, height: int = 700) -> None:
    """The Replay surface.

    IMPORTANT: the HTML handed to components.html is CONSTANT for a given
    camera. It contains no live frame and no recording data, so Streamlit's
    auto-refresh cannot change it - which means the iframe is never re-mounted
    and the <video> element survives. Previously the payload changed on every
    250 ms rerun, the iframe was rebuilt several times a second, and playback
    could never start. The player now feeds itself by polling
    ``live/index.json``, exactly like a real NVR client.
    """
    html = """
<div id="nvr">
  <div class="stage" id="stage">
    <img  id="live"  class="layer" style="display:block">
    <img  id="still" class="layer" style="display:none">
    <video id="vid"  class="layer" playsinline preload="auto" style="display:none"></video>
    <video id="vid2" class="layer" playsinline preload="auto" style="display:none"></video>
    <canvas id="ov" class="layer overlay"></canvas>
    <div class="badge live" id="badge">● LIVE</div>
    <div class="badge cam">__CAMLABEL__</div>
    <div class="badge clock" id="clock">--:--:--</div>
    <div class="spinner" id="spin"></div>
  </div>

  <div class="timeline-wrap">
    <div class="tl" id="tl">
      <div class="tl-fill"></div>
      <div class="tl-markers" id="tlmark"></div>
      <div class="tl-head" id="tlhead"></div>
    </div>
    <div class="tl-labels">
      <span id="tlstart">--:--</span>
      <span id="tlnow" class="now">--:--:--</span>
      <span id="tlend">--:--</span>
    </div>
  </div>

  <div class="bar">
    <button id="b-back" title="אחורה 10 שניות">⏮</button>
    <button id="b-play" class="primary" title="נגן / השהה">▶</button>
    <button id="b-fwd"  title="קדימה 10 שניות">⏭</button>
    <button id="b-live" class="live-btn on" title="חזרה לשידור חי">⏺ LIVE</button>
    <span class="sep"></span>
    <div class="speeds" id="speeds">
      <button data-s="1" class="on">1x</button><button data-s="2">2x</button>
      <button data-s="4">4x</button><button data-s="8">8x</button>
    </div>
    <span class="sep"></span>
    <button id="b-ov"   title="שכבת AI">🧠 AI</button>
    <button id="b-snap" title="צילום מסך">📷</button>
    <button id="b-dl"   title="הורדת הקטע">⬇</button>
    <button id="b-full" title="מסך מלא">⛶</button>
    <span class="grow"></span>
    <span class="meta" id="meta">טוען…</span>
  </div>
</div>

<style>
  #nvr { font-family:'Segoe UI',-apple-system,sans-serif; color:#e6edf3; direction:ltr; }
  #nvr .stage { position:relative; width:100%; height:520px; background:#05070a;
    border-radius:16px; overflow:hidden; border:1px solid rgba(255,255,255,.08);
    box-shadow:0 18px 50px rgba(0,0,0,.55); }
  #nvr .stage:fullscreen { height:100vh; border-radius:0; }
  #nvr .layer { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
  #nvr .overlay { pointer-events:none; }
  #nvr .badge { position:absolute; padding:5px 12px; border-radius:999px; font-size:11px;
    letter-spacing:.06em; font-weight:700; backdrop-filter:blur(10px);
    background:rgba(13,17,23,.62); border:1px solid rgba(255,255,255,.10); }
  #nvr .badge.live { top:12px; left:12px; background:rgba(200,56,60,.85); }
  #nvr .badge.live.rep { background:rgba(88,101,242,.85); }
  #nvr .badge.cam { top:12px; right:12px; }
  #nvr .badge.clock { bottom:12px; left:12px; font-variant-numeric:tabular-nums; }
  #nvr .spinner { position:absolute; inset:0; margin:auto; width:34px; height:34px;
    display:none; border:3px solid rgba(255,255,255,.15); border-top-color:#3ddc97;
    border-radius:50%; animation:spin .8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg);} }
  #nvr .timeline-wrap { margin-top:10px; }
  #nvr .tl { position:relative; height:30px; border-radius:10px; cursor:pointer;
    background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.02));
    border:1px solid rgba(255,255,255,.08); overflow:hidden; }
  #nvr .tl-fill { position:absolute; inset:0;
    background:linear-gradient(90deg,rgba(61,220,151,.14),rgba(61,220,151,.05)); }
  #nvr .tl-markers { position:absolute; inset:0; }
  #nvr .tl-markers i { position:absolute; top:6px; width:3px; height:22px; border-radius:2px;
    transform:translateX(-1px); pointer-events:auto; transition:height .15s,top .15s; }
  #nvr .tl-markers i:hover { top:2px; height:30px; }
  #nvr .tl-head { position:absolute; top:0; bottom:0; width:2px; background:#fff;
    box-shadow:0 0 10px rgba(255,255,255,.85); }
  #nvr .tl-labels { display:flex; justify-content:space-between; font-size:11px;
    color:#7d8590; margin-top:6px; font-variant-numeric:tabular-nums; }
  #nvr .tl-labels .now { color:#3ddc97; font-weight:700; }
  #nvr .bar { display:flex; align-items:center; flex-wrap:wrap; gap:7px;
    margin-top:10px; padding:9px 11px;
    border-radius:14px; background:rgba(22,27,34,.55);
    border:1px solid rgba(255,255,255,.07); backdrop-filter:blur(14px); }
  #nvr .bar button { background:rgba(255,255,255,.05); color:#e6edf3;
    border:1px solid rgba(255,255,255,.09); border-radius:9px; padding:7px 13px;
    font-size:13px; cursor:pointer; transition:background .15s,transform .15s; }
  #nvr .bar button:hover { background:rgba(255,255,255,.12); transform:translateY(-1px); }
  #nvr .bar button.primary { background:#238636; border-color:#2ea043; min-width:46px; }
  #nvr .bar button.live-btn { background:rgba(200,56,60,.22); border-color:rgba(200,56,60,.5); }
  #nvr .bar button.live-btn.on { background:#c8383c; }
  #nvr .bar button#b-ov.on { background:#1f6feb; border-color:#388bfd; }
  #nvr .speeds { display:flex; gap:4px; }
  #nvr .speeds button { padding:6px 10px; font-size:12px; }
  #nvr .speeds button.on { background:#1f6feb; border-color:#388bfd; }
  #nvr .sep { width:1px; height:22px; background:rgba(255,255,255,.10); }
  #nvr .grow { flex:1; }
  #nvr .meta { font-size:11px; color:#7d8590; font-variant-numeric:tabular-nums; }
</style>

<script>
(function(){
  const CAM = "__CAMID__", LABEL = "__CAMLABEL__";
  const base = (function(){ try { return (window.parent && window.parent.location.origin)
                                        || location.origin; } catch(e){ return location.origin; } })();
  const ROOT = base + "/app/static/recordings/" + CAM + "/live/";

  const $=id=>document.getElementById(id);
  const live=$('live'), still=$('still'), ov=$('ov'), badge=$('badge'),
        clock=$('clock'), spin=$('spin'), tl=$('tl'), tlhead=$('tlhead'),
        tlmark=$('tlmark'), meta=$('meta');

  // ---------------------------------------------------------------------
  // DVR PLAYBACK ENGINE
  //
  // Replay is a SECOND, INDEPENDENT reader of the same timeline the recorder
  // is writing. It never talks to the recorder and the recorder never talks
  // to it. The only thing Replay does is walk the wall clock forward from
  // wherever the user dropped the play head, opening whatever file happens to
  // cover that instant. When a file runs out it opens the next one; if the
  // next one has not been flushed to disk yet it WAITS for it. It does not
  // stop, it does not black out, and it never snaps back to LIVE on its own -
  // only the LIVE button does that.
  //
  // Seamlessness comes from double buffering: two <video> elements. While A
  // plays the tail of the current clip, B is already loading and decoding the
  // next clip. At the boundary we swap which one is visible. Because B is
  // primed, the swap costs no network round trip and shows no blank frame.
  // ---------------------------------------------------------------------
  const vidA=$('vid'), vidB=$('vid2');
  let front=vidA, back=vidB;

  let IDX=null, isLive=true, cursor=0, playing=false, speed=1;
  let curSeg=null;      // segment currently in the FRONT buffer
  let nextSeg=null;     // segment currently primed in the BACK buffer
  let waiting=false;    // played to the end of the timeline, awaiting new footage
  let armed=false;      // guards against firing the boundary handler twice
  // AI overlay starts OFF: fetching metadata per frame was the main source of
  // the slowness the user hit. It is opt-in via the 🧠 AI button.
  let overlayOn=false;
  let metaCache=new Map(), lastMetaFetch=0, lastMarkerSig="";

  const EPS=1.5;        // tolerance when matching clip boundaries (seconds)

  const fmt=t=>new Date(t*1000).toLocaleTimeString('he-IL',{hour12:false});
  const fmtS=t=>new Date(t*1000).toLocaleTimeString('he-IL',{hour12:false,hour:'2-digit',minute:'2-digit'});
  const EVC={person:'#8b9bb4',tam:'#e3b341',sam:'#3b82f6',som:'#3ddc97',
             rec_start:'#c8383c',rec_stop:'#6e7681'};
  function fmtDelay(sec){
    sec=Math.max(0,Math.round(sec));
    const h=Math.floor(sec/3600), m=Math.floor((sec%3600)/60), s=sec%60;
    const p=n=>String(n).padStart(2,'0');
    return h?`${h}:${p(m)}:${p(s)}`:`${m}:${p(s)}`;
  }

  // Returns the key frame's EXACT integer-millisecond id, which is also its
  // filename. Never round it: kf_list holds real filenames, not timestamps.
  // O(log n) so scrubbing stays instant with thousands of frames.
  function nearestKFms(tSec){
    const a=IDX && IDX.kf_list; if(!a||!a.length) return null;
    const t=tSec*1000;
    let lo=0, hi=a.length-1;
    if(t<=a[0]) return a[0];
    if(t>=a[hi]) return a[hi];
    while(lo<=hi){ const m=(lo+hi)>>1;
      if(a[m]===t) return a[m];
      if(a[m]<t) lo=m+1; else hi=m-1; }
    const c1=a[Math.max(0,hi)], c2=a[Math.min(a.length-1,lo)];
    return Math.abs(c1-t)<=Math.abs(c2-t)?c1:c2;
  }
  function playables(){
    if(!IDX) return [];
    return IDX.segments.filter(s=>s.playable).sort((a,b)=>a.start-b.start);
  }
  function segFor(t){
    for(const s of playables()) if(t>=s.start && t<=s.end) return s;
    return null;
  }
  // The clip that carries the timeline FORWARD from t. Either the clip that
  // covers t, or - when t lands in a hole where nothing was recorded - the
  // next clip that starts after t. Returning the latter is what lets playback
  // step over gaps instead of dying in them.
  function segForward(t){
    const pl=playables();
    for(const s of pl) if(t>=s.start && t<=s.end) return s;
    for(const s of pl) if(s.start>t) return s;
    return null;
  }
  // The clip that follows `seg` on the wall clock. This is the whole basis of
  // continuous playback: we do not care that clips are separate files, only
  // that one begins where the previous stopped.
  function segAfter(seg){
    if(!seg) return null;
    for(const s of playables()) if(s.start >= seg.end-EPS && s.file!==seg.file) return s;
    return null;
  }
  // Nearest clip that can actually be DECODED. Play must never leave the user
  // looking at the key-frame slideshow when real video exists nearby.
  function nearestPlayable(t){
    const pl=playables();
    if(!pl.length) return null;
    const cover=pl.find(s=>t>=s.start&&t<=s.end);
    if(cover) return cover;
    let best=null,bd=Infinity;
    for(const s of pl){ const d=t<s.start?s.start-t:t-s.end;
      if(d<bd){bd=d;best=s;} }
    return best;
  }
  function newestPlayable(){
    const pl=playables();
    return pl.length?pl[pl.length-1]:null;
  }
  function show(el){ [live,still,vidA,vidB].forEach(x=>{ if(x!==el) x.style.display='none'; });
                     el.style.display='block'; }
  // A clip's file length can differ slightly from the wall-clock span it
  // covers, so map between the two instead of assuming 1:1.
  function toFileTime(seg,t){ const w=Math.max(0.001,seg.end-seg.start);
    return Math.max(0,Math.min(seg.pb||w,(t-seg.start)*((seg.pb||w)/w))); }
  function toWallTime(seg,ft){ const w=Math.max(0.001,seg.end-seg.start);
    return seg.start + ft*(w/(seg.pb||w)); }
  function pct(t){ if(!IDX) return 100;
    const sp=Math.max(1,IDX.end-IDX.start);
    return Math.max(0,Math.min(100,((t-IDX.start)/sp)*100)); }

  function markers(){
    if(!IDX) return;
    const sig=IDX.events.length+"|"+IDX.start+"|"+IDX.end;
    if(sig===lastMarkerSig) return;           // avoid pointless DOM churn
    lastMarkerSig=sig;
    const sp=Math.max(1,IDX.end-IDX.start);
    tlmark.innerHTML=IDX.events.map(e=>{
      const p=((e.t-IDX.start)/sp)*100; if(p<0||p>100) return '';
      return `<i style="left:${p}%;background:${EVC[e.k]||'#888'}" title="${e.k} · ${fmt(e.t)}" data-t="${e.t}"></i>`;
    }).join('');
    tlmark.querySelectorAll('i').forEach(el=>el.addEventListener('click',ev=>{
      ev.stopPropagation(); seekTo(parseFloat(el.dataset.t)); }));
  }

  // ---- buffer plumbing -------------------------------------------------
  function stopBack(){
    try{ back.pause(); }catch(e){}
    back.removeAttribute('src');
    try{ back.load(); }catch(e){}
    nextSeg=null;
  }
  // Load `seg` into the BACK buffer and park it at `atWall`, ready to be shown
  // instantly. Nothing about the front buffer is disturbed, so whatever the
  // user is watching keeps playing untouched while this happens.
  function primeBack(seg, atWall){
    if(!seg) return;
    if(nextSeg && nextSeg.file===seg.file) return;   // already primed
    nextSeg=seg;
    back.src=ROOT+encodeURIComponent(seg.file);
    back.playbackRate=speed;
    back.muted=true;
    try{ back.load(); }catch(e){}
    back.addEventListener('loadedmetadata',function once(){
      back.removeEventListener('loadedmetadata',once);
      if(nextSeg!==seg) return;                      // superseded meanwhile
      try{ back.currentTime=toFileTime(seg, Math.max(atWall, seg.start)); }catch(e){}
      back.playbackRate=speed;
    });
  }
  // Promote the primed BACK buffer to FRONT. This is the seamless cut: the
  // element we reveal already holds decoded frames, so there is no reload,
  // no flash and no gap in motion.
  function promote(){
    const oldFront=front;
    front=back; back=oldFront;
    curSeg=nextSeg; nextSeg=null;
    armed=false; waiting=false;
    spin.style.display='none';
    show(front);
    front.playbackRate=speed;
    if(playing) front.play().catch(()=>{});
    try{ back.pause(); }catch(e){}
    back.removeAttribute('src');
    try{ back.load(); }catch(e){}
    // Immediately start fetching the clip after this one.
    primeBack(segAfter(curSeg), curSeg?curSeg.end:cursor);
    render();
  }

  // Open `seg` directly in the FRONT buffer at wall time `t`.
  // `soft` = do not flash the key-frame still first (used for automatic
  // continuation, where any visual change at all would be noticed).
  function openFront(seg, t, soft){
    curSeg=seg; armed=false; waiting=false;
    if(!soft) spin.style.display='block';
    front.src=ROOT+encodeURIComponent(seg.file);
    front.muted=true;
    try{ front.load(); }catch(e){}
    front.addEventListener('loadedmetadata',function once(){
      front.removeEventListener('loadedmetadata',once);
      if(curSeg!==seg) return;
      try{ front.currentTime=toFileTime(seg,Math.max(t,seg.start)); }catch(e){}
      front.playbackRate=speed;
      spin.style.display='none';
      show(front);
      if(playing) front.play().catch(()=>{});
      primeBack(segAfter(seg), seg.end);
    });
  }

  // The end of the current clip has been reached. Continue the timeline.
  // Under NO circumstances does this return to LIVE or blank the stage.
  function onBoundary(){
    if(isLive || !curSeg || armed) return;
    armed=true;
    const nx = (nextSeg && nextSeg.start >= curSeg.end-EPS) ? nextSeg : segAfter(curSeg);
    if(nx && nextSeg && nx.file===nextSeg.file && back.readyState>=2){
      promote();                       // primed and decodable -> seamless cut
      return;
    }
    if(nx){                            // exists but not primed yet -> open it
      primeBack(nx, nx.start);
      if(back.readyState>=2){ promote(); return; }
      // Give the back buffer a moment; the watchdog will promote it.
      waiting=true; armed=false;
      render();
      return;
    }
    // Nothing written past this point YET. The recorder is still running, so
    // the next clip will exist within a segment length. Hold the last frame
    // (never black, never LIVE) and let the watchdog resume us.
    waiting=true; armed=false;
    cursor=curSeg.end;
    render();
  }

  // Runs 4x/second. Owns every "keep going" decision.
  function watchdog(){
    if(isLive || !IDX) return;

    // Resume from a wait as soon as the recorder has produced the next clip.
    if(waiting){
      const nx = nextSeg || segAfter(curSeg) || segForward(cursor);
      if(nx){
        if(nextSeg && nx.file===nextSeg.file && back.readyState>=2){ promote(); }
        else if(!nextSeg || nextSeg.file!==nx.file){ primeBack(nx, nx.start); }
        else if(back.readyState>=2){ promote(); }
      }
      render();
      return;
    }

    if(!playing || !curSeg) return;

    // Some browsers drop the 'ended' event when a clip is swapped under load.
    if(front.ended || (front.duration && front.currentTime >= front.duration-0.06)){
      onBoundary(); return;
    }
    // Prime the next clip early. Faster speeds need more lead time.
    if(!nextSeg && front.duration){
      const left=(front.duration-front.currentTime)/Math.max(1,speed);
      if(left < 5) primeBack(segAfter(curSeg), curSeg.end);
    }
    // A stalled network must not look like a dead player.
    if(playing && front.paused && !front.ended && front.readyState>=2){
      front.play().catch(()=>{});
    }
  }

  function goLive(){
    isLive=true; playing=false; curSeg=null; waiting=false; armed=false;
    try{ front.pause(); }catch(e){}
    stopBack();
    show(live); badge.textContent='● LIVE'; badge.classList.remove('rep');
    $('b-live').classList.add('on'); $('b-play').textContent='▶';
    spin.style.display='none';
    if(IDX) cursor=IDX.end;
    paint(null); render();
  }

  // User-initiated jump. Shows the key-frame preview while the clip opens,
  // which is the existing scrub feel. Never auto-returns to LIVE.
  function seekTo(t, autoplay){
    if(!IDX){ return; }
    if(autoplay) playing=true;
    t=Math.max(IDX.start,Math.min(IDX.end,t));
    isLive=false; cursor=t; waiting=false; armed=false;
    badge.textContent='⏪ REPLAY'; badge.classList.add('rep');
    $('b-live').classList.remove('on');

    const kfms=nearestKFms(t);
    if(kfms!==null){ still.src=ROOT+"frames/"+kfms+".jpg"; show(still); }

    stopBack();
    const seg=segForward(t);
    if(seg){
      if(t<seg.start) cursor=t=seg.start;      // landed in a gap -> next clip
      openFront(seg,t,false);
    } else {
      // Ahead of everything recorded so far. Stay in REPLAY and wait for the
      // recorder to catch up rather than snapping to LIVE.
      curSeg=null; spin.style.display='none';
      if(playing) waiting=true;
    }
    render();
  }

  function fetchMeta(t){
    if(!overlayOn) return;
    const now=performance.now();
    if(now-lastMetaFetch<250) return;      // throttle
    lastMetaFetch=now;
    const key=nearestKFms(t); if(key===null) return;
    if(metaCache.has(key)){ paint(metaCache.get(key)); return; }
    fetch(ROOT+"meta/"+key+".json").then(r=>r.ok?r.json():null).then(m=>{
      if(metaCache.size>400) metaCache.clear();
      metaCache.set(key,m); paint(m);
    }).catch(()=>{});
  }

  function paint(m){
    const r=ov.getBoundingClientRect();
    if(ov.width!==r.width||ov.height!==r.height){ ov.width=r.width; ov.height=r.height; }
    const g=ov.getContext('2d'); g.clearRect(0,0,ov.width,ov.height);
    if(!overlayOn||!m) return;
    const sw=m.w||640, sh=m.h||480, sc=Math.min(ov.width/sw,ov.height/sh);
    const ox=(ov.width-sw*sc)/2, oy=(ov.height-sh*sc)/2;
    const X=x=>ox+x*sc, Y=y=>oy+y*sc;
    g.lineWidth=1.5; g.font='12px Segoe UI';
    if(m.tam_zone){ g.strokeStyle='rgba(227,179,65,.85)'; g.beginPath();
      m.tam_zone.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));
      g.closePath(); g.stroke(); }
    if(m.som_zone){ g.strokeStyle='rgba(61,220,151,.9)'; g.beginPath();
      m.som_zone.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));
      if(m.som_closed) g.closePath(); g.stroke(); }
    (m.people||[]).forEach(p=>{ const[x1,y1,x2,y2]=p.box;
      g.strokeStyle=p.som?'#3ddc97':(p.sam?'#3b82f6':(p.in_tam?'#e3b341':'#8b9bb4'));
      g.strokeRect(X(x1),Y(y1),(x2-x1)*sc,(y2-y1)*sc);
      const tag=`ID ${p.id} · ${(p.conf*100).toFixed(0)}%`+(p.in_tam?` · ${p.dwell}s`:'');
      g.fillStyle='rgba(0,0,0,.65)'; g.fillRect(X(x1),Y(y1)-17,g.measureText(tag).width+8,15);
      g.fillStyle='#fff'; g.fillText(tag,X(x1)+4,Y(y1)-5); });
  }

  function render(){
    if(!IDX) return;
    tlhead.style.left=pct(cursor)+'%';
    clock.textContent=fmt(cursor); $('tlnow').textContent=fmt(cursor);
    $('tlstart').textContent=fmtS(IDX.start); $('tlend').textContent=fmtS(IDX.end);
    if(isLive){
      meta.textContent = curSeg
        ? `${curSeg.file} · ${curSeg.w||'?'}x${curSeg.h||'?'} · ${(curSeg.size/1048576).toFixed(1)}MB · ${curSeg.codec}`
        : `${IDX.segments.length} clips · ${IDX.kf_count} frames`;
    } else {
      // In REPLAY the useful number is how far behind LIVE we are - that is
      // the whole point of a DVR - so surface it in the existing meta slot.
      const delay=fmtDelay(IDX.end-cursor);
      meta.textContent = waiting
        ? `⏳ ממתין להקלטה הבאה · פיגור ${delay} מהשידור החי`
        : `⏪ פיגור ${delay} מהשידור החי${curSeg?' · '+curSeg.file:''}`;
    }
    fetchMeta(cursor);
  }

  // ---- self-refresh: the player polls, Streamlit never re-mounts it ----
  function poll(){
    fetch(ROOT+"index.json?_="+Date.now()).then(r=>r.ok?r.json():null).then(j=>{
      if(!j) return;
      IDX=j; markers();
      if(isLive){
        cursor=j.end;
        if(j.latest_kf) live.src=ROOT+"frames/"+j.latest_kf+".jpg";
      }
      // A fresh index may be exactly the clip a waiting player needs.
      if(!isLive && waiting) watchdog();
      render();
    }).catch(()=>{});
  }
  setInterval(poll, 1000);
  setInterval(watchdog, 250);
  poll();

  // ---- controls ----
  tl.addEventListener('click',e=>{ if(!IDX) return;
    const r=tl.getBoundingClientRect();
    seekTo(IDX.start+((e.clientX-r.left)/r.width)*(IDX.end-IDX.start)); });
  let drag=false;
  tl.addEventListener('mousedown',()=>drag=true);
  window.addEventListener('mouseup',()=>drag=false);
  tl.addEventListener('mousemove',e=>{ if(!drag||!IDX) return;
    const r=tl.getBoundingClientRect();
    seekTo(IDX.start+((e.clientX-r.left)/r.width)*(IDX.end-IDX.start)); });

  $('b-live').onclick=goLive;
  $('b-back').onclick=()=>seekTo(cursor-10, playing);
  $('b-fwd').onclick =()=>seekTo(cursor+10, playing);
  $('b-play').onclick=()=>{
    if(!IDX) return;
    if(playing){                       // pause
      playing=false; $('b-play').textContent='▶';
      try{ front.pause(); }catch(e){}
      return;
    }
    // Resuming inside REPLAY must never re-enter LIVE: keep the play head
    // exactly where the user left it.
    if(!isLive && curSeg){
      playing=true; $('b-play').textContent='⏸';
      front.playbackRate=speed;
      front.play().catch(()=>{});
      primeBack(segAfter(curSeg), curSeg.end);
      render();
      return;
    }
    // Choose a start point that has DECODABLE video, so Play always produces
    // motion rather than the key-frame slideshow.
    let target = isLive ? (newestPlayable() ? newestPlayable().start : IDX.end-30)
                        : cursor;
    const seg = nearestPlayable(target);
    if(!seg){
      meta.textContent = 'אין עדיין קטע וידאו מוכן לניגון — ממתין לסגירת הקטע הראשון';
      return;
    }
    if(target < seg.start || target > seg.end) target = seg.start + 0.1;
    $('b-play').textContent='⏸';
    seekTo(target, true);
  };
  document.querySelectorAll('#speeds button').forEach(b=>b.onclick=()=>{
    document.querySelectorAll('#speeds button').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); speed=parseFloat(b.dataset.s);
    // Speed is a property of the SESSION, not of a file, so both buffers carry
    // it and it survives every clip change.
    vidA.playbackRate=speed; vidB.playbackRate=speed; });
  $('b-ov').onclick=()=>{ overlayOn=!overlayOn;
    $('b-ov').classList.toggle('on',overlayOn);
    if(!overlayOn) paint(null); else fetchMeta(cursor); };
  $('b-full').onclick=()=>{ const st=$('stage');
    if(document.fullscreenElement) document.exitFullscreen();
    else st.requestFullscreen&&st.requestFullscreen(); };
  $('b-snap').onclick=()=>{
    const src=isLive?live:(front.style.display!=='none'?front:still);
    const c=document.createElement('canvas');
    c.width=src.videoWidth||src.naturalWidth||1280;
    c.height=src.videoHeight||src.naturalHeight||720;
    const g=c.getContext('2d'); g.drawImage(src,0,0,c.width,c.height);
    if(overlayOn) g.drawImage(ov,0,0,c.width,c.height);
    g.fillStyle='rgba(0,0,0,.6)'; g.fillRect(10,c.height-40,380,30);
    g.fillStyle='#fff'; g.font='16px Segoe UI';
    g.fillText(LABEL+"  "+fmt(cursor),18,c.height-19);
    const a=document.createElement('a');
    a.download=LABEL+"_"+fmt(cursor).replace(/:/g,'-')+".png";
    a.href=c.toDataURL('image/png'); a.click(); };
  $('b-dl').onclick=()=>{ if(!curSeg){ alert('גררו את ציר הזמן אחורה כדי לבחור קטע.'); return; }
    const a=document.createElement('a'); a.href=ROOT+encodeURIComponent(curSeg.file);
    a.download=curSeg.file; a.click(); };

  // ---- media events (bound to BOTH buffers) ----
  [vidA,vidB].forEach(v=>{
    v.addEventListener('playing',()=>{ if(v===front){ show(front);
      $('b-play').textContent = playing?'⏸':'▶'; spin.style.display='none'; } });
    v.addEventListener('pause',()=>{ if(v===front && !playing) $('b-play').textContent='▶'; });
    v.addEventListener('timeupdate',()=>{
      if(isLive||v!==front||!curSeg) return;
      cursor=toWallTime(curSeg,v.currentTime);
      render();
    });
    // End of a file is NOT the end of the timeline - it is just a page turn.
    v.addEventListener('ended',()=>{ if(v===front) onBoundary(); });
    // A broken clip must not stop the session: skip past it and carry on.
    v.addEventListener('error',()=>{
      if(v!==front || isLive) return;
      spin.style.display='none';
      const kfms=nearestKFms(cursor);
      if(kfms!==null){ still.src=ROOT+"frames/"+kfms+".jpg"; show(still); }
      const nx=segAfter(curSeg);
      if(nx && playing) openFront(nx, nx.start, true);
      else waiting=true;
    });
  });
  // A missing key frame must not leave a broken-image icon on a black stage,
  // and must NEVER fall through to the live image while we are in REPLAY.
  still.addEventListener('error',()=>{ still.style.display='none';
    if(isLive) show(live);
    else if(front.readyState>=2) show(front); });
})();
</script>
"""
    html = html.replace("__CAMID__", camera_id).replace("__CAMLABEL__", camera_label)
    components.html(html, height=height, scrolling=False)


def render_replay_page(
    camera_manager: CameraManager,
    store_display_name: str,
    telemetry: Optional[Dict[str, Any]] = None,
) -> None:
    """Production Replay System (CCTV/NVR).

    Opens on LIVE, never shows a black screen, scrubs on a REAL CLOCK
    timeline, plays browser-native H.264 clips, reconstructs the AI overlay
    from the metadata track, and lists every AI event.
    """
    _inject_replay_theme()
    telemetry = telemetry or {}

    st.markdown('<div class="cw-replay-header">מערכת שידור חוזר</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="cw-replay-subheader">ניטור והיסטוריה · {store_display_name}</div>',
        unsafe_allow_html=True,
    )

    cameras = camera_manager.list_cameras()
    if not cameras:
        st.markdown(
            '<div class="cw-replay-panel"><div class="cw-empty-state">'
            'לא הוקצו מצלמות לחנות זו עדיין.</div></div>', unsafe_allow_html=True)
        return

    labels = [c.label for c in cameras]

    # ── Multi-camera selector (Task 12) ────────────────────────────────────
    saved = st.session_state.get("cw_replay_camera")
    idx = labels.index(saved) if saved in labels else 0
    camera_label = st.selectbox("מצלמה", labels, index=idx, key="cw_replay_camera")
    # The AI overlay is now the 🧠 AI button INSIDE the player and defaults to
    # OFF. Toggling it here would re-run Streamlit and re-mount the iframe,
    # which is exactly what was killing playback.

    camera_id = get_camera_id_for_label(cameras, camera_label)
    if not camera_id:
        st.error(
            "לא נמצא מזהה למצלמה שנבחרה — לכן הנגן אינו מוצג. "
            f"תוויות זמינות: {labels}"
        )
        return

    # The player is fed by live/index.json, which the engine rewrites once a
    # second, so this call is cheap and its HTML never changes between reruns.
    try:
        _render_nvr_player(camera_id, camera_label)
    except Exception as exc:
        import traceback
        st.error(f"רינדור הנגן נכשל: {exc}")
        st.code(traceback.format_exc(), language="text")

    idx_file = rbuf.live_dir(camera_id) / "index.json"
    if idx_file.is_file():
        try:
            idx = json.loads(idx_file.read_text(encoding="utf-8"))
            st.caption(
                f"מצלמה `{camera_id}` · {len(idx.get('segments', []))} קטעים · "
                f"{idx.get('kf_count', 0)} תמונות מפתח · {len(idx.get('events', []))} אירועים"
            )
        except Exception:
            pass
    else:
        st.info(
            f"אין עדיין מניפסט הקלטה עבור `{camera_id}`. "
            "הפעילו את מנוע ה-AI (`python main.py`) — הנגן יתמלא תוך שניות. "
            "אם זה נמשך, הריצו `python diagnose_replay.py`."
        )

    # ── Smart search (Task 13) + event list (Task 14) ─────────────────────
    st.markdown("")
    ev_col, info_col = st.columns([3, 2], gap="medium")

    with ev_col:
        st.markdown('<div class="cw-replay-panel">', unsafe_allow_html=True)
        st.markdown("<h4>אירועים</h4>", unsafe_allow_html=True)

        kinds = st.multiselect(
            "סינון לפי סוג",
            [rbuf.EV_PERSON, rbuf.EV_TAM, rbuf.EV_SAM, rbuf.EV_SOM,
             rbuf.EV_REC_START, rbuf.EV_REC_STOP],
            default=[rbuf.EV_TAM, rbuf.EV_SAM, rbuf.EV_SOM],
            format_func=lambda k: f"{_EVENT_STYLE.get(k, ('•',))[0]} {_EVENT_STYLE.get(k, ('', '', k))[2]}",
            key="cw_replay_ev_kinds",
        )
        f1, f2 = st.columns(2)
        with f1:
            day = st.date_input("תאריך", value=date.today(), key="cw_replay_ev_date")
        with f2:
            hour_from, hour_to = st.select_slider(
                "טווח שעות", options=list(range(25)), value=(0, 24),
                key="cw_replay_ev_hours",
            )

        day_start = datetime.combine(day, dt_time(0, 0)).timestamp()
        since = day_start + hour_from * 3600
        until = day_start + hour_to * 3600
        events = rbuf.list_events(camera_id, since=since, until=until,
                                  kinds=kinds or None, limit=300)

        if not events:
            st.markdown('<div class="cw-empty-state">לא נמצאו אירועים תואמים.</div>',
                        unsafe_allow_html=True)
        else:
            st.caption(f"{len(events)} אירועים")
            for ev in reversed(events[-80:]):
                icon, color, label = _EVENT_STYLE.get(ev["kind"], ("•", "#8b9bb4", ev["kind"]))
                ts = datetime.fromtimestamp(ev["t"]).strftime("%H:%M:%S")
                detail = " · ".join(f"{k}={v}" for k, v in (ev.get("data") or {}).items())
                st.markdown(
                    f'<div class="cw-rec-row"><span>{icon} <b>{ts}</b> &nbsp; '
                    f'<span style="color:{color}">{label}</span></span>'
                    f'<span class="cw-rec-meta">{detail}</span></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="cw-replay-panel">', unsafe_allow_html=True)
        st.markdown("<h4>מידע הקלטה</h4>", unsafe_allow_html=True)
        segs = rbuf.list_segments(camera_id)
        bounds = rbuf.timeline_bounds(camera_id)
        if bounds:
            unplayable = [x for x in segs if not x["browser_playable"]]
            st.caption(
                ("🔴 מקליט כעת" if rbuf.has_live_footage(camera_id) else "⚪ אין הקלטה פעילה")
                + "  \n"
                f"מצלמה: {camera_label}  \n"
                f"תחילת היסטוריה: {datetime.fromtimestamp(bounds[0]):%d/%m/%Y %H:%M:%S}  \n"
                f"סוף היסטוריה: {datetime.fromtimestamp(bounds[1]):%d/%m/%Y %H:%M:%S}  \n"
                f"משך כולל: {(bounds[1] - bounds[0]) / 60:.1f} דקות  \n"
                f"קטעים: {len(segs)}  \n"
                f"תמונות מפתח: {len(rbuf.list_keyframes(camera_id))}  \n"
                f"נפח אחסון: {rbuf.storage_usage_bytes(camera_id) / 1048576:.1f} MB"
            )
            if unplayable:
                # Report the ACTUAL encoder state. This message used to claim
                # "new recordings are H.264" unconditionally, which was false on
                # any machine without an H.264 encoder - the unplayable count
                # kept growing while the UI insisted the problem was historical.
                enc = rbuf.encoder_status()
                if enc["h264"]:
                    st.warning(
                        f"{len(unplayable)} קטעים ישנים בקודק mp4v שדפדפנים אינם מנגנים. "
                        f"הנגן יציג עבורם תמונות מפתח.\n\n"
                        f"מקודד פעיל: {enc['label']} — הקלטות חדשות אכן H.264.\n\n"
                        "להמרת הישנים:  `python migrate_recordings.py --convert`"
                    )
                else:
                    st.error(
                        f"**אין מקודד H.264 במערכת — גם ההקלטות החדשות אינן ניתנות לניגון.**\n\n"
                        f"{len(unplayable)} קטעים במצב זה, והמספר ימשיך לגדול.\n\n"
                        f"מקודד פעיל: {enc['label']}\n\n"
                        "**תיקון (ללא הרשאות מנהל):**\n"
                        "```\npip install imageio-ffmpeg\n```\n"
                        "התקינו לאותו מפרש פייתון שמריץ את `main.py`, הפעילו אותו מחדש, "
                        "ואז המירו את הישנים:\n"
                        "```\npython migrate_recordings.py --convert\n```"
                    )
        else:
            st.caption(f"מצלמה: {camera_label}  \nאין עדיין הקלטות.")
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# CAMERA SETTINGS PAGE  (per-camera, independent feature ON/OFF controls)
# =========================================================================
def _inject_camera_settings_theme() -> None:
    st.markdown(
        """
        <style>
            .cw-cams-header {
                font-size: 1.6rem; font-weight: 700; letter-spacing: 0.04em;
                color: #e6edf3; margin-bottom: 0;
            }
            .cw-cams-subheader {
                color: #7d8590; font-size: 0.85rem; letter-spacing: 0.03em;
                margin-top: -4px; margin-bottom: 18px;
            }
            div[class*="st-key-cw_cam_card_"] {
                background: #0d1117;
                border: 1px solid #1c2128;
                border-radius: 16px;
                padding: 6px 20px 14px 20px;
                margin-bottom: 16px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            }
            .cw-cam-title {
                display: flex; align-items: center; gap: 10px;
                font-size: 1.1rem; font-weight: 700; color: #e6edf3;
                margin: 6px 0 2px 0;
            }
            .cw-cam-status-dot {
                width: 9px; height: 9px; border-radius: 50%; display: inline-block;
            }
            .cw-cam-meta { color: #7d8590; font-size: 0.78rem; margin-bottom: 6px; }
            .cw-cams-empty {
                color: #7d8590; font-size: 0.9rem; text-align: center;
                padding: 40px 16px; border: 1px dashed #1c2128; border-radius: 14px;
                line-height: 1.6;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
 
 
# =========================================================================
# CAMERA SETUP WIZARD + CALIBRATION EDITOR
#
# The wizard runs automatically the first time the engine reports a camera
# with no saved identity: Store -> Name -> Type -> Calibration -> Save. Once
# saved it never asks again. Already-known cameras get a "Calibration"
# editor instead, so calibration can always be redrawn later without ever
# touching a config file. Both share the same point-editor / live-preview
# building blocks below.
# =========================================================================
_WIZARD_STATE_KEYS = [
    "cw_wizard_camera_id", "cw_wizard_step", "cw_wizard_mode",
    "cw_wizard_store_id", "cw_wizard_name", "cw_wizard_type",
    "cw_wizard_description", "cw_wizard_location",
    "cw_wizard_tam", "cw_wizard_sam", "cw_wizard_som",
    "cw_wizard_active_zone",
    # --- Professional polygon editor state (undo/redo history, edit mode,
    # the point currently selected for a "move", and the baseline snapshot
    # captured on entering the calibration step, used by "Reset") ---
    "cw_wizard_undo", "cw_wizard_redo", "cw_wizard_edit_mode",
    "cw_wizard_move_selected", "cw_wizard_baseline",
]


def _wizard_reset() -> None:
    for key in _WIZARD_STATE_KEYS:
        st.session_state.pop(key, None)


def _wizard_start(camera_id: str, default_store_id: str, mode: str = "new_camera") -> None:
    """Load an existing identity (calibration edits) or start fresh (new camera)."""
    existing = get_camera_identity(camera_id)
    st.session_state["cw_wizard_camera_id"] = camera_id
    st.session_state["cw_wizard_mode"] = mode
    st.session_state["cw_wizard_store_id"] = existing.store_id if existing else default_store_id
    st.session_state["cw_wizard_name"] = existing.name if existing else ""
    st.session_state["cw_wizard_type"] = existing.camera_type if existing else CAMERA_TYPES[0]
    # New metadata fields
    st.session_state["cw_wizard_description"] = existing.description if existing else ""
    st.session_state["cw_wizard_location"] = existing.location if existing else ""
    st.session_state["cw_wizard_tam"] = list(existing.calibration.tam_area) if existing else []
    st.session_state["cw_wizard_sam"] = list(existing.calibration.sam_area) if existing else []
    st.session_state["cw_wizard_som"] = list(existing.calibration.som_line) if existing else []
    st.session_state["cw_wizard_step"] = 4 if mode == "calibration_only" else 1
    # Dedup guard: tracks the last processed click per zone so that Streamlit
    # reruns don't re-process the same widget value and add duplicate points.
    st.session_state["cw_last_click_tam"] = None
    st.session_state["cw_last_click_sam"] = None
    st.session_state["cw_last_click_som"] = None
    # Polygon editor: fresh undo/redo history and edit mode for this session.
    st.session_state["cw_wizard_undo"] = []
    st.session_state["cw_wizard_redo"] = []
    st.session_state["cw_wizard_edit_mode"] = "add"
    st.session_state["cw_wizard_move_selected"] = None
    st.session_state["cw_wizard_baseline"] = None


def _calibration_preview_html(
    frame_b64: Optional[str],
    tam: list[list[float]],
    sam: list[list[float]],
    som: list[list[float]],
    width: int = 620,
    height: int = 349,
) -> str:
    """One self-contained image+SVG block: the live frame with the TAM area
    (amber), SAM area (blue), and SOM entrance line (green) drawn on top in
    real time as their points change -- the "live camera preview" the
    calibration is drawn on."""

    def to_px(pts: list[list[float]]) -> str:
        return " ".join(f"{(p[0] / 100) * width:.1f},{(p[1] / 100) * height:.1f}" for p in pts)

    def markers(pts: list[list[float]], color: str) -> str:
        return "".join(
            f'<circle cx="{(p[0] / 100) * width:.1f}" cy="{(p[1] / 100) * height:.1f}" '
            f'r="5" fill="{color}" stroke="#05070a" stroke-width="1.5" />'
            for p in pts
        )

    if frame_b64:
        media = (
            f'<img src="data:image/jpeg;base64,{frame_b64}" width="{width}" height="{height}" '
            f'style="display:block;border-radius:12px;object-fit:cover;" />'
        )
    else:
        media = (
            f'<div style="width:{width}px;height:{height}px;background:#0d1117;border-radius:12px;'
            f'display:flex;align-items:center;justify-content:center;color:#7d8590;'
            f'font-size:0.85rem;border:1px dashed #1c2128;">ממתין לשידור מהמצלמה…</div>'
        )

    svg_parts = []
    if len(tam) >= 2:
        closed = len(tam) >= 3
        tag = "polygon" if closed else "polyline"
        svg_parts.append(
            f'<{tag} points="{to_px(tam)}" fill="rgba(227,179,65,0.16)" '
            f'stroke="#e3b341" stroke-width="2.5" />'
        )
    svg_parts.append(markers(tam, "#e3b341"))
    if len(sam) >= 2:
        closed = len(sam) >= 3
        tag = "polygon" if closed else "polyline"
        svg_parts.append(
            f'<{tag} points="{to_px(sam)}" fill="rgba(59,130,246,0.16)" '
            f'stroke="#3b82f6" stroke-width="2.5" />'
        )
    svg_parts.append(markers(sam, "#3b82f6"))
    if len(som) >= 3:
        # Closed entrance polygon (any vertex count).
        svg_parts.append(
            f'<polygon points="{to_px(som)}" fill="rgba(61,220,151,0.16)" '
            f'stroke="#3ddc97" stroke-width="3" />'
        )
    elif len(som) == 2:
        # Open entrance line.
        svg_parts.append(
            f'<polyline points="{to_px(som)}" stroke="#3ddc97" stroke-width="4" '
            f'stroke-linecap="round" />'
        )
    svg_parts.append(markers(som, "#3ddc97"))

    svg = (
        f'<svg width="{width}" height="{height}" '
        f'style="position:absolute;top:0;left:0;pointer-events:none;">{"".join(svg_parts)}</svg>'
    )
    return f'<div style="position:relative;width:{width}px;">{media}{svg}</div>'

 
def _draw_calibration_on_image(
    frame_b64: Optional[str],
    tam: list[list[float]],
    sam: list[list[float]],
    som: list[list[float]],
    width: int = 620,
    height: int = 349,
    highlight: Optional[tuple[str, int]] = None,
) -> Image.Image:
    """Build a PIL image with TAM/SAM/SOM zones drawn on top of the live frame.
 
    Used by the click-to-calibrate flow so that the user both sees the current
    calibration state AND can click directly on the composite image to add points
    — eliminating the need to type coordinates manually.
    """
    if frame_b64:
        try:
            img_bytes = base64.b64decode(frame_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGBA").resize((width, height))
        except Exception:
            img = Image.new("RGBA", (width, height), (13, 17, 23, 255))
    else:
        img = Image.new("RGBA", (width, height), (13, 17, 23, 255))
 
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
 
    def to_px(pts: list[list[float]]) -> list[tuple[int, int]]:
        return [(int(p[0] / 100 * width), int(p[1] / 100 * height)) for p in pts]
 
    # TAM — amber
    if len(tam) >= 3:
        draw.polygon(to_px(tam), fill=(227, 179, 65, 45), outline=(227, 179, 65, 220))
    elif len(tam) == 2:
        draw.line(to_px(tam), fill=(227, 179, 65, 220), width=2)
    for px, py in to_px(tam):
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(227, 179, 65, 255))
 
    # SAM — blue
    if len(sam) >= 3:
        draw.polygon(to_px(sam), fill=(59, 130, 246, 45), outline=(59, 130, 246, 220))
    elif len(sam) == 2:
        draw.line(to_px(sam), fill=(59, 130, 246, 220), width=2)
    for px, py in to_px(sam):
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(59, 130, 246, 255))
 
    # SOM — green. Any vertex count: 3+ = closed polygon, 2 = open line.
    if len(som) >= 3:
        draw.polygon(to_px(som), fill=(61, 220, 151, 40), outline=(61, 220, 151, 255))
        draw.line(to_px(som) + [to_px(som)[0]], fill=(61, 220, 151, 255), width=3)
    elif len(som) == 2:
        draw.line(to_px(som), fill=(61, 220, 151, 255), width=4)
    for px, py in to_px(som):
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(61, 220, 151, 255))

    # Selection ring — the point currently picked up for a "move"
    if highlight is not None:
        zone_pts = {"tam": tam, "sam": sam, "som": som}.get(highlight[0])
        if zone_pts and 0 <= highlight[1] < len(zone_pts):
            hx, hy = to_px(zone_pts)[highlight[1]]
            draw.ellipse([hx - 10, hy - 10, hx + 10, hy + 10], outline=(255, 255, 255, 255), width=3)

    composite = Image.alpha_composite(img, overlay).convert("RGB")
    return composite


# =========================================================================
# POLYGON EDITOR — geometry helpers + undo/redo history
#
# These give the click-based calibration canvas a "professional map editor"
# feel without needing a new custom JS component: an edit-mode selector
# (Add / Move / Insert / Delete) reinterprets what a click on the live
# preview does, and every mutation is snapshotted so Undo/Redo/Reset can
# restore a previous shape exactly.
# =========================================================================
def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    return (ax - bx) ** 2 + (ay - by) ** 2


def _point_segment_dist2(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Squared distance from point (px,py) to the segment a→b."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return _dist2(px, py, ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return _dist2(px, py, ax + t * dx, ay + t * dy)


def _nearest_point_index(pts: list[list[float]], x: float, y: float, threshold_pct: float = 4.0) -> Optional[int]:
    """Index of the closest existing point to (x, y), if within threshold_pct
    (percent-of-frame units), else None."""
    if not pts:
        return None
    best_i, best_d = None, None
    for i, p in enumerate(pts):
        d = _dist2(p[0], p[1], x, y)
        if best_d is None or d < best_d:
            best_d, best_i = d, i
    if best_i is not None and best_d is not None and best_d <= threshold_pct ** 2:
        return best_i
    return None


def _nearest_edge_insert_index(pts: list[list[float]], x: float, y: float) -> int:
    """Where to insert a new point so it lands on the polygon edge closest
    to (x, y). Falls back to appending at the end for 0-1 existing points."""
    n = len(pts)
    if n < 2:
        return n
    closed = n >= 3
    edges = n if closed else n - 1
    best_idx, best_d = n, None
    for i in range(edges):
        a = pts[i]
        b = pts[(i + 1) % n]
        d = _point_segment_dist2(x, y, a[0], a[1], b[0], b[1])
        if best_d is None or d < best_d:
            best_d, best_idx = d, i + 1
    return best_idx


def _wizard_snapshot() -> Dict[str, list]:
    return {
        "tam": [list(p) for p in st.session_state["cw_wizard_tam"]],
        "sam": [list(p) for p in st.session_state["cw_wizard_sam"]],
        "som": [list(p) for p in st.session_state["cw_wizard_som"]],
    }


def _wizard_apply_snapshot(snapshot: Dict[str, list]) -> None:
    st.session_state["cw_wizard_tam"] = [list(p) for p in snapshot["tam"]]
    st.session_state["cw_wizard_sam"] = [list(p) for p in snapshot["sam"]]
    st.session_state["cw_wizard_som"] = [list(p) for p in snapshot["som"]]


def _wizard_push_history() -> None:
    """Call BEFORE mutating tam/sam/som — snapshots the pre-mutation shape
    onto the undo stack and clears redo (a fresh edit invalidates redo)."""
    undo_stack = st.session_state.setdefault("cw_wizard_undo", [])
    undo_stack.append(_wizard_snapshot())
    if len(undo_stack) > 50:  # cap history depth so session state stays small
        undo_stack.pop(0)
    st.session_state["cw_wizard_redo"] = []


def _wizard_undo() -> bool:
    undo_stack = st.session_state.get("cw_wizard_undo", [])
    if not undo_stack:
        return False
    st.session_state.setdefault("cw_wizard_redo", []).append(_wizard_snapshot())
    _wizard_apply_snapshot(undo_stack.pop())
    st.session_state["cw_wizard_move_selected"] = None
    return True


def _wizard_redo() -> bool:
    redo_stack = st.session_state.get("cw_wizard_redo", [])
    if not redo_stack:
        return False
    st.session_state.setdefault("cw_wizard_undo", []).append(_wizard_snapshot())
    _wizard_apply_snapshot(redo_stack.pop())
    st.session_state["cw_wizard_move_selected"] = None
    return True


def _render_point_list(points_key: str, color: str, exact_points: Optional[int] = None) -> None:
    """Show the current list of calibration points with individual delete buttons."""
    points: list[list[float]] = st.session_state[points_key]
    if points:
        for i, pt in enumerate(list(points)):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.caption(f"נקודה {i + 1}: X={pt[0]:.1f}%, Y={pt[1]:.1f}%")
            with c2:
                if st.button("✕", key=f"cw_pt_del_{points_key}_{i}"):
                    _wizard_push_history()
                    st.session_state[points_key].pop(i)
                    st.session_state["cw_wizard_move_selected"] = None
                    st.rerun()
    else:
        st.caption("לחצו על התמונה להוספת נקודות.")


def render_camera_wizard(camera: "CameraDescriptor", telemetry: Dict[str, Any]) -> None:
    step = st.session_state["cw_wizard_step"]
    mode = st.session_state["cw_wizard_mode"]
    is_full_wizard = mode == "new_camera"

    with st.container(key=f"cw_wizard_card_{camera.camera_id}"):
        st.markdown(
            f'<div class="cw-cam-title">🧭 {"הגדרת מצלמה חדשה" if is_full_wizard else "עריכת כיול"} '
            f'· {camera.camera_id}</div>',
            unsafe_allow_html=True,
        )

        if is_full_wizard:
            steps_he = ["חנות", "שם מצלמה", "סוג מצלמה", "כיול", "שמירה"]
            st.caption(" → ".join(
                f"**{s}**" if i + 1 == step else s for i, s in enumerate(steps_he)
            ))

        if step == 1:
            store_ids = [s.store_id for s in STORES]
            current = st.session_state["cw_wizard_store_id"]
            index = store_ids.index(current) if current in store_ids else 0
            choice = st.selectbox(
                "לאיזו חנות שייכת המצלמה?",
                store_ids,
                index=index,
                format_func=lambda sid: get_store(sid).display_name if get_store(sid) else sid,
                key=f"cw_wizard_store_select_{camera.camera_id}",
            )
            st.session_state["cw_wizard_store_id"] = choice
            if st.button("הבא →", key=f"cw_wizard_next_{camera.camera_id}_1"):
                st.session_state["cw_wizard_step"] = 2
                st.rerun()

        elif step == 2:
            name = st.text_input(
                "שם המצלמה",
                value=st.session_state["cw_wizard_name"],
                placeholder="לדוגמה: כניסה ראשית",
                key=f"cw_wizard_name_input_{camera.camera_id}",
            )
            st.session_state["cw_wizard_name"] = name

            # New metadata fields: description and location (optional)
            description = st.text_area(
                "תיאור (אופציונלי)",
                value=st.session_state.get("cw_wizard_description", ""),
                placeholder="לדוגמה: מצלמה בקומת קרקע, משמאל לכניסה",
                key=f"cw_wizard_description_{camera.camera_id}",
                height=80,
            )
            st.session_state["cw_wizard_description"] = description

            location = st.text_input(
                "מיקום פיזי (אופציונלי)",
                value=st.session_state.get("cw_wizard_location", ""),
                placeholder="לדוגמה: כניסה צפונית",
                key=f"cw_wizard_location_{camera.camera_id}",
            )
            st.session_state["cw_wizard_location"] = location

            c1, c2 = st.columns(2)
            with c1:
                if st.button("← חזור", key=f"cw_wizard_back_{camera.camera_id}_2"):
                    st.session_state["cw_wizard_step"] = 1
                    st.rerun()
            with c2:
                if st.button("הבא →", key=f"cw_wizard_next_{camera.camera_id}_2", disabled=not name.strip()):
                    st.session_state["cw_wizard_step"] = 3
                    st.rerun()

        elif step == 3:
            current_type = st.session_state["cw_wizard_type"]
            choice = st.radio(
                "סוג המצלמה",
                CAMERA_TYPES,
                index=CAMERA_TYPES.index(current_type) if current_type in CAMERA_TYPES else 0,
                format_func=lambda t: CAMERA_TYPE_LABELS_HE.get(t, t),
                key=f"cw_wizard_type_radio_{camera.camera_id}",
            )
            st.session_state["cw_wizard_type"] = choice
            c1, c2 = st.columns(2)
            with c1:
                if st.button("← חזור", key=f"cw_wizard_back_{camera.camera_id}_3"):
                    st.session_state["cw_wizard_step"] = 2
                    st.rerun()
            with c2:
                if st.button("הבא →", key=f"cw_wizard_next_{camera.camera_id}_3"):
                    st.session_state["cw_wizard_step"] = 4
                    st.rerun()

        elif step == 4:
            # ── Click-to-calibrate (mode-aware) ─────────────────────────────
            analytics_mode = st.session_state.get("analytics_mode", "tam_sam_som")
            use_sam_zone = (analytics_mode == "tam_sam_som")

            st.session_state.setdefault("cw_wizard_active_zone", "tam")
            active_zone = st.session_state["cw_wizard_active_zone"]

            # In TAM+SOM mode the SAM zone is derived from dwell time;
            # reset active_zone to "tam" if it was "sam" and mode changed.
            if not use_sam_zone and active_zone == "sam":
                st.session_state["cw_wizard_active_zone"] = "tam"
                active_zone = "tam"

            # Baseline snapshot for "Reset" — captured once, the first time
            # this step is shown, so Reset always restores what calibration
            # looked like when editing began (not an empty shape).
            if st.session_state.get("cw_wizard_baseline") is None:
                st.session_state["cw_wizard_baseline"] = _wizard_snapshot()

            zone_defs = [
                ("tam", "🟡 TAM", "#e3b341", "כיול TAM — לחצו להוסיף נקודות לאיזור עוברי האורח"),
                ("sam", "🔵 SAM", "#3b82f6", "כיול SAM — לחצו להוסיף נקודות לאיזור המתעניינים"),
                ("som", "🟢 SOM", "#3ddc97", "כיול SOM — 2 נקודות = קו כניסה, 3 ומעלה = מצולע כניסה (ללא הגבלת נקודות)"),
            ]
            visible_zones = [z for z in zone_defs if z[0] != "sam" or use_sam_zone]

            zone_cols = st.columns(len(visible_zones))
            for col, (zone_id, zone_btn, zone_color, zone_desc) in zip(zone_cols, visible_zones):
                with col:
                    btn_type = "primary" if active_zone == zone_id else "secondary"
                    if st.button(zone_btn, key=f"cw_zone_sel_{zone_id}_{camera.camera_id}",
                                 type=btn_type, use_container_width=True):
                        st.session_state["cw_wizard_active_zone"] = zone_id
                        st.rerun()

            if not use_sam_zone:
                _wiz_sam_settings = get_sam_min_stay_settings(st.session_state["cw_wizard_store_id"])
                if _wiz_sam_settings["enabled"]:
                    st.caption(
                        "מצב TAM+SOM: איזור ה-SAM יחושב אוטומטית ממי ששהה ב-TAM מעל "
                        f"{_wiz_sam_settings['seconds']} שניות (ניתן לשינוי בהגדרות)."
                    )
                else:
                    st.caption("מצב TAM+SOM: זמן השהייה המינימלי כבוי — כל מבקר ב-TAM נחשב SAM מיד.")

            # Show zone-specific instruction and current point counts
            _, active_label, active_color, active_desc = next(
                d for d in zone_defs if d[0] == active_zone
            )
            tam_ok = len(st.session_state["cw_wizard_tam"]) >= 3
            sam_ok = len(st.session_state["cw_wizard_sam"]) >= 3 if use_sam_zone else True
            som_ok = len(st.session_state["cw_wizard_som"]) >= 2

            # SOM can now be an entrance polyline with 2 or more points.
            if use_sam_zone:
                counts_txt = (
                    f"TAM: {len(st.session_state['cw_wizard_tam'])} {'✅' if tam_ok else 'נקודות (דרושות 3+)'} · "
                    f"SAM: {len(st.session_state['cw_wizard_sam'])} {'✅' if sam_ok else 'נקודות (דרושות 3+)'} · "
                    f"SOM: {len(st.session_state['cw_wizard_som'])} {'✅' if som_ok else 'נקודות (דרושות 2+)'}"
                )
            else:
                counts_txt = (
                    f"TAM: {len(st.session_state['cw_wizard_tam'])} {'✅' if tam_ok else 'נקודות (דרושות 3+)'} · "
                    f"SOM: {len(st.session_state['cw_wizard_som'])} {'✅' if som_ok else 'נקודות (דרושות 2+)'}"
                )

            st.markdown(
                f'<div style="color:{active_color};font-weight:700;font-size:0.85rem;'
                f'margin:8px 0 2px 0;">● {active_desc}</div>',
                unsafe_allow_html=True,
            )
            st.caption(counts_txt)

            # ── Professional polygon editor toolbar ─────────────────────────
            # No point-count limit any more: TAM/SAM support unlimited points.
            # The edit mode changes what a click on the preview does; Undo /
            # Redo / Reset / Delete-polygon act as a real history stack so an
            # edit is never destructive until Save is pressed.
            EDIT_MODES = [
                ("add", "➕ הוספה"),
                ("move", "✥ הזזה"),
                ("insert", "⤵ הכנסה על הקו"),
                ("delete", "🗑 מחיקה בלחיצה"),
            ]
            mode_ids = [m[0] for m in EDIT_MODES]
            current_mode = st.session_state.get("cw_wizard_edit_mode", "add")
            if current_mode not in mode_ids:
                current_mode = "add"
            edit_mode = st.radio(
                "מצב עריכה",
                mode_ids,
                index=mode_ids.index(current_mode),
                format_func=lambda m: dict(EDIT_MODES)[m],
                horizontal=True,
                key=f"cw_edit_mode_radio_{camera.camera_id}",
                label_visibility="collapsed",
            )
            if edit_mode != current_mode:
                st.session_state["cw_wizard_move_selected"] = None
            st.session_state["cw_wizard_edit_mode"] = edit_mode

            mode_hints = {
                "add": "לחצו על התמונה כדי להוסיף נקודה חדשה בסוף הצורה.",
                "move": "לחצו על נקודה קיימת לבחירה, ואז לחצו על המיקום החדש כדי להזיז אותה.",
                "insert": "לחצו ליד אחת הצלעות כדי להכניס נקודה חדשה בדיוק שם.",
                "delete": "לחצו ליד נקודה קיימת כדי למחוק אותה.",
            }
            st.caption(f"💡 {mode_hints[edit_mode]}")

            tb1, tb2, tb3, tb4 = st.columns(4)
            with tb1:
                if st.button("↶ בטל", key=f"cw_undo_{camera.camera_id}", use_container_width=True):
                    if _wizard_undo():
                        st.rerun()
                    else:
                        st.toast("אין מה לבטל.")
            with tb2:
                if st.button("↷ בצע שוב", key=f"cw_redo_{camera.camera_id}", use_container_width=True):
                    if _wizard_redo():
                        st.rerun()
                    else:
                        st.toast("אין מה לבצע שוב.")
            with tb3:
                if st.button("⟲ איפוס איזור", key=f"cw_reset_zone_{camera.camera_id}", use_container_width=True):
                    zone_key_map = {"tam": "cw_wizard_tam", "sam": "cw_wizard_sam", "som": "cw_wizard_som"}
                    baseline = st.session_state.get("cw_wizard_baseline") or _wizard_snapshot()
                    _wizard_push_history()
                    st.session_state[zone_key_map[active_zone]] = [list(p) for p in baseline[active_zone]]
                    st.session_state["cw_wizard_move_selected"] = None
                    st.rerun()
            with tb4:
                if st.button("✕ מחיקת איזור", key=f"cw_delete_zone_{camera.camera_id}", use_container_width=True):
                    zone_key_map = {"tam": "cw_wizard_tam", "sam": "cw_wizard_sam", "som": "cw_wizard_som"}
                    _wizard_push_history()
                    st.session_state[zone_key_map[active_zone]] = []
                    st.session_state["cw_wizard_move_selected"] = None
                    st.rerun()

            # Build the composite image (live frame + calibration overlay)
            frame_b64 = telemetry.get("live_frame_jpeg_b64")
            sam_for_preview = st.session_state["cw_wizard_sam"] if use_sam_zone else []
            move_selected = st.session_state.get("cw_wizard_move_selected")
            highlight = move_selected if (move_selected and move_selected[0] == active_zone) else None
            calib_img = _draw_calibration_on_image(
                frame_b64,
                st.session_state["cw_wizard_tam"],
                sam_for_preview,
                st.session_state["cw_wizard_som"],
                highlight=highlight,
            )

            # Render the clickable image — a click returns {x, y} in pixels.
            # ROOT-CAUSE FIX: streamlit_image_coordinates keeps the last-clicked
            # value in widget state across reruns.  Without a dedup guard every
            # st.rerun() (triggered after adding a point) sees the same value and
            # adds another point endlessly.  We track the last processed pixel
            # coordinate in session state and ignore any repeat of that value.
            _click_last_key = f"cw_last_click_{active_zone}"
            if _click_last_key not in st.session_state:
                st.session_state[_click_last_key] = None

            click_val = streamlit_image_coordinates(
                calib_img,
                key=f"cw_calib_click_{camera.camera_id}_{active_zone}",
            )
            if click_val is not None:
                # Represent click as a tuple so it can be compared reliably
                _click_tuple = (click_val["x"], click_val["y"])
                _already_processed = (st.session_state[_click_last_key] == _click_tuple)

                if not _already_processed:
                    # Mark as processed BEFORE st.rerun() so the next rerun skips it
                    st.session_state[_click_last_key] = _click_tuple

                    x_pct = round((click_val["x"] / calib_img.width) * 100, 1)
                    y_pct = round((click_val["y"] / calib_img.height) * 100, 1)
                    zone_key_map = {
                        "tam": "cw_wizard_tam",
                        "sam": "cw_wizard_sam",
                        "som": "cw_wizard_som",
                    }
                    pts_key = zone_key_map[active_zone]
                    pts = st.session_state[pts_key]

                    if edit_mode == "add":
                        # Allow multi-point SOM polylines; no special warning.
                        _wizard_push_history()
                        pts.append([x_pct, y_pct])
                        st.rerun()

                    elif edit_mode == "move":
                        if highlight is None:
                            idx = _nearest_point_index(pts, x_pct, y_pct)
                            if idx is None:
                                st.warning("לא נמצאה נקודה קרובה מספיק ללחיצה. נסו ללחוץ קרוב יותר לנקודה.")
                            else:
                                st.session_state["cw_wizard_move_selected"] = (active_zone, idx)
                                st.rerun()
                        else:
                            _, sel_idx = highlight
                            if 0 <= sel_idx < len(pts):
                                _wizard_push_history()
                                pts[sel_idx] = [x_pct, y_pct]
                            st.session_state["cw_wizard_move_selected"] = None
                            st.rerun()

                    elif edit_mode == "insert":
                        # Allow multi-point SOM polylines now; no special warning needed.
                        _wizard_push_history()
                        insert_at = _nearest_edge_insert_index(pts, x_pct, y_pct)
                        pts.insert(insert_at, [x_pct, y_pct])
                        st.rerun()

                    elif edit_mode == "delete":
                        idx = _nearest_point_index(pts, x_pct, y_pct)
                        if idx is None:
                            st.warning("לא נמצאה נקודה קרובה מספיק ללחיצה למחיקה.")
                        else:
                            _wizard_push_history()
                            pts.pop(idx)
                            st.rerun()

            # Editable point list (delete individual points)
            with st.expander("📍 נקודות שהוגדרו — לחצו להסרה"):
                for zone_id, _, z_color, _ in visible_zones:
                    zone_key = {"tam": "cw_wizard_tam", "sam": "cw_wizard_sam", "som": "cw_wizard_som"}[zone_id]
                    pts = st.session_state[zone_key]
                    if pts:
                        st.markdown(
                            f'<div style="color:{z_color};font-weight:700;font-size:0.8rem;'
                            f'margin-top:8px;">{zone_id.upper()}</div>',
                            unsafe_allow_html=True,
                        )
                        _render_point_list(zone_key, z_color)

            calib_ok = tam_ok and sam_ok and som_ok
            if not calib_ok:
                if use_sam_zone:
                    st.info("יש להשלים את שלושת האיזורים: TAM ו-SAM (3+ נקודות כל אחד) ו-SOM (2 נקודות).")
                else:
                    st.info("יש להשלים כיול: TAM (3+ נקודות) ו-SOM (2 נקודות).")

            if is_full_wizard:
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("← חזור", key=f"cw_wizard_back_{camera.camera_id}_4", use_container_width=True):
                        st.session_state["cw_wizard_step"] = 3
                        st.rerun()
                with c2:
                    if st.button("✕ ביטול", key=f"cw_wizard_cancel_full_{camera.camera_id}_4",
                                 use_container_width=True):
                        _wizard_reset()  # abandon the whole setup; nothing is saved
                        st.rerun()
                with c3:
                    if st.button("הבא →", key=f"cw_wizard_next_{camera.camera_id}_4",
                                 disabled=not calib_ok, type="primary", use_container_width=True):
                        if not use_sam_zone:
                            st.session_state["cw_wizard_sam"] = list(st.session_state["cw_wizard_tam"])
                        st.session_state["cw_wizard_step"] = 5
                        st.rerun()
            else:
                # calibration_only mode: Cancel (discard) · Save (commit)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✕ ביטול (ללא שמירה)", key=f"cw_wizard_cancel_{camera.camera_id}_4",
                                 use_container_width=True):
                        _wizard_reset()   # discard all in-progress points; original calibration is untouched
                        st.rerun()
                with c2:
                    if st.button("💾 שמירת כיול", key=f"cw_wizard_next_{camera.camera_id}_4",
                                 disabled=not calib_ok, type="primary", use_container_width=True):
                        # In TAM+SOM mode, auto-fill SAM from TAM before saving.
                        if not use_sam_zone:
                            st.session_state["cw_wizard_sam"] = list(st.session_state["cw_wizard_tam"])
                        identity = CameraIdentity(
                            camera_id=camera.camera_id,
                            store_id=st.session_state["cw_wizard_store_id"],
                            name=st.session_state["cw_wizard_name"],
                            camera_type=st.session_state["cw_wizard_type"],
                            calibration=CameraCalibration(
                                tam_area=st.session_state["cw_wizard_tam"],
                                sam_area=st.session_state["cw_wizard_sam"],
                                som_line=st.session_state["cw_wizard_som"],
                            ),
                        )
                        save_camera_identity(identity)
                        _wizard_reset()
                        st.success("הכיול נשמר. הגדרות חדשות יישלחו לבינה המלאכותית בסיבוב הבא.")
                        st.rerun()

        elif step == 5:
            store = get_store(st.session_state["cw_wizard_store_id"])
            st.markdown(
                f"**סיכום**\n\n"
                f"- חנות: {store.display_name if store else '—'}\n"
                f"- שם מצלמה: {st.session_state['cw_wizard_name']}\n"
                f"- סוג: {CAMERA_TYPE_LABELS_HE.get(st.session_state['cw_wizard_type'], '')}\n"
                f"- כיול: הושלם ✅"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("← חזור לכיול", key=f"cw_wizard_back_{camera.camera_id}_5"):
                    st.session_state["cw_wizard_step"] = 4
                    st.rerun()
            with c2:
                if st.button("✅ סיום והפעלה", key=f"cw_wizard_save_{camera.camera_id}"):
                    identity = CameraIdentity(
                        camera_id=camera.camera_id,
                        store_id=st.session_state["cw_wizard_store_id"],
                        name=st.session_state["cw_wizard_name"],
                        camera_type=st.session_state["cw_wizard_type"],
                        calibration=CameraCalibration(
                            tam_area=st.session_state["cw_wizard_tam"],
                            sam_area=st.session_state["cw_wizard_sam"],
                            som_line=st.session_state["cw_wizard_som"],
                        ),
                    )
                    save_camera_identity(identity)
                    _wizard_reset()
                    st.success(f'המצלמה "{identity.name}" מוכנה לשימוש.')
                    st.rerun()


def render_camera_settings_page(
    camera_manager: CameraManager, store_id: str, store_display_name: str, telemetry: Dict[str, Any]
) -> None:
    """Per-camera identity, calibration, and feature controls.

    Features are global to the system (``CAMERA_FEATURES``), but every camera
    has its own independent ON/OFF for each one, its own permanent identity
    (store/name/type), and its own TAM/SAM/SOM calibration -- all persisted
    per camera so nothing is ever shared between cameras.
    """
    _inject_camera_settings_theme()

    st.markdown('<div class="cw-cams-header">הגדרות מצלמות</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="cw-cams-subheader">שליטה עצמאית בכל מצלמה · {store_display_name}</div>',
        unsafe_allow_html=True,
    )

    all_cameras = camera_manager.list_cameras()

    # A wizard already in progress takes priority over everything else on
    # this page, whichever camera it's for.
    active_wizard_camera_id = st.session_state.get("cw_wizard_camera_id")
    if active_wizard_camera_id:
        active_cam = next((c for c in all_cameras if c.camera_id == active_wizard_camera_id), None)
        if active_cam is not None:
            render_camera_wizard(active_cam, telemetry)
            return
        _wizard_reset()  # the camera that was being set up is no longer reported

    # A brand-new, never-configured camera auto-launches the Setup Wizard --
    # this is what replaces "plug in a camera and it just starts working".
    unassigned = find_unassigned_camera(all_cameras)
    if unassigned is not None:
        st.markdown(
            '<div class="cw-cams-empty" style="border-color:#e3b341;color:#e3b341;">'
            f'📷 זוהתה מצלמה חדשה ({unassigned.camera_id}) — יש להגדיר אותה לפני שתתחיל לפעול.</div>',
            unsafe_allow_html=True,
        )
        if st.button("🧭 התחל הגדרת מצלמה", key=f"cw_start_wizard_{unassigned.camera_id}"):
            _wizard_start(unassigned.camera_id, default_store_id=store_id, mode="new_camera")
            st.rerun()

    cameras = filter_cameras_by_store(all_cameras, store_id)
    if not cameras:
        if unassigned is None:
            st.markdown(
                '<div class="cw-cams-empty">אין כרגע מצלמות זמינות עבור חנות זו.<br>'
                'המצלמות אינן מחוברות או שאינן פעילות כעת — ההגדרות יופיעו אוטומטית '
                'עבור כל מצלמה ברגע שתתחבר.</div>',
                unsafe_allow_html=True,
            )
        return

    status_colors = {"online": "#3ddc97", "paused": "#e3b341", "offline": "#f85149"}

    for camera in cameras:
        identity = get_camera_identity(camera.camera_id)
        display_name = identity.name if identity and identity.name else camera.label
        with st.container(key=f"cw_cam_card_{camera.camera_id}"):
            dot = status_colors.get(camera.status, "#7d8590")
            type_label = CAMERA_TYPE_LABELS_HE.get(identity.camera_type, "") if identity else ""
            st.markdown(
                f'<div class="cw-cam-title">'
                f'<span class="cw-cam-status-dot" style="background:{dot}"></span>'
                f'📷 {display_name}</div>'
                f'<div class="cw-cam-meta">מזהה: {camera.camera_id} · '
                f'סוג: {type_label or "—"} · סטטוס: {camera.status}</div>',
                unsafe_allow_html=True,
            )

            settings = get_camera_feature_settings(store_id, camera.camera_id)
            cols = st.columns(len(CAMERA_FEATURES))
            for col, feature in zip(cols, CAMERA_FEATURES):
                with col:
                    new_value = st.toggle(
                        f"{feature.icon} {feature.label_he}",
                        value=settings[feature.key],
                        key=f"cw_camtoggle_{camera.camera_id}_{feature.key}",
                        help=feature.help_he or None,
                    )
                    if new_value != settings[feature.key]:
                        set_camera_feature(store_id, camera.camera_id, feature.key, new_value)

            with st.expander("🎛 כיול (TAM / SAM / SOM)"):
                calib_complete = identity and identity.calibration.is_complete()
                if calib_complete:
                    st.markdown(
                        _calibration_preview_html(
                            telemetry.get("live_frame_jpeg_b64"),
                            identity.calibration.tam_area,
                            identity.calibration.sam_area,
                            identity.calibration.som_line,
                            width=420, height=236,
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("לא הוגדר כיול עדיין עבור מצלמה זו.")

                ec1, ec2, ec3, ec4 = st.columns(4)
                with ec1:
                    if st.button("✏️ עריכת כיול", key=f"cw_edit_calib_{camera.camera_id}",
                                 use_container_width=True):
                        _wizard_start(camera.camera_id, default_store_id=store_id, mode="calibration_only")
                        st.rerun()
                with ec2:
                    if st.button("🗑 מחיקת כיול",
                                 key=f"cw_del_calib_{camera.camera_id}",
                                 disabled=not calib_complete,
                                 use_container_width=True):
                        st.session_state[f"cw_confirm_delete_{camera.camera_id}"] = True
                        st.rerun()
                with ec3:
                    wizard_active = st.session_state.get("cw_wizard_camera_id") == camera.camera_id
                    if st.button("✕ ביטול",
                                 key=f"cw_cancel_calib_{camera.camera_id}",
                                 disabled=not wizard_active,
                                 use_container_width=True):
                        _wizard_reset()
                        st.rerun()
                with ec4:
                    # Disconnect camera instead of deleting to preserve metadata and allow reconnection.
                    if identity and identity.disconnected:
                        if st.button("🔌 חיבור מחדש",
                                     key=f"cw_reconnect_camera_{camera.camera_id}",
                                     use_container_width=True):
                            reconnect_camera(camera.camera_id)
                            st.success("המצלמה מחוברת מחדש.")
                            st.rerun()
                    else:
                        if st.button("🔌 נתק מצלמה",
                                     key=f"cw_disconnect_camera_{camera.camera_id}",
                                     use_container_width=True):
                            st.session_state[f"cw_confirm_disconnect_camera_{camera.camera_id}"] = True
                            st.rerun()

                if st.session_state.pop(f"cw_confirm_disconnect_camera_{camera.camera_id}", False):
                    st.warning(
                        "האם אתם בטוחים שברצונכם לנתק את המצלמה הזו? פעולה זו תעצור את הפעילות החיה אך תשמור על ההגדרות.")
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        if st.button("✅ כן, נתק מצלמה",
                                     key=f"cw_disconnect_camera_confirm_yes_{camera.camera_id}",
                                     type="primary", use_container_width=True):
                            if st.session_state.get("cw_wizard_camera_id") == camera.camera_id:
                                _wizard_reset()
                            disconnect_camera(camera.camera_id)
                            st.success("המצלמה נותקה אך שמרנו את ההגדרות.")
                            st.rerun()
                    with dc2:
                        if st.button("← חזור",
                                     key=f"cw_disconnect_camera_confirm_no_{camera.camera_id}",
                                     use_container_width=True):
                            st.rerun()

                # Preserve a separate hard-delete option for admins; keep it but hide behind an extra confirmation
                if st.session_state.pop(f"cw_confirm_delete_camera_{camera.camera_id}", False):
                    st.warning(
                        "האם אתם בטוחים שברצונכם למחוק את המצלמה הזו לצמיתות? פעולה זו תמחק את כל ההגדרות ולא ניתנת לביטול.")
                    cam_cd1, cam_cd2 = st.columns(2)
                    with cam_cd1:
                        if st.button("✅ כן, מחק מצלמה לצמיתות",
                                     key=f"cw_del_camera_confirm_yes_{camera.camera_id}",
                                     type="primary", use_container_width=True):
                            if st.session_state.get("cw_wizard_camera_id") == camera.camera_id:
                                _wizard_reset()
                            delete_camera(store_id, camera.camera_id)
                            st.success("המצלמה נמחקה לצמיתות.")
                            st.rerun()
                    with cam_cd2:
                        if st.button("← חזור",
                                     key=f"cw_del_camera_confirm_no_{camera.camera_id}",
                                     use_container_width=True):
                            st.rerun()

                if st.session_state.pop(f"cw_confirm_delete_{camera.camera_id}", False):
                    st.warning("האם אתם בטוחים? פעולה זו תמחק את הכיול ולא ניתן לבטל אותה.")
                    cd1, cd2 = st.columns(2)
                    with cd1:
                        if st.button("✅ כן, מחק",
                                     key=f"cw_del_confirm_yes_{camera.camera_id}",
                                     type="primary", use_container_width=True):
                            if st.session_state.get("cw_wizard_camera_id") == camera.camera_id:
                                _wizard_reset()
                            delete_camera_calibration(camera.camera_id)
                            st.success("הכיול נמחק.")
                            st.rerun()
                    with cd2:
                        if st.button("← חזור",
                                     key=f"cw_del_confirm_no_{camera.camera_id}",
                                     use_container_width=True):
                            st.rerun()

                if st.session_state.pop(f"cw_confirm_delete_camera_{camera.camera_id}", False):
                    st.warning(
                        "האם אתם בטוחים שברצונכם למחוק את המצלמה הזו? פעולה זו תמחק "
                        "את שיוך המצלמה, את הכיול המלא שלה ואת כל ההגדרות הספציפיות "
                        "לה (טשטוש פנים, ראיית לילה, הקלטה). מצלמות אחרות לא יושפעו. "
                        "לא ניתן לבטל פעולה זו."
                    )
                    cam_cd1, cam_cd2 = st.columns(2)
                    with cam_cd1:
                        if st.button("✅ כן, מחק מצלמה",
                                     key=f"cw_del_camera_confirm_yes_{camera.camera_id}",
                                     type="primary", use_container_width=True):
                            if st.session_state.get("cw_wizard_camera_id") == camera.camera_id:
                                _wizard_reset()
                            delete_camera(store_id, camera.camera_id)
                            st.success("המצלמה נמחקה.")
                            st.rerun()
                    with cam_cd2:
                        if st.button("← חזור",
                                     key=f"cw_del_camera_confirm_no_{camera.camera_id}",
                                     use_container_width=True):
                            st.rerun()

# =========================================================================
# STORE SELECTION SCREEN  (Netflix-style picker + password login dialog)
# =========================================================================
_AUTH = AuthenticationManager()
 
 
def _placeholder_logo_data_uri(display_name: str) -> str:
    """Used only if a logo PNG is missing, so a bad file path can never crash the app."""
    initials = "".join(w[0] for w in display_name.split()[:2]).upper()
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240">
        <rect width="240" height="240" rx="24" fill="#161b22"/>
        <text x="50%" y="52%" text-anchor="middle" dominant-baseline="middle"
              font-family="Arial, sans-serif" font-size="64" fill="#4a5260">{initials}</text>
    </svg>"""
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"
 
 
@st.cache_data(show_spinner=False)
def _logo_data_uri(path_str: str, display_name: str) -> str:
    path = Path(path_str)
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        return _placeholder_logo_data_uri(display_name)
 
 
def _inject_selector_theme() -> None:
    st.markdown(
        """
        <style>
            header { visibility: hidden; }
            footer { visibility: hidden; }
            [data-testid="stSidebar"] { display: none; }
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(1200px 600px at 15% -10%, rgba(61,220,151,0.08), transparent 60%),
                    radial-gradient(1200px 700px at 110% 10%, rgba(59,130,246,0.10), transparent 55%),
                    linear-gradient(180deg, #05070a 0%, #060a10 55%, #05070a 100%);
            }
            .cw-select-logo {
                text-align: center;
                font-size: 2.6rem;
                font-weight: 800;
                letter-spacing: 0.32em;
                color: #eef2f6;
                margin-top: 3.5rem;
                margin-bottom: 0.2rem;
                text-shadow: 0 0 40px rgba(61,220,151,0.18);
            }
            .cw-select-tagline {
                text-align: center;
                color: #4a5260;
                font-size: 0.78rem;
                letter-spacing: 0.22em;
                text-transform: uppercase;
                margin-bottom: 3.2rem;
            }
 
            .st-key-cw_store_carousel {
                position: relative;
            }
            .st-key-cw_store_carousel [data-testid="stHorizontalBlock"] {
                display: flex;
                flex-wrap: nowrap !important;
                gap: 28px;
                overflow-x: auto;
                overflow-y: hidden;
                padding: 12px 64px 28px 64px;
                scroll-behavior: smooth;
                scroll-snap-type: x proximity;
                scrollbar-width: none;
                -ms-overflow-style: none;
            }
            .st-key-cw_store_carousel [data-testid="stHorizontalBlock"]::-webkit-scrollbar {
                display: none;
            }
            .st-key-cw_store_carousel [data-testid="stColumn"] {
                flex: 0 0 180px !important;
                width: 180px !important;
                min-width: 180px !important;
                max-width: 180px !important;
                scroll-snap-align: center;
            }
 
            .cw-carousel-arrow {
                position: absolute;
                top: 50%;
                transform: translateY(-50%);
                z-index: 20;
                width: 44px;
                height: 44px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(13,17,23,0.75);
                border: 1px solid rgba(255,255,255,0.12);
                color: #eef2f6;
                font-size: 1.4rem;
                cursor: pointer;
                backdrop-filter: blur(4px);
                transition: background 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
                user-select: none;
            }
            .cw-carousel-arrow:hover {
                background: rgba(61,220,151,0.18);
                border-color: rgba(61,220,151,0.55);
                transform: translateY(-50%) scale(1.08);
            }
            .cw-carousel-arrow-left { left: 4px; }
            .cw-carousel-arrow-right { right: 4px; }
 
            div[class*="st-key-cw_store_card_"] {
                animation: cw-card-in 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
            }
            @keyframes cw-card-in {
                from { opacity: 0; transform: translateY(18px) scale(0.96); }
                to   { opacity: 1; transform: translateY(0) scale(1); }
            }
 
            div[class*="st-key-cw_store_card_"] button {
                width: 180px !important;
                height: 180px !important;
                border-radius: 22px;
                border: 1px solid rgba(255,255,255,0.08);
                background-color: #0d1117;
                background-repeat: no-repeat;
                background-position: center;
                background-size: 62%;
                box-shadow: 0 14px 34px rgba(0,0,0,0.45);
                color: transparent !important;
                font-size: 0 !important;
                transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1),
                            box-shadow 0.28s ease,
                            border-color 0.28s ease;
            }
            div[class*="st-key-cw_store_card_"] button:hover {
                transform: translateY(-6px) scale(1.08);
                border-color: rgba(61,220,151,0.55);
                box-shadow: 0 22px 46px rgba(0,0,0,0.55), 0 0 30px rgba(61,220,151,0.20);
            }
            div[class*="st-key-cw_store_card_"] button:active {
                transform: translateY(-2px) scale(1.02);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
 
 
def _render_carousel_arrows() -> None:
    # The arrow elements themselves are plain, styled markup. Streamlit
    # sanitizes inline event handlers (onclick/onerror) out of markdown, so
    # the previous approach of attaching the scroll logic directly to these
    # divs never actually ran — that was the "arrows don't work" bug. The
    # click handlers are now wired up from a real <script> (rendered via
    # streamlit.components — see `_render_carousel_scroll_script`), which can
    # reach the arrows and the scroll track in the parent document.
    st.markdown(
        """
        <div class="cw-carousel-arrow cw-carousel-arrow-left" data-cw-arrow="left">&#8249;</div>
        <div class="cw-carousel-arrow cw-carousel-arrow-right" data-cw-arrow="right">&#8250;</div>
        """,
        unsafe_allow_html=True,
    )
 
 
def _render_carousel_scroll_script() -> None:
    """Wire the carousel arrows to horizontal scrolling of only the store row.
 
    This runs as a genuine <script> inside a zero-height component iframe. The
    iframe is same-origin with the app, so it can reach the arrows and the
    scroll track in the parent document and move exactly one card-pitch (card
    width + the track's real column gap) per click, in sync with the snap
    points. The page itself never scrolls — only the carousel row does.
    """
    components.html(
        """
        <script>
        (function () {
            const doc = window.parent.document;
 
            function getTrack() {
                return doc.querySelector(
                    '.st-key-cw_store_carousel [data-testid="stHorizontalBlock"]'
                );
            }
 
            function stepSize(track) {
                const card = track.querySelector('[data-testid="stColumn"]');
                if (!card) return 208;
                const rect = card.getBoundingClientRect();
                const cs = window.getComputedStyle(track);
                const gap = parseFloat(cs.columnGap || cs.gap || '0') || 0;
                return rect.width + gap;
            }
 
            function scrollCarousel(direction) {
                const track = getTrack();
                if (!track) return;
                const step = stepSize(track);
                const maxScroll = track.scrollWidth - track.clientWidth;
                let target = track.scrollLeft + direction * step;
                target = Math.max(0, Math.min(maxScroll, target));
                track.scrollTo({ left: target, behavior: 'smooth' });
            }
 
            function bind() {
                const left = doc.querySelector('[data-cw-arrow="left"]');
                const right = doc.querySelector('[data-cw-arrow="right"]');
                if (left && !left.dataset.cwBound) {
                    left.dataset.cwBound = '1';
                    left.addEventListener('click', function () { scrollCarousel(-1); });
                }
                if (right && !right.dataset.cwBound) {
                    right.dataset.cwBound = '1';
                    right.addEventListener('click', function () { scrollCarousel(1); });
                }
            }
 
            bind();
            // Re-bind after Streamlit reruns re-render the arrows.
            setInterval(bind, 400);
        })();
        </script>
        """,
        height=0,
    )
 
 
def _render_carousel(stores: list[Store]) -> None:
    missing = [s for s in stores if not s.logo_path.exists()]
    if missing:
        names = ", ".join(s.logo_path.name for s in missing)
        st.warning(
            f"Missing logo file(s): {names}\n\n"
            f"Expected inside: {ASSETS_DIR}\n\n"
            "Showing placeholder cards for those stores until the PNG files are added there.",
            icon="⚠️",
        )
 
    with st.container(key="cw_store_carousel"):
        _render_carousel_arrows()
        cols = st.columns(len(stores))
        for col, store in zip(cols, stores):
            with col:
                with st.container(key=f"cw_store_card_{store.store_id}"):
                    logo_uri = _logo_data_uri(str(store.logo_path), store.display_name)
                    st.markdown(
                        f"""
                        <style>
                            .st-key-cw_store_card_{store.store_id} button {{
                                background-image: url('{logo_uri}');
                            }}
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    clicked = st.button(
                        store.display_name,
                        key=f"cw_store_btn_{store.store_id}",
                        width="stretch",
                    )
                    if clicked:
                        session_set_selected_store_id(store.store_id)
                        session_set_login_error(None)
                        st.rerun()
 
 
def _inject_login_theme() -> None:
    """Premium, self-contained login overlay (RTL, Hebrew)."""
    st.markdown(
        """
        <style>
            @keyframes cw-login-in {
                from { opacity: 0; transform: translateY(14px) scale(0.97); }
                to   { opacity: 1; transform: translateY(0) scale(1); }
            }
            @keyframes cw-shake {
                10%, 90% { transform: translateX(-1px); }
                20%, 80% { transform: translateX(2px); }
                30%, 50%, 70% { transform: translateX(-4px); }
                40%, 60% { transform: translateX(4px); }
            }
 
            /* Full-screen dimmed backdrop that sits above the store picker. */
            .st-key-cw_login_overlay {
                position: fixed !important;
                inset: 0;
                z-index: 1000;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(3, 5, 8, 0.72);
                backdrop-filter: blur(6px);
                -webkit-backdrop-filter: blur(6px);
            }
 
            /* The login card itself. */
            .st-key-cw_login_overlay .st-key-cw_login_card {
                direction: rtl;
                width: 380px !important;
                max-width: 90vw;
                margin: 0 auto;
                background: linear-gradient(180deg, #10151c 0%, #0d1117 100%);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 22px;
                padding: 30px 30px 26px 30px;
                box-shadow: 0 30px 80px rgba(0,0,0,0.6), 0 0 40px rgba(61,220,151,0.06);
                animation: cw-login-in 0.45s cubic-bezier(0.16, 1, 0.3, 1) both;
            }
            .st-key-cw_login_card.cw-login-error-shake {
                animation: cw-shake 0.5s cubic-bezier(.36,.07,.19,.97) both;
            }
            .cw-login-logo {
                display: block;
                width: 84px;
                height: 84px;
                object-fit: contain;
                margin: 0 auto 14px auto;
                border-radius: 20px;
                background: #0a0d12;
                border: 1px solid rgba(255,255,255,0.06);
                padding: 8px;
            }
            .cw-login-store {
                text-align: center;
                font-size: 1.35rem;
                font-weight: 800;
                color: #eef2f6;
                margin-bottom: 2px;
            }
            .cw-login-sub {
                text-align: center;
                color: #7d8590;
                font-size: 0.82rem;
                margin-bottom: 18px;
            }
            .cw-login-error {
                color: #f85149;
                background: rgba(248,81,73,0.08);
                border: 1px solid rgba(248,81,73,0.5);
                border-radius: 10px;
                padding: 9px 12px;
                margin: 4px 0 2px 0;
                font-size: 0.86rem;
                text-align: center;
            }
            /* Login input: readable, high-contrast (white field, dark text). */
            .st-key-cw_login_card input {
                background: #ffffff !important;
                color: #0d1117 !important;
                border-radius: 10px !important;
                text-align: right;
            }
            .st-key-cw_login_card input::placeholder { color: #8a94a3 !important; }
            .st-key-cw_login_card [data-testid="stForm"] {
                border: none !important;
                padding: 0 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
 
 
def render_login_overlay(store: Store) -> None:
    """Inline, premium login overlay.
 
    Deliberately does NOT use ``st.dialog``: a dialog kept the password
    screen visually "stuck" after a correct password (it had to be closed
    with the X). As a plain overlay, a successful login simply stops
    rendering the overlay on the next rerun, so the user drops straight
    into the dashboard, and a wrong password shows a clear inline error.
    """
    _inject_login_theme()
    error = session_get_login_error()
 
    with st.container(key="cw_login_overlay"):
        with st.container(key="cw_login_card"):
            logo_uri = _logo_data_uri(str(store.logo_path), store.display_name)
            st.markdown(f'<img class="cw-login-logo" src="{logo_uri}"/>', unsafe_allow_html=True)
            st.markdown(f'<div class="cw-login-store">{store.display_name}</div>', unsafe_allow_html=True)
            st.markdown('<div class="cw-login-sub">הזינו סיסמה כדי להיכנס למערכת</div>', unsafe_allow_html=True)
 
            with st.form(key="cw_login_form", border=False):
                password = st.text_input(
                    "סיסמה", type="password", key="cw_password_input",
                    placeholder="הקלידו את הסיסמה", label_visibility="collapsed",
                )
                if error:
                    st.markdown(f'<div class="cw-login-error">{error}</div>', unsafe_allow_html=True)
 
                # Login is defined first so pressing Enter in the password
                # field submits Login (the browser triggers the first submit
                # button), not Cancel. Columns keep Login on the right (RTL).
                c1, c2 = st.columns(2)
                with c1:
                    cancel = st.form_submit_button("ביטול", width="stretch")
                with c2:
                    submit = st.form_submit_button("כניסה", type="primary", width="stretch")
 
    if cancel:
        session_set_selected_store_id(None)
        session_set_login_error(None)
        st.rerun()
 
    if submit:
        result = _AUTH.attempt_login(store.store_id, password)
        if result.success:
            session_mark_authenticated(store.store_id)
            session_set_login_error(None)
            st.rerun()
        else:
            session_set_login_error(result.message)
            st.rerun()
 
 
def _inject_store_search_theme() -> None:
    st.markdown(
        """
        <style>
            .st-key-cw_store_search { max-width: 560px; margin: 0 auto 4px auto; }
            .st-key-cw_store_search input {
                direction: rtl;
                text-align: right;
                background: #0d1117 !important;
                color: #eef2f6 !important;
                border: 1px solid rgba(255,255,255,0.12) !important;
                border-radius: 999px !important;
                padding: 14px 22px !important;
                font-size: 1.02rem !important;
                box-shadow: 0 10px 30px rgba(0,0,0,0.45);
            }
            .st-key-cw_store_search input:focus {
                border-color: rgba(61,220,151,0.55) !important;
                box-shadow: 0 0 0 3px rgba(61,220,151,0.15), 0 10px 30px rgba(0,0,0,0.5) !important;
            }
            .st-key-cw_store_search input::placeholder { color: #6b7480 !important; }

            .cw-suggest-wrap { max-width: 560px; margin: 6px auto 10px auto; }
            div[class*="st-key-cw_suggest_btn_"] button {
                width: 100% !important;
                text-align: right !important;
                direction: rtl;
                background: #0d1117 !important;
                border: 1px solid rgba(255,255,255,0.08) !important;
                border-radius: 14px !important;
                padding: 10px 16px !important;
                margin-bottom: 6px;
                color: #eef2f6 !important;
                font-weight: 600 !important;
                transition: border-color 0.2s ease, transform 0.15s ease, background 0.2s ease;
            }
            div[class*="st-key-cw_suggest_btn_"] button:hover {
                border-color: rgba(61,220,151,0.55) !important;
                background: rgba(61,220,151,0.06) !important;
                transform: translateY(-1px);
            }
            .cw-suggest-row { display: flex; align-items: center; gap: 12px; direction: rtl; }
            .cw-suggest-logo {
                width: 34px; height: 34px; border-radius: 8px; object-fit: contain;
                background: #0a0d12; border: 1px solid rgba(255,255,255,0.06); padding: 3px; flex: 0 0 auto;
            }
            .cw-suggest-name { color: #eef2f6; font-size: 0.98rem; font-weight: 700; }
            .cw-suggest-meta { color: #7d8590; font-size: 0.78rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_store_search() -> bool:
    """Google-style store autocomplete for the landing page.

    Live suggestions render as the user types — no Enter required (Streamlit
    reruns on each keystroke, and we recompute suggestions from the prebuilt
    in-memory index, not by rescanning all stores). Clicking a suggestion
    selects that store immediately (the login overlay then appears).

    Returns True if a suggestion was clicked (caller should st.rerun()).
    """
    _inject_store_search_theme()
    with st.container(key="cw_store_search"):
        query = st.text_input(
            "חיפוש חנות",
            key="cw_store_search_input",
            placeholder="חיפוש חנות לפי שם, סניף או עיר…",
            label_visibility="collapsed",
        )

    if not query or not query.strip():
        return False

    index = get_store_search_index()
    suggestions = index.suggest(query, limit=8)

    if not suggestions:
        st.markdown(
            '<div class="cw-suggest-wrap"><div class="cw-suggest-meta" '
            'style="text-align:center;">אין תוצאות מתאימות</div></div>',
            unsafe_allow_html=True,
        )
        return False

    clicked = False
    with st.container(key="cw_suggest_list"):
        st.markdown('<div class="cw-suggest-wrap">', unsafe_allow_html=True)
        for sug in suggestions:
            meta_bits = [b for b in (sug.branch, sug.city) if b]
            meta = " · ".join(meta_bits)
            logo_html = (
                f'<img class="cw-suggest-logo" src="{sug.logo_uri}"/>' if sug.logo_uri else ""
            )
            st.markdown(
                f'<div class="cw-suggest-row">{logo_html}'
                f'<div><div class="cw-suggest-name">{sug.name}</div>'
                f'<div class="cw-suggest-meta">{meta}</div></div></div>',
                unsafe_allow_html=True,
            )
            if st.button(
                f"פתח {sug.name}",
                key=f"cw_suggest_btn_{sug.store_id}",
                width="stretch",
            ):
                session_set_selected_store_id(sug.store_id)
                session_set_login_error(None)
                session_set_active_page("Dashboard")
                clicked = True
        st.markdown("</div>", unsafe_allow_html=True)

    return clicked


def render_store_selector() -> None:
    """Full-screen store picker. Call this before any dashboard code runs."""
    _inject_selector_theme()
    st.markdown('<div class="cw-select-logo">COREWISE</div>', unsafe_allow_html=True)
    st.markdown('<div class="cw-select-tagline">בחרו את החנות שלכם</div>', unsafe_allow_html=True)
    # Store Search sits at the TOP of the landing page (Task 1), above the
    # store carousel — not in the sidebar.
    if render_store_search():
        st.rerun()
    _render_carousel(get_all_stores())
    _render_carousel_scroll_script()
 
 
def require_store_selection() -> str | None:
    """
    Entry point for app.py.
 
    Returns the authenticated store_id once a store has been picked and
    the correct password entered, or None while the picker/login is
    still in progress (caller should stop rendering the dashboard).
    """
    session_init()
 
    selected_id = session_get_selected_store_id()
 
    if selected_id and session_is_authenticated(selected_id):
        return selected_id
 
    render_store_selector()
 
    if selected_id:
        store = get_store(selected_id)
        if store is not None:
            render_login_overlay(store)
 
    return None
 
 
# =========================================================================
# ORIGINAL DASHBOARD  (unchanged - only gated behind store selection in main())
# =========================================================================
REFRESH_INTERVAL_MS = 250
MAX_HISTORY_POINTS = 300
 
MODEL_OPTIONS = {
    "מהיר (yolov8n)": "yolov8n.pt",
    "מאוזן (yolov8s)": "yolov8s.pt",
    "מדויק (yolov8m)": "yolov8m.pt",
}
 
WEATHER_CODES = {
    0: "שמיים בהירים", 1: "בהיר בעיקר", 2: "מעונן חלקית", 3: "מעונן",
    45: "ערפל", 48: "ערפל", 51: "טפטוף קל", 53: "טפטוף", 55: "טפטוף כבד",
    61: "גשם קל", 63: "גשם", 65: "גשם כבד", 71: "שלג קל",
    73: "שלג", 75: "שלג כבד", 80: "ממטרי גשם", 81: "ממטרי גשם",
    82: "ממטרי גשם עזים", 95: "סופת רעמים", 96: "סופת רעמים",
    99: "סופת רעמים",
}
 
st.set_page_config(
    page_title="Corewise | בינה קמעונאית",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# Store operating hours shown on the hourly traffic graph (08:00 - 20:00).
OPERATING_HOURS: list[int] = list(range(8, 21))
 
 
# =========================================================================
# SESSION STATE
# =========================================================================
def init_session_state() -> None:
    defaults = {
        "history": [],          # list of {"time": datetime, "inside": int, "tam": int}
        "dark_mode": True,
        "last_sent_control": None,
        "analytics_mode": "tam_sam_som",  # "tam_sam_som" | "tam_som"
        # --- Hourly analytics state (see _ensure_hourly_state); feeds the
        # hourly traffic chart, Today's Opportunities, and the AI Assistant ---
        "cw_hourly_day": None,          # date the buckets below belong to
        "cw_hourly_store_id": None,     # store the buckets below belong to
        "cw_hourly_tam": {},            # hour(int) -> new-TAM this hour (delta of engine's cumulative counter)
        "cw_hourly_som": {},            # hour(int) -> new-SOM (entries) this hour
        "cw_hourly_peak_inside": {},    # hour(int) -> max concurrent "inside" observed this hour
        "cw_last_tam_som": None,        # (tam, som) snapshot from the previous tick, for delta math
        "cw_zero_seconds_today": 0.0,   # cumulative seconds today with 0 people inside
        "cw_zero_streak_since": None,   # datetime the current zero-traffic streak started (or None)
        "cw_stay_time_samples": [],     # list of average_stay_time readings observed today (session-local)
        # --- AI Assistant state ---
        "cw_current_store_id": None,
        "cw_ai_messages": [],           # chat feed: [{"role","type","text","time"}]
        "cw_ai_last_emit_ts": None,     # datetime of the last auto-generated message (throttling)
        "cw_ai_emitted_keys": {},       # insight-key -> datetime last emitted (dedup/cooldown)
        "cw_ai_greeted_store": None,    # store_id the greeting message was already sent for
        "cw_ai_open": True,             # whether the Copilot side panel is expanded
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
 
 
init_session_state()
 
 
# =========================================================================
# THEME / CSS  (Tesla / Apple / Palantir-inspired minimal dark theme)
# =========================================================================
def inject_theme(dark_mode: bool) -> None:
    bg = "#05070a" if dark_mode else "#f5f6f8"
    panel = "#0d1117" if dark_mode else "#ffffff"
    text = "#e6edf3" if dark_mode else "#0d1117"
    subtext = "#7d8590" if dark_mode else "#57606a"
    accent = "#3ddc97"
    border = "#1c2128" if dark_mode else "#e2e2e2"
 
    st.markdown(
        f"""
        <style>
            .stApp {{ background: {bg}; color: {text}; }}
            /* Sidebar fix v2: hiding the whole <header> (as before) also hid the
               sidebar's native reopen control, whose internal data-testid turned
               out to differ from what was targeted last time -- so once closed,
               the sidebar had no way back. Instead of guessing testids again,
               the header itself is left alone (so its built-in sidebar toggle
               keeps working exactly as Streamlit ships it) and only the specific
               chrome pieces we actually want hidden are targeted individually. */
            header[data-testid="stHeader"] {{ background: transparent; }}
            #MainMenu {{ visibility: hidden; }}
            [data-testid="stToolbar"] {{ visibility: hidden; }}
            [data-testid="stDecoration"] {{ display: none; }}
            footer {{ visibility: hidden; }}
            [data-testid="stSidebar"] {{
                background: {panel};
                border-right: 1px solid {border};
            }}
            .corewise-header {{
                font-size: 2rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                color: {text};
                margin-bottom: 0;
            }}
            .corewise-subheader {{
                color: {subtext};
                font-size: 0.9rem;
                letter-spacing: 0.03em;
                margin-top: -6px;
            }}
            .cw-card {{
                background: {panel};
                border: 1px solid {border};
                border-radius: 16px;
                padding: 18px 20px;
                margin-bottom: 12px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.12);
            }}
            .cw-card h3 {{ color: {text}; margin: 0 0 4px 0; }}
            .cw-card p {{ color: {subtext}; margin: 0; }}
            .cw-metric-label {{
                color: {subtext};
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            .cw-metric-value {{
                color: {text};
                font-size: 2.1rem;
                font-weight: 700;
                line-height: 1.2;
            }}
            .cw-metric-value-small {{
                color: {text};
                font-size: 1.3rem;
                font-weight: 700;
            }}
            .cw-accent {{ color: {accent}; }}
            .cw-status-dot {{
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            .cw-divider {{ border-top: 1px solid {border}; margin: 18px 0; }}
 
            /* --- Executive Insights / Today's Opportunities (glassmorphism) --- */
            @keyframes cw-fade-in-up {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to   {{ opacity: 1; transform: translateY(0); }}
            }}
            .cw-fade-in {{ animation: cw-fade-in-up 0.5s cubic-bezier(0.16, 1, 0.3, 1) both; }}
 
            .cw-glass-card {{
                background: {"rgba(13,17,23,0.55)" if dark_mode else "rgba(255,255,255,0.65)"};
                backdrop-filter: blur(14px) saturate(140%);
                -webkit-backdrop-filter: blur(14px) saturate(140%);
                border: 1px solid {"rgba(255,255,255,0.08)" if dark_mode else "rgba(0,0,0,0.06)"};
                border-radius: 18px;
                padding: 20px 22px;
                margin-bottom: 14px;
                box-shadow: 0 18px 40px rgba(0,0,0,{0.28 if dark_mode else 0.08});
                transition: transform 0.25s ease, box-shadow 0.25s ease;
            }}
            .cw-glass-card:hover {{
                box-shadow: 0 22px 50px rgba(0,0,0,{0.34 if dark_mode else 0.12});
            }}
 
            .cw-exec-subtitle {{
                display: block;
                font-size: 0.82rem;
                color: {subtext};
                letter-spacing: 0.02em;
                margin-top: 2px;
            }}
 
            .cw-exec-section-title {{
                font-size: 1.1rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                color: {text};
                margin: 6px 0 12px 0;
            }}
            .cw-opportunity-card {{ min-height: 100%; }}
            .cw-opp-label {{
                font-size: 0.68rem;
                letter-spacing: 0.09em;
                text-transform: uppercase;
                color: {subtext};
                margin-top: 10px;
            }}
            .cw-opp-label:first-child {{ margin-top: 0; }}
            .cw-opp-title {{
                font-size: 1.05rem;
                font-weight: 700;
                color: {text};
                margin-top: 2px;
            }}
            .cw-opp-body {{
                font-size: 0.92rem;
                color: {text};
                margin-top: 2px;
            }}
            .cw-opp-reason {{
                font-size: 0.8rem;
                color: {subtext};
                margin-top: 12px;
                padding-top: 10px;
                border-top: 1px dashed {"rgba(255,255,255,0.08)" if dark_mode else "rgba(0,0,0,0.08)"};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
 
 
# =========================================================================
# LOCATION / WEATHER  (best-effort, never blocks the dashboard if offline)
# =========================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_location() -> Dict[str, Any]:
    """Best-effort IP-based location lookup for the store. Fails gracefully offline."""
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            city = data.get("city") or "לא ידוע"
            country = data.get("country_name") or ""
            label = f"{city}, {country}".strip(", ")
            return {"label": label or "לא ידוע", "lat": data.get("latitude"), "lon": data.get("longitude")}
    except Exception:
        pass
    return {"label": "לא זמין", "lat": None, "lon": None}
 
 
@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(lat: float, lon: float) -> Dict[str, Any]:
    """Best-effort weather lookup via the free Open-Meteo API."""
    if lat is None or lon is None:
        return {"temp": None, "desc": "לא זמין"}
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            current = resp.json().get("current_weather", {})
            return {"temp": current.get("temperature"), "desc": WEATHER_CODES.get(current.get("weathercode"), "לא זמין")}
    except Exception:
        pass
    return {"temp": None, "desc": "לא זמין"}
 
 
# =========================================================================
# CLIENT / HISTORY
# =========================================================================
def get_client() -> CorewiseClient:
    """Create (once) and reuse the WebSocket client across Streamlit reruns."""
    if "corewise_client" not in st.session_state:
        client = CorewiseClient(role="dashboard")
        client.start()
        st.session_state["corewise_client"] = client
    return st.session_state["corewise_client"]
 
 
def record_history_point(telemetry: Dict[str, Any]) -> None:
    """Append the current live counts to the in-memory history (capped length)."""
    history = st.session_state["history"]
    history.append(
        {
            "time": datetime.now(),
            "inside": int(telemetry.get("inside", 0)),
            "tam": int(telemetry.get("tam", 0)),
        }
    )
    if len(history) > MAX_HISTORY_POINTS:
        del history[: len(history) - MAX_HISTORY_POINTS]
 
 
# =========================================================================
# DAILY HISTORY PERSISTENCE  (feeds the AI Assistant's day-over-day and
# same-weekday comparisons)
#
# The live telemetry stream only ever tells us "as of now" - once a
# calendar day ends, the hourly buckets built up during that day would be
# lost on the next reset. To let the assistant answer things like
# "compare today with yesterday" or "today's performance vs. the average
# Monday" honestly, each day's finished bucket set is written to a small
# per-store JSON file on disk before the buckets are cleared. This is the
# only extra state introduced for the assistant - it is real, previously
# observed data, never fabricated or simulated.
# =========================================================================
DATA_DIR = Path(__file__).parent / "data" / "history"
 
 
def _history_file(store_id: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{store_id}.json"
 
 
def _load_store_history(store_id: str) -> Dict[str, Any]:
    path = _history_file(store_id)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
 
 
def _save_day_record(store_id: str, day_str: str, record: Dict[str, Any]) -> None:
    """Best-effort persistence - never allowed to break the live dashboard."""
    try:
        data = _load_store_history(store_id)
        data[day_str] = record
        if len(data) > 120:  # keep roughly the last 4 months
            for stale_key in sorted(data.keys())[: len(data) - 120]:
                del data[stale_key]
        path = _history_file(store_id)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)
    except Exception:
        pass
 
 
def get_yesterday_record(store_id: str) -> Optional[Dict[str, Any]]:
    hist = _load_store_history(store_id)
    return hist.get((date.today() - timedelta(days=1)).isoformat())
 
 
def get_weekday_history(store_id: str, weekday: int, exclude_today: bool = True) -> list[Dict[str, Any]]:
    """All previously recorded full days that fall on the given weekday (0=Mon..6=Sun)."""
    hist = _load_store_history(store_id)
    today_str = date.today().isoformat()
    return [
        rec for day_str, rec in hist.items()
        if rec.get("weekday") == weekday and not (exclude_today and day_str == today_str)
    ]
 
 
def _weekday_name(idx: int) -> str:
    return ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"][idx]
 
 
# =========================================================================
# HOURLY ANALYTICS  (feeds the hourly traffic graph, the AI Assistant's
# live insights, and Today's Opportunities)
#
# The engine (CorewiseClient/telemetry) only streams live cumulative
# counters - tam/som/inside/average_stay_time "as of now". There is no
# per-hour storage on the wire, so this layer derives real hourly buckets
# by diffing successive ticks and accumulating the deltas by wall-clock
# hour, for the current calendar day. Buckets reset automatically at
# midnight (and are re-scoped whenever the active store changes, so one
# store's numbers can never leak into another's). This is genuine data
# derived from the live stream - nothing here is mocked or randomly
# generated.
# =========================================================================
def _ensure_hourly_state() -> None:
    today = date.today()
    current_store_id = st.session_state.get("cw_current_store_id")
    stored_store_id = st.session_state.get("cw_hourly_store_id")
 
    day_changed = st.session_state["cw_hourly_day"] != today
    store_changed = current_store_id is not None and stored_store_id not in (None, current_store_id)
 
    if day_changed or store_changed:
        prev_day = st.session_state["cw_hourly_day"]
        prev_store = stored_store_id
        prev_tam = st.session_state["cw_hourly_tam"]
        if prev_day is not None and prev_store is not None and prev_tam:
            samples = st.session_state["cw_stay_time_samples"]
            record = {
                "date": prev_day.isoformat(),
                "weekday": prev_day.weekday(),
                "hourly_tam": {str(h): v for h, v in prev_tam.items()},
                "hourly_som": {str(h): v for h, v in st.session_state["cw_hourly_som"].items()},
                "peak_inside": {str(h): v for h, v in st.session_state["cw_hourly_peak_inside"].items()},
                "total_tam": sum(prev_tam.values()),
                "total_som": sum(st.session_state["cw_hourly_som"].values()),
                "avg_stay_time": (sum(samples) / len(samples)) if samples else None,
            }
            _save_day_record(prev_store, prev_day.isoformat(), record)
 
        st.session_state["cw_hourly_day"] = today
        st.session_state["cw_hourly_store_id"] = current_store_id
        st.session_state["cw_hourly_tam"] = {}
        st.session_state["cw_hourly_som"] = {}
        st.session_state["cw_hourly_peak_inside"] = {}
        st.session_state["cw_last_tam_som"] = None
        st.session_state["cw_zero_seconds_today"] = 0.0
        st.session_state["cw_zero_streak_since"] = None
        st.session_state["cw_stay_time_samples"] = []
 
 
def record_hourly_point(telemetry: Dict[str, Any]) -> None:
    """Fold the current telemetry tick into today's hourly buckets."""
    _ensure_hourly_state()
    now = datetime.now()
    hour = now.hour
    tam = int(telemetry.get("tam", 0))
    som = int(telemetry.get("som", 0))
    inside = int(telemetry.get("inside", 0))
 
    last = st.session_state["cw_last_tam_som"]
    if last is not None:
        last_tam, last_som = last
        d_tam = max(0, tam - last_tam)
        d_som = max(0, som - last_som)
        if d_tam:
            st.session_state["cw_hourly_tam"][hour] = st.session_state["cw_hourly_tam"].get(hour, 0) + d_tam
        if d_som:
            st.session_state["cw_hourly_som"][hour] = st.session_state["cw_hourly_som"].get(hour, 0) + d_som
    st.session_state["cw_last_tam_som"] = (tam, som)
 
    peak = st.session_state["cw_hourly_peak_inside"]
    peak[hour] = max(peak.get(hour, 0), inside)
 
    # Zero-traffic streak tracking (feeds the "store was empty for ~X" insight).
    if inside == 0:
        if st.session_state["cw_zero_streak_since"] is None:
            st.session_state["cw_zero_streak_since"] = now
    else:
        streak_start = st.session_state["cw_zero_streak_since"]
        if streak_start is not None:
            st.session_state["cw_zero_seconds_today"] += (now - streak_start).total_seconds()
            st.session_state["cw_zero_streak_since"] = None
 
    stay_time = telemetry.get("average_stay_time")
    if stay_time is not None:
        samples = st.session_state["cw_stay_time_samples"]
        samples.append(float(stay_time))
        if len(samples) > 500:
            del samples[: len(samples) - 500]
 
 
def _current_zero_traffic_seconds() -> float:
    """Total empty-store seconds today, including an in-progress streak."""
    total = st.session_state["cw_zero_seconds_today"]
    streak_start = st.session_state["cw_zero_streak_since"]
    if streak_start is not None:
        total += (datetime.now() - streak_start).total_seconds()
    return total
 
 
def _hour_capture_rate(hour: int) -> Optional[float]:
    tam = st.session_state["cw_hourly_tam"].get(hour, 0)
    som = st.session_state["cw_hourly_som"].get(hour, 0)
    if tam <= 0:
        return None
    return som / tam
 
 
def render_history_chart(dark_mode: bool) -> None:
    """People-over-time line chart (Inside vs. cumulative TAM), Altair-themed."""
    history = st.session_state["history"]
    if not history:
        st.info("אין עדיין תנועה - הגרף יתמלא ככל שיזוהו מבקרים.")
        return
 
    df = pd.DataFrame(history).melt(
        id_vars="time", value_vars=["inside", "tam"], var_name="metric", value_name="count"
    )
    df["metric"] = df["metric"].map({"inside": "בפנים כעת", "tam": "TAM (מצטבר)"})
 
    text_color = "#e6edf3" if dark_mode else "#111827"
    grid_color = "#1c2128" if dark_mode else "#e5e7eb"
 
    chart = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("time:T", title="שעה", axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color)),
            y=alt.Y("count:Q", title="אנשים", axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color)),
            color=alt.Color("metric:N", legend=alt.Legend(title=None, labelColor=text_color)),
            tooltip=[alt.Tooltip("time:T", title="שעה"), alt.Tooltip("metric:N", title="מדד"), alt.Tooltip("count:Q", title="כמות")],
        )
        .properties(height=260, background="transparent")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)
 
 
def render_hourly_traffic_chart(dark_mode: bool) -> None:
    """
    Hourly visitor-traffic graph (08:00 - 20:00), replacing the old
    per-minute chart as the dashboard's primary trend view.
 
    Each bar/point is the real number of new visitors (TAM delta) the
    engine reported during that hour of the current day - see
    `record_hourly_point`. Hours in the future (relative to now) are
    omitted rather than padded with zeros, so the chart never implies
    data that doesn't exist yet.
    """
    _ensure_hourly_state()
    hourly_tam = st.session_state["cw_hourly_tam"]
    current_hour = datetime.now().hour
 
    rows = [
        {"hour": f"{h:02d}:00", "hour_num": h, "traffic": hourly_tam.get(h, 0)}
        for h in OPERATING_HOURS
        if h <= current_hour
    ]
 
    if not rows or sum(r["traffic"] for r in rows) == 0:
        st.info("עדיין לא נרשמה תנועה היום - הגרף השעתי יתמלא ככל שיזוהו מבקרים.")
        return
 
    df = pd.DataFrame(rows)
    text_color = "#e6edf3" if dark_mode else "#111827"
    grid_color = "#1c2128" if dark_mode else "#e5e7eb"
    accent = "#3ddc97"
 
    chart = (
        alt.Chart(df)
        .mark_area(
            line={"color": accent, "strokeWidth": 2.5},
            interpolate="monotone",
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color=accent, offset=0),
                    alt.GradientStop(color="rgba(61,220,151,0.02)", offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
            opacity=0.85,
        )
        .encode(
            x=alt.X("hour:N", title="שעה", sort=[f"{h:02d}:00" for h in OPERATING_HOURS],
                    axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color)),
            y=alt.Y("traffic:Q", title="מבקרים חדשים",
                    axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color)),
            tooltip=[alt.Tooltip("hour:N", title="שעה"), alt.Tooltip("traffic:Q", title="מבקרים חדשים")],
        )
        .properties(height=280, background="transparent")
        .configure_view(strokeWidth=0)
    )
    st.markdown('<div class="cw-fade-in">', unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
 
 
# =========================================================================
# TODAY'S OPPORTUNITIES  (rule-based recommendations from the same real
# hourly data used elsewhere — no fabricated impact numbers, only
# directional expectations tied to the pattern that triggered the card).
# =========================================================================
def build_opportunities(telemetry: Dict[str, Any]) -> list[Dict[str, str]]:
    _ensure_hourly_state()
    hourly_tam = st.session_state["cw_hourly_tam"]
    current_hour = datetime.now().hour
    opportunities: list[Dict[str, str]] = []
 
    past = {h: c for h, c in hourly_tam.items() if h <= current_hour}
    if past and max(past.values()) > 0:
        peak = max(past.values())
        quiet_hours = sorted(h for h, c in past.items() if c <= peak * 0.3 and h in OPERATING_HOURS)
        if len(quiet_hours) >= 2:
            span = f"{quiet_hours[0]:02d}:00–{quiet_hours[-1] + 1:02d}:00"
            opportunities.append({
                "opportunity_he": f"תנועה נמוכה בין {span}",
                "action_he": "שקלו מבצע ממוקד בשעות אלו",
                "impact_he": "צפי לעלייה במספר המבקרים",
                "reason_he": f"התנועה בחלון זמן זה נמוכה מ-30% משעת השיא היום ({peak} מבקרים).",
            })
 
    this_rate = _hour_capture_rate(current_hour)
    if this_rate is not None and hourly_tam.get(current_hour, 0) >= 10 and this_rate < 0.2:
        opportunities.append({
            "opportunity_he": "שיעור כניסה נמוך",
            "action_he": "שפרו את חלון הראווה או את נראות הכניסה",
            "impact_he": "הגדלת מספר הנכנסים לחנות",
            "reason_he": f"רק {this_rate * 100:.0f}% מהעוברים ושבים נכנסו לחנות בשעה זו.",
        })
 
    avg_stay = telemetry.get("average_stay_time")
    if avg_stay is not None and float(avg_stay) >= 600 and int(telemetry.get("inside", 0)) >= 5:
        opportunities.append({
            "opportunity_he": "שהייה ארוכה בחנות לצד תפוסה גבוהה",
            "action_he": "שקלו לפתוח עמדת קופה או שירות נוספת",
            "impact_he": "צמצום זמן ההמתנה בתור",
            "reason_he": f"זמן השהייה הממוצע הוא {float(avg_stay) / 60:.1f} דקות עם {int(telemetry.get('inside', 0))} אנשים כרגע בחנות.",
        })
 
    evening_hours = [h for h in past if h >= 17]
    if evening_hours:
        evening_avg = sum(past[h] for h in evening_hours) / len(evening_hours)
        day_hours = [h for h in past if h < 17]
        day_avg = sum(past[h] for h in day_hours) / len(day_hours) if day_hours else 0
        if day_avg > 0 and evening_avg >= day_avg * 1.4:
            opportunities.append({
                "opportunity_he": "תנועה גבוהה בשעות הערב",
                "action_he": "תאמו יותר עובדים למשמרת הערב",
                "impact_he": "שיפור חוויית הלקוח בשעות העומס",
                "reason_he": f"תנועת הערב גבוהה פי {evening_avg / day_avg:.1f} מרמת שעות היום.",
            })
 
    return opportunities
 
 
def render_opportunities_section(telemetry: Dict[str, Any]) -> None:
    opportunities = build_opportunities(telemetry)
    st.markdown('<div class="cw-exec-section-title">הזדמנויות להיום</div>', unsafe_allow_html=True)
 
    if not opportunities:
        st.markdown(
            '<div class="cw-glass-card cw-fade-in">'
            '<span class="cw-exec-subtitle">אין כרגע פעולות מומלצות - הביצועים בטווח הרגיל.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        return
 
    cols = st.columns(min(3, len(opportunities)))
    for idx, opp in enumerate(opportunities):
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="cw-glass-card cw-opportunity-card cw-fade-in">
                    <div class="cw-opp-label">הזדמנות</div>
                    <div class="cw-opp-title">{opp['opportunity_he']}</div>
                    <div class="cw-opp-label">פעולה מומלצת</div>
                    <div class="cw-opp-body">{opp['action_he']}</div>
                    <div class="cw-opp-label">השפעה משוערת</div>
                    <div class="cw-opp-body cw-accent">{opp['impact_he']}</div>
                    <div class="cw-opp-reason">{opp['reason_he']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
 
# =========================================================================
# RENDER HELPERS
# =========================================================================
def metric_card(label: str, value: str, accent: bool = False) -> None:
    value_class = "cw-metric-value cw-accent" if accent else "cw-metric-value"
    st.markdown(
        f'<div class="cw-card"><div class="cw-metric-label">{label}</div>'
        f'<div class="{value_class}">{value}</div></div>',
        unsafe_allow_html=True,
    )
 
 
# =========================================================================
# COREWISE BI ASSISTANT  (replaces the old static Executive Insights card)
#
# A permanent, always-visible chat panel that:
#   - proactively posts new messages as real patterns emerge in today's
#     analytics (reusing/extending the same rule-based logic the old
#     Executive Insights card used, plus day-over-day and same-weekday
#     comparisons backed by the persisted history above), and
#   - answers free-text questions, computed on demand from the same live
#     telemetry + hourly buckets + persisted history. There is no external
#     LLM call and no fabricated content: every sentence is built from a
#     number that was actually observed. If there isn't enough data yet to
#     answer honestly, the assistant says so instead of guessing.
# =========================================================================
AI_MIN_SECONDS_BETWEEN_AUTO_MESSAGES = 45   # overall pacing of the live feed
AI_KEY_COOLDOWN_SECONDS = 20 * 60           # don't repeat the same insight too often
 
AI_TYPE_META = {
    "insight":        {"icon": "◆", "color": "#58a6ff", "label": "תובנה"},
    "recommendation": {"icon": "★", "color": "#3ddc97", "label": "המלצה"},
    "alert":          {"icon": "▲", "color": "#e3b341", "label": "התראה"},
    "prediction":     {"icon": "◇", "color": "#a371f7", "label": "תחזית"},
}
 
 
def _today_running_totals() -> tuple[int, int]:
    _ensure_hourly_state()
    current_hour = datetime.now().hour
    hourly_tam = st.session_state["cw_hourly_tam"]
    hourly_som = st.session_state["cw_hourly_som"]
    tam = sum(c for h, c in hourly_tam.items() if h <= current_hour)
    som = sum(c for h, c in hourly_som.items() if h <= current_hour)
    return tam, som
 
 
def compare_today_vs_yesterday(store_id: str) -> Optional[Dict[str, Any]]:
    """Same-time-of-day comparison: today's traffic-so-far vs. yesterday's traffic through the same hours."""
    yrec = get_yesterday_record(store_id)
    if not yrec:
        return None
    current_hour = datetime.now().hour
    y_hourly = {int(h): v for h, v in yrec.get("hourly_tam", {}).items()}
    y_tam_so_far = sum(v for h, v in y_hourly.items() if h <= current_hour)
    today_tam, _ = _today_running_totals()
    if y_tam_so_far <= 0:
        return None
    pct = (today_tam - y_tam_so_far) / y_tam_so_far * 100
    return {"today_tam": today_tam, "yesterday_tam_so_far": y_tam_so_far, "pct": pct}
 
 
def compare_today_vs_weekday_average(store_id: str) -> Optional[Dict[str, Any]]:
    """Today-so-far vs. the average of previously recorded days on the same weekday."""
    today_wd = date.today().weekday()
    records = get_weekday_history(store_id, today_wd)
    if len(records) < 2:  # need at least 2 prior samples to call it an "average" honestly
        return None
    current_hour = datetime.now().hour
    totals = []
    for rec in records:
        hourly = {int(k): v for k, v in rec.get("hourly_tam", {}).items()}
        totals.append(sum(v for h, v in hourly.items() if h <= current_hour))
    avg = sum(totals) / len(totals)
    today_tam, _ = _today_running_totals()
    if avg <= 0:
        return None
    pct = (today_tam - avg) / avg * 100
    return {"today_tam": today_tam, "avg": avg, "pct": pct, "sample_days": len(records), "weekday": today_wd}
 
 
def predict_next_hour_trend() -> Optional[str]:
    """Simple momentum read on the last two completed hours - "up"/"down"/None."""
    _ensure_hourly_state()
    hourly_tam = st.session_state["cw_hourly_tam"]
    current_hour = datetime.now().hour
    completed = sorted(h for h in hourly_tam if h < current_hour and h in OPERATING_HOURS)
    if len(completed) < 2:
        return None
    v1, v2 = hourly_tam[completed[-2]], hourly_tam[completed[-1]]
    if v1 <= 0:
        return None
    change = (v2 - v1) / v1
    if change >= 0.25:
        return "up"
    if change <= -0.25:
        return "down"
    return None
 
 
def _staffing_alert_text(telemetry: Dict[str, Any]) -> Optional[str]:
    _ensure_hourly_state()
    hourly_tam = st.session_state["cw_hourly_tam"]
    current_hour = datetime.now().hour
    past = {h: c for h, c in hourly_tam.items() if h <= current_hour and h in OPERATING_HOURS}
    if len(past) < 3:
        return None
    avg = sum(past.values()) / len(past)
    current = past.get(current_hour, 0)
    inside = int(telemetry.get("inside", 0))
    if avg > 0 and current >= avg * 1.5 and inside >= 5:
        return (
            f"נפח המבקרים בשעה זו ({current}) גבוה משמעותית מהממוצע היומי ({avg:.0f}), "
            f"ו-{inside} אנשים נמצאים כרגע בחנות — ייתכן שכדאי להוסיף עובד בשעה זו."
        )
    return None
 
 
def generate_ai_insight_events(store_id: str, telemetry: Dict[str, Any]) -> list[Dict[str, str]]:
    """Every candidate message the assistant could post right now, each tagged with a stable
    dedup key so `_maybe_emit_ai_messages` won't repeat the same observation too often."""
    _ensure_hourly_state()
    today = date.today()
    events: list[Dict[str, str]] = []
    hourly_tam = st.session_state["cw_hourly_tam"]
    hourly_som = st.session_state["cw_hourly_som"]
    current_hour = datetime.now().hour
 
    # Busiest hour so far.
    past_hours = {h: c for h, c in hourly_tam.items() if h <= current_hour and c > 0}
    if past_hours:
        busiest_hour = max(past_hours, key=past_hours.get)
        events.append({
            "key": f"busiest-{today}-{busiest_hour}-{past_hours[busiest_hour]}",
            "type": "insight",
            "text": f"השעה העמוסה ביותר היום עד כה — {busiest_hour:02d}:00 עם {past_hours[busiest_hour]} מבקרים חדשים.",
        })
 
    # Capture-rate trend vs. previous hour.
    this_rate = _hour_capture_rate(current_hour)
    prev_rate = _hour_capture_rate(current_hour - 1) if current_hour > 0 else None
    if this_rate is not None and prev_rate is not None and prev_rate > 0:
        change_pct = (this_rate - prev_rate) / prev_rate * 100
        if abs(change_pct) >= 5:
            direction = "עולה" if change_pct > 0 else "יורד"
            events.append({
                "key": f"capture-{today}-{current_hour}-{'up' if change_pct > 0 else 'down'}",
                "type": "insight",
                "text": f"שיעור הכניסה (Capture Rate) {direction} בהשוואה לשעה הקודמת ({abs(change_pct):.0f}%).",
            })
 
    # People slowing down but not entering (high TAM, low SOM this hour).
    tam_this_hour = hourly_tam.get(current_hour, 0)
    som_this_hour = hourly_som.get(current_hour, 0)
    if tam_this_hour >= 10 and som_this_hour <= max(1, tam_this_hour * 0.15):
        events.append({
            "key": f"gap-{today}-{current_hour}",
            "type": "alert",
            "text": f"אנשים רבים האטו מול החנות בשעה זו ({tam_this_hour} עוברים ושבים) אך לא נכנסו — רק {som_this_hour} נכנסו פנימה.",
        })
 
    # Empty-store duration.
    zero_seconds = _current_zero_traffic_seconds()
    if zero_seconds >= 1800:
        events.append({
            "key": f"empty-{today}-{int(zero_seconds // 1800)}",
            "type": "alert",
            "text": f"בחנות לא היו מבקרים במשך כ-{zero_seconds / 3600:.1f} שעות היום.",
        })
 
    # Stay-time trend vs. today's running average.
    samples = st.session_state["cw_stay_time_samples"]
    current_stay = telemetry.get("average_stay_time")
    if current_stay is not None and len(samples) >= 20:
        running_avg = sum(samples) / len(samples)
        if running_avg > 0:
            delta_pct = (float(current_stay) - running_avg) / running_avg * 100
            if delta_pct <= -20:
                events.append({
                    "key": f"stay-down-{today}-{current_hour}",
                    "type": "alert",
                    "text": "העניין ירד בשעה האחרונה — המבקרים שוהים בחנות זמן קצר משמעותית מהממוצע היומי.",
                })
            elif delta_pct >= 20:
                events.append({
                    "key": f"stay-up-{today}-{current_hour}",
                    "type": "insight",
                    "text": "שיעור הכניסה משתפר בהשוואה למוקדם יותר היום, והמבקרים שוהים זמן ארוך מהממוצע היומי.",
                })
 
    # Yesterday, same-time-of-day comparison.
    cmp_y = compare_today_vs_yesterday(store_id)
    if cmp_y and abs(cmp_y["pct"]) >= 10:
        direction = "higher" if cmp_y["pct"] >= 0 else "lower"
        direction_he = "גבוהה" if cmp_y["pct"] >= 0 else "נמוכה"
        events.append({
            "key": f"yesterday-{today}-{current_hour}-{direction}",
            "type": "insight",
            "text": f"התנועה כרגע {direction_he} ב-{abs(cmp_y['pct']):.0f}% לעומת אתמול באותה נקודת זמן ביום.",
        })
 
    # Same-weekday historical average comparison.
    cmp_w = compare_today_vs_weekday_average(store_id)
    if cmp_w and abs(cmp_w["pct"]) >= 10:
        stronger = cmp_w["pct"] >= 0
        events.append({
            "key": f"weekday-{today}-{current_hour}-{'up' if stronger else 'down'}",
            "type": "insight",
            "text": (
                f"הביצועים היום {'חזקים' if stronger else 'חלשים'} מהממוצע של יום "
                f"{_weekday_name(cmp_w['weekday'])} ({abs(cmp_w['pct']):.0f}% {'מעל' if stronger else 'מתחת'} לממוצע, "
                f"מבוסס על {cmp_w['sample_days']} ימי {_weekday_name(cmp_w['weekday'])} שנרשמו)."
            ),
        })
 
    # Momentum-based prediction.
    trend = predict_next_hour_trend()
    if trend == "up":
        events.append({
            "key": f"predict-up-{today}-{current_hour}", "type": "prediction",
            "text": "על פי המגמות הנוכחיות, צפויה עלייה בתנועה בשעה הקרובה.",
        })
    elif trend == "down":
        events.append({
            "key": f"predict-down-{today}-{current_hour}", "type": "prediction",
            "text": "על פי המגמות הנוכחיות, צפויה ירידה בתנועה בשעה הקרובה.",
        })
 
    # Staffing alert.
    staffing_text = _staffing_alert_text(telemetry)
    if staffing_text:
        events.append({"key": f"staffing-{today}-{current_hour}", "type": "alert", "text": staffing_text})
 
    # Recommendations (same source data as the Today's Opportunities cards).
    for opp in build_opportunities(telemetry):
        title = opp["opportunity_he"]
        action = opp["action_he"]
        reason = opp["reason_he"]
        events.append({
            "key": f"opp-{today}-{title}",
            "type": "recommendation",
            "text": f"{title}: {action} — {reason}",
        })
 
    return events
 
 
def _ensure_ai_state() -> None:
    for key, value in {
        "cw_ai_messages": [], "cw_ai_last_emit_ts": None,
        "cw_ai_emitted_keys": {}, "cw_ai_greeted_store": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value
 
 
def _maybe_greet(store_id: str, store_display_name: str) -> None:
    _ensure_ai_state()
    if st.session_state["cw_ai_greeted_store"] != store_id:
        st.session_state["cw_ai_messages"] = []
        st.session_state["cw_ai_emitted_keys"] = {}
        st.session_state["cw_ai_last_emit_ts"] = None
        st.session_state["cw_ai_messages"].append({
            "role": "assistant", "type": "insight",
            "text": (
                f"שלום! אני העוזר החכם של {store_display_name} ועוקב אחר הנתונים בזמן אמת. "
                "אפרסם כאן תובנות חדשות ככל שהיום מתקדם — שאלו אותי כל דבר על ביצועי החנות היום."
            ),
            "time": datetime.now(),
        })
        st.session_state["cw_ai_greeted_store"] = store_id
 
 
def _maybe_emit_ai_messages(store_id: str, telemetry: Dict[str, Any]) -> None:
    _ensure_ai_state()
    now = datetime.now()
    last_emit = st.session_state["cw_ai_last_emit_ts"]
    if last_emit is not None and (now - last_emit).total_seconds() < AI_MIN_SECONDS_BETWEEN_AUTO_MESSAGES:
        return
 
    emitted_keys = st.session_state["cw_ai_emitted_keys"]
    for event in generate_ai_insight_events(store_id, telemetry):
        last_seen = emitted_keys.get(event["key"])
        if last_seen is not None and (now - last_seen).total_seconds() < AI_KEY_COOLDOWN_SECONDS:
            continue
        st.session_state["cw_ai_messages"].append({
            "role": "assistant", "type": event["type"], "text": event["text"], "time": now,
        })
        emitted_keys[event["key"]] = now
        st.session_state["cw_ai_last_emit_ts"] = now
        break  # one new auto-message per cycle keeps the feed readable
 
 
def _camera_status_answer(store_id: str, telemetry: Dict[str, Any]) -> str:
    """Honest, data-grounded answer about camera status. Never fabricated."""
    status = telemetry.get("camera_status")
    manager = get_camera_manager(store_id, telemetry)
    cameras = manager.list_cameras()
 
    if not status or status == "offline" or not cameras:
        return (
            "אין כרגע מידע זמין על המצלמות מכיוון שהמצלמות אינן מקוונות. "
            "ברגע שהמצלמות יחזרו לפעול אוכל להציג את הסטטוס והנתונים החיים."
        )
 
    status_he = {"online": "מקוונת", "paused": "מושהית"}.get(status, status)
    fps = telemetry.get("fps", 0) or 0
    count = len(cameras)
    parts = [f"סטטוס המצלמות: {status_he}."]
    if count == 1:
        parts.append("מחוברת מצלמה אחת פעילה")
    else:
        parts.append(f"מחוברות {count} מצלמות פעילות")
    if fps:
        parts.append(f"בקצב של {float(fps):.1f} פריימים לשנייה")
    return " ".join(parts) + "."
 
 
def answer_ai_question(question: str, store_id: str, telemetry: Dict[str, Any]) -> str:
    """Grounded Hebrew Q&A over real analytics only — no external model and no
    fabricated numbers. Each branch either computes an answer from data actually
    on hand, or clearly says (in Hebrew) that there isn't enough real data yet.
    Both Hebrew and English phrasings are recognised so questions can be asked
    naturally."""
    q = question.lower()
 
    def pct(p: float) -> str:
        return f"{abs(p):.0f}%"
 
    def has(*words: str) -> bool:
        return any(w in q for w in words)
 
    _ensure_hourly_state()
    current_hour = datetime.now().hour
 
    # --- Camera status (explicit offline grounding) ---
    if has("מצלמ", "camera", "cctv", "וידאו"):
        return _camera_status_answer(store_id, telemetry)
 
    # --- Yesterday comparison ---
    if has("אתמול", "yesterday"):
        cmp_y = compare_today_vs_yesterday(store_id)
        if not cmp_y:
            return "אין לי עדיין נתונים מאתמול עבור חנות זו, ולכן איני יכול לבצע השוואה אמינה. נסו שוב מחר."
        direction = "גבוהה" if cmp_y["pct"] >= 0 else "נמוכה"
        return (
            f"עד כה היום, התנועה {direction} ב-{pct(cmp_y['pct'])} לעומת אתמול באותה נקודת זמן "
            f"({cmp_y['today_tam']} מול {cmp_y['yesterday_tam_so_far']} מבקרים באותן שעות)."
        )
 
    # --- Busiest hour ---
    if has("שעה", "hour") and has("עמוס", "שיא", "busiest", "הכי"):
        hourly_tam = st.session_state["cw_hourly_tam"]
        past = {h: c for h, c in hourly_tam.items() if h <= current_hour and c > 0}
        if not past:
            return "עדיין לא נרשמו נתוני מבקרים היום."
        busiest = max(past, key=past.get)
        return f"השעה העמוסה ביותר היום עד כה הייתה {busiest:02d}:00, עם {past[busiest]} מבקרים חדשים."
 
    # --- Why is Capture Rate lower ---
    if has("למה", "why") and has("capture", "שיעור כניסה", "כניסה"):
        this_rate = _hour_capture_rate(current_hour)
        prev_rate = _hour_capture_rate(current_hour - 1) if current_hour > 0 else None
        if this_rate is None:
            return "אין עדיין מספיק נתוני עוברים ושבים/כניסות בשעה זו כדי להעריך את שיעור הכניסה."
        answer = f"שיעור הכניסה בשעה זו הוא {this_rate * 100:.0f}%."
        if prev_rate:
            change = (this_rate - prev_rate) / prev_rate * 100
            if change < 0:
                answer += f" זו ירידה של {pct(change)} מהשעה הקודמת — פחות מהעוברים ושבים נכנסים פנימה."
            else:
                answer += f" זו למעשה עלייה של {pct(change)} מהשעה הקודמת."
        return answer
 
    # --- Why is traffic lower/down ---
    if has("למה", "why") and has("נמוך", "ירד", "פחות", "lower", "down") and has("תנוע", "מבקר", "כניס", "traffic", "visitor", "entr"):
        cmp_y = compare_today_vs_yesterday(store_id)
        cmp_w = compare_today_vs_weekday_average(store_id)
        parts = []
        if cmp_y and cmp_y["pct"] < 0:
            parts.append(f"התנועה נמוכה ב-{pct(cmp_y['pct'])} לעומת אתמול באותה נקודת זמן")
        if cmp_w and cmp_w["pct"] < 0:
            parts.append(
                f"{pct(cmp_w['pct'])} מתחת לממוצע של יום {_weekday_name(cmp_w['weekday'])} "
                f"({cmp_w['sample_days']} ימים שנרשמו)"
            )
        if parts:
            return "כן — " + " וגם ".join(parts) + "."
        if cmp_y is None and cmp_w is None:
            return "אין לי עדיין מספיק נתונים היסטוריים כדי לומר אם היום נמוך מהרגיל — דרוש לפחות יום קודם אחד שנרשם."
        return "מספר הכניסות היום למעשה תואם, או אף גבוה, מהנתונים ההיסטוריים באותה נקודת זמן."
 
    # --- Staffing ---
    if has("עובד", "עובדים", "כוח אדם", "משמרת", "staff", "employee"):
        hourly_tam = st.session_state["cw_hourly_tam"]
        past = {h: c for h, c in hourly_tam.items() if h <= current_hour and h in OPERATING_HOURS}
        if len(past) < 2:
            return "אין עדיין מספיק נתוני תנועה היום כדי להמליץ על שינויי כוח אדם."
        avg = sum(past.values()) / len(past)
        busy_hours = sorted(h for h, c in past.items() if c >= avg * 1.4)
        if not busy_hours:
            return "אף שעה היום לא בלטה עדיין כזקוקה לתגבור כוח אדם."
        hours_str = ", ".join(f"{h:02d}:00" for h in busy_hours)
        return f"על פי דפוס התנועה היום, השעות הבאות היו הרבה מעל הממוצע וכדאי לשקול בהן תגבור עובדים: {hours_str}."
 
    # --- People currently inside ---
    if has("כרגע", "עכשיו", "כמה אנשים", "בפנים", "inside", "right now", "currently"):
        if telemetry.get("camera_status") in (None, "offline"):
            return "אין כרגע מידע זמין מהמצלמות מכיוון שהן אינן מקוונות, ולכן איני יכול לדעת כמה אנשים בחנות עכשיו."
        inside = int(telemetry.get("inside", 0))
        return f"כרגע נמצאים בחנות {inside} אנשים."
 
    # --- Conversion rate ---
    if has("המרה", "conversion"):
        if telemetry.get("camera_status") in (None, "offline"):
            return "אין כרגע נתונים חיים מהמצלמות (הן אינן מקוונות), ולכן איני יכול לדווח על שיעור ההמרה הנוכחי."
        conv = telemetry.get("conversion_rate", 0)
        return f"שיעור ההמרה הנוכחי הוא {float(conv):.1f}%."
 
    # --- Average stay time ---
    if has("שהייה", "שהות", "זמן ממוצע", "stay", "dwell"):
        if telemetry.get("camera_status") in (None, "offline"):
            return "אין כרגע נתונים חיים מהמצלמות (הן אינן מקוונות), ולכן איני יכול לדווח על זמן השהייה."
        avg_stay = telemetry.get("average_stay_time")
        if not avg_stay:
            return "עדיין אין מספיק נתונים כדי לחשב את זמן השהייה הממוצע."
        return f"זמן השהייה הממוצע בחנות הוא {float(avg_stay) / 60:.1f} דקות."
 
    # --- Prediction / next hour ---
    if has("תחזית", "צפוי", "השעה הבאה", "predict", "forecast", "next hour"):
        trend = predict_next_hour_trend()
        if trend == "up":
            return "על פי המגמות הנוכחיות, צפויה עלייה בתנועה בשעה הקרובה."
        if trend == "down":
            return "על פי המגמות הנוכחיות, צפויה ירידה בתנועה בשעה הקרובה."
        return "אין עדיין מגמה ברורה מספיק כדי לתת תחזית אמינה לשעה הקרובה."
 
    # --- Summary ---
    if has("סיכום", "סכם", "תסכם", "מצב", "מה קורה", "summar", "overview"):
        today_tam, today_som = _today_running_totals()
        inside = int(telemetry.get("inside", 0))
        avg_stay = telemetry.get("average_stay_time")
        hourly_tam = st.session_state["cw_hourly_tam"]
        past = {h: c for h, c in hourly_tam.items() if h <= current_hour and c > 0}
        busiest = max(past, key=past.get) if past else None
        parts = [f"עד כה היום: {today_tam} עוברים ושבים, {today_som} כניסות"]
        if today_tam > 0:
            parts.append(f"שיעור כניסה של {today_som / today_tam * 100:.0f}%")
        if busiest is not None:
            parts.append(f"השעה העמוסה ביותר הייתה {busiest:02d}:00 ({past[busiest]} מבקרים)")
        parts.append(f"{inside} אנשים כרגע בחנות")
        if avg_stay:
            parts.append(f"זמן שהייה ממוצע {float(avg_stay) / 60:.1f} דקות")
        cmp_y = compare_today_vs_yesterday(store_id)
        if cmp_y:
            direction = "גבוהה" if cmp_y["pct"] >= 0 else "נמוכה"
            parts.append(f"התנועה {direction} ב-{pct(cmp_y['pct'])} לעומת אתמול בנקודה זו")
        return "; ".join(parts) + "."
 
    return (
        "אפשר לשאול אותי דברים כמו: \"השווה את היום לאתמול\", \"מה הייתה השעה העמוסה ביותר?\", "
        "\"מה סטטוס המצלמות?\", \"כמה אנשים בחנות עכשיו?\", \"למה שיעור הכניסה נמוך?\", "
        "\"אילו שעות דורשות יותר עובדים?\" או \"סכם את הביצועים של היום\" — הכול מחושב מנתונים חיים "
        "והיסטוריים אמיתיים של החנות בלבד."
    )
 
 
AI_PANEL_WIDTH = 350   # compact Copilot panel width in px
 
 
def _inject_ai_panel_theme(dark_mode: bool, is_open: bool) -> None:
    panel_bg = "#0d1117" if dark_mode else "#ffffff"
    border = "#1c2128" if dark_mode else "#e2e2e2"
    text = "#e6edf3" if dark_mode else "#0d1117"
    subtext = "#7d8590" if dark_mode else "#57606a"
    bubble_bg = "rgba(255,255,255,0.05)" if dark_mode else "rgba(0,0,0,0.04)"
    user_bubble_bg = "rgba(61,220,151,0.16)"
    # Only reserve space on the right while the panel is open, so a closed
    # panel never leaves the dashboard feeling empty or pushed around.
    reserve = f"{AI_PANEL_WIDTH + 24}px" if is_open else "0px"
 
    st.markdown(
        f"""
        <style>
            [data-testid="stAppViewContainer"] > .main,
            section[data-testid="stMain"] {{
                padding-right: {reserve};
                transition: padding-right 0.25s ease;
            }}
 
            /* ---- Collapsed launcher (floating round button) ---- */
            .st-key-cw_ai_launcher {{
                position: fixed !important;
                right: 22px; bottom: 22px; z-index: 1000;
                width: auto;
            }}
            .st-key-cw_ai_launcher button {{
                border-radius: 999px !important;
                background: linear-gradient(135deg, #3ddc97, #58a6ff) !important;
                color: #04240f !important;
                font-weight: 800 !important;
                border: none !important;
                box-shadow: 0 12px 30px rgba(0,0,0,0.35) !important;
                padding: 12px 20px !important;
            }}
 
            /* ---- Open Copilot panel ---- */
            .st-key-cw_ai_panel {{
                position: fixed !important;
                top: 14px;
                right: 14px;
                width: {AI_PANEL_WIDTH}px;
                height: calc(100vh - 28px) !important;
                z-index: 999;
                background: {panel_bg};
                border: 1px solid {border};
                border-radius: 18px;
                box-shadow: -10px 0 40px rgba(0,0,0,0.28);
                overflow: hidden;
                direction: rtl;
                animation: cw-ai-slide 0.3s cubic-bezier(0.16, 1, 0.3, 1) both;
            }}
            @keyframes cw-ai-slide {{
                from {{ opacity: 0; transform: translateX(24px); }}
                to   {{ opacity: 1; transform: translateX(0); }}
            }}
 
            .cw-ai-header {{
                display: flex; align-items: center; gap: 10px;
                padding: 14px 16px 12px 16px;
                background: {panel_bg};
                border-bottom: 1px solid {border};
            }}
            .cw-ai-avatar {{
                width: 38px; height: 38px; border-radius: 10px; flex: 0 0 auto;
                object-fit: contain; background: #0a0d12;
                border: 1px solid rgba(255,255,255,0.06); padding: 3px;
            }}
            .cw-ai-title {{ font-size: 0.98rem; font-weight: 800; color: {text}; }}
            .cw-ai-subtitle {{
                font-size: 0.72rem; color: {subtext};
                display: flex; align-items: center; gap: 5px; margin-top: 1px;
            }}
            .cw-ai-live-dot {{
                width: 7px; height: 7px; border-radius: 50%; background: #3ddc97;
                display: inline-block; animation: cw-ai-pulse 1.6s ease-in-out infinite;
            }}
            @keyframes cw-ai-pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} }}
 
            /* Close (✕) button in the header */
            .st-key-cw_ai_close_btn button {{
                background: transparent !important;
                border: none !important;
                color: {subtext} !important;
                font-size: 1.1rem !important;
                padding: 2px 8px !important;
                min-height: 0 !important;
            }}
            .st-key-cw_ai_close_btn button:hover {{ color: {text} !important; }}
 
            .cw-ai-bubble {{
                padding: 9px 12px;
                border-radius: 12px;
                background: {bubble_bg};
                margin: 8px 14px;
                font-size: 0.85rem;
                color: {text};
                line-height: 1.5;
                border-right: 3px solid var(--cw-ai-accent, {subtext});
                text-align: right;
                animation: cw-fade-in-up 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
            }}
            .cw-ai-bubble-user {{
                background: {user_bubble_bg};
                margin-right: 46px;
                margin-left: 14px;
                border-right: none;
                border-left: 3px solid #3ddc97;
                color: {text};
            }}
            .cw-ai-badge {{
                font-size: 0.62rem; letter-spacing: 0.05em;
                font-weight: 800; margin-bottom: 3px; display: block;
            }}
            .cw-ai-time {{
                font-size: 0.64rem; color: {subtext}; margin-top: 5px; display: block;
            }}
 
            /* ---- Chat input: readable black text on a high-contrast field ---- */
            .st-key-cw_ai_panel [data-testid="stChatInput"],
            .st-key-cw_ai_panel [data-baseweb="textarea"] {{
                background: #ffffff !important;
                border: 1px solid #c9d1d9 !important;
                border-radius: 12px !important;
            }}
            .st-key-cw_ai_panel [data-testid="stChatInput"] textarea {{
                color: #0d1117 !important;
                background: #ffffff !important;
                -webkit-text-fill-color: #0d1117 !important;
                caret-color: #0d1117 !important;
                text-align: right;
                direction: rtl;
            }}
            .st-key-cw_ai_panel [data-testid="stChatInput"] textarea::placeholder {{
                color: #6e7781 !important;
                -webkit-text-fill-color: #6e7781 !important;
            }}
            .st-key-cw_ai_panel [data-testid="stChatInputSubmitButton"] svg {{
                fill: #0d1117 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
 
 
def render_ai_assistant_panel(
    store_id: str, store_display_name: str, telemetry: Dict[str, Any], dark_mode: bool
) -> None:
    """Corewise's per-store AI Copilot — a small, collapsible Hebrew side panel.
 
    It is branded per store (logo + "AI <store>"), proactively narrates real
    analytics as the day unfolds, and answers free-text questions using only
    this store's live and historical data. It can be opened and closed and,
    when closed, collapses to a small floating launcher so it never covers the
    dashboard."""
    is_open = bool(st.session_state.get("cw_ai_open", True))
    _inject_ai_panel_theme(dark_mode, is_open)
    _maybe_greet(store_id, store_display_name)
    _maybe_emit_ai_messages(store_id, telemetry)
 
    if not is_open:
        with st.container(key="cw_ai_launcher"):
            if st.button(f"💬  AI {store_display_name}", key="cw_ai_open_btn"):
                st.session_state["cw_ai_open"] = True
                st.rerun()
        return
 
    store = get_store(store_id)
    logo_uri = _logo_data_uri(str(store.logo_path), store_display_name) if store else ""
 
    with st.container(key="cw_ai_panel", height=900):
        head_l, head_r = st.columns([6, 1])
        with head_l:
            st.markdown(
                f"""
                <div class="cw-ai-header">
                    <img class="cw-ai-avatar" src="{logo_uri}"/>
                    <div>
                        <div class="cw-ai-title">AI {store_display_name}</div>
                        <div class="cw-ai-subtitle"><span class="cw-ai-live-dot"></span> עוקב אחר {store_display_name} בזמן אמת</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with head_r:
            if st.button("✕", key="cw_ai_close_btn", help="סגירה"):
                st.session_state["cw_ai_open"] = False
                st.rerun()
 
        for msg in st.session_state["cw_ai_messages"]:
            ts = msg["time"].strftime("%H:%M")
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="cw-ai-bubble cw-ai-bubble-user">{msg["text"]}'
                    f'<span class="cw-ai-time">{ts}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                meta = AI_TYPE_META.get(msg["type"], AI_TYPE_META["insight"])
                st.markdown(
                    f'<div class="cw-ai-bubble" style="--cw-ai-accent:{meta["color"]}">'
                    f'<span class="cw-ai-badge" style="color:{meta["color"]}">{meta["icon"]} {meta["label"]}</span>'
                    f'{msg["text"]}'
                    f'<span class="cw-ai-time">{ts}</span></div>',
                    unsafe_allow_html=True,
                )
 
        question = st.chat_input("שאלו אותי על ביצועי החנות היום…", key="cw_ai_chat_input")
        if question:
            now = datetime.now()
            st.session_state["cw_ai_messages"].append({"role": "user", "type": "question", "text": question, "time": now})
            answer = answer_ai_question(question, store_id, telemetry)
            st.session_state["cw_ai_messages"].append({"role": "assistant", "type": "insight", "text": answer, "time": datetime.now()})
            st.rerun()
 
 
def render_sidebar(client: CorewiseClient, state: Dict[str, Any], workspace: StoreWorkspace) -> None:
    st.sidebar.markdown('<div class="corewise-header">COREWISE</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="corewise-subheader">פלטפורמת בינה קמעונאית</div>', unsafe_allow_html=True)
    st.sidebar.caption(f"חנות: {workspace.store.display_name}")
    st.sidebar.markdown('<div class="cw-divider"></div>', unsafe_allow_html=True)
 
    menu_options = ["Dashboard", "Camera Settings", "Replay System", "Reports", "Settings", "System Health"]
    active_page = session_get_active_page()
    if active_page not in menu_options:
        active_page = "Dashboard"
    st.sidebar.markdown("### תפריט")
    menu_labels_he = {
        "Dashboard": "לוח בקרה",
        "Camera Settings": "הגדרות מצלמות",
        "Replay System": "מערכת שידור חוזר",
        "Reports": "דוחות",
        "Settings": "הגדרות",
        "System Health": "מצב מערכת",
    }
    active_page = st.sidebar.radio(
        "תפריט",
        menu_options,
        index=menu_options.index(active_page),
        format_func=lambda opt: menu_labels_he.get(opt, opt),
        label_visibility="collapsed",
    )
    session_set_active_page(active_page)
 
    if st.sidebar.button("החלפת חנות", width="stretch"):
        session_sign_out_current_store()
        st.rerun()

    # NOTE: Store Search now lives on the app landing page (the store
    # selector), not in the sidebar — see render_store_search() called from
    # render_store_selector(). This keeps the in-store dashboard sidebar
    # focused on operating the current store.

    st.sidebar.markdown('<div class="cw-divider"></div>', unsafe_allow_html=True)
 
    dark_mode = st.sidebar.toggle("מצב כהה", value=st.session_state.get("dark_mode", True))
    st.session_state["dark_mode"] = dark_mode
 
    st.sidebar.markdown("### בקרות חיות")
    st.sidebar.caption("טשטוש פנים וראיית לילה מוגדרים כעת לכל מצלמה בנפרד תחת 'הגדרות מצלמות'.")
    camera_running = st.sidebar.toggle("מצלמה פעילה", value=bool(state.get("camera_running", True)))
    detection_paused = st.sidebar.toggle("השהיית זיהוי", value=bool(state.get("detection_paused", False)))
 
    # Face Blur / Night Vision are controlled per-camera now. The engine
    # command still carries these flags for backward compatibility: they are
    # derived from the store's primary live camera's saved settings (single
    # camera engines have exactly one), falling back to off when no camera is
    # connected. This never fabricates a camera — it only reads real settings.
    cameras = get_camera_manager(workspace.store_id, state).list_cameras()
    if cameras:
        primary_conf = get_camera_feature_settings(workspace.store_id, cameras[0].camera_id)
        blur_faces = bool(primary_conf.get("face_blur", False))
        night_vision = bool(primary_conf.get("night_vision", False))
    else:
        blur_faces = bool(state.get("blur_faces", False))
        night_vision = bool(state.get("night_vision", False))
 
    st.sidebar.markdown('<div class="cw-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("### תצוגת וידאו")
    st.sidebar.caption("ברירת המחדל היא תמונה נקייה לגמרי. הפעילו רק את מה שאתם רוצים לראות. אינו משפיע על הזיהוי, הספירה או ההקלטה.")
    show_zones = st.sidebar.toggle(
        "הצגת איזורי כיול", value=st.session_state.get("cw_show_zones", False),
        key="cw_show_zones",
        help="קווי TAM ו-SOM על התמונה. כבו לתצוגה נקייה לגמרי.",
    )
    show_labels = st.sidebar.toggle(
        "הצגת תיבות ותוויות", value=st.session_state.get("cw_show_labels", False),
        key="cw_show_labels",
        help="תיבות סביב אנשים, מזהה וזמן שהייה.",
    )
    show_overlay = st.sidebar.toggle(
        "הצגת פס נתונים", value=st.session_state.get("cw_show_overlay", False),
        key="cw_show_overlay",
        help="פס TAM/SAM/SOM בתחתית התמונה.",
    )

    st.sidebar.markdown('<div class="cw-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("### מצב אנליטיקה")
    sam_settings = get_sam_min_stay_settings(workspace.store_id)
    analytics_mode = st.sidebar.radio(
        "מצב אנליטיקה",
        ["tam_sam_som", "tam_som"],
        index=0 if st.session_state.get("analytics_mode", "tam_sam_som") == "tam_sam_som" else 1,
        format_func=lambda m: "TAM + SAM + SOM" if m == "tam_sam_som" else "TAM + SOM (ללא איזור SAM נפרד)",
        label_visibility="collapsed",
        help=(
            "TAM+SAM+SOM: כל שלושת האיזורים. TAM+SOM: SAM מחושב אוטומטית לפי זמן שהייה ב-TAM "
            f"({sam_settings['seconds']} שנ׳, ניתן לשינוי למטה)."
        ),
    )
    st.session_state["analytics_mode"] = analytics_mode

    # --- SAM minimum stay time — only meaningful in TAM+SOM mode, where SAM
    # is derived from dwell time instead of its own polygon. Never hardcoded:
    # persisted per store and sent to the engine with every control update.
    if analytics_mode == "tam_som":
        st.sidebar.markdown("### הגדרות זמן שהייה ל-SAM")
        sam_enabled = st.sidebar.toggle(
            "הפעלת זמן שהייה מינימלי",
            value=sam_settings["enabled"],
            help="כאשר כבוי, מבקר נחשב SAM מיד עם הכניסה ל-TAM, ללא דרישת זמן שהייה.",
            key="cw_sam_min_stay_enabled_toggle",
        )
        sam_seconds = sam_settings["seconds"]
        if sam_enabled:
            sam_seconds = st.sidebar.number_input(
                "זמן שהייה מינימלי (שניות)",
                min_value=0.0,
                max_value=3600.0,
                value=float(sam_settings["seconds"]),
                step=0.1,
                format="%.2f",
                help="כל ערך עשרוני חיובי אפשרי — לדוגמה 0, 0.5, 1.2, 2.75, 3, 10, 30, 120.",
                key="cw_sam_min_stay_seconds_input",
            )
        if sam_enabled != sam_settings["enabled"] or float(sam_seconds) != sam_settings["seconds"]:
            save_sam_min_stay_settings(workspace.store_id, sam_enabled, float(sam_seconds))
        sam_settings = {"enabled": sam_enabled, "seconds": float(sam_seconds)}

    st.sidebar.markdown("### הגדרות בינה מלאכותית")
    current_model_file = state.get("model", "yolov8n.pt")
    current_label = next(
        (label for label, fname in MODEL_OPTIONS.items() if fname == current_model_file),
        list(MODEL_OPTIONS.keys())[0],
    )
    model_label = st.sidebar.selectbox(
        "איכות מודל הבינה המלאכותית",
        list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(current_label),
        help="מודלים גדולים יותר מדויקים יותר בתאורה חלשה, הסתרות וצפיפות, במחיר מהירות.",
    )
    model = MODEL_OPTIONS[model_label]
 
    confidence = st.sidebar.slider(
        "סף ביטחון", min_value=0.1, max_value=0.95,
        value=float(state.get("confidence", 0.5)), step=0.05,
    )

    # --- Human Skeleton (Pose) — optional module, per the spec -----------
    st.sidebar.markdown("### שלד אנושי (Pose)")
    pose_settings = get_pose_settings(workspace.store_id)
    pose_enabled = st.sidebar.toggle(
        "זיהוי שלד אנושי",
        value=pose_settings["enabled"],
        key="cw_pose_enabled_toggle",
        help=(
            "כבוי: מודל ה-Pose לא נטען כלל ואין שום עלות GPU/CPU. "
            "פעיל: המנוע טוען מודל Pose, מזהה 17 נקודות שלד "
            "(ראש, כתפיים, מרפקים, שורשי כף יד, ירכיים, ברכיים, קרסוליים) "
            "ומצמיד אותן לזהות הגלובלית של האדם."
        ),
    )
    pose_model = pose_settings["model"]
    if pose_enabled:
        _pose_label = next((l for l, f in POSE_MODEL_OPTIONS.items()
                            if f == pose_model), list(POSE_MODEL_OPTIONS.keys())[0])
        _pose_label = st.sidebar.selectbox(
            "מודל שלד", list(POSE_MODEL_OPTIONS.keys()),
            index=list(POSE_MODEL_OPTIONS.keys()).index(_pose_label),
            key="cw_pose_model_select",
        )
        pose_model = POSE_MODEL_OPTIONS[_pose_label]
    if (pose_enabled != pose_settings["enabled"]
            or pose_model != pose_settings["model"]):
        save_pose_settings(workspace.store_id, pose_enabled, pose_model)

    # Build calibration payload — take the primary camera's saved calibration
    # and send it with the command so the engine never reads from disk files.
    calib_payload: Optional[Dict[str, Any]] = None
    if cameras:
        primary_identity = get_camera_identity(cameras[0].camera_id)
        if primary_identity and primary_identity.calibration.is_complete():
            c = primary_identity.calibration
            calib_payload = {
                "tam_area": c.tam_area,
                "sam_area": c.sam_area,
                "som_line": c.som_line,
            }

    new_control = {
        "blur_faces": blur_faces,
        "night_vision": night_vision,
        "camera_running": camera_running,
        "detection_paused": detection_paused,
        "model": model,
        "confidence": confidence,
        "analytics_mode": analytics_mode,
        "calibration": calib_payload,
        # New, additive fields — an engine that doesn't read these yet keeps
        # working exactly as before; once the backend is integrated it can
        # start honoring them without any other payload shape changing.
        "sam_min_stay_enabled": sam_settings["enabled"],
        "sam_min_stay_seconds": sam_settings["seconds"],
        # Human skeleton module — OFF means the engine never loads the model.
        "pose_enabled": bool(pose_enabled),
        "pose_model": pose_model,
        # Cosmetic display switches - they change only what is DRAWN on the
        # frame, never detection, counting or recording.
        "show_zones": show_zones,
        "show_labels": show_labels,
        "show_overlay": show_overlay,
    }
    if new_control != st.session_state.get("last_sent_control"):
        client.send("command", new_control)
        st.session_state["last_sent_control"] = new_control
 
 
def render_settings_page(camera_manager: CameraManager, store_id: str, store_display_name: str, telemetry: Dict[str, Any]) -> None:
    """Comprehensive Settings page with tabs for SAM, Camera, AI, Analytics, Replay, Dashboard, Storage."""
    st.markdown('<div class="cw-cams-header">הגדרות מערכת</div>', unsafe_allow_html=True)
    tabs = st.tabs(["SAM", "Camera", "AI", "Analytics", "Replay", "Dashboard", "Storage"])

    # SAM tab
    with tabs[0]:
        st.markdown("### הגדרות SAM (לפי חנות)")
        sam = get_sam_min_stay_settings(store_id)
        enabled = st.toggle("הפעלת זמן שהייה מינימלי", value=sam["enabled"], key=f"settings_sam_enabled_{store_id}")
        seconds = st.number_input("זמן שהייה מינימלי (שניות)", min_value=0.0, value=float(sam["seconds"]), step=0.1, key=f"settings_sam_seconds_{store_id}")
        if st.button("שמור הגדרות SAM", key=f"settings_sam_save_{store_id}"):
            save_sam_min_stay_settings(store_id, bool(enabled), float(seconds))
            st.success("הגדרות SAM נשמרו")

    # Camera tab
    with tabs[1]:
        st.markdown("### הגדרות מצלמה")
        cams = camera_manager.list_cameras()
        if not cams:
            st.info("אין מצלמות זמינות לעדכון הגדרות.")
        for cam in cams:
            ident = get_camera_identity(cam.camera_id)
            name = st.text_input("שם מצלמה", value=ident.name if ident else cam.label, key=f"settings_cam_name_{cam.camera_id}")
            desc = st.text_area("תיאור", value=ident.description if ident else "", key=f"settings_cam_desc_{cam.camera_id}")
            loc = st.text_input("מיקום פיזי", value=ident.location if ident else "", key=f"settings_cam_loc_{cam.camera_id}")
            cam_type = st.selectbox("סוג מצלמה", CAMERA_TYPES, index=CAMERA_TYPES.index(ident.camera_type) if ident and ident.camera_type in CAMERA_TYPES else 0, key=f"settings_cam_type_{cam.camera_id}")
            cols = st.columns(2)
            with cols[0]:
                if st.button("שמירת זהות מצלמה", key=f"settings_cam_save_{cam.camera_id}"):
                    new_identity = CameraIdentity(
                        camera_id=cam.camera_id,
                        store_id=ident.store_id if ident else store_id,
                        name=str(name),
                        camera_type=str(cam_type),
                        description=str(desc),
                        location=str(loc),
                        calibration=ident.calibration if ident else CameraCalibration(),
                        disconnected=bool(ident.disconnected) if ident else False,
                        created_at=ident.created_at if ident else "",
                    )
                    save_camera_identity(new_identity)
                    st.success("זהות המצלמה נשמרה.")
            with cols[1]:
                if st.button("איפוס הכיול למחדל", key=f"settings_cam_resetcal_{cam.camera_id}"):
                    delete_camera_calibration(cam.camera_id)
                    st.success("כיול מאופס.")

    # AI tab
    with tabs[2]:
        st.markdown("### הגדרות בינה מלאכותית")
        current_model = telemetry.get("model", "yolov8n.pt")
        model_label = next((label for label, fname in MODEL_OPTIONS.items() if fname == current_model), list(MODEL_OPTIONS.keys())[0])
        new_model_label = st.selectbox("מודל ברירת מחדל", list(MODEL_OPTIONS.keys()), index=list(MODEL_OPTIONS.keys()).index(model_label), key=f"settings_ai_model_{store_id}")
        new_conf = st.slider("סף ביטחון (ברירת מחדל)", min_value=0.1, max_value=0.95, value=float(telemetry.get("current_confidence", 0.5)), step=0.05, key=f"settings_ai_conf_{store_id}")
        if st.button("שמור הגדרות AI", key=f"settings_ai_save_{store_id}"):
            # Push to engine via control command
            client = get_client()
            client.send("command", {"model": MODEL_OPTIONS[new_model_label], "confidence": float(new_conf)})
            st.success("הגדרות AI נשלחו למנוע.")

        st.markdown("### שלד אנושי (Pose)")
        st.caption(
            "מודול אופציונלי. כבוי — מודל ה-Pose לא נטען ואין עלות משאבים. "
            "פעיל — המנוע מזהה 17 נקודות שלד לכל אדם ומצמיד אותן לזהות "
            "הגלובלית שלו (Global Person ID), לא לתיבה זמנית."
        )
        _pose = get_pose_settings(store_id)
        _pose_enabled = st.toggle("הפעלת זיהוי שלד", value=_pose["enabled"],
                                  key=f"settings_pose_enabled_{store_id}")
        _pose_label = next((l for l, f in POSE_MODEL_OPTIONS.items()
                            if f == _pose["model"]),
                           list(POSE_MODEL_OPTIONS.keys())[0])
        _pose_label = st.selectbox("מודל שלד", list(POSE_MODEL_OPTIONS.keys()),
                                   index=list(POSE_MODEL_OPTIONS.keys()).index(_pose_label),
                                   key=f"settings_pose_model_{store_id}")
        if st.button("שמור הגדרות שלד", key=f"settings_pose_save_{store_id}"):
            save_pose_settings(store_id, bool(_pose_enabled),
                               POSE_MODEL_OPTIONS[_pose_label])
            get_client().send("command", {
                "pose_enabled": bool(_pose_enabled),
                "pose_model": POSE_MODEL_OPTIONS[_pose_label]})
            st.success("הגדרות השלד נשמרו ונשלחו למנוע.")
        if telemetry.get("pose_enabled"):
            st.info("סטטוס מנוע: מודל השלד טעון ופעיל" if telemetry.get("pose_model_loaded")
                    else "סטטוס מנוע: הפעלה התקבלה, המודל בטעינה...")

    # Analytics tab
    with tabs[3]:
        st.markdown("### הגדרות אנליטיקה")
        mode = st.selectbox("מצב אנליטיקה", ["tam_sam_som", "tam_som"], format_func=lambda m: "TAM + SAM + SOM" if m == "tam_sam_som" else "TAM + SOM", index=0 if st.session_state.get("analytics_mode", "tam_sam_som") == "tam_sam_som" else 1, key=f"settings_analytics_mode_{store_id}")
        if st.button("שמור הגדרות אנליטיקה", key=f"settings_analytics_save_{store_id}"):
            st.session_state["analytics_mode"] = mode
            client = get_client()
            client.send("command", {"analytics_mode": mode})
            st.success("הגדרות אנליטיקה נשמרו ונשלחו למנוע.")

    # Replay tab
    with tabs[4]:
        st.markdown("### הגדרות שידור חוזר")
        replay = _load_replay_settings(store_id)
        rec_mode = st.selectbox("מצב הקלטה", ["continuous", "motion"], index=0 if replay.get("recording_mode", "continuous") == "continuous" else 1, key=f"settings_replay_mode_{store_id}")
        video_quality = st.selectbox("איכות וידאו", ["low", "medium", "high"], index=["low","medium","high"].index(replay.get("video_quality","medium")), key=f"settings_replay_quality_{store_id}")
        if st.button("שמור הגדרות שידור חוזר", key=f"settings_replay_save_{store_id}"):
            _save_replay_settings(store_id, {"recording_mode": rec_mode, "video_quality": video_quality})
            st.success("הגדרות שידור חוזר נשמרו.")

    # Dashboard tab
    with tabs[5]:
        st.markdown("### הגדרות לוח בקרה")
        st.info("ממשק ניהול לוח הבקרה — הגדרות מצורפות כאן (למשל שפה, תצוגה) — כרגע גלויה לצפייה בלבד.")

    # Storage tab
    with tabs[6]:
        st.markdown("### הגדרות אחסון")
        st.caption("מיקום הקלטות ושימוש בדיסק")
        try:
            import recording_paths as _rp
            root = st.text_input("נתיב אחסון הקלטות", value=str(_rp.RECORDINGS_DIR), key=f"settings_storage_root_{store_id}")
            if st.button("בדוק ומירה", key=f"settings_storage_save_{store_id}"):
                st.success("הגדרות האחסון נשמרו זמנית (דרוש יישום ספציפי למנוע).")
        except Exception:
            st.info("מודול storage אינו זמין כרגע.")


def _replay_settings_file(store_id: str):
    path = Path(__file__).parent / "data" / "replay_settings"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{store_id}.json"


def _load_replay_settings(store_id: str) -> Dict[str, Any]:
    p = _replay_settings_file(store_id)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_replay_settings(store_id: str, data: Dict[str, Any]) -> None:
    p = _replay_settings_file(store_id)
    try:
        tmp = p.with_suffix('.tmp')
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(p)
    except Exception:
        pass


def _report_period_label(period: str) -> str:
    return {"daily": "יומי", "weekly": "שבועי", "monthly": "חודשי"}.get(period, period)


def _render_report_body(report: Dict[str, Any]) -> None:
    """Render one report's metrics using the dashboard's existing cards."""
    metrics = report.get("metrics", {}) or {}
    rng = report.get("range", {}) or {}

    start, end = rng.get("start"), rng.get("end")
    if start and end and start != end:
        st.caption(f"טווח: {start} — {end} · נוצר: {report.get('generated_at', '—')}")
    elif start:
        st.caption(f"תאריך: {start} · נוצר: {report.get('generated_at', '—')}")

    cols = st.columns(4)
    with cols[0]:
        metric_card("מבקרים", str(int(metrics.get("total_visitors", 0) or 0)), accent=True)
    with cols[1]:
        metric_card("TAM", str(int(metrics.get("tam", 0) or 0)))
    with cols[2]:
        metric_card("SAM", str(int(metrics.get("sam", 0) or 0)))
    with cols[3]:
        metric_card("SOM", str(int(metrics.get("som", 0) or 0)))

    cols2 = st.columns(3)
    with cols2[0]:
        metric_card("אחוז המרה", f"{float(metrics.get('conversion_rate', 0) or 0):.1f}%")
    with cols2[1]:
        stay = metrics.get("avg_stay_time")
        metric_card("זמן שהייה ממוצע", f"{float(stay):.1f} שנ׳" if stay is not None else "—")
    with cols2[2]:
        metric_card("ימים בדוח", str(int(metrics.get("days_counted", 0) or 0)))

    busy = metrics.get("busy_hours") or []
    st.markdown("#### שעות עומס")
    if busy:
        st.write(" · ".join(f"{int(b['hour']):02d}:00 ({int(b['value'])})" for b in busy))
    else:
        st.write("אין מספיק נתונים לשעות עומס.")

    hourly_tam = metrics.get("hourly_tam") or {}
    if any(int(v or 0) for v in hourly_tam.values()):
        st.markdown("#### פעילות לקוחות לפי שעה (TAM)")
        df = pd.DataFrame(
            {"hour": [f"{h:02d}:00" for h in range(24)],
             "tam": [int(hourly_tam.get(str(h), 0) or 0) for h in range(24)]}
        )
        chart = (
            alt.Chart(df)
            .mark_bar(color="#3ddc97")
            .encode(x=alt.X("hour:N", title="שעה"), y=alt.Y("tam:Q", title="מבקרים חדשים"))
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)

    # Export scaffolding — CSV download is wired now; future formats (PDF /
    # email) can reuse the saved structured report without regeneration.
    try:
        csv_text = reports_mod.report_to_csv(report)
        st.download_button(
            "ייצוא CSV",
            data=csv_text.encode("utf-8"),
            file_name=f"{report.get('store_id','store')}_{report.get('period','')}_{report.get('key','')}.csv",
            mime="text/csv",
            key=f"cw_report_csv_{report.get('period','')}_{report.get('key','')}",
        )
    except Exception:
        pass


def render_reports_page(store_id: str, store_display_name: str) -> None:
    """Automatic Reports page: Daily / Weekly / Monthly (Task 4).

    Reports are generated automatically from persisted daily history and
    saved to disk, so previous reports can always be reopened. This function
    triggers generation for any newly-complete periods on each visit, then
    lets the user browse saved reports.
    """
    st.markdown('<div class="cw-cams-header">דוחות</div>', unsafe_allow_html=True)
    st.caption(f"חנות: {store_display_name}")

    # Auto-generate any missing reports for completed periods.
    try:
        created = reports_mod.ensure_reports_up_to_date(store_id)
        if created:
            st.toast(f"נוצרו {created} דוחות חדשים", icon="🗂")
    except Exception:
        # Report generation must never break the page; expected failure is
        # simply "no history yet".
        created = 0

    tabs = st.tabs(["יומי", "שבועי", "חודשי"])
    for tab, period in zip(tabs, reports_mod.PERIODS):
        with tab:
            keys = reports_mod.list_reports(store_id, period)
            if not keys:
                st.info(
                    "אין עדיין דוחות זמינים לתקופה זו. דוחות נוצרים אוטומטית "
                    "בסיום כל תקופה (יום/שבוע/חודש) מרגע שנאספים נתונים."
                )
                # Allow generating a partial report for the current period on
                # demand, if there is any data for it.
                if st.button(
                    "צור דוח לתקופה הנוכחית",
                    key=f"cw_report_gen_now_{period}",
                ):
                    today = date.today()
                    key = {
                        "daily": today.isoformat(),
                        "weekly": reports_mod._week_key(today),
                        "monthly": reports_mod._month_key(today),
                    }[period]
                    rep = reports_mod.generate_report_now(store_id, period, key)
                    if rep:
                        st.rerun()
                    else:
                        st.warning("אין עדיין נתונים לתקופה הנוכחית.")
                continue

            selected = st.selectbox(
                f"בחר דוח {_report_period_label(period)}",
                keys,
                key=f"cw_report_select_{period}",
            )
            report = reports_mod.load_report(store_id, period, selected)
            if report is None:
                st.warning("לא ניתן לטעון את הדוח שנבחר.")
                continue
            _render_report_body(report)


def _system_resource_stats() -> Dict[str, Optional[float]]:
    """CPU/RAM/Disk percentages, each None if that specific probe is
    unavailable. Never raises and never blocks the render loop: psutil's
    cpu_percent is called non-blocking (interval=None) so the page stays
    responsive under the dashboard's fast auto-refresh."""
    stats: Dict[str, Optional[float]] = {"cpu": None, "ram": None, "disk": None}
    try:
        import psutil  # optional dependency
        # interval=None -> non-blocking; returns usage since the previous
        # call. We prime it once per session so the first reading is real.
        if not st.session_state.get("_cw_cpu_primed"):
            psutil.cpu_percent(interval=None)
            st.session_state["_cw_cpu_primed"] = True
        stats["cpu"] = float(psutil.cpu_percent(interval=None))
        stats["ram"] = float(psutil.virtual_memory().percent)
        stats["disk"] = float(psutil.disk_usage(".").percent)
    except Exception:
        # psutil missing or a probe failed — fall back to disk only via
        # the stdlib, leaving the rest as safe placeholders.
        try:
            import shutil
            total, used, _free = shutil.disk_usage(".")
            stats["disk"] = round(used / total * 100.0, 1) if total else None
        except Exception:
            pass
    return stats


def _gpu_stats() -> list[Dict[str, str]]:
    """Best-effort GPU utilization via nvidia-smi. Empty list if no GPU /
    nvidia-smi (an expected, non-error condition on most stores)."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
    except Exception:
        return []
    if res.returncode != 0 or not res.stdout.strip():
        return []
    gpus = []
    for line in res.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3:
            gpus.append({"util": parts[0], "mem_used": parts[1], "mem_total": parts[2]})
    return gpus


def render_system_health_page(store_id: str, telemetry: Dict[str, Any], connected: bool = False) -> None:
    """System Status page.

    Renders fully whether or not telemetry is available and whether or not
    the WebSocket is connected. Every metric has a safe placeholder, so the
    page can never go blank. Only expected failures (missing psutil, no
    GPU, offline engine) are handled — unexpected errors are allowed to
    surface via Streamlit rather than being hidden.
    """
    st.markdown('<div class="cw-cams-header">מצב מערכת</div>', unsafe_allow_html=True)

    telemetry = telemetry or {}

    # --- Connection banner (always rendered) ---
    if connected:
        link_label, link_accent = "מחובר", True
    else:
        link_label, link_accent = "מנותק", False
    last_update = telemetry.get("last_update")
    if last_update:
        try:
            age = max(0, int(time.time() - float(last_update)))
            freshness = f"עודכן לפני {age} שנ׳"
        except Exception:
            freshness = "—"
    else:
        freshness = "אין נתוני טלמטריה"

    conn_cols = st.columns(3)
    with conn_cols[0]:
        metric_card("חיבור לשרת", link_label, accent=link_accent)
    with conn_cols[1]:
        cam_status = telemetry.get("camera_status", "offline")
        status_he = {"online": "מקוונת", "paused": "מושהית", "offline": "לא מקוונת"}.get(cam_status, cam_status)
        metric_card("מצב מצלמה", status_he)
    with conn_cols[2]:
        metric_card("טלמטריה", freshness)

    if not connected:
        st.warning("החיבור לשרת נותק. הנתונים המוצגים הם האחרונים שהתקבלו.", icon="🔌")

    # --- Host resources (always rendered, placeholders when unavailable) ---
    st.markdown("### משאבי מערכת")
    stats = _system_resource_stats()

    def _pct(value: Optional[float]) -> str:
        return f"{value:.0f}%" if value is not None else "לא זמין"

    res_cols = st.columns(3)
    with res_cols[0]:
        metric_card("CPU", _pct(stats["cpu"]))
    with res_cols[1]:
        metric_card("RAM", _pct(stats["ram"]))
    with res_cols[2]:
        metric_card("דיסק", _pct(stats["disk"]))

    if all(v is None for v in stats.values()):
        st.caption("סטטיסטיקות מערכת אינן זמינות בסביבה זו (מומלץ להתקין psutil).")

    # --- GPU (only shown if present) ---
    gpus = _gpu_stats()
    if gpus:
        st.markdown("### GPU")
        for idx, gpu in enumerate(gpus):
            g_cols = st.columns(2)
            with g_cols[0]:
                metric_card(f"GPU {idx} ניצול", f"{gpu['util']}%")
            with g_cols[1]:
                metric_card(f"GPU {idx} זיכרון", f"{gpu['mem_used']} / {gpu['mem_total']} MB")

    # --- Live engine metrics (safe defaults) ---
    st.markdown("### נתוני מנוע חיים")
    eng_cols = st.columns(3)
    with eng_cols[0]:
        fps = telemetry.get("fps", 0) or 0
        metric_card("FPS", f"{float(fps):.1f}")
    with eng_cols[1]:
        metric_card("בפנים כעת", str(int(telemetry.get("inside", 0) or 0)))
    with eng_cols[2]:
        metric_card("מבקרים היום", str(int(telemetry.get("people_today", 0) or 0)))

    # --- Per-camera health (safe when telemetry has no camera list) ---
    st.markdown("### מצב מצלמות")
    cameras = telemetry.get("cameras")
    if isinstance(cameras, list) and cameras:
        for cam in cameras:
            label = cam.get("label") or cam.get("camera_id") or "מצלמה"
            status = cam.get("status", "unknown")
            st.write(f"• {label}: {status}")
    else:
        primary = telemetry.get("camera_status", "offline")
        primary_he = {"online": "מקוונת", "paused": "מושהית", "offline": "לא מקוונת"}.get(primary, primary)
        st.write(f"• מצלמה ראשית: {primary_he}")


def render_status_row(telemetry: Dict[str, Any], connected: bool) -> None:
    camera_status = telemetry.get("camera_status", "offline")
    status_color = {"online": "#3ddc97", "paused": "#e3b341", "offline": "#f85149"}.get(camera_status, "#f85149")
 
    status_labels_he = {"online": "מקוונת", "paused": "מושהית", "offline": "לא מקוונת"}
    status_label = status_labels_he.get(camera_status, camera_status.upper())

    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f'<div class="cw-card"><span class="cw-status-dot" style="background:{status_color}"></span>'
            f'<span class="cw-metric-label">מצלמה</span><br>'
            f'<span class="cw-metric-value-small">{status_label}</span></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        metric_card("FPS", f"{telemetry.get('fps', 0):.1f}")
    with cols[2]:
        link_status = "פעיל" if connected else "מתחבר מחדש"
        metric_card("חיבור לשרת", link_status, accent=connected)
    with cols[3]:
        metric_card("שעה", datetime.now().strftime("%H:%M:%S"))
 
 
def render_environment_row(location: Dict[str, Any], weather: Dict[str, Any]) -> None:
    cols = st.columns(2)
    with cols[0]:
        metric_card("מיקום החנות", location["label"])
    with cols[1]:
        temp = f"{weather['temp']}°C" if weather["temp"] is not None else "לא זמין"
        metric_card("מזג אוויר", f"{weather['desc']} · {temp}")
 
 
def render_main_stats(telemetry: Dict[str, Any]) -> None:
    cols = st.columns(4)
    with cols[0]:
        metric_card("TAM", str(telemetry.get("tam", 0)), accent=True)
    with cols[1]:
        metric_card("SAM", str(telemetry.get("sam", 0)))
    with cols[2]:
        metric_card("SOM", str(telemetry.get("som", 0)))
    with cols[3]:
        metric_card("בפנים כעת", str(telemetry.get("inside", 0)))
 
    cols2 = st.columns(3)
    with cols2[0]:
        metric_card("זמן שהייה ממוצע", f"{telemetry.get('average_stay_time', 0):.1f} שנ׳")
    with cols2[1]:
        metric_card("אחוז המרה", f"{telemetry.get('conversion_rate', 0):.1f}%")
    with cols2[2]:
        metric_card("מבקרים היום", str(telemetry.get("people_today", 0)))
 
 
def render_live_view(telemetry: Dict[str, Any]) -> None:
    st.markdown("### תצוגת מצלמה חיה")
    frame_b64 = telemetry.get("live_frame_jpeg_b64")
    if frame_b64:
        try:
            st.image(base64.b64decode(frame_b64), use_container_width=True)
            return
        except Exception:
            pass
    st.info("ממתין לשידור מהמצלמה...")
 
 
# =========================================================================
# MAIN
# =========================================================================
def main() -> None:
    store_id = require_store_selection()
    if store_id is None:
        # Store carousel / password dialog is being shown; the main
        # dashboard must not render (and no store-scoped client/data
        # should be touched) until a store is authenticated.
        return
 
    workspace = build_workspace(store_id)
    st.session_state["cw_current_store_id"] = store_id
 
    client = get_client()
    # The Replay page refreshes ITSELF (the player polls live/index.json once a
    # second). Re-running Streamlit at 250 ms there would re-mount the player's
    # iframe several times a second and destroy the <video> element mid-playback
    # — which is why Replay could not be played at all. Every other page keeps
    # the fast refresh it needs for live telemetry.
    _page_now = session_get_active_page()
    st_autorefresh(
        interval=30000 if _page_now == "Replay System" else REFRESH_INTERVAL_MS,
        key="corewise_autorefresh",
    )
 
    telemetry = client.get_state()
    record_history_point(telemetry)
    record_hourly_point(telemetry)
    inject_theme(st.session_state["dark_mode"])
 
    render_sidebar(client, telemetry, workspace)
 
    active_page = session_get_active_page()
 
    if active_page == "Replay System":
        render_replay_page(
            get_camera_manager(workspace.store_id, telemetry),
            workspace.store.display_name,
            telemetry,
        )
    elif active_page == "Camera Settings":
        render_camera_settings_page(
            get_camera_manager(workspace.store_id, telemetry),
            workspace.store_id,
            workspace.store.display_name,
            telemetry,
        )
    elif active_page == "Settings":
        render_settings_page(
            get_camera_manager(workspace.store_id, telemetry),
            workspace.store_id,
            workspace.store.display_name,
            telemetry,
        )
    elif active_page == "Reports":
        render_reports_page(workspace.store_id, workspace.store.display_name)
    elif active_page == "System Health":
        render_system_health_page(workspace.store_id, telemetry, client.is_connected())
    else:
        # --- Above-the-fold: KPIs + store performance first, so the dashboard
        # never opens on an empty screen. Only layout/order is changed here;
        # the analytics themselves are untouched. ---
        st.markdown('<div class="corewise-header">בינת חנות</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="corewise-subheader">נתוני תנועה והמרה בזמן אמת</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="cw-divider"></div>', unsafe_allow_html=True)
 
        render_status_row(telemetry, client.is_connected())
        render_main_stats(telemetry)
 
        st.write("")
        render_opportunities_section(telemetry)
 
        st.write("")
        st.markdown(
            '<div class="cw-card"><h3>📊 תנועה שעתית</h3>'
            '<p>מבקרים חדשים לפי שעה, היום.</p></div>',
            unsafe_allow_html=True,
        )
        render_hourly_traffic_chart(st.session_state["dark_mode"])
 
        st.write("")
        location = get_location()
        weather = get_weather(location["lat"], location["lon"])
        render_environment_row(location, weather)
 
        st.write("")
        render_live_view(telemetry)
 
        for event in client.pop_events():
            st.toast(f"אדם נכנס · SOM {event.get('som', '')}", icon="🟢")
 
    # The AI Copilot panel is fixed-positioned and overlays the dashboard, so
    # it's rendered last (after the page content) to guarantee the dashboard
    # content itself starts at the very top of the viewport.
    render_ai_assistant_panel(store_id, workspace.store.display_name, telemetry, st.session_state["dark_mode"])
 
 
if __name__ == "__main__":
    main()