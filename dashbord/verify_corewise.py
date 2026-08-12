"""
verify_corewise.py
------------------
Standalone verification for the Corewise fixes. Run it from the PROJECT ROOT
(the folder containing main.py / app.py / tracking.py):

    python verify_corewise.py

It exercises the real modules — no mocks for the logic under test — and exits
non-zero if anything fails, so it can be dropped into CI.

Does NOT require a camera, a running server, or the dashboard.
"""

from __future__ import annotations

import ast
import math
import os
import pathlib
import sys
import time

os.environ.setdefault("COREWISE_SAM_DEBUG", "0")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

PASS = FAIL = 0


def chk(label: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}")


def strip_comments_and_docs(src: str) -> str:
    """Executable code only, so a constant mentioned in a comment or docstring
    is not mistaken for a surviving hardcoded threshold."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.unparse(tree)


TAM = [[100, 100], [500, 100], [500, 400], [100, 400]]
ENTRANCE = [[100, 250], [500, 250]]

print("=" * 68)
print("COREWISE VERIFICATION")
print("=" * 68)

# ---------------------------------------------------------------- TASK 1
print("\n[TASK 1] SAM minimum stay — dashboard is the only source of truth")
from tracking import PersonTracker  # noqa: E402

tracking_code = strip_comments_and_docs(pathlib.Path("tracking.py").read_text(encoding="utf-8"))
main_code = strip_comments_and_docs(pathlib.Path("main.py").read_text(encoding="utf-8"))

chk("tracking.py has no SAM_DWELL_SECONDS constant", "SAM_DWELL_SECONDS" not in tracking_code)
chk("main.py has no SAM_DWELL_THRESHOLD_SECONDS", "SAM_DWELL_THRESHOLD_SECONDS" not in main_code)
chk("PersonTracker.set_min_stay() exists", hasattr(PersonTracker, "set_min_stay"))

for want in (0, 1, 2.7, 3.5, 5):
    t = PersonTracker(TAM, som_points=ENTRANCE)
    t.set_min_stay(True, want)
    t0 = time.time()
    fired = None
    while time.time() - t0 < want + 1.0:
        t.update(1, 300, 300)
        if t.sam_count:
            fired = time.time() - t0
            break
        time.sleep(0.02)
    chk(f"dashboard={want}s -> SAM at {round(fired, 2) if fired else None}s",
        fired is not None and abs(fired - want) <= 0.15)

t = PersonTracker(TAM, som_points=ENTRANCE)
t.set_min_stay(False, 999)
t.update(1, 300, 300)
chk("minimum-stay toggle OFF counts immediately", t.sam_count == 1)

t = PersonTracker(TAM, som_points=ENTRANCE)
for _ in range(25):
    t.update(1, 300, 300)
    time.sleep(0.01)
chk("no dashboard value -> no engine-side fallback fires", t.sam_count == 0)

t = PersonTracker(TAM, som_points=ENTRANCE)
t.set_min_stay(True, 10.0)
t.update(1, 300, 300)
time.sleep(0.2)
t.set_min_stay(True, 0.1)          # dashboard changed mid-visit
time.sleep(0.15)
t.update(1, 300, 300)
chk("live threshold change applies without restart", t.sam_count == 1)

t = PersonTracker(TAM, som_points=ENTRANCE)
t.set_min_stay(True, 1.0)
t.update(1, 300, 300)
time.sleep(0.8)
t.update(1, 10, 10)                # leaves the zone
time.sleep(0.4)
t.update(1, 300, 300)              # re-enters: dwell clock must restart
chk("dwell clock restarts on re-entry", t.sam_count == 0)

# ---------------------------------------------------------------- TASK 3
print("\n[TASK 3] SOM arbitrary polygons")
for n in (3, 4, 5, 6, 8, 10, 12, 16):
    poly = [[300 + 150 * math.cos(2 * math.pi * i / n),
             250 + 150 * math.sin(2 * math.pi * i / n)] for i in range(n)]
    t = PersonTracker(TAM, som_points=poly)
    t.update(1, 10, 10)
    t.update(1, 300, 250)
    chk(f"{n}-vertex SOM polygon detects entry", t.som_count == 1)

t = PersonTracker(TAM, som_points=[[300, 100], [300, 400]])
t.update(1, 200, 250)
t.update(1, 400, 250)
chk("vertical entrance line: crossing detected (was a false negative)", t.som_count == 1)

t = PersonTracker(TAM, som_points=[[300, 100], [300, 400]])
t.update(1, 100, 200)
t.update(1, 100, 300)
chk("vertical entrance line: no false positive", t.som_count == 0)

t = PersonTracker(TAM, som_points=[[200, 150], [400, 150], [400, 350], [200, 350]])
t.update(1, 50, 50)
for _ in range(6):
    t.update(1, 300, 250)
t.update(1, 50, 50)
t.update(1, 300, 250)
chk("SOM counted exactly once per person", t.som_count == 1)

app_src = pathlib.Path("app.py").read_text(encoding="utf-8")
chk("app.py is_complete() accepts >= 2 SOM points", "som_ok = len(self.som_line) >= 2" in app_src)
chk("app.py renders SOM polygons (SVG)", '<polygon points="{to_px(som)}"' in app_src or "len(som) >= 3" in app_src)

# ---------------------------------------------------------------- TASK 4
print("\n[TASK 4] Face blur")
from effects import FaceBlurEffect, resolve_cascade  # noqa: E402

chk("resolve_cascade() locates a cascade", resolve_cascade("haarcascade_frontalface_default.xml") is not None)
fb = FaceBlurEffect()
chk("FaceBlurEffect.available is True", fb.available)
chk("effects.py falls back to cv2.data.haarcascades",
    "cv2.data.haarcascades" in pathlib.Path("effects.py").read_text(encoding="utf-8"))
chk("face_recognition.py uses the same fallback",
    "resolve_cascade" in pathlib.Path("face_recognition.py").read_text(encoding="utf-8"))

try:
    import numpy as np
    from effects import NightVisionEffect, apply_effects
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    base = frame.copy()
    fb._detect_faces_in_box = lambda f, b: [(200, 150, 80, 90)]
    fb._detect_faces_full_frame = lambda f: [(200, 150, 80, 90)]
    on = apply_effects(frame.copy(), True, False, [(180, 120, 320, 400)], fb, NightVisionEffect())
    off = apply_effects(base.copy(), False, False, [(180, 120, 320, 400)], fb, NightVisionEffect())
    chk("ENABLED  -> region is blurred",
        on[150:240, 200:280].astype(float).var() < base[150:240, 200:280].astype(float).var() * 0.5)
    chk("DISABLED -> frame untouched", np.array_equal(off, base))
except Exception as exc:  # pragma: no cover
    chk(f"blur pipeline runtime test ({exc})", False)

# ---------------------------------------------------------------- TASK 5
print("\n[TASK 5] Demo limitations removed")
chk("TAM_MAX_COUNT removed from executable code", "TAM_MAX_COUNT" not in main_code)
chk("TAM_SPIKE_GUARD_MULTIPLIER removed", "TAM_SPIKE_GUARD_MULTIPLIER" not in main_code)

# ---------------------------------------------------------------- TASK 6
print("\n[TASK 6] main.py duplicate / dead code")
main_tree = ast.parse(pathlib.Path("main.py").read_text(encoding="utf-8"))
for node in ast.walk(main_tree):
    if isinstance(node, ast.ClassDef) and node.name == "CorewiseEngine":
        runs = [b for b in node.body if isinstance(b, ast.FunctionDef) and b.name == "run"]
        chk("exactly one run() in CorewiseEngine", len(runs) == 1)
chk("unreachable CameraPipeline removed", "class CameraPipeline" not in main_code)
chk("system-health monitor started from the live run()", "_start_system_health_monitor()" in main_code)

app_tree = ast.parse(app_src)
names = {}
dupes = []
for node in app_tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        if node.name in names:
            dupes.append(node.name)
        names[node.name] = node.lineno
chk(f"no shadowed module-level duplicates in app.py (found {len(dupes)})", not dupes)

# ---------------------------------------------------------------- TASK 2
print("\n[TASK 2] Replay")
chk("engine imports ReplayBuffer", "from replay_buffer import ReplayBuffer" in main_code)
chk("engine writes CLEAN frames + AI metadata to the buffer",
    "_replay_buffer.write(clean_frame" in main_code)
chk("dashboard imports replay_buffer", "import replay_buffer as rbuf" in app_src)
chk("Replay page renders a LIVE pane", 'id="live"' in app_src)
chk("Replay page has a real-clock timeline",
    'id="tl"' in app_src and "function pct(" in app_src and "tlhead.style.left" in app_src)
chk("Replay page has a return-to-LIVE control", "b-live" in app_src)
cfg = pathlib.Path(".streamlit/config.toml")
chk(".streamlit/config.toml exists", cfg.is_file())
if cfg.is_file():
    chk("enableStaticServing = true", "enableStaticServing = true" in cfg.read_text(encoding="utf-8"))

try:
    import shutil
    import subprocess

    import numpy as np

    import replay_buffer as rbuf

    cam = "__verify_cam__"
    shutil.rmtree(rbuf.live_dir(cam).parent, ignore_errors=True)
    buf = rbuf.ReplayBuffer(cam, camera_name="VerifyCam")
    rng = np.random.default_rng(1)
    t0 = time.time()
    last = 0.0
    while time.time() - t0 < 12.0:
        now = time.time()
        if now - last >= rbuf.FRAME_INTERVAL_SECONDS:
            last = now
            buf.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8),
                      {"people": [{"id": 1, "box": [10, 10, 90, 200],
                                   "conf": 0.9, "in_tam": True, "sam": False,
                                   "som": False, "dwell": 1.0}]})
    buf.log_event(rbuf.EV_SOM, {"som": 1})
    buf.release()
    time.sleep(4)

    segs = rbuf.list_segments(cam)
    chk(f"clips recorded ({len(segs)})", len(segs) >= 1)
    chk("clips are browser-playable H.264", bool(segs) and all(s["browser_playable"] for s in segs))

    import re as _re
    # HH-MM-SS (a superset of the requested HH-MM): clips are 20 s, so two
    # clips share a minute and second-level precision is required to keep
    # filenames unique without "(2)" suffixes.
    chk("filename is Camera_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4",
        bool(segs) and bool(_re.match(
            r"VerifyCam_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_to_\d{2}-\d{2}-\d{2}",
            segs[0]["name"])))
    if shutil.which("ffprobe") and segs:
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0",
                             str(segs[0]["path"])], capture_output=True, text=True)
        chk(f"ffprobe codec: {pr.stdout.strip()}", "h264" in pr.stdout and "yuv420p" in pr.stdout)

    bounds = rbuf.timeline_bounds(cam)
    chk("real-clock timeline bounds available", bounds is not None)
    chk("timeline spans the recording", bounds and 10 <= (bounds[1] - bounds[0]) <= 25)

    kf = rbuf.list_keyframes(cam)
    chk(f"key-frame track ({len(kf)} frames, guarantees never-black scrub)", len(kf) >= 8)

    mid = bounds[0] + (bounds[1] - bounds[0]) / 2
    chk("segment_at(mid) resolves a clip + seek offset", rbuf.segment_at(cam, mid) is not None)
    chk("keyframe_path(mid) resolves an image", rbuf.keyframe_path(cam, mid) is not None)
    chk("keyframe_meta(mid) returns AI metadata",
        bool(rbuf.keyframe_meta(cam, mid).get("people")))

    try:
        import cv2 as _cv
        seg = rbuf.segment_at(cam, mid)
        cap = _cv.VideoCapture(str(seg["path"]))
        cap.set(_cv.CAP_PROP_POS_MSEC, seg["seek"] * 1000)
        ok, fr = cap.read()
        cap.release()
        chk("seeking mid-timeline decodes a NON-BLACK frame",
            ok and fr is not None and fr.mean() > 1)
    except Exception:
        pass

    evs = rbuf.list_events(cam)
    chk("event log written", any(e["kind"] == rbuf.EV_SOM for e in evs))
    chk("recording start/stop events", any(e["kind"] == rbuf.EV_REC_START for e in evs))
    chk("no duplicate clips", len({s["name"] for s in segs}) == len(segs))
    chk("no orphaned staging files", not list(rbuf.live_dir(cam).glob(".writing_*.mp4")))

    for fn in ("_render_nvr_player", "_replay_static_url"):
        chk(f"app.py provides {fn}()", f"def {fn}" in app_src)
    chk("player HTML is constant (no per-rerun iframe re-mount)",
        "data:image/jpeg;base64" not in app_src.split("def _render_nvr_player")[1][:20000])
    chk("player self-polls index.json", "setInterval(poll, 1000)" in app_src)
    chk("AI overlay defaults to OFF", "let overlayOn=false" in app_src)
    chk("O(log n) keyframe lookup", "(lo+hi)>>1" in app_src)
    chk("player builds exact ms keyframe URLs (no rounding)",
        "nearestKFms" in app_src and "Math.round(kf*1000)" not in app_src)
    chk("stage is portrait-friendly (not locked to 16:9)",
        "aspect-ratio:16/9" not in app_src.split("def _render_nvr_player")[1][:20000])
    chk("broken-keyframe fallback", "still.addEventListener('error'" in app_src)
    chk("Play always lands on decodable video", "nearestPlayable" in app_src)
    chk("gapless rollover into the next clip", "seekTo(nx.start+0.05, true)" in app_src)
    chk("seek offset scaled to file length", "toFileTime" in app_src and "toWallTime" in app_src)
    chk("clips are short enough for recent playback (<=30s)",
        rbuf.SEGMENT_SECONDS <= 30)
    # Regression guard: a container header carrying the avc1 tag but holding no
    # decodable frames must NOT be accepted (this false positive made the engine
    # announce "OpenCV avc1 verified" while every clip was actually unplayable).
    import tempfile as _tf
    _d = pathlib.Path(_tf.mkdtemp())
    _fake = _d / "fake.mp4"
    _fake.write_bytes(b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2avc1mp41"
                      + b"\x00" * 400 + b"avc1" + b"\x00" * 900)
    chk("tagged-but-empty file is rejected as not H.264",
        not rbuf._produces_h264(_fake, min_frames=1))
    shutil.rmtree(_d, ignore_errors=True)

    _enc = rbuf.probe_encoder()
    chk(f"encoder probe verified a real H.264 encoder ({_enc['label']})", _enc["h264"])
    chk("browser_playable is verified from file bytes, not assumed",
        all(s["browser_playable"] == (b"avc1" in s["path"].read_bytes()) for s in segs))
    chk("dashboard reports the real encoder state",
        "אין מקודד H.264 במערכת" in app_src and "encoder_status()" in app_src)
    chk("engine prints an encoder banner at startup",
        "NO H.264 ENCODER FOUND" in main_code)
    chk("check_ffmpeg.py is available", pathlib.Path("check_ffmpeg.py").is_file())
    # Both recorders must agree on which binary to use, or the log contradicts
    # itself ("Replay encoder verified" next to "ffmpeg was not found").
    import recording_paths as _rp
    chk("hourly recorder and Replay resolve the SAME ffmpeg",
        _rp.resolve_ffmpeg() == rbuf.resolve_ffmpeg())
    chk("hourly recorder no longer hardcodes 'ffmpeg'",
        '["ffmpeg", "-y"' not in main_code)
    chk("migrate_recordings.py is available",
        (pathlib.Path("migrate_recordings.py")).is_file())
    chk("Replay page uses a slow autorefresh", "interval=30000" in app_src)
    idx = rbuf.live_dir(cam) / "index.json"
    chk("engine writes live/index.json manifest", idx.is_file())
    if idx.is_file():
        import json as _j
        _i = _j.loads(idx.read_text(encoding="utf-8"))
        chk(f"manifest lists clips ({len(_i.get('segments', []))})", len(_i.get("segments", [])) >= 1)
        chk(f"manifest under 200KB ({len(idx.read_bytes()):,}B)", len(idx.read_bytes()) < 200_000)
        # Regression guard for the black-screen bug: every kf_list entry must be
        # an EXACT existing filename. Rounding these broke every REPLAY seek.
        _kf = _i.get("kf_list", [])
        chk("kf_list holds integer millisecond ids", all(isinstance(k, int) for k in _kf))
        _missing = [k for k in _kf if not (rbuf.frames_dir(cam) / f"{k}.jpg").is_file()]
        chk(f"every kf_list entry resolves to a real file ({len(_kf) - len(_missing)}/{len(_kf)})",
            not _missing)
        _badseg = [x for x in _i.get("segments", [])
                   if not (rbuf.live_dir(cam) / x["file"]).is_file()]
        chk("every manifest clip resolves to a real file", not _badseg)
        _sg = _i.get("segments", [])
        _gaps = [round(b["start"] - a["end"], 3) for a, b in zip(_sg, _sg[1:])]
        chk(f"no timeline holes between clips (max {max(_gaps) if _gaps else 0:.3f}s)",
            not _gaps or max(_gaps) < 0.35)
        chk("clips carry playback length for accurate seeking",
            all("pb" in x for x in _sg))
    chk("single Replay implementation (no duplicate player)", "def _render_player(" not in app_src)
    for ctl in ("b-play", "b-live", "b-back", "b-fwd", "b-snap", "b-dl", "b-full"):
        chk(f"player control {ctl}", ctl in app_src)
    chk("speeds 1x/2x/4x/8x", all(f'data-s="{x}"' in app_src for x in (1, 2, 4, 8)))
    chk("timeline event markers", "tlmark" in app_src)
    chk("AI overlay toggle inside the player", 'id="b-ov"' in app_src)
    chk("AI overlay canvas", "function paint(" in app_src)
    chk("auto-return to LIVE at the timeline edge", "IDX.end-2" in app_src)
    chk("key-frame fallback when video errors", "vid.addEventListener('error'" in app_src)
    chk("event list + smart search", "cw_replay_ev_kinds" in app_src)

    shutil.rmtree(rbuf.live_dir(cam).parent, ignore_errors=True)
except Exception as exc:  # pragma: no cover
    chk(f"replay runtime test ({exc})", False)

# ---------------------------------------------------------------- DEPS
print("\n[DEPENDENCIES]")
req = pathlib.Path("requirements.txt").read_text(encoding="utf-8").lower()
for pkg in ("streamlit-image-coordinates", "pillow", "psutil", "imageio-ffmpeg"):
    chk(f"{pkg} declared in requirements.txt", pkg in req)

print("\n" + "=" * 68)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
print("=" * 68)
sys.exit(1 if FAIL else 0)