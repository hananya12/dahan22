"""
diagnose_replay.py
------------------
Run this from your PROJECT ROOT (the folder with app.py / main.py):

    python diagnose_replay.py

It inspects YOUR files and reports exactly why the Replay page is rendering
empty. It changes nothing - read-only.
"""

from __future__ import annotations

import ast
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
PROBLEMS: list[str] = []


def head(t: str) -> None:
    print("\n" + t)
    print("-" * len(t))


print("=" * 70)
print("COREWISE REPLAY DIAGNOSTIC")
print("=" * 70)
print(f"project root: {ROOT}")

# ---------------------------------------------------------------- files
head("1. Files present")
for name in ("app.py", "main.py", "replay_buffer.py", "recording_paths.py",
             "tracking.py", "effects.py"):
    p = ROOT / name
    print(f"   {'OK     ' if p.is_file() else 'MISSING'} {name}"
          + (f"  ({p.stat().st_size:,} bytes)" if p.is_file() else ""))
    if not p.is_file():
        PROBLEMS.append(f"{name} is missing")

cfg = ROOT / ".streamlit" / "config.toml"
print(f"   {'OK     ' if cfg.is_file() else 'MISSING'} .streamlit/config.toml")
if cfg.is_file():
    txt = cfg.read_text(encoding="utf-8")
    # Match the actual SETTING, ignoring any comment that mentions the name.
    import re as _re
    ok = bool(_re.search(r"^\s*enableStaticServing\s*=\s*true",
                         txt, _re.MULTILINE | _re.IGNORECASE))
    print(f"   {'OK     ' if ok else 'PROBLEM'} enableStaticServing = true")
    if not ok:
        PROBLEMS.append("enableStaticServing is not true -> video will 404 (black)")
else:
    PROBLEMS.append(".streamlit/config.toml missing -> video will 404 (black)")

app = ROOT / "app.py"
if not app.is_file():
    print("\nCannot continue without app.py.")
    sys.exit(1)

src = app.read_text(encoding="utf-8")
try:
    tree = ast.parse(src)
except SyntaxError as e:
    print(f"\napp.py HAS A SYNTAX ERROR at line {e.lineno}: {e.msg}")
    sys.exit(1)

# ---------------------------------------------- duplicate definitions
head("2. Duplicate / shadowed definitions in app.py")
defs = collections.defaultdict(list)
for n in tree.body:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        defs[n.name].append(n.lineno)

dupes = {k: v for k, v in defs.items() if len(v) > 1}
if dupes:
    for name, lines in dupes.items():
        print(f"   PROBLEM  {name}() defined {len(lines)}x at lines {lines}")
        print(f"            -> Python keeps line {lines[-1]}; the others are DEAD")
        PROBLEMS.append(f"{name}() is defined {len(lines)} times - line {lines[-1]} wins")
else:
    print("   OK       no duplicates")

# ------------------------------------------------ required functions
head("3. Replay functions in app.py")
required = {
    "render_replay_page": "the Replay page itself",
    "_render_nvr_player": "the NVR player component",
    "_build_replay_payload": "builds segments/keyframes/events for the player",
    "_replay_static_url": "turns file paths into browser URLs",
}
for fn, why in required.items():
    if fn in defs:
        print(f"   OK       {fn}()  - {why}   [line {defs[fn][-1]}]")
    else:
        print(f"   MISSING  {fn}()  - {why}")
        PROBLEMS.append(f"{fn}() is missing from app.py -> you are running an OLD app.py")

if "_render_player" in defs:
    print("   PROBLEM  _render_player() still exists - that is the OLD single-file player")
    PROBLEMS.append("_render_player() (old implementation) is still present")

# -------------------------------------------- what the page actually does
head("4. What render_replay_page() actually calls")
target = None
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name == "render_replay_page":
        target = n                      # last definition wins, same as Python
if target is None:
    print("   render_replay_page() not found at module level.")
else:
    print(f"   using the definition at line {target.lineno} "
          f"(lines {target.lineno}-{target.end_lineno})")
    called = set()
    for sub in ast.walk(target):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                called.add(f.id)
            elif isinstance(f, ast.Attribute):
                called.add(f.attr)
    for fn in ("_render_nvr_player", "_build_replay_payload"):
        if fn in called:
            print(f"   OK       calls {fn}()")
        else:
            print(f"   PROBLEM  does NOT call {fn}()")
            PROBLEMS.append(f"the active render_replay_page() never calls {fn}()"
                            " -> you are running an OLD page")
    if "expander" in called:
        print("   NOTE     this page renders an st.expander - the NEW page has none,"
              " so this is an OLD/merged version")
        PROBLEMS.append("the active Replay page renders an expander -> not the new page")

    sig = [a.arg for a in target.args.args]
    print(f"   signature: render_replay_page({', '.join(sig)})")
    if "telemetry" not in sig:
        print("   PROBLEM  no 'telemetry' parameter -> LIVE frames cannot reach the player")
        PROBLEMS.append("render_replay_page() has no telemetry parameter (old signature)")

# --------------------------------------------------- call site in main()
head("5. Call site in main()")
found_call = False
for n in ast.walk(tree):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
            and n.func.id == "render_replay_page":
        found_call = True
        print(f"   line {n.lineno}: render_replay_page(...) with "
              f"{len(n.args)} positional args")
        if len(n.args) < 3:
            print("   PROBLEM  called with fewer than 3 args -> telemetry not passed,"
                  " LIVE view will be empty")
            PROBLEMS.append("render_replay_page() is called without telemetry")
if not found_call:
    print("   PROBLEM  render_replay_page() is never called")
    PROBLEMS.append("render_replay_page() is never called from main()")

# ------------------------------------------------------ recorded data
head("6. Recorded data on disk")
try:
    sys.path.insert(0, str(ROOT))
    import recording_paths as rp
    import replay_buffer as rbuf

    root = rp.RECORDINGS_DIR
    print(f"   recordings root: {root}")
    print(f"   exists: {root.is_dir()}")
    if root.is_dir():
        cams = [d.name for d in root.iterdir() if d.is_dir()]
        print(f"   cameras on disk: {cams or 'NONE'}")
        if not cams:
            PROBLEMS.append("no camera folders under static/recordings -> engine never recorded")
        for cam in cams:
            segs = rbuf.list_segments(cam)
            kfs = rbuf.list_keyframes(cam)
            evs = rbuf.list_events(cam)
            bounds = rbuf.timeline_bounds(cam)
            print(f"\n   [{cam}]")
            print(f"      clips      : {len(segs)}")
            print(f"      keyframes  : {len(kfs)}")
            print(f"      events     : {len(evs)}")
            if bounds:
                import datetime as _dt
                print(f"      timeline   : "
                      f"{_dt.datetime.fromtimestamp(bounds[0]):%Y-%m-%d %H:%M:%S} -> "
                      f"{_dt.datetime.fromtimestamp(bounds[1]):%H:%M:%S}")
            bad = [s for s in segs if not s["browser_playable"]]
            if bad:
                print(f"      PROBLEM    {len(bad)} clips are mp4v (browsers cannot play)")
                PROBLEMS.append(f"{cam}: {len(bad)} clips use mp4v - install ffmpeg")
            if segs:
                print(f"      newest     : {segs[-1]['name']}")
            if not segs and not kfs:
                PROBLEMS.append(f"{cam}: no clips and no keyframes -> nothing to replay")
    else:
        PROBLEMS.append("static/recordings does not exist -> the engine has not recorded")
except Exception as exc:
    print(f"   could not inspect recordings: {exc}")

# ------------------------------------------------------------- ffmpeg
head("7. ffmpeg")
import shutil as _sh
for tool in ("ffmpeg", "ffprobe"):
    p = _sh.which(tool)
    print(f"   {'OK     ' if p else 'MISSING'} {tool}" + (f"  ({p})" if p else ""))
    if not p and tool == "ffmpeg":
        PROBLEMS.append("ffmpeg not on PATH -> clips fall back to unplayable mp4v")

# ------------------------------------------------------------ verdict
print("\n" + "=" * 70)
if PROBLEMS:
    print(f"FOUND {len(PROBLEMS)} PROBLEM(S):")
    for i, p in enumerate(PROBLEMS, 1):
        print(f"  {i}. {p}")
else:
    print("No problems detected. If Replay still looks empty, hard-refresh the")
    print("browser (Ctrl+F5) and check the browser console for blocked requests.")
print("=" * 70)