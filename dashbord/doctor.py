"""
integration/doctor.py
---------------------
Run this from your PROJECT ROOT (the folder with main.py and app.py):

    python doctor.py

It tells you exactly why the identity engine is not running, and what to do
about it. Read-only — it changes nothing.

The most common answer, by a wide margin, is #2: the ``vision/`` folder is
not next to ``main.py``, so ``from vision.shadow import ShadowRunner`` fails,
the guard in main.py catches it, and everything silently carries on with the
old counters. That silence is deliberate (Phase 0 must never break the
engine) but it does mean a misplaced folder looks exactly like "it doesn't
work".
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
PROBLEMS: list[tuple[str, str]] = []


def ok(label: str, detail: str = "") -> None:
    print(f"  OK       {label}" + (f"  {detail}" if detail else ""))


def bad(label: str, fix: str, detail: str = "") -> None:
    print(f"  PROBLEM  {label}" + (f"  {detail}" if detail else ""))
    PROBLEMS.append((label, fix))


print("=" * 72)
print("COREWISE IDENTITY ENGINE — DOCTOR")
print("=" * 72)
print(f"project root : {ROOT}")
print(f"python       : {sys.executable}")

# ---------------------------------------------------------------- 1. project
print("\n1. Am I in the right folder?")
if (ROOT / "main.py").is_file() and (ROOT / "app.py").is_file():
    ok("main.py and app.py found")
else:
    bad("main.py / app.py not here",
        "cd into the folder that contains main.py and app.py, then run this again.")
    print("\nCannot continue from here.")
    raise SystemExit(1)

# ---------------------------------------------------------- 2. package place
print("\n2. Is the vision/ package where Python can find it?")
vision_dir = ROOT / "vision"
if not vision_dir.is_dir():
    found = list(ROOT.glob("*/vision/identity")) + list(ROOT.glob("*/*/vision/identity"))
    hint = f"\n           I found one at: {found[0].parent}" if found else ""
    bad("no vision/ folder next to main.py",
        "Move the vision/ folder so it sits BESIDE main.py:\n"
        f"             {ROOT / 'main.py'}\n"
        f"             {ROOT / 'vision' / 'identity' / '...'}\n"
        "           The download has it inside a corewise/ folder — copy the\n"
        "           INNER vision/ folder out, not the corewise/ wrapper." + hint)
elif not (vision_dir / "identity" / "__init__.py").is_file():
    bad("vision/ exists but vision/identity/ is incomplete",
        "Re-copy the whole vision/ folder from the download.")
else:
    ok("vision/identity/ present")

# ---------------------------------------------------------------- 3. imports
print("\n3. Does it import?")
sys.path.insert(0, str(ROOT))
pipeline_ok = False
try:
    from vision.shadow import ShadowRunner          # noqa
    ok("from vision.shadow import ShadowRunner")
    try:
        from vision.pipeline import IdentityPipeline  # noqa
        pipe = IdentityPipeline(camera_id="doctor")
        backend = pipe.reid.backend.name
        ok("IdentityPipeline constructs", f"backend={backend}")
        pipeline_ok = True
        if backend == "color_histogram":
            print("           NOTE: colour-histogram baseline. It works, but it will not")
            print("           re-identify someone reliably after they leave the frame.")
            print("           pip install torch torchreid   then   COREWISE_REID_BACKEND=osnet")
    except Exception as exc:
        bad("IdentityPipeline failed to construct",
            "Paste the error below to me — this one is a real bug, not a setup issue.",
            f"{exc.__class__.__name__}: {exc}")
except Exception as exc:
    bad("cannot import vision.shadow",
        "Almost always the folder placement in step 2. If vision/ IS beside main.py,\n"
        "           check that you are running the same Python that runs main.py.",
        f"{exc.__class__.__name__}: {exc}")

for mod in ("numpy", "cv2", "ultralytics"):
    if importlib.util.find_spec(mod) is None:
        bad(f"{mod} missing", f"{sys.executable} -m pip install "
            + {"cv2": "opencv-contrib-python"}.get(mod, mod))

# ------------------------------------------------------------- 4. main.py
print("\n4. Is main.py wired up?")
source = (ROOT / "main.py").read_text(encoding="utf-8", errors="ignore")
hooks = source.count("COREWISE PHASE 0")
if hooks == 0:
    bad("main.py has no shadow hooks",
        "python integration/install_shadow.py --install")
elif hooks < 6:
    bad(f"main.py has only {hooks}/6 hooks",
        "python integration/install_shadow.py --uninstall\n"
        "           python integration/install_shadow.py --install")
else:
    ok(f"all {hooks} hooks present")

# ------------------------------------------------------------ 5. kill switch
print("\n5. Is it switched off?")
flag = os.environ.get("COREWISE_SHADOW")
if flag in ("0", "false", "False", "no"):
    bad(f"COREWISE_SHADOW={flag}", "unset COREWISE_SHADOW  (or set it to 1)")
else:
    ok("COREWISE_SHADOW is not disabling it")

# ---------------------------------------------------------------- 6. output
print("\n6. Has it actually produced anything?")
log_root = ROOT / "data" / "shadow"
if not log_root.is_dir():
    bad("no data/shadow/ folder",
        "The engine has not run with the shadow active yet. Start main.py, stand in\n"
        "           front of the camera for a minute, then run this again.")
else:
    logs = sorted(log_root.rglob("*.jsonl"))
    if not logs:
        bad("data/shadow/ exists but is empty",
            "The runner started and then disabled itself. Look at the engine console\n"
            "           for a line beginning [SHADOW] — it says why.")
    else:
        newest = max(logs, key=lambda p: p.stat().st_mtime)
        age = time.time() - newest.stat().st_mtime
        rows = sum(1 for line in newest.read_text(encoding="utf-8").splitlines() if line.strip())
        ok(f"{newest.relative_to(ROOT)}", f"{rows} samples, last written {age/60:.0f} min ago")
        print("\n           python tools/shadow_report.py     <- your numbers are in here")

# ------------------------------------------------------------ 7. expectation
print("\n7. What you should EXPECT to see")
print("  Phase 0 is deliberately invisible on the dashboard. TAM / SAM / SOM keep")
print("  showing the OLD numbers — that is the whole point: it cannot break what")
print("  already works. The new numbers arrive as separate shadow_* telemetry keys")
print("  and in data/shadow/*.jsonl.")
print("")
print("  If what you wanted is for the DASHBOARD to show the corrected count, that")
print("  is Phase 2, and it is one flag:  COREWISE_IDENTITY_LIVE=1")
print("  Do not turn that on until tools/shadow_report.py looks healthy.")

# ------------------------------------------------------------------ verdict
print("\n" + "=" * 72)
if PROBLEMS:
    print(f"FOUND {len(PROBLEMS)} PROBLEM(S)\n")
    for i, (label, fix) in enumerate(PROBLEMS, 1):
        print(f"  {i}. {label}")
        print(f"     -> {fix}\n")
else:
    print("No setup problems found.")
    print("If the counts still look wrong, run tools/shadow_report.py and send me")
    print("the output — at that point it is a tuning question, not a wiring one.")
print("=" * 72)
