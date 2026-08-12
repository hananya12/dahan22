"""
check_ffmpeg.py
---------------
Standalone check: does THIS Python interpreter have a working H.264 encoder
for the Replay System, and is it the interpreter that actually runs Corewise?

Imports nothing heavy (no cv2, no ultralytics, no streamlit), so it runs even
on an interpreter where the project's dependencies are missing.

    python check_ffmpeg.py
    py -3.14 check_ffmpeg.py
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

print("=" * 66)
print("COREWISE — H.264 ENCODER CHECK")
print("=" * 66)

print("\n1. Which interpreter is this?")
print(f"   executable : {sys.executable}")
print(f"   version    : {sys.version.split()[0]}")
print(f"   cwd        : {os.getcwd()}")

print("\n2. Are the project's dependencies installed HERE?")
mods = {
    "cv2": "opencv-contrib-python  (engine + recording)",
    "numpy": "numpy",
    "ultralytics": "ultralytics  (YOLO detection)",
    "streamlit": "streamlit  (dashboard)",
    "websockets": "websockets  (server)",
    "imageio_ffmpeg": "imageio-ffmpeg  (bundled H.264 encoder)",
}
missing = []
for mod, label in mods.items():
    ok = importlib.util.find_spec(mod) is not None
    print(f"   {'OK     ' if ok else 'MISSING'} {label}")
    if not ok:
        missing.append(mod)

core_missing = [m for m in missing if m in ("cv2", "numpy", "ultralytics", "streamlit")]
if core_missing:
    print("\n   >> This interpreter CANNOT run Corewise.")
    print("      You are checking the wrong Python. Find the right one with:")
    print("         where python")
    print("         py -0")
    print("      then run this script with that exact executable, e.g.:")
    print(r'         "C:\Users\<you>\AppData\Local\Python\pythoncore-3.14-64\python.exe" check_ffmpeg.py')

print("\n3. ffmpeg on PATH")
sys_ffmpeg = shutil.which("ffmpeg")
print(f"   {'OK     ' if sys_ffmpeg else 'MISSING'} ffmpeg" + (f"  ({sys_ffmpeg})" if sys_ffmpeg else ""))

print("\n4. Bundled ffmpeg (imageio-ffmpeg wheel)")
bundled = None
try:
    import imageio_ffmpeg
    bundled = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"   OK      {bundled}")
except Exception as exc:
    print(f"   MISSING ({exc.__class__.__name__})")
    print("           install with:  pip install imageio-ffmpeg")

chosen = sys_ffmpeg or bundled
print("\n5. Encoder Corewise will actually use")
print(f"   {chosen or 'NONE'}")

if not chosen:
    print("\n" + "=" * 66)
    print("RESULT: NO H.264 ENCODER")
    print("Recordings will fall back to OpenCV mp4v, which no browser can play,")
    print("so Replay will show key frames instead of video.")
    print("\nFix (no admin rights needed):")
    print(f'   "{sys.executable}" -m pip install imageio-ffmpeg')
    print("=" * 66)
    raise SystemExit(1)

print("\n6. Does that encoder really produce H.264?")
out = Path("__ffmpeg_selftest.mp4")
try:
    r = subprocess.run(
        [chosen, "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x240:rate=10:duration=1",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out)],
        capture_output=True, text=True, timeout=90,
    )
    if r.returncode != 0 or not out.is_file():
        print(f"   FAILED: {r.stderr.strip()[:300]}")
        raise SystemExit(1)

    raw = out.read_bytes()
    has_avc1 = b"avc1" in raw
    print(f"   encoded {len(raw):,} bytes")
    print(f"   contains avc1 (H.264): {'YES' if has_avc1 else 'NO'}")

    probe = chosen.replace("ffmpeg", "ffprobe")
    if shutil.which(probe) or Path(probe).is_file():
        p = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, timeout=30)
        print(f"   ffprobe: {p.stdout.strip()}")

    print("\n" + "=" * 66)
    if has_avc1:
        print("RESULT: OK — Replay will record browser-playable H.264.")
        if core_missing:
            print("        (but this interpreter still cannot run Corewise — see step 2)")
    else:
        print("RESULT: encoder ran but did not produce H.264. Reinstall ffmpeg.")
    print("=" * 66)
    raise SystemExit(0 if has_avc1 else 1)
finally:
    out.unlink(missing_ok=True)
