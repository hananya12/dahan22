"""
integration/install_shadow.py
-----------------------------
Install Phase 0 into the running engine — or take it back out.

    python integration/install_shadow.py --check      # what would change
    python integration/install_shadow.py --install    # do it (writes a .bak)
    python integration/install_shadow.py --uninstall  # restore the .bak

Why a script instead of a diff
------------------------------
A unified diff against a 1,129-line file that is still being edited would go
stale in a day and fail with a rejected hunk. This does anchored replacement:
it finds five short, distinctive snippets in ``main.py`` and inserts next to
them. If any anchor is missing or already patched, it refuses and changes
nothing — so it is safe to run twice, and safe to run on a file that has
moved on since this was written.

Every inserted line is guarded. Shadow mode cannot raise into the frame loop,
cannot slow it (its work is on a background thread behind a bounded queue),
and cannot alter a single existing telemetry key. Disable at runtime with
``COREWISE_SHADOW=0`` without editing anything back out.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

MARKER = "# --- COREWISE PHASE 0 (shadow mode) ---"

# (label, anchor, insertion, position)  position: "after" | "before"
EDITS: List[Tuple[str, str, str, str]] = [
    (
        "import",
        "class CorewiseEngine:",
        f'''{MARKER}
# Phase 0 runs the V2 identity engine beside the existing one for comparison.
# It is fully optional: if the package is absent, or anything inside it fails,
# the engine below runs exactly as it did before.
try:
    from vision.shadow import ShadowRunner as _ShadowRunner
except Exception as _shadow_import_error:   # pragma: no cover
    _ShadowRunner = None
    print(f"[SHADOW] not available ({{_shadow_import_error.__class__.__name__}}) "
          "- running without Phase 0.")
# --- end Phase 0 ---


''',
        "before",
    ),
    (
        "construct",
        "        self._system_health_thread: Optional[threading.Thread] = None",
        f'''
        {MARKER}
        self._shadow = None
        if _ShadowRunner is not None:
            try:
                self._shadow = _ShadowRunner(rp.default_camera_id()).start()
            except Exception as exc:
                print(f"[SHADOW] disabled at startup: {{exc}}")
                self._shadow = None
        # --- end Phase 0 ---''',
        "after",
    ),
    (
        "calibration",
        "        if self.tracker is not None:\n            self._apply_sam_min_stay(sam_min_stay_enabled, sam_min_stay_seconds)",
        f'''
        {MARKER}
        if self._shadow is not None:
            self._shadow.set_min_stay(sam_min_stay_enabled, sam_min_stay_seconds)
        # --- end Phase 0 ---''',
        "after",
    ),
    (
        "zones",
        "                if self.tam_zone_np is not None:\n                        draw_zone(frame, self.tam_zone_np, COLOR_TAM, True)",
        "",   # resolved dynamically below
        "skip",
    ),
    (
        "telemetry",
        '        if frame is not None:\n            small = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))',
        f'''        {MARKER}
        if getattr(self, "_shadow", None) is not None:
            try:
                payload.update(self._shadow.telemetry())
            except Exception:
                pass
        # --- end Phase 0 ---

''',
        "before",
    ),
    (
        "submit",
        "                self.maybe_send_telemetry(camera_status=\"online\", frame=frame)\n\n                cv2.imshow(\"Corewise AI\", frame)",
        f'''                {MARKER}
                # clean_frame, not frame: the shadow must see the picture WITHOUT
                # boxes and overlays drawn on it. Non-blocking; drops frames when
                # behind rather than holding up the loop.
                if self._shadow is not None:
                    self._shadow.configure(
                        tam_polygon_px=(self.tam_zone_np.tolist()
                                        if self.tam_zone_np is not None else None),
                        som_points_px=self.som_points_px or None,
                        som_is_polygon=self.som_is_polygon,
                    )
                    self._shadow.submit(clean_frame, detections, _stats_now,
                                        now=time.time(), fps=self._current_fps)
                # --- end Phase 0 ---

''',
        "before",
    ),
    (
        "shutdown",
        "        finally:\n            self.camera.release()",
        f'''        finally:
            {MARKER}
            if getattr(self, "_shadow", None) is not None:
                try:
                    self._shadow.stop()
                except Exception:
                    pass
            # --- end Phase 0 ---''',
        "replace_prefix",
    ),
]


def find_main(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"not found: {path}")
        return path
    for candidate in (Path.cwd() / "main.py",
                      Path(__file__).resolve().parents[2] / "main.py"):
        if candidate.is_file():
            return candidate
    raise SystemExit("main.py not found — pass --main /path/to/main.py")


def apply(source: str) -> Tuple[str, List[str], List[str]]:
    applied: List[str] = []
    missing: List[str] = []
    out = source

    for label, anchor, insertion, position in EDITS:
        if position == "skip":
            continue
        if anchor not in out:
            missing.append(label)
            continue
        if position == "before":
            out = out.replace(anchor, insertion + anchor, 1)
        elif position == "after":
            out = out.replace(anchor, anchor + insertion, 1)
        elif position == "replace_prefix":
            out = out.replace(anchor, insertion + "\n            self.camera.release()", 1)
        applied.append(label)

    return out, applied, missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Install/remove Phase 0 shadow mode in main.py")
    ap.add_argument("--main", help="path to main.py")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    path = find_main(args.main)
    source = path.read_text(encoding="utf-8")
    backup = path.with_suffix(".py.pre-shadow.bak")

    if args.uninstall:
        if not backup.is_file():
            raise SystemExit(f"no backup at {backup}")
        shutil.copy2(backup, path)
        print(f"restored {path} from {backup}")
        return 0

    if MARKER in source:
        print(f"{path.name} already has Phase 0 installed "
              f"({source.count(MARKER)} hooks). Nothing to do.")
        print("Disable it at runtime instead:  COREWISE_SHADOW=0")
        return 0

    patched, applied, missing = apply(source)

    print(f"target : {path}")
    print(f"hooks  : {', '.join(applied) or 'none'}")
    if missing:
        print(f"MISSING: {', '.join(missing)}")
        print("\nThose anchors were not found, which means main.py has changed since")
        print("this installer was written. Nothing has been modified. Install the")
        print("missing hooks by hand — each one is a few guarded lines, listed above.")
        return 1

    try:
        ast.parse(patched)
    except SyntaxError as exc:
        raise SystemExit(f"patched file would not parse (line {exc.lineno}): {exc.msg}")

    if args.check or not args.install:
        print("\n--check only, nothing written. Re-run with --install to apply.")
        return 0

    shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")
    print(f"\ninstalled. backup at {backup}")
    print("Phase 0 is now live. It writes data/shadow/<camera>/<date>.jsonl")
    print("Analyse with:  python tools/shadow_report.py")
    print("Turn off with: COREWISE_SHADOW=0   (no code change needed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())