"""
tools/shadow_report.py
----------------------
Read the Phase 0 shadow log and answer the only question Phase 0 exists to
answer: **is the new engine ready to replace the old counters?**

    python tools/shadow_report.py                       # today, all cameras
    python tools/shadow_report.py --date 2026-07-27
    python tools/shadow_report.py --camera cam_1 --days 7

The headline number is the divergence between ``legacy_tam`` and
``shadow_tam``. That gap is not noise — it is the double-counting the store
is living with today, measured on the store's own traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SHADOW_DIR = ROOT / "data" / "shadow"


def load_rows(camera: Optional[str], dates: List[str]) -> Dict[str, List[dict]]:
    if not SHADOW_DIR.is_dir():
        raise SystemExit(f"no shadow logs at {SHADOW_DIR} — is COREWISE_SHADOW enabled?")
    out: Dict[str, List[dict]] = defaultdict(list)
    for cam_dir in sorted(SHADOW_DIR.iterdir()):
        if not cam_dir.is_dir() or (camera and cam_dir.name != camera):
            continue
        for date in dates:
            path = cam_dir / f"{date}.jsonl"
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            out[cam_dir.name].append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    return out


def analyse(rows: List[dict]) -> Dict[str, Any]:
    last = rows[-1]
    legacy, shadow = last.get("legacy", {}), last.get("shadow", {})

    verdicts = Counter()
    scores = Counter()
    gate = Counter()
    errors = 0
    dropped = 0
    p95: List[float] = []
    for row in rows:
        verdicts.update(row.get("verdicts", {}))
        scores.update(row.get("score_histogram", {}))
        gate.update((row.get("reid", {}) or {}).get("gate_reasons", {}) or {})
        errors = max(errors, row.get("errors", 0))
        dropped = max(dropped, (row.get("cost", {}) or {}).get("dropped", 0))
        cost = (row.get("cost", {}) or {}).get("p95_ms")
        if cost:
            p95.append(cost)

    total_verdicts = sum(verdicts.values()) or 1
    reid = last.get("reid", {}) or {}
    embedded, gated = reid.get("embedded", 0), reid.get("gated_out", 0)

    return {
        "samples": len(rows),
        "hours": round((rows[-1]["t"] - rows[0]["t"]) / 3600.0, 2) if len(rows) > 1 else 0.0,
        "legacy": legacy,
        "shadow": shadow,
        "divergence": {
            key: _pct(legacy.get(key), shadow.get(key))
            for key in ("tam", "sam", "som")
        },
        "verdicts": dict(verdicts),
        "ambiguous_rate": round(verdicts.get("ambiguous", 0) / total_verdicts * 100, 1),
        "score_histogram": dict(sorted(scores.items())),
        "gate": {"embedded": embedded, "rejected": gated,
                 "reject_rate": round(gated / (embedded + gated) * 100, 1)
                                if (embedded + gated) else None,
                 "reasons": dict(gate.most_common(6))},
        "health": {"errors": errors, "dropped_frames": dropped,
                   "p95_ms": round(max(p95), 1) if p95 else None,
                   "stride": (last.get("cost", {}) or {}).get("stride")},
    }


def _pct(legacy: Any, shadow: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(legacy, (int, float)) or not isinstance(shadow, (int, float)):
        return None
    delta = shadow - legacy
    return {"legacy": legacy, "shadow": shadow, "delta": round(delta, 1),
            "pct": round(delta / legacy * 100, 1) if legacy else None}


def report(camera: str, data: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(f"SHADOW REPORT — {camera}   ({data['samples']} samples over {data['hours']}h)")
    print("=" * 70)

    print("\nCOUNTS: old (by track id)  vs  new (by person)")
    for key, d in data["divergence"].items():
        if not d:
            continue
        arrow = "↓" if d["delta"] < 0 else "↑" if d["delta"] > 0 else "="
        pct = f"{d['pct']:+.0f}%" if d["pct"] is not None else "n/a"
        print(f"   {key.upper():4} {d['legacy']:>7} → {d['shadow']:>7}   {arrow} {pct}")

    tam = data["divergence"].get("tam")
    if tam and tam["pct"] is not None and tam["delta"] < 0:
        print(f"\n   >> the old counter reported {abs(tam['delta']):.0f} more visitors than")
        print(f"      there were people. That is {abs(tam['pct']):.0f}% double-counting,")
        print("      measured on this store's own traffic.")

    print("\nMATCHING")
    print(f"   verdicts       {data['verdicts']}")
    print(f"   ambiguous rate {data['ambiguous_rate']}%")
    if data["ambiguous_rate"] > 25:
        print("   >> high. The appearance model is unsure a lot: usually low resolution,")
        print("      strong backlight, or people dressed alike. Check the score histogram")
        print("      before touching thresholds.")
    if data["score_histogram"]:
        print("   score distribution:")
        peak = max(data["score_histogram"].values()) or 1
        for bucket, n in data["score_histogram"].items():
            bar = "█" * max(1, int(n / peak * 34))
            print(f"      {bucket:>10} {bar} {n}")
        print("      A healthy distribution is BIMODAL: a hump of non-matches low down,")
        print("      a hump of true matches high up, and a clear valley between them.")
        print("      Put the threshold in the valley. One smooth hump means the model")
        print("      cannot separate people at this camera angle.")

    gate = data["gate"]
    print(f"\nQUALITY GATE   embedded {gate['embedded']}, rejected {gate['rejected']}"
          + (f" ({gate['reject_rate']}%)" if gate["reject_rate"] is not None else ""))
    for reason, n in gate["reasons"].items():
        print(f"      {reason:28} {n}")
    if gate["reject_rate"] and gate["reject_rate"] > 90:
        print("   >> almost everything is rejected. Reframe or move the camera;")
        print("      no threshold change can fix crops this poor.")

    health = data["health"]
    print(f"\nHEALTH   errors {health['errors']}   dropped {health['dropped_frames']}   "
          f"p95 {health['p95_ms']} ms   stride {health['stride']}")

    print("\nCUTOVER CHECKLIST")
    checks = [
        ("no errors", health["errors"] == 0),
        ("ambiguous rate under 25%", data["ambiguous_rate"] < 25),
        ("gate rejecting under 90%", (gate["reject_rate"] or 0) < 90),
        ("at least 4h of samples", data["hours"] >= 4),
        ("TAM divergence is stable and explainable", tam is not None),
    ]
    for label, ok in checks:
        print(f"   [{'x' if ok else ' '}] {label}")
    if all(ok for _, ok in checks):
        print("\n   Ready for Phase 2 on this camera.")
    else:
        print("\n   Not yet. Keep Phase 0 running.")
    print("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyse Phase 0 shadow logs.")
    ap.add_argument("--camera")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--json", help="write the analysis here")
    args = ap.parse_args()

    if args.date:
        dates = [args.date]
    else:
        dates = [time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400 * i))
                 for i in range(args.days)]

    by_camera = load_rows(args.camera, dates)
    if not by_camera:
        raise SystemExit(f"no shadow rows for {dates}")

    out = {}
    for camera, rows in by_camera.items():
        rows.sort(key=lambda r: r.get("t", 0))
        analysis = analyse(rows)
        out[camera] = analysis
        report(camera, analysis)

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())