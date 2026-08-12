"""
vision/identity/heatmap_engine.py
---------------------------------
Two heat maps in one grid, and the distinction is the whole point:

    dwell     — seconds of human presence per cell (time-weighted)
    visitors  — DISTINCT people per cell (a set of global ids)

A stationary employee is 1 visitor with a huge dwell; a busy aisle is many
visitors with modest dwell each. A frame-counting heat map cannot tell those
apart — an identity-keyed one can.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from events import EventType, IdentityEvent
from person_profile import PersonProfile

Point = Tuple[float, float]


class HeatmapEngine:
    def __init__(self, grid_size: Tuple[int, int] = (24, 32),
                 frame_size: Tuple[int, int] = (720, 1280)) -> None:
        self.rows, self.cols = grid_size
        self.frame_size = frame_size            # (h, w); pipeline updates it
        self.dwell: List[List[float]] = [[0.0] * self.cols
                                         for _ in range(self.rows)]
        self.visitors: Dict[int, Set[str]] = {}  # cell index -> global ids

    # ------------------------------------------------------------------

    def cell_of(self, point: Point) -> int:
        fh, fw = self.frame_size
        r = int(min(self.rows - 1, max(0, point[1] / max(fh, 1) * self.rows)))
        c = int(min(self.cols - 1, max(0, point[0] / max(fw, 1) * self.cols)))
        return r * self.cols + c

    def record(self, profile: PersonProfile, foot: Point, now: float) -> None:
        # Called BEFORE profile.touch(now), so last_seen is the previous
        # frame — the elapsed time is real presence, not a frame count.
        dt = max(0.0, min(1.0, now - (profile.last_seen or now)))
        cell = self.cell_of(foot)
        r, c = divmod(cell, self.cols)
        self.dwell[r][c] += dt
        self.visitors.setdefault(cell, set()).add(profile.global_id)

    # EventSink protocol: a merge must not leave the absorbed id in any cell.
    def emit(self, event: IdentityEvent) -> None:
        if event.type is not EventType.IDENTITY_MERGED:
            return
        absorbed = event.data.get("absorbed")
        if not absorbed:
            return
        for ids in self.visitors.values():
            if absorbed in ids:
                ids.discard(absorbed)
                ids.add(event.global_id)

    # ------------------------------------------------------------------

    def hotspots(self, n: int = 5) -> List[Dict[str, Any]]:
        cells = []
        for cell, ids in self.visitors.items():
            r, c = divmod(cell, self.cols)
            cells.append({"cell": cell, "row": r, "col": c,
                          "visitors": len(ids),
                          "dwell_seconds": round(self.dwell[r][c], 1)})
        cells.sort(key=lambda x: x["dwell_seconds"], reverse=True)
        return cells[:n]

    def normalised(self) -> List[List[float]]:
        peak = max((max(row) for row in self.dwell), default=0.0) or 1.0
        return [[round(v / peak, 4) for v in row] for row in self.dwell]

    def snapshot(self) -> Dict[str, Any]:
        return {"grid": self.normalised(),
                "grid_size": [self.rows, self.cols],
                "hotspots": self.hotspots(5)}

    @property
    def grid_size(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def reset(self) -> None:
        self.dwell = [[0.0] * self.cols for _ in range(self.rows)]
        self.visitors.clear()
