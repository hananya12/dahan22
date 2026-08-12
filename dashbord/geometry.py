"""
vision/geometry.py
------------------
Pure 2D geometry for Corewise. NO cv2, NO numpy, NO I/O.

Why this file exists
--------------------
``tracking.py`` today owns ``segments_intersect`` and uses
``cv2.pointPolygonTest`` for zone membership. The Identity Engine needs the
exact same geometry for entry/exit decisions. Duplicating it would guarantee
that one day the two answers disagree on the same pixel — the single worst
class of bug in a counting system, because it is silent.

So the geometry lives here once, depends on nothing, and is unit-testable in
microseconds without a camera or an OpenCV install.

Migration note: ``tracking.py`` should eventually
``from vision.geometry import segments_intersect, point_in_polygon`` and
delete its private copies. Until then the implementations are kept
bit-for-bit identical on purpose.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

Point = Tuple[float, float]
EPS = 1e-9


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    """Cross product AB x AC. >0 = counter-clockwise, <0 = clockwise, 0 = collinear."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> bool:
    return (
        min(ax, bx) - EPS <= px <= max(ax, bx) + EPS
        and min(ay, by) - EPS <= py <= max(ay, by) + EPS
    )


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if segment P1P2 intersects segment Q1Q2 (touching counts)."""
    (p1x, p1y), (p2x, p2y) = p1, p2
    (q1x, q1y), (q2x, q2y) = q1, q2

    d1 = orientation(q1x, q1y, q2x, q2y, p1x, p1y)
    d2 = orientation(q1x, q1y, q2x, q2y, p2x, p2y)
    d3 = orientation(p1x, p1y, p2x, p2y, q1x, q1y)
    d4 = orientation(p1x, p1y, p2x, p2y, q2x, q2y)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    if abs(d1) < EPS and _on_segment(q1x, q1y, q2x, q2y, p1x, p1y):
        return True
    if abs(d2) < EPS and _on_segment(q1x, q1y, q2x, q2y, p2x, p2y):
        return True
    if abs(d3) < EPS and _on_segment(p1x, p1y, p2x, p2y, q1x, q1y):
        return True
    if abs(d4) < EPS and _on_segment(p1x, p1y, p2x, p2y, q2x, q2y):
        return True
    return False


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon. Matches cv2.pointPolygonTest(...) >= 0
    for all practical retail geometry, without needing OpenCV."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or EPS) + xi
            if x <= x_cross:
                inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Directional crossing — the heart of Entry/Exit
# ---------------------------------------------------------------------------

def crossing_direction(
    prev: Point,
    curr: Point,
    polyline: Sequence[Point],
) -> int:
    """Direction in which a movement crossed an oriented polyline.

    Returns:
        +1  crossed from the polyline's LEFT side to its RIGHT side
        -1  crossed from RIGHT to LEFT
         0  did not cross

    "Left"/"right" are defined by the polyline's own point order, so the
    dashboard's calibration decides which direction means "entered". This is
    what makes a single line able to produce both Entered and Exited instead
    of needing two separate lines.
    """
    if len(polyline) < 2:
        return 0

    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        if not segments_intersect(prev, curr, a, b):
            continue
        side_before = orientation(a[0], a[1], b[0], b[1], prev[0], prev[1])
        side_after = orientation(a[0], a[1], b[0], b[1], curr[0], curr[1])
        if side_before > 0 and side_after < 0:
            return -1
        if side_before < 0 and side_after > 0:
            return +1
        # Started or ended exactly on the line: ambiguous, ignore this segment.
    return 0


def foot_point(bbox: Tuple[float, float, float, float]) -> Point:
    """Ground contact point of a bounding box (bottom-centre).

    Every zone decision in Corewise uses this point rather than the box
    centre, because a person's position on the shop floor is where their
    feet are, not where their chest is. Getting this wrong shifts every
    zone boundary by half a body height.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)
