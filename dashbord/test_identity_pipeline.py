"""
tests/test_identity_pipeline.py
-------------------------------
The regressions that define the product.

Everything here drives the REAL engines through ``IdentityPipeline`` —
identity, zones, retail, occupancy, journey, heatmap. Only the appearance
model is faked, so the logic under test is the logic that ships. The suite
runs in well under a second with no camera, no GPU, no YOLO and no torch,
which is the point: counting logic must be verifiable on every commit, not
only in front of a live camera.

    python corewise/tests/test_identity_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from events import EventBus
from config import IdentityConfig
from memory_engine import MemoryEngine
from contracts import Side
from zone_engine import TAM_ZONE
from reid_engine import assess_crop
from pipeline import IdentityPipeline

FRAME_W, FRAME_H = 1280, 720

# The TAM zone / store floor. "Outside" is a point outside this polygon but
# still well inside the FRAME: a person at the image edge is a truncated crop,
# the quality gate refuses to embed them, and they never get an identity at
# all. Real cameras have exactly this property — which is an installation
# requirement, not a code problem.
TAM_POLYGON = [(300, 150), (1150, 150), (1150, 690), (300, 690)]
ZONES = {
    "Vegetables": [(320, 170), (700, 170), (700, 450), (320, 450)],
    "Drinks":     [(720, 170), (1140, 170), (1140, 450), (720, 450)],
    "Checkout":   [(320, 470), (1140, 470), (1140, 680), (320, 680)],
}
OUT_X, VEG, DRINK, CHECKOUT = 120.0, (500.0, 400.0), (900.0, 400.0), (700.0, 600.0)


class FakeAppearanceBackend:
    """Deterministic stand-in for OSNet.

    Reads a person id out of the crop and returns a near-orthogonal vector per
    person. Same person -> cosine ~0.99, different people -> ~0.0. The tests
    therefore assert on the DECISION LOGIC, not on how good a particular ReID
    checkpoint happens to be.
    """

    name = "fake"
    dim = 64

    def __init__(self, noise: float = 0.02, seed: int = 7) -> None:
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._bases: dict[int, np.ndarray] = {}

    def _base(self, person: int) -> np.ndarray:
        if person not in self._bases:
            v = np.zeros(self.dim, dtype=np.float32)
            v[person % self.dim] = 1.0
            v[(person * 7 + 3) % self.dim] = 0.5
            self._bases[person] = v / np.linalg.norm(v)
        return self._bases[person]

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(crops), self.dim), dtype=np.float32)
        for i, crop in enumerate(crops):
            v = self._base(int(crop[0, 0, 0]))
            v = v + self.rng.normal(0, self.noise, self.dim).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class Scene:
    """Paints people into a frame and produces detection.py's exact output."""

    HEIGHT = 240.0

    def __init__(self) -> None:
        self.frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self._n = 0

    def detections(self, people: Sequence[Tuple[int, int, float, float]]) -> List[dict]:
        self.frame[:] = 0
        self._n += 1
        out = []
        for track_id, person, x, y in people:
            h = self.HEIGHT
            w = h / 2.5
            x1, y1, x2, y2 = int(x - w / 2), int(y - h), int(x + w / 2), int(y)
            crop = np.random.default_rng(person * 1000 + self._n).integers(
                0, 255, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
            crop[0, 0, 0] = person
            self.frame[y1:y2, x1:x2] = crop
            out.append({"id": track_id, "box": (float(x1), float(y1),
                                                float(x2), float(y2), 0.92)})
        return out


def build(min_stay=None, funnel=None) -> IdentityPipeline:
    pipe = IdentityPipeline(camera_id="cam_1", backend=FakeAppearanceBackend(),
                            funnel_steps=funnel)
    pipe.configure(tam_polygon_px=TAM_POLYGON, named_zones_px=ZONES)
    if min_stay is not None:
        pipe.set_min_stay(*min_stay)
    return pipe


def run(pipe: IdentityPipeline, scene: Scene, script, t0=None, dt=0.2):
    """script: iterable of lists of (track_id, person, x, y). [] = nobody visible."""
    t = t0 or time.time()
    stats = {}
    for people in script:
        stats = pipe.process(scene.frame, scene.detections(people), now=t)
        t += dt
    return stats, t


def at(track, person, position):
    return (track, person, position[0], position[1])


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_same_person_returning_keeps_one_identity():
    """THE bug: person leaves frame, returns with a new track id."""
    pipe, scene = build(), Scene()
    _, t = run(pipe, scene, [[at(15, 1, VEG)]] * 10)
    first = pipe.identity.global_id_for(15)
    assert first, "track 15 never committed to an identity"

    _, t = run(pipe, scene, [[]] * 40, t0=t)                     # gone 8 seconds
    _, t = run(pipe, scene, [[at(61, 1, DRINK)]] * 10, t0=t)     # back, new id

    second = pipe.identity.global_id_for(61)
    assert second == first, f"re-identification FAILED: {second} != {first}"
    assert pipe.memory.stats["created_total"] == 1
    assert pipe.telemetry()["tam"] == 1, pipe.telemetry()["tam"]
    print(f"   one person, two tracker ids -> {first}   TAM=1")


def test_two_different_people_stay_separate():
    """The opposite failure: never merge two real people into one."""
    pipe, scene = build(), Scene()
    run(pipe, scene, [[at(1, 1, VEG), at(2, 2, DRINK)]] * 12)
    a, b = pipe.identity.global_id_for(1), pipe.identity.global_id_for(2)
    assert a and b and a != b, f"FALSE MERGE: {a} == {b}"
    assert pipe.telemetry()["tam"] == 2
    print(f"   two people -> {a} / {b}   TAM=2")


# ---------------------------------------------------------------------------
# TAM / SAM / SOM — over GlobalPerson
# ---------------------------------------------------------------------------

def test_tam_counts_people_not_tracks():
    """One shopper, five tracker ids, TAM must stay 1.

    This is what replaces _tam_dedup_count and TAM_RECOUNT_COOLDOWN_SECONDS
    in main.py. No cooldown is involved, so two genuinely different people
    arriving seconds apart are BOTH counted — which the cooldown suppressed.
    """
    pipe, scene = build(), Scene()
    t = time.time()
    for track_id in (11, 22, 33, 44, 55):                # tracker churn
        _, t = run(pipe, scene, [[at(track_id, 1, VEG)]] * 6, t0=t)
        _, t = run(pipe, scene, [[]] * 3, t0=t)

    stats = pipe.telemetry()
    assert stats["tam"] == 1, f"TAM inflated by tracker churn: {stats['tam']}"
    profile = pipe.memory.hot_candidates()[0]
    assert len(profile.track_ids_seen) == 5, profile.track_ids_seen
    print(f"   5 tracker ids absorbed into 1 person   TAM={stats['tam']}")


def test_sam_dwell_survives_tracker_loss():
    """The measurable win of identity-based dwell.

    A shopper is observed 2s, lost by the tracker for 6s, then re-acquired
    under a new id. Track-based SAM restarts the clock at zero and never
    reaches a 3s threshold from a 2s + 2s visit. Person-based dwell banks the
    first 2s and resumes.
    """
    pipe, scene = build(min_stay=(True, 3.0)), Scene()

    _, t = run(pipe, scene, [[at(7, 1, VEG)]] * 10)              # ~2.0s observed
    gid = pipe.identity.global_id_for(7)
    profile = pipe.memory.get(gid)
    banked = profile.zone_dwell(TAM_ZONE, t)
    assert 1.0 < banked < 3.0, banked
    assert not profile.retail.counted_sam, "SAM fired before the threshold"

    _, t = run(pipe, scene, [[]] * 30, t0=t)                     # lost for 6s
    after_gap = profile.zone_dwell(TAM_ZONE, t)
    assert abs(after_gap - banked) < 0.5, (
        f"dwell reset during the tracker gap: {banked:.2f} -> {after_gap:.2f}")

    _, t = run(pipe, scene, [[at(93, 1, VEG)]] * 20, t0=t)       # back, new id
    assert pipe.identity.global_id_for(93) == gid
    assert profile.retail.counted_sam, (
        f"SAM never counted; dwell={profile.zone_dwell(TAM_ZONE, t):.2f} "
        f"reason={profile.retail.sam_reason}")
    assert pipe.telemetry()["sam"] == 1
    print(f"   dwell survived a 6s gap ({banked:.1f}s -> "
          f"{profile.zone_dwell(TAM_ZONE, t):.1f}s), SAM counted")


def test_sam_not_counted_before_dashboard_configures_it():
    """No engine-side default may ever win silently."""
    pipe, scene = build(), Scene()                                # no set_min_stay
    run(pipe, scene, [[at(1, 1, VEG)]] * 15)
    stats = pipe.telemetry()
    assert stats["tam"] == 1 and stats["sam"] == 0, stats
    profile = pipe.memory.hot_candidates()[0]
    assert "dashboard" in profile.retail.sam_reason
    print(f"   SAM withheld: {profile.retail.sam_reason!r}")


def test_som_line_crossing_counts_each_person_once():
    pipe, scene = IdentityPipeline(camera_id="cam_1",
                                   backend=FakeAppearanceBackend()), Scene()
    pipe.configure(tam_polygon_px=TAM_POLYGON,
                   som_points_px=[(300, 500), (1150, 500)],   # horizontal entrance
                   som_is_polygon=False, som_direction=-1)
    t = time.time()
    # Below the line, then above it: one crossing, then loiter across it again.
    _, t = run(pipe, scene, [[at(1, 1, (700.0, 600.0))]] * 6, t0=t)
    _, t = run(pipe, scene, [[at(1, 1, (700.0, 400.0))]] * 6, t0=t)
    _, t = run(pipe, scene, [[at(1, 1, (700.0, 600.0))]] * 4, t0=t)
    _, t = run(pipe, scene, [[at(1, 1, (700.0, 400.0))]] * 4, t0=t)
    stats = pipe.telemetry()
    assert stats["som"] == 1, f"SOM counted a person more than once: {stats['som']}"
    assert stats["som_shape"] == "polyline"
    print(f"   SOM={stats['som']} after two crossings by the same person")


# ---------------------------------------------------------------------------
# Occupancy, journey, heatmap
# ---------------------------------------------------------------------------

def test_occupancy_and_hysteresis():
    pipe, scene = build(), Scene()
    t = time.time()
    _, t = run(pipe, scene, [[at(1, 1, (OUT_X, 400.0))]] * 8, t0=t)      # outside
    _, t = run(pipe, scene, [[at(1, 1, VEG)]] * 8, t0=t)                 # walks in
    assert pipe.entry_exit.currently_inside == 1, pipe.entry_exit.stats()
    assert pipe.entry_exit.stats()["entered"] == 1

    # One stray frame outside must not register as an exit.
    _, t = run(pipe, scene, [[at(1, 1, VEG)], [at(1, 1, (OUT_X, 400.0))],
                             [at(1, 1, VEG)], [at(1, 1, VEG)]], t0=t)
    assert pipe.entry_exit.stats()["exited"] == 0, "hysteresis failed"

    t += 3.0
    _, t = run(pipe, scene, [[at(1, 1, (OUT_X, 400.0))]] * 8, t0=t)      # real exit
    stats = pipe.entry_exit.stats()
    assert stats["exited"] == 1 and stats["currently_inside"] == 0, stats
    print(f"   entered={stats['entered']} exited={stats['exited']} inside=0")


def test_journey_is_per_person_with_dwell_and_funnel():
    pipe, scene = build(funnel=["Vegetables", "Checkout"]), Scene()
    t = time.time()
    # Person 1 does the full route; person 2 only browses Drinks.
    _, t = run(pipe, scene, [[at(1, 1, VEG), at(2, 2, DRINK)]] * 10, t0=t)
    _, t = run(pipe, scene, [[at(1, 1, DRINK), at(2, 2, DRINK)]] * 8, t0=t)
    _, t = run(pipe, scene, [[at(1, 1, CHECKOUT), at(2, 2, DRINK)]] * 8, t0=t)

    gid1 = pipe.identity.global_id_for(1)
    path = pipe.journey.path_of(gid1)
    assert path == ["Vegetables", "Drinks", "Checkout"], path

    steps = pipe.journey.path_with_dwell(gid1)
    assert steps[0]["dwell"] > 0.5, steps

    funnel = pipe.journey.funnel(["Vegetables", "Checkout"])
    assert funnel[0]["people"] == 1 and funnel[1]["people"] == 1, funnel
    flows = {(f["from"], f["to"]): f["count"] for f in pipe.journey.flows()}
    assert flows.get(("Vegetables", "Drinks")) == 1, flows
    print(f"   journey {path}  funnel={[f['people'] for f in funnel]}")


def test_heatmap_separates_dwell_from_unique_traffic():
    """A stationary employee must not look like a thousand customers."""
    pipe, scene = build(), Scene()
    t = time.time()
    _, t = run(pipe, scene, [[at(1, 1, CHECKOUT)]] * 100, t0=t)   # one person, 20s

    cell = pipe.heatmap.cell_of(CHECKOUT)
    r, c = divmod(cell, pipe.heatmap.cols)
    assert len(pipe.heatmap.visitors[cell]) == 1, "unique traffic inflated by frames"
    assert pipe.heatmap.dwell[r][c] > 5.0, pipe.heatmap.dwell[r][c]

    hotspot = pipe.heatmap.hotspots(1)[0]
    assert hotspot["visitors"] == 1 and hotspot["dwell_seconds"] > 5.0, hotspot
    print(f"   1 visitor, {hotspot['dwell_seconds']}s dwell in one cell "
          "(frame-counting would have said 100)")


# ---------------------------------------------------------------------------
# Merging and architecture
# ---------------------------------------------------------------------------

def test_merge_reconciles_every_engine_at_once():
    """Late evidence proves two ids were one person. Nothing may be orphaned."""
    pipe, scene = build(min_stay=(False, None)), Scene()
    t = time.time()
    _, t = run(pipe, scene, [[at(1, 1, VEG)]] * 10, t0=t)
    a = pipe.identity.global_id_for(1)          # capture while the track is live
    _, t = run(pipe, scene, [[]] * 3, t0=t)
    _, t = run(pipe, scene, [[at(2, 2, CHECKOUT)]] * 10, t0=t)
    b = pipe.identity.global_id_for(2)

    assert a and b and a != b
    assert pipe.telemetry()["tam"] == 2

    kept = pipe.memory.merge(a, b, t)

    stats = pipe.telemetry()
    assert stats["tam"] == 1, f"TAM not reconciled after merge: {stats['tam']}"
    assert stats["sam"] == 1, f"SAM not reconciled after merge: {stats['sam']}"
    assert pipe.journey.path_of(kept.global_id) == ["Vegetables", "Checkout"], \
        pipe.journey.path_of(kept.global_id)
    heat_ids = set().union(*pipe.heatmap.visitors.values())
    assert b not in heat_ids, "heatmap kept the absorbed id"
    assert kept.global_id in heat_ids, "heatmap lost the surviving id"
    assert pipe.memory.get(b) is None, "absorbed identity still resolvable"
    print(f"   merge folded TAM/SAM/journey/heatmap into {kept.global_id}")


def test_no_engine_below_identity_touches_track_id():
    """Architecture test: the layering is checked, not just documented.

    ``track_id`` may appear only where a tracker id genuinely exists — the
    contracts, the manager that resolves it, the ReID scheduler and the
    adapter. Any other module referencing it has started keying state by
    tracker id again, which is the exact bug this system was built to remove.
    """
    import ast
    below_identity = ["zone_engine.py", "retail_engine.py", "journey_engine.py",
                      "heatmap_engine.py", "entry_exit_engine.py",
                      "analytics_engine.py", "memory_engine.py",
                      "matching_engine.py"]
    # person_profile.py and events.py CARRY a track_id as inert diagnostic
    # data (track_ids_seen / IdentityEvent.track_id); neither keys state by
    # it, so they sit with contracts.py on the allowed side of the line.
    root = Path(__file__).resolve().parent
    offenders = []
    for path in [root / name for name in below_identity]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = (node.id if isinstance(node, ast.Name)
                    else node.attr if isinstance(node, ast.Attribute)
                    else node.arg if isinstance(node, ast.arg) else None)
            if name == "track_id":
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"track_id leaked below the identity layer: {offenders}"
    print(f"   {len(below_identity)} modules verified free of track-id state")


def test_quality_gate_rejects_unusable_crops():
    from config import ReIDConfig
    from contracts import TrackObservation
    cfg = ReIDConfig()

    def obs(x, y, h=240.0, conf=0.92):
        w = h / 2.5
        crop = np.random.default_rng(1).integers(0, 255, (int(h), int(w), 3),
                                                 dtype=np.uint8)
        return TrackObservation(1, (x - w / 2, y - h, x + w / 2, y), conf, 0,
                                time.time(), FRAME_W, FRAME_H, crop), crop

    tiny, crop = obs(640, 400, h=40)
    assert not assess_crop(tiny, cfg, crop).passed
    edge, crop = obs(20, 400)
    assert not assess_crop(edge, cfg, crop).passed, "truncated crop passed"
    good, crop = obs(640, 400)
    report = assess_crop(good, cfg, crop)
    assert report.passed and 0 < report.score <= 1
    print(f"   quality gate ok (good crop scored {report.score})")


def test_memory_demotes_to_warm_instead_of_deleting():
    config = IdentityConfig()
    config.memory.hot_ttl_seconds = 1.0
    memory = MemoryEngine(config.memory, "cam_1", EventBus())
    profile = memory.create(time.time() - 5.0)
    result = memory.sweep()
    assert result["demoted"] == 1 and result["warm"] == 1
    assert memory.get(profile.global_id) is not None, "identity deleted, not demoted"
    assert memory.promote(profile.global_id, time.time()).is_returning
    print("   hot -> warm -> promoted back as a returning visitor")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    print("=" * 70)
    print("COREWISE V2 — IDENTITY-FIRST PIPELINE")
    print("=" * 70)
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}\n      {exc}")
        except Exception as exc:
            failures += 1
            import traceback
            failures_tb = traceback.format_exc().strip().splitlines()[-3:]
            print(f"ERROR {fn.__name__}\n      " + "\n      ".join(failures_tb))
    print("=" * 70)
    print(f"{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)