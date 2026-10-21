from __future__ import annotations

from speed_trap.config import FusionConfig, LiteConfig, RoadConfig, SpeedConfig
from speed_trap.geometry import RadarModel
from speed_trap.lite import RadarTargetLinker
from speed_trap.types import RadarScan, RadarTarget


def ts(seconds: float) -> int:
    return round(seconds * 1e9)


def linker() -> RadarTargetLinker:
    radar = RadarModel((0.0, 0.0), 1.6, 0.0, 1, 20.0, 1.0, 100.0)
    return RadarTargetLinker(
        radar, FusionConfig(), LiteConfig(lost_track_s=1.5), SpeedConfig(), RoadConfig()
    )


def tg(t: float, dist: int, approaching: bool = True, angle: int = 0) -> RadarTarget:
    return RadarTarget(ts(t), angle, dist, approaching, 40, 10, 0)


def scan(t: float, *targets: RadarTarget) -> RadarScan:
    return RadarScan(ts(t), targets)


def test_one_subject_per_passing_target() -> None:
    lk = linker()
    seen = {}
    for i in range(10):
        active, fin = lk.update([scan(i * 0.1, tg(i * 0.1, 30 - i))], ts(i * 0.1))
        assert fin == []
        seen.update({s.subject_id: s for s in active})
    assert len(seen) == 1
    assert len(seen[1].matches) == 10 and seen[1].direction == "coming"


def test_opposite_directions_two_subjects() -> None:
    lk = linker()
    for i in range(5):
        t = i * 0.1
        active, _ = lk.update([scan(t, tg(t, 25 - i), tg(t, 20 + i, approaching=False))], ts(t))
    assert len(active) == 2
    assert {s.direction for s in active} == {"coming", "going"}


def test_subject_finishes_after_lost_track_s() -> None:
    lk = linker()
    lk.update([scan(0.0, tg(0.0, 25))], 0)
    active, fin = lk.update([], ts(1.0))
    assert len(active) == 1 and fin == []
    active, fin = lk.update([], ts(1.6))
    assert active == [] and len(fin) == 1


def test_targets_beyond_max_range_ignored() -> None:
    lk = linker()
    active, fin = lk.update([scan(0.0, tg(0.0, 60))], 0)
    assert active == [] and fin == []
