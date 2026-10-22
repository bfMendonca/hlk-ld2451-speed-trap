from __future__ import annotations

import pytest

from speed_trap.config import CaptureConfig, SpeedConfig
from speed_trap.lite import RadarSubject
from speed_trap.monitor import SpeedingMonitor
from speed_trap.types import Box, Direction, RadarMatch, RadarTarget

W, H = 1296, 972


def ts(seconds: float) -> int:
    return round(seconds * 1e9)


def monitor(speed: SpeedConfig | None = None) -> SpeedingMonitor:
    return SpeedingMonitor(
        speed or SpeedConfig(limit_kmh=40), CaptureConfig(min_consecutive_over=2), W, H
    )


class Subj:
    def __init__(
        self,
        speeds: list[float],
        box: Box | None,
        reliable: bool = True,
        direction: Direction = "coming",
    ) -> None:
        self.subject_id = 1
        self.direction: Direction | None = direction
        self.reliable = reliable
        self.xyxy = box
        self.confidence = 1.0
        self.matches = [mk(i, v) for i, v in enumerate(speeds)]

    @property
    def max_speed_kmh(self) -> float | None:
        return max(m.speed_kmh for m in self.matches)

    @property
    def mean_speed_kmh(self) -> float | None:
        return sum(m.speed_kmh for m in self.matches) / len(self.matches)


def mk(i: int, speed: float, dist: int = 20, verified: bool = True) -> RadarMatch:
    t = RadarTarget(ts(i), 0, dist, True, int(speed), 10, 0)
    return RadarMatch(ts(i), t, (float(dist), 0.0), speed, int(speed), verified)


def subject(
    speeds: list[float],
    box: Box | None = (100, 100, 300, 250),
    reliable: bool = True,
    direction: Direction = "coming",
) -> Subj:
    return Subj(speeds, box, reliable, direction)


def test_two_consecutive_over_triggers_and_event_on_finish() -> None:
    m = monitor()
    s = subject([38, 45, 47], box=(100, 100, 300, 250))
    m.update([s], 0, 10, ts(1))
    ev = m.finish([s])
    assert ev[0].speed_max_kmh == 47 and ev[0].speed_mean_kmh == pytest.approx(130 / 3)
    assert ev[0].speed_at_capture_kmh == 45
    assert m.triggered_ids == {1}


def test_single_spike_does_not_trigger() -> None:
    m = monitor()
    s = subject([45, 30])
    m.update([s], 0, 1, ts(1))
    assert m.finish([s]) == []


def test_unreliable_subject_never_triggers() -> None:
    m = monitor()
    s = subject([50, 50, 50], reliable=False)
    assert m.update([s], 0, 1, ts(1)) == []
    assert m.finish([s]) == []
    assert m.triggered_ids == set()


def test_per_direction_limit_override() -> None:
    m = monitor(SpeedConfig(limit_kmh=40, limit_kmh_going=60))
    assert m.limit_for("going") == 60 and m.limit_for("coming") == 40
    s = subject([50, 50], direction="going")
    m.update([s], 0, 1, ts(1))
    assert m.finish([s]) == []


def test_best_frame_is_largest_box_not_touching_edge() -> None:
    m = monitor()
    s = subject([50, 50, 50], box=(0, 0, 100, 100))
    s.xyxy = (100, 100, 200, 200)
    m.update([s], 0, 1, ts(1))
    s.xyxy = (W - 200, 100, W, 300)  # touches right edge, area 4e4
    m.update([s], 0, 2, ts(2))
    s.xyxy = (100, 100, 300, 200)  # area 2e4
    m.update([s], 0, 3, ts(3))
    ev = m.finish([s])
    assert ev[0].capture.index == 3


def test_lite_subject_best_is_closest_reading() -> None:
    m = monitor()
    s = subject([50, 50], box=None)
    m.update([s], 0, 1, ts(1))
    s.matches.append(mk(2, 50, dist=12))
    m.update([s], 0, 2, ts(2))
    s.matches.append(mk(3, 50, dist=15))
    m.update([s], 0, 3, ts(3))
    ev = m.finish([s])
    assert ev[0].capture.index == 2 and ev[0].capture.xyxy is None


def test_radar_subject_satisfies_protocol() -> None:
    m = monitor()
    s = RadarSubject(1, [mk(0, 50), mk(1, 50)], "coming")
    m.update([s], 0, 1, ts(1))
    assert len(m.finish([s])) == 1


def test_unverified_over_limit_matches_never_trigger() -> None:
    m = monitor()
    s = subject([50, 50, 50])
    s.matches = [mk(i, 50, verified=False) for i in range(3)]
    m.update([s], 0, 1, ts(1))
    assert m.triggered_ids == set()
    assert m.finish([s]) == []


def test_one_verified_match_among_the_over_limit_run_triggers() -> None:
    m = monitor()
    s = subject([50, 50])
    s.matches = [mk(0, 50, verified=False), mk(1, 50, verified=True)]
    m.update([s], 0, 1, ts(1))
    assert m.triggered_ids == {1}


def test_verified_match_outside_the_run_does_not_count() -> None:
    m = monitor()
    s = subject([50, 50, 50])
    s.matches = [mk(0, 50, verified=True), mk(1, 50, verified=False), mk(2, 50, verified=False)]
    m.update([s], 0, 1, ts(1))
    assert m.triggered_ids == set()


def test_event_confidence_is_the_capture_frames() -> None:
    m = monitor()
    s = subject([50, 50, 50], box=(100, 100, 300, 300))
    s.confidence = 0.8
    m.update([s], 0, 1, ts(1))  # best frame: largest box
    s.xyxy, s.confidence = (100, 100, 150, 150), 0.95
    m.update([s], 0, 2, ts(2))
    ev = m.finish([s])
    assert ev[0].capture.index == 1
    assert ev[0].capture.confidence == 0.8 and ev[0].confidence == 0.8
