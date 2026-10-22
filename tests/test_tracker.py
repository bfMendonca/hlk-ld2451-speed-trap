from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from speed_trap.config import TrackerConfig
from speed_trap.tracker import Tracker, TrackerOutput
from speed_trap.types import Detection

FPS = 25.0
DT = 40_000_000


def box_at(x: float, cls: str = "car", y: float = 100.0) -> Detection:
    return Detection((x, y, x + 80.0, y + 50.0), 0.9, cls)


def run_frames(
    tracker: Tracker, n: int, t0: int = 0, boxes: list[float] | None = None
) -> list[TrackerOutput]:
    outs = []
    for i in range(n):
        xs = boxes if boxes is not None else [10.0 + 5 * i]
        outs.append(tracker.update([box_at(x) for x in xs], t0 + i * DT))
    return outs


def make() -> Tracker:
    return Tracker(TrackerConfig(), FPS)


def test_moving_box_keeps_one_id() -> None:
    outs = run_frames(make(), 30)
    ids = {a.track_id for o in outs for a in o.active}
    assert len(ids) == 1
    assert all(len(o.active) == 1 for o in outs[1:])  # first frame is tentative


def test_two_boxes_two_ids() -> None:
    outs = run_frames(make(), 10, boxes=[10.0, 400.0])
    assert len(outs[-1].active) == 2
    assert len({o.track_id for o in outs[-1].active}) == 2


def test_time_gap_finishes_and_new_ids_are_unique() -> None:
    tracker = make()
    out1 = run_frames(tracker, 10)
    a = out1[-1].active[0].track_id
    t = 10 * DT + 3_000_000_000
    out2 = tracker.update([box_at(0)], t)
    assert a in out2.finished
    out3 = tracker.update([box_at(5)], t + DT)
    assert out3.active[0].track_id != a


def test_none_detections_predict_constant_velocity_box() -> None:
    tracker = make()
    outs = run_frames(tracker, 10)  # moves 5 px per frame
    prev, last = outs[-2].active[0], outs[-1].active[0]
    for k in (1, 2):
        out = tracker.update(None, (9 + k) * DT)
        assert len(out.active) == 1
        assert out.active[0].track_id == last.track_id
        assert not out.active[0].detected
        step = [b - a for a, b in zip(prev.xyxy, last.xyxy, strict=True)]
        expected = [v + k * d for v, d in zip(last.xyxy, step, strict=True)]
        assert list(out.active[0].xyxy) == pytest.approx(expected, abs=1e-3)


def test_none_detections_hold_a_box_seen_once() -> None:
    tracker = make()
    tracker._bt = _StickyByteTrack()  # type: ignore[assignment]
    first = tracker.update([box_at(10.0)], 0).active[0]
    out = tracker.update(None, DT)
    assert [o.xyxy for o in out.active] == [first.xyxy]


def test_none_detections_skip_tracks_lost_on_the_last_detected_frame() -> None:
    tracker = make()
    run_frames(tracker, 10, boxes=[10.0, 400.0])
    out = tracker.update([box_at(10.0)], 10 * DT)  # the second box is gone
    assert len(out.active) == 1 and out.finished == []
    kept = out.active[0].track_id
    out = tracker.update(None, 11 * DT)
    assert [o.track_id for o in out.active] == [kept]


def test_lost_track_finishes_after_timeout() -> None:
    tracker = make()
    lost_ns = TrackerConfig().lost_track_s * 1e9
    run_frames(tracker, 10)
    finished: list[int] = []
    for i in range(10, 10 + 60):
        t = i * DT
        finished += tracker.update([], t).finished
        if finished:
            elapsed = t - 9 * DT
            assert lost_ns < elapsed <= lost_ns + 2 * DT
            break
    assert len(finished) == 1


def test_majority_class_and_finish_all() -> None:
    tracker = make()
    for i, cls in enumerate(["car", "truck", "truck", "truck", "car"]):
        out = tracker.update([box_at(10.0 + 5 * i, cls)], i * DT)
    assert out.active[0].class_name == "truck"
    tid = out.active[0].track_id
    assert tracker.finish_all() == [tid]
    assert tracker.finish_all() == []


class _StickyByteTrack:
    """Stub that keeps reporting sv id 0, like a ByteTrack that outlives wrapper expiry."""

    def update(self, dets: sv.Detections, timestamp: float | None = None) -> sv.Detections:
        if len(dets) == 0:
            return sv.Detections.empty()
        dets.tracker_id = np.zeros(len(dets), dtype=int)
        return dets

    def reset(self) -> None:
        pass


def test_expired_id_never_returns() -> None:
    tracker = make()
    tracker._bt = _StickyByteTrack()  # type: ignore[assignment]
    a = tracker.update([box_at(10.0)], 0).active[0].track_id
    finished: list[int] = []
    t = DT
    for _ in range(50):  # skipped frames past lost_track_s
        finished += tracker.update(None, t).finished
        t += DT
    assert a in finished
    out = tracker.update([box_at(10.0)], t)
    assert out.active[0].track_id != a
    assert tracker.update([box_at(10.0)], t + DT).active[0].track_id == out.active[0].track_id
