from __future__ import annotations

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


def test_none_detections_hold_last_box() -> None:
    tracker = make()
    last = run_frames(tracker, 10)[-1].active[0]
    out = tracker.update(None, 10 * DT)
    assert len(out.active) == 1
    assert out.active[0].xyxy == last.xyxy
    assert out.active[0].track_id == last.track_id
    assert not out.active[0].detected


def test_lost_track_finishes_after_timeout() -> None:
    tracker = make()
    run_frames(tracker, 10)
    finished: list[int] = []
    for i in range(10, 10 + 60):
        t = i * DT
        finished += tracker.update([], t).finished
        if finished:
            assert t - 9 * DT <= (1.5 + 0.04 * 2) * 1e9
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
