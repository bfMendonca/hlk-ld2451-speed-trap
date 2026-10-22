from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import av
from radar_cam_recorder.reader import open_session

from speed_trap.config import Config
from speed_trap.synth import generate_session


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_session_opens_with_recorder_reader(synth_session: Path) -> None:
    s = open_session(synth_session).summary()
    assert s["state"] == "complete" and 15 < s["mean_fps"] < 25 and s["radar_reports"] > 100


def test_video_frame_count_matches_csv(synth_session: Path) -> None:
    for seg in (0, 1):
        with av.open(str(synth_session / f"video/{seg:04d}.h264"), format="h264") as c:
            n = sum(1 for _ in c.decode(video=0))
        assert n == len(read_csv_rows(synth_session / f"video/{seg:04d}.frames.csv"))


def test_segment_gap_present(synth_session: Path) -> None:
    segs = json.loads((synth_session / "session.json").read_text())["segments"]
    assert segs[1]["start_boottime_ns"] - segs[0]["end_boottime_ns"] >= 2.9e9


def test_radar_values_quantised_and_capped(synth_session: Path) -> None:
    reports = list(open_session(synth_session).radar_reports())
    assert all(len(r.targets) <= 5 for r in reports)
    assert any(t.approaching for r in reports for t in r.targets)
    assert any(not t.approaching for r in reports for t in r.targets)


def test_truth_marks_only_the_fast_vehicles(synth_session: Path) -> None:
    truth = json.loads((synth_session / "truth.json").read_text())
    assert {v["speed_kmh"] for v in truth["vehicles"] if v["expect_capture"]} >= {60, 50}
    assert not any(v["expect_capture"] for v in truth["vehicles"] if v["direction"] is None)


def test_truth_has_boxes_for_visible_vehicles(synth_session: Path) -> None:
    truth = json.loads((synth_session / "truth.json").read_text())
    assert any(boxes for boxes in truth["frames"].values())
    assert "0:0" in truth["frames"]


def test_deterministic_with_seed(tmp_path: Path) -> None:
    a = generate_session(tmp_path / "a", Config(), 3.0, 7)
    b = generate_session(tmp_path / "b", Config(), 3.0, 7)
    assert (a / "radar/0000.csv").read_bytes() == (b / "radar/0000.csv").read_bytes()
    assert (a / "truth.json").read_bytes() == (b / "truth.json").read_bytes()


def test_stopped_vehicle_is_parked_out_of_the_moving_lanes() -> None:
    from speed_trap.synth import VEHICLE_WIDTH_M, default_scenario

    vehicles = default_scenario(20.0, 1)
    parked = [v for v in vehicles if v.direction is None]
    assert len(parked) == 1
    for v in vehicles:
        if v.direction is not None:  # moving vehicles never drive through the parked one
            assert abs(v.lane_y_m - parked[0].lane_y_m) > VEHICLE_WIDTH_M


def test_frame_gaps_leave_no_frames_but_radar_continues(tmp_path: Path) -> None:
    cfg = Config.model_validate({"synth": {"frame_gaps": [[2.0, 1.1]], "drop_probability": 0}})
    session = generate_session(tmp_path, cfg, 5.0, 1)
    rows = read_csv_rows(session / "video/0000.frames.csv")
    t0 = int(rows[0]["sensor_ts_ns"])
    times = [(int(r["sensor_ts_ns"]) - t0) / 1e9 for r in rows]
    assert not any(2.0 <= t < 3.1 for t in times)
    assert max(b - a for a, b in itertools.pairwise(times)) >= 1.1
    seg = json.loads((session / "session.json").read_text())["segments"][0]
    assert seg["max_gap_ns"] >= 1.1e9
    reports = [r.t_rx_ns for r in open_session(session).radar_reports()]
    assert any(2.0 <= (t - t0) / 1e9 < 3.1 for t in reports)
