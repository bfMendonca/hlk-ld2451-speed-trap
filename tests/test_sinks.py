from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from speed_trap.config import CaptureConfig, Config, config_hash
from speed_trap.fusion import FusedTrack
from speed_trap.lite import RadarSubject
from speed_trap.monitor import Candidate, CaptureEvent
from speed_trap.sinks import CaptureSink, TracksSink, VideoSink, write_run_info
from speed_trap.types import RadarMatch, RadarTarget

MS = 1_000_000


def decoded_pts_ms(path: Path) -> list[float]:
    out: list[float] = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for f in c.decode(s):
            assert f.pts is not None
            out.append(float(f.pts * f.time_base * 1000))
    return out


def imread(path: str) -> np.ndarray:
    img = cv2.imread(path)
    assert img is not None
    return img


def blank() -> np.ndarray:
    return np.zeros((240, 320, 3), np.uint8)


def mk(i: int, speed: float = 45.0) -> RadarMatch:
    t = RadarTarget(i * MS, 5, 20, True, int(speed), 12, 0)
    return RadarMatch(i * MS, t, (20.0 + i, 1.0), speed, int(speed))


def test_video_sink_vfr_timing(tmp_path: Path) -> None:
    ts = [0, 40, 80, 200, 240]  # ms, one 120 ms drop
    sink = VideoSink(tmp_path, 320, 240)
    for t in ts:
        sink.write(0, t * MS, blank())
    sink.close()
    assert decoded_pts_ms(tmp_path / "0000.overlay.mp4") == pytest.approx(ts, abs=1)


def test_video_sink_pts_relative_to_segment_start(tmp_path: Path) -> None:
    sink = VideoSink(tmp_path, 320, 240)
    for t in (5000, 5040, 5080):
        sink.write(0, t * MS, blank())
    sink.close()
    assert decoded_pts_ms(tmp_path / "0000.overlay.mp4") == pytest.approx([0, 40, 80], abs=1)


def test_video_sink_one_file_per_segment(tmp_path: Path) -> None:
    sink = VideoSink(tmp_path, 320, 240)
    for seg, base in ((0, 0), (0, 40), (2, 9000), (2, 9040)):
        sink.write(seg, base * MS, blank())
    sink.close()
    assert sorted(p.name for p in tmp_path.glob("*.mp4")) == [
        "0000.overlay.mp4",
        "0002.overlay.mp4",
    ]
    assert len(decoded_pts_ms(tmp_path / "0002.overlay.mp4")) == 2
    with av.open(str(tmp_path / "0000.overlay.mp4")) as c:
        assert c.streams.video[0].codec_context.name == "h264"


def test_tracks_sink_one_line_per_subject(tmp_path: Path) -> None:
    track = FusedTrack(7, "car", (10, 20, 110, 90), 0.9, 0, 3 * MS)
    track.boxes = [(0, (10.0, 20.0, 110.0, 90.0)), (MS, (12.0, 20.0, 112.0, 90.0))]
    track.road_points = [(0, (5.0, 1.0)), (MS, None)]
    track.matches = [mk(1), mk(2, 50.0)]
    track.reliable_ever = True
    track.direction = "coming"
    radar = RadarSubject(3, [mk(1, 30.0)], "going")
    path = tmp_path / "tracks.jsonl"
    sink = TracksSink(path)
    sink.write([track, radar])
    sink.write([])
    sink.close()
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(lines) == 2
    a, b = lines
    assert (a["id"], a["kind"], a["class"], a["direction"]) == (7, "track", "car", "coming")
    assert a["first_ts_ns"] == 0 and a["last_ts_ns"] == 3 * MS and a["reliable_ever"] is True
    assert a["max_speed_kmh"] == 50.0 and a["mean_speed_kmh"] == 47.5
    assert a["boxes"][1] == [MS, 12.0, 20.0, 112.0, 90.0]
    assert a["road"] == [[0, 5.0, 1.0], None]  # per box, null where back-projection fails
    assert a["radar_road"] == [[MS, 21.0, 1.0], [2 * MS, 22.0, 1.0]]
    assert a["matches"][0] == {
        "t_ns": MS,
        "speed_kmh": 45.0,
        "speed_raw_kmh": 45,
        "distance_m": 20,
        "angle_deg": 5,
        "snr": 12,
        "approaching": True,
    }
    assert (b["id"], b["kind"], b["class"], b["boxes"]) == (3, "radar", None, [])
    assert b["first_ts_ns"] == MS and b["last_ts_ns"] == MS and b["road"] == []
    assert b["radar_road"] == [[MS, 21.0, 1.0]]


def event(
    sid: int = 1, xyxy: tuple[float, float, float, float] | None = (100, 100, 300, 200)
) -> CaptureEvent:
    cand = Candidate(sid, 2, 17, 5 * MS, xyxy, 0.9)
    return CaptureEvent(sid, "coming", 40.0, 47.0, 44.0, 46.0, cand, 0.8, (mk(1), mk(2)))


def capture_sink(tmp_path: Path, ntp: bool = True, **kw: object) -> CaptureSink:
    return CaptureSink(tmp_path, "sess", CaptureConfig(**kw), ntp, "abc123")


def test_capture_crop_and_sidecar(tmp_path: Path) -> None:
    sink = capture_sink(tmp_path)
    raw = np.full((480, 640, 3), 100, np.uint8)
    wall = 1_700_000_000_000_000_000
    ev = event()
    sink.offer([ev.capture], raw, raw + 1, wall)
    paths = sink.write([ev])
    jpg = tmp_path / "captures" / "0002_t1_47kmh.jpg"
    side = jpg.with_suffix(".json")
    assert set(paths) == {jpg, side}
    img = imread(str(jpg))
    assert img.shape == (120, 240, 3)
    d = json.loads(side.read_text())
    assert d["speed_kmh"] == {"max": 47.0, "mean": 44.0, "at_capture": 46.0}
    assert d["event"] == "speeding" and d["segment"] == 2
    assert d["limit_kmh"] == 40.0 and d["direction"] == "coming" and d["track_id"] == 1
    assert d["session"] == "sess" and d["config_sha256"] == "abc123"
    wall_iso = datetime.fromtimestamp(1_700_000_000, timezone.utc).isoformat()
    assert d["capture"] == {
        "frame_index": 17,
        "sensor_ts_ns": 5 * MS,
        "wall_clock": wall_iso,
        "wall_clock_reliable": True,
        "bbox_xyxy": [100, 100, 300, 200],
        "detector_conf": 0.8,
    }
    assert len(d["radar_history"]) == 2 and d["radar_history"][0]["speed_kmh"] == 45.0
    assert set(d["radar_history"][0]) == {
        "t_ns",
        "speed_kmh",
        "speed_raw_kmh",
        "distance_m",
        "angle_deg",
        "snr",
        "approaching",
    }
    assert not (tmp_path / "captures" / "0002_t1_47kmh_full.jpg").exists()


def test_capture_crop_clipped_and_full_frame(tmp_path: Path) -> None:
    sink = capture_sink(tmp_path, save_full_frame=True)
    raw = np.zeros((480, 640, 3), np.uint8)
    ev = event(xyxy=(0, 0, 100, 100))
    sink.offer([ev.capture], raw, raw, None)
    sink.write([ev])
    assert imread(str(tmp_path / "captures" / "0002_t1_47kmh.jpg")).shape == (110, 110, 3)
    assert imread(str(tmp_path / "captures" / "0002_t1_47kmh_full.jpg")).shape == (480, 640, 3)


def test_capture_lite_uses_rendered_frame(tmp_path: Path) -> None:
    sink = capture_sink(tmp_path)
    raw = np.zeros((480, 640, 3), np.uint8)
    rendered = np.full((480, 640, 3), 200, np.uint8)
    ev = event(xyxy=None)
    sink.offer([ev.capture], raw, rendered, None)
    sink.write([ev])
    jpg = tmp_path / "captures" / "0002_t1_47kmh.jpg"
    assert imread(str(jpg)).shape == (480, 640, 3)
    d = json.loads(jpg.with_suffix(".json").read_text())
    assert d["capture"]["bbox_xyxy"] is None and d["capture"]["wall_clock"] is None


def test_capture_keeps_only_best_candidate_memory(tmp_path: Path) -> None:
    sink = capture_sink(tmp_path)
    first = np.full((480, 640, 3), 10, np.uint8)
    second = np.full((480, 640, 3), 250, np.uint8)
    c1 = Candidate(1, 2, 16, 4 * MS, (100, 100, 200, 200), 0.9)
    ev = event()
    sink.offer([c1], first, first, None)
    sink.offer([ev.capture], second, second, None)
    assert len(sink._pending) == 1
    sink.write([ev])
    img = imread(str(tmp_path / "captures" / "0002_t1_47kmh.jpg"))
    assert img.mean() > 200
    assert sink._pending == {}


def test_capture_discard_and_missing(tmp_path: Path) -> None:
    sink = capture_sink(tmp_path)
    raw = np.zeros((480, 640, 3), np.uint8)
    ev = event()
    sink.offer([ev.capture], raw, raw, None)
    sink.discard([1])
    assert sink.write([ev]) == []
    assert not (tmp_path / "captures").exists() or not list((tmp_path / "captures").iterdir())


@pytest.mark.parametrize("ntp", [True, False])
def test_capture_wall_clock_reliability_flag(tmp_path: Path, ntp: bool) -> None:
    sink = capture_sink(tmp_path, ntp=ntp)
    raw = np.zeros((480, 640, 3), np.uint8)
    ev = event()
    sink.offer([ev.capture], raw, raw, 1_700_000_000_000_000_000)
    sink.write([ev])
    d = json.loads((tmp_path / "captures" / "0002_t1_47kmh.json").read_text())
    assert d["capture"]["wall_clock_reliable"] is ntp


def test_run_info_contains_hash_and_counters(tmp_path: Path) -> None:
    cfg = Config()
    path = tmp_path / "run.json"
    write_run_info(
        path,
        cfg,
        Path("/data/session"),
        {"frames": 10, "tracks": 2},
        1,
        2_000_000_001,
        ["w"],
        model="rfdetr-small",
    )
    d = json.loads(path.read_text())
    assert d["config_hash"] == config_hash(cfg)
    assert d["config"]["speed"]["limit_kmh"] == cfg.speed.limit_kmh
    assert d["counters"] == {"frames": 10, "tracks": 2}
    assert d["warnings"] == ["w"] and d["session_path"] == "/data/session"
    assert d["started_ns"] == 1 and d["finished_ns"] == 2_000_000_001
    assert d["model"] == "rfdetr-small"
    write_run_info(path, cfg, Path("/s"), {}, 1, 2, [], model=None)  # lite mode
    assert json.loads(path.read_text())["model"] is None
    assert "speed_trap" in d["versions"] and "ld2451" in d["versions"]


def test_capture_box_outside_frame_skipped(tmp_path: Path) -> None:
    sink = capture_sink(tmp_path)
    raw = np.zeros((480, 640, 3), np.uint8)
    ev = event(xyxy=(700, 100, 800, 200))
    sink.offer([ev.capture], raw, raw, None)
    assert sink.write([ev]) == []


def test_video_sink_rejects_bad_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="even"):
        VideoSink(tmp_path, 321, 240)
    sink = VideoSink(tmp_path, 320, 240)
    with pytest.raises(ValueError, match="shape"):
        sink.write(0, 0, np.zeros((100, 320, 3), np.uint8))


def test_video_sink_rejects_unknown_codec(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="codec"):
        VideoSink(tmp_path, 64, 48, codec="vp9")


def test_capture_filename_floors_the_speed(tmp_path: Path) -> None:
    sink = capture_sink(tmp_path)
    raw = np.zeros((480, 640, 3), np.uint8)
    ev = event()
    ev = CaptureEvent(1, "coming", 40.0, 47.6, 44.0, 46.0, ev.capture, 0.8, ev.history)
    sink.offer([ev.capture], raw, raw, None)
    sink.write([ev])
    assert (tmp_path / "captures" / "0002_t1_47kmh.jpg").exists()
