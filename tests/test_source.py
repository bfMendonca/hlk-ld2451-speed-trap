from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from radar_cam_recorder.reader import open_session

from speed_trap.config import RadarConfig
from speed_trap.source import RecordedSessionSource, SourceError


def copy_session(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "copy"
    shutil.copytree(src, dst)
    return dst


def append_csv_rows(path: Path, n: int) -> None:
    lines = path.read_text().splitlines()
    last = lines[-1].split(",")
    index, ts = int(last[0]), int(last[1])
    with open(path, "a") as fh:
        for k in range(1, n + 1):
            fh.write(
                f"{index + k},{ts + k * 40_000_000},{(ts + k * 40_000_000) // 1000},20000,100.0\n"
            )


def count_reports_within_video_span(session: Path) -> int:
    s = open_session(session)
    last = max(f.sensor_ts_ns for f in s.frames())
    return sum(1 for r in s.radar_reports() if r.t_rx_ns <= last)


def test_frames_in_order_with_images(synth_session: Path) -> None:
    b = list(RecordedSessionSource(synth_session, RadarConfig()))
    assert [x.ts_ns for x in b] == sorted(x.ts_ns for x in b)
    assert b[0].image.shape == (972, 1296, 3) and b[0].image.dtype == np.uint8


def test_properties(synth_session: Path) -> None:
    src = RecordedSessionSource(synth_session, RadarConfig())
    assert (src.width, src.height) == (1296, 972)
    assert src.segments == [0, 1]
    assert src.has_radar and not src.ntp_synced
    assert src.radar_max_distance_m == 100 and src.radar_direction_param == 2
    assert src.fps == pytest.approx(25.0)
    assert src.warnings == []


def test_every_scan_delivered_once_after_its_time(synth_session: Path) -> None:
    src = RecordedSessionSource(synth_session, RadarConfig())
    bundles = list(src)
    scans = [s for b in bundles for s in b.scans]
    assert len(scans) == count_reports_within_video_span(synth_session)
    assert all(s.t_ns <= b.ts_ns for b in bundles for s in b.scans)


def test_clock_offset_shifts_scans(synth_session: Path) -> None:
    base = [s.t_ns for b in RecordedSessionSource(synth_session, RadarConfig()) for s in b.scans]
    shifted = [
        s.t_ns
        for b in RecordedSessionSource(synth_session, RadarConfig(clock_offset_ms=50.0))
        for s in b.scans
    ]
    # Formula in the brief: t = t_rx - latency - clock_offset_ms * 1e6 (so +50 ms is earlier).
    assert shifted and set(shifted) & {t - 50_000_000 for t in base}
    assert min(shifted) == min(base) - 50_000_000


def test_approaching_byte_flips_direction(synth_session: Path) -> None:
    def flags(cfg: RadarConfig) -> list[bool]:
        return [
            t.approaching
            for b in RecordedSessionSource(synth_session, cfg)
            for s in b.scans
            for t in s.targets
        ]

    a, b = flags(RadarConfig()), flags(RadarConfig(approaching_byte=1))
    assert a and len(a) == len(b)
    assert all(x != y for x, y in zip(a, b, strict=True))


def test_truncated_video_warns_and_stops(tmp_path: Path, synth_session: Path) -> None:
    s = copy_session(synth_session, tmp_path)
    append_csv_rows(s / "video/0000.frames.csv", 5)
    src = RecordedSessionSource(s, RadarConfig())
    n = len(list(src))
    assert any("segment 0: video has" in w for w in src.warnings)
    assert n == 234 + 243


def test_missing_radar_dir(tmp_path: Path, synth_session: Path) -> None:
    s = copy_session(synth_session, tmp_path)
    shutil.rmtree(s / "radar")
    src = RecordedSessionSource(s, RadarConfig())
    assert not src.has_radar and all(b.scans == () for b in src)
    assert any("radar" in w for w in src.warnings)


def test_bad_time_range_names_valid_range(synth_session: Path) -> None:
    with pytest.raises(SourceError, match=r"0\.\.\d+"):
        RecordedSessionSource(synth_session, RadarConfig(), start_s=999)


def test_unknown_segment(synth_session: Path) -> None:
    with pytest.raises(SourceError, match="segments: 0, 1"):
        RecordedSessionSource(synth_session, RadarConfig(), segments=[7])


def test_segment_and_time_selection(synth_session: Path) -> None:
    only1 = list(RecordedSessionSource(synth_session, RadarConfig(), segments=[1]))
    assert {b.segment for b in only1} == {1} and len(only1) == 243
    sub = list(RecordedSessionSource(synth_session, RadarConfig(), start_s=1.0, end_s=2.0))
    assert 24 <= len(sub) <= 27
    assert sub[0].ts_ns - 100_000_000_000 >= 1_000_000_000


def test_wall_time_from_frames_csv(synth_session: Path) -> None:
    first = next(iter(RecordedSessionSource(synth_session, RadarConfig())))
    row = (synth_session / "video/0000.frames.csv").read_text().splitlines()[1].split(",")
    assert first.wall_time_ns == int(row[2]) * 1000


def test_wall_time_from_clock_samples(tmp_path: Path, synth_session: Path) -> None:
    s = copy_session(synth_session, tmp_path)
    for f in (s / "video").glob("*.frames.csv"):
        f.write_text(
            "\n".join(
                ",".join([*r.split(",")[:2], "", *r.split(",")[3:]])
                for r in f.read_text().splitlines()
            )
            + "\n"
        )
    b = list(RecordedSessionSource(s, RadarConfig()))
    assert b[0].wall_time_ns == 1792562400000000000
    assert b[1].wall_time_ns == 1792562400040000000


def test_radar_latency_is_subtracted(tmp_path: Path, synth_session: Path) -> None:
    s = copy_session(synth_session, tmp_path)
    meta = json.loads((s / "session.json").read_text())
    meta["radar_latency_ns"] = 20_000_000
    (s / "session.json").write_text(json.dumps(meta))
    base = [t.t_ns for b in RecordedSessionSource(synth_session, RadarConfig()) for t in b.scans]
    lat = [t.t_ns for b in RecordedSessionSource(s, RadarConfig()) for t in b.scans]
    assert min(lat) == min(base) - 20_000_000


def test_frame_counts_follow_selection(synth_session: Path) -> None:
    src = RecordedSessionSource(synth_session, RadarConfig(), segments=[1], start_s=12.0)
    assert set(src.frame_counts) == {1}
    assert src.frame_counts[1] == len(list(src))


def test_empty_selection_names_segment_ranges(synth_session: Path) -> None:
    with pytest.raises(SourceError, match=r"no frames.*segment 0: 0\.0\.\.\d+\.\d s"):
        RecordedSessionSource(synth_session, RadarConfig(), segments=[0], start_s=15.0)


def test_missing_video_file_is_a_source_error(synth_session: Path, tmp_path: Path) -> None:
    session = copy_session(synth_session, tmp_path)
    (session / "video" / "0001.h264").unlink()
    RecordedSessionSource(session, RadarConfig(), segments=[0])  # not selected: fine
    with pytest.raises(SourceError, match=r"0001\.h264"):
        RecordedSessionSource(session, RadarConfig())
