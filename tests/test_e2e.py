"""End to end: synthetic session (20 s, seed 1, limit 40) -> ``speed-trap process``."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import av
import pytest

from speed_trap.cli import main
from speed_trap.config import Config
from speed_trap.pipeline import Pipeline
from speed_trap.source import RecordedSessionSource
from speed_trap.types import Detection, FrameBundle

Box = tuple[float, float, float, float]


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text("speed:\n  limit_kmh: 40\n", encoding="utf-8")
    return p


def truth_detector(session: Path) -> list[str]:
    weights = session / "truth.json"
    return ["--set", "detector.backend=ground_truth", "--set", f"detector.weights={weights}"]


def truth(session: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((session / "truth.json").read_text(encoding="utf-8"))
    return doc


def expected_vehicles(session: Path) -> list[dict[str, Any]]:
    return [v for v in truth(session)["vehicles"] if v["expect_capture"]]


def decoded_frame_count(path: Path) -> int:
    with av.open(str(path)) as c:
        return sum(1 for _ in c.decode(video=0))


def csv_rows(path: Path) -> int:
    with open(path, encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def load_sidecars(captures: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(captures.glob("*.json"))]


def load_tracks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def truth_vehicle_at(session: Path, segment: int, index: int, box: Box) -> int | None:
    """Truth vehicle id whose box best overlaps ``box`` on that frame (IoU > 0.5)."""
    best, best_iou = None, 0.5
    for item in truth(session)["frames"].get(f"{segment}:{index}", []):
        v = iou(box, tuple(item["xyxy"]))
        if v > best_iou:
            best, best_iou = item["vehicle_id"], v
    return best


def frame_keys(session: Path) -> dict[int, tuple[int, int]]:
    """sensor_ts_ns -> (segment, index) for every frame of the session."""
    out: dict[int, tuple[int, int]] = {}
    for p in sorted((session / "video").glob("*.frames.csv")):
        seg = int(p.name.split(".")[0])
        with open(p, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                out[int(row["sensor_ts_ns"])] = (seg, int(row["index"]))
    return out


def tracks_matching(session: Path, tracks: list[dict[str, Any]], vehicle_id: int) -> list[Any]:
    """Tracks whose boxes mostly sit on ``vehicle_id``'s truth boxes."""
    keys = frame_keys(session)
    out = []
    for tr in tracks:
        hits = [
            truth_vehicle_at(session, *keys[b[0]], tuple(b[1:]))
            for b in tr["boxes"]
            if b[0] in keys
        ]
        if hits and hits.count(vehicle_id) > len(hits) / 2:
            out.append(tr)
    return out


def test_lite_end_to_end(tmp_path: Path, synth_session: Path, cfg_file: Path) -> None:
    out = tmp_path / "out"
    args = ["process", str(synth_session), "--config", str(cfg_file), "--mode", "lite"]
    assert main([*args, "--out", str(out), "--capture"]) == 0

    for seg in (0, 1):
        assert decoded_frame_count(out / f"{seg:04d}.overlay.mp4") == csv_rows(
            synth_session / "video" / f"{seg:04d}.frames.csv"
        )
    events = load_sidecars(out / "captures")
    expected = expected_vehicles(synth_session)
    assert expected
    for v in expected:
        assert any(
            e["direction"] == v["direction"] and abs(e["speed_kmh"]["max"] - v["speed_kmh"]) <= 2
            for e in events
        ), (v, [(e["direction"], e["speed_kmh"]["max"]) for e in events])
    assert len(events) == len(expected)  # one event per passing vehicle
    assert all(e["speed_kmh"]["max"] > e["limit_kmh"] for e in events)
    assert all(
        (out / "captures" / f"{p.stem}.jpg").exists() for p in (out / "captures").glob("*.json")
    )
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["counters"]["captures"] == len(events)
    assert run["counters"]["frames"] == sum(
        csv_rows(p) for p in (synth_session / "video").glob("*.frames.csv")
    )
    assert run["counters"]["scans"] > 0
    assert any("intrinsics" in w for w in run["warnings"])


def test_full_end_to_end_with_truth_detector(
    tmp_path: Path, synth_session: Path, cfg_file: Path
) -> None:
    out = tmp_path / "out"
    args = ["process", str(synth_session), "--config", str(cfg_file), "--out", str(out)]
    rc = main([*args, "--capture", *truth_detector(synth_session)])
    assert rc == 0
    events = load_sidecars(out / "captures")
    vehicles = {v["vehicle_id"]: v for v in truth(synth_session)["vehicles"]}
    expected = {v["vehicle_id"] for v in expected_vehicles(synth_session)}
    matched = [
        truth_vehicle_at(
            synth_session,
            e["segment"],
            e["capture"]["frame_index"],
            tuple(e["capture"]["bbox_xyxy"]),
        )
        for e in events
    ]
    assert len(set(matched)) == len(matched), matched  # one event per vehicle
    assert set(matched) == expected
    for e, vid in zip(events, matched, strict=True):
        assert vid is not None
        assert e["speed_kmh"]["max"] <= vehicles[vid]["speed_kmh"] + 1  # never overstated
        assert e["direction"] == vehicles[vid]["direction"]

    tracks = load_tracks(out / "tracks.jsonl")
    stopped = [v["vehicle_id"] for v in vehicles.values() if v["direction"] is None]
    assert len(stopped) == 1
    stopped_tracks = tracks_matching(synth_session, tracks, stopped[0])
    assert stopped_tracks
    assert not any(t["reliable_ever"] for t in stopped_tracks)

    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["counters"]["captures"] == len(events)
    assert run["counters"]["tracks"] == len(tracks)
    assert run["counters"]["matches"] == sum(len(t["matches"]) for t in tracks)
    assert run["counters"]["matches"] > 0


def test_session_without_radar_all_gray(
    tmp_path: Path, synth_session: Path, cfg_file: Path
) -> None:
    session = tmp_path / "s000001"
    shutil.copytree(synth_session, session)
    shutil.rmtree(session / "radar")
    out = tmp_path / "out"
    args = ["process", str(session), "--config", str(cfg_file), "--out", str(out)]
    rc = main([*args, "--capture", "--segment", "0", *truth_detector(session)])
    assert rc == 0
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["counters"]["scans"] == 0
    assert run["counters"]["matches"] == 0
    assert run["counters"]["captures"] == 0
    assert any("no radar" in w for w in run["warnings"])
    tracks = load_tracks(out / "tracks.jsonl")
    assert tracks
    assert not any(t["reliable_ever"] for t in tracks)
    assert not list(out.glob("captures/*.json"))


class CountingDetector:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, bundle: FrameBundle) -> list[Detection]:
        self.calls += 1
        return []


def test_detector_runs_every_n_frames(tmp_path: Path, synth_session: Path) -> None:
    cfg = Config.model_validate({"detector": {"every_n_frames": 3}})
    source = RecordedSessionSource(synth_session, cfg.radar, segments=[1])
    det = CountingDetector()
    progress: list[tuple[int, int]] = []
    counters = Pipeline(cfg, source, tmp_path / "out", "full", det).run(
        lambda seg, n, _fps: progress.append((seg, n))
    )
    frames = source.frame_counts[1]
    assert counters["frames"] == frames
    assert det.calls == (frames + 2) // 3
    assert progress == [(1, frames)]
