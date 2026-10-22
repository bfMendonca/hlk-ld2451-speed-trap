import builtins
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from speed_trap.config import ConfigError, DetectorConfig, RadarConfig
from speed_trap.detect import apply_filters, make_detector
from speed_trap.source import RecordedSessionSource
from speed_trap.types import Detection


def truth_boxes(session: Path, segment: int, index: int) -> list[dict[str, Any]]:
    doc = json.loads((session / "truth.json").read_text(encoding="utf-8"))
    frames: dict[str, list[dict[str, Any]]] = doc["frames"]
    return frames.get(f"{segment}:{index}", [])


def test_ground_truth_detector_returns_truth_boxes(synth_session: Path) -> None:
    det = make_detector(
        DetectorConfig(backend="ground_truth", weights=str(synth_session / "truth.json"))
    )
    count = 0
    for bundle in RecordedSessionSource(synth_session, RadarConfig()):
        expected = truth_boxes(synth_session, bundle.segment, bundle.index)
        out = det.detect(bundle)
        assert len(out) == len(expected)
        for d, t in zip(out, expected, strict=True):
            assert d.class_name == "car"
            assert d.confidence == 0.9
            assert all(abs(a - b) <= 2.0 * 6 for a, b in zip(d.xyxy, t["xyxy"], strict=True))
        count += len(expected)
        if count > 5:
            break
    assert count > 0


def test_roi_filters_by_bottom_centre() -> None:
    cfg = DetectorConfig(roi=[(0, 0), (100, 0), (100, 100), (0, 100)])
    inside = Detection((10, 10, 50, 90), 0.9, "car")  # bottom-centre (30, 90)
    outside = Detection((10, 10, 50, 150), 0.9, "car")  # bottom-centre (30, 150)
    assert apply_filters([inside, outside], cfg) == [inside]


def test_class_and_conf_filter() -> None:
    cfg = DetectorConfig(conf=0.35, classes=["car"])
    keep = Detection((0, 0, 1, 1), 0.5, "car")
    dets = [keep, Detection((0, 0, 1, 1), 0.9, "person"), Detection((0, 0, 1, 1), 0.2, "car")]
    assert apply_filters(dets, cfg) == [keep]


def test_missing_extra_message(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"rfdetr", "ultralytics"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ConfigError, match="uv sync --extra rfdetr"):
        make_detector(DetectorConfig(backend="rfdetr"))
    with pytest.raises(ConfigError, match="uv sync --extra yolo"):
        make_detector(DetectorConfig(backend="ultralytics", weights="yolo11n.pt"))


def _black_bundle() -> Any:
    from speed_trap.types import FrameBundle

    return FrameBundle(np.zeros((972, 1296, 3), np.uint8), 0, 0, 0, None, ())


@pytest.mark.skipif(importlib.util.find_spec("rfdetr") is None, reason="rfdetr extra not installed")
def test_rfdetr_smoke() -> None:
    det = make_detector(DetectorConfig(backend="rfdetr", weights="rfdetr-nano", device="cpu"))
    assert isinstance(det.detect(_black_bundle()), list)


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None, reason="yolo extra not installed"
)
def test_yolo_smoke() -> None:
    det = make_detector(DetectorConfig(backend="ultralytics", weights="yolo11n.pt", device="cpu"))
    assert isinstance(det.detect(_black_bundle()), list)


def test_default_imgsz_is_rfdetr_small_native_resolution() -> None:
    assert DetectorConfig().imgsz == 512


@pytest.mark.skipif(importlib.util.find_spec("rfdetr") is None, reason="rfdetr extra not installed")
def test_rfdetr_rejects_imgsz_not_divisible_by_the_block_size() -> None:
    cfg = DetectorConfig(backend="rfdetr", weights="rfdetr-nano", device="cpu", imgsz=500)
    with pytest.raises(ConfigError, match=r"detector\.imgsz.*multiple of 32"):
        make_detector(cfg)
