"""Vehicle detectors: a common interface plus RF-DETR, Ultralytics and ground-truth backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from speed_trap.config import ConfigError, DetectorConfig
from speed_trap.types import Detection, FrameBundle


class Detector(Protocol):
    name: str

    def detect(self, bundle: FrameBundle) -> list[Detection]: ...


def apply_filters(dets: list[Detection], cfg: DetectorConfig) -> list[Detection]:
    """Keep configured classes, enough confidence, and boxes with bottom-centre inside the ROI."""
    classes = set(cfg.classes)
    roi = None if cfg.roi is None else np.asarray(cfg.roi, dtype=np.float32).reshape(-1, 1, 2)
    out: list[Detection] = []
    for d in dets:
        if d.class_name not in classes or d.confidence < cfg.conf:
            continue
        if roi is not None:
            x1, _, x2, y2 = d.xyxy
            if cv2.pointPolygonTest(roi, ((x1 + x2) / 2.0, float(y2)), False) < 0:
                continue
        out.append(d)
    return out


def make_detector(cfg: DetectorConfig) -> Detector:
    if cfg.backend == "ground_truth":
        from speed_trap.detect.truth import GroundTruthDetector

        return GroundTruthDetector(Path(cfg.weights))
    if cfg.backend == "rfdetr":
        from speed_trap.detect.rfdetr import RfDetrDetector

        return RfDetrDetector(cfg)
    if cfg.backend == "ultralytics":
        from speed_trap.detect.yolo import YoloDetector

        return YoloDetector(cfg)
    raise ConfigError(f"unknown detector.backend {cfg.backend!r}")


__all__ = ["Detector", "apply_filters", "make_detector"]
