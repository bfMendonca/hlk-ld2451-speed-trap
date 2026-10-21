"""Ultralytics YOLO detector backend (``uv sync --extra yolo``; AGPL-3.0 terms apply)."""

from __future__ import annotations

from typing import Any

from speed_trap.config import ConfigError, DetectorConfig
from speed_trap.detect import apply_filters
from speed_trap.types import Detection, FrameBundle


class YoloDetector:
    name = "ultralytics"

    def __init__(self, cfg: DetectorConfig) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ConfigError(
                "detector.backend 'ultralytics' needs: uv sync --extra yolo"
            ) from None
        self._cfg = cfg
        self._model: Any = YOLO(cfg.weights)

    def detect(self, bundle: FrameBundle) -> list[Detection]:
        cfg = self._cfg
        results = self._model.predict(
            bundle.image,  # BGR ndarray, the format Ultralytics expects
            imgsz=cfg.imgsz,
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device,
            verbose=False,
        )
        names: dict[int, str] = self._model.names
        dets: list[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            dets.append(Detection((x1, y1, x2, y2), float(box.conf[0]), names[int(box.cls[0])]))
        return apply_filters(dets, cfg)
