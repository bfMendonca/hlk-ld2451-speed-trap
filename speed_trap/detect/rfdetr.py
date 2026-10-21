"""RF-DETR detector backend (``uv sync --extra rfdetr``)."""

from __future__ import annotations

from typing import Any

import cv2

from speed_trap.config import ConfigError, DetectorConfig
from speed_trap.detect import apply_filters
from speed_trap.types import Detection, FrameBundle

_VARIANTS = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-base": "RFDETRBase",
    "rfdetr-large": "RFDETRLarge",
}
_INSTALL_HINT = "detector.backend 'rfdetr' needs: uv sync --extra rfdetr"


class RfDetrDetector:
    name = "rfdetr"

    def __init__(self, cfg: DetectorConfig) -> None:
        try:
            import rfdetr
            from rfdetr.assets.coco_classes import COCO_CLASSES
        except ImportError:
            raise ConfigError(_INSTALL_HINT) from None
        self._cfg = cfg
        # RF-DETR's pretrained heads use sparse COCO category ids (1-90), not 0-based positions.
        self._coco: dict[int, str] = dict(COCO_CLASSES)
        cls_name = _VARIANTS.get(cfg.weights.lower())
        if cls_name is not None:
            self._model: Any = getattr(rfdetr, cls_name)(device=cfg.device)
        else:
            self._model = getattr(rfdetr, "RFDETRBase")(  # noqa: B009
                pretrain_weights=cfg.weights, device=cfg.device
            )

    def detect(self, bundle: FrameBundle) -> list[Detection]:
        rgb = cv2.cvtColor(bundle.image, cv2.COLOR_BGR2RGB)
        # RF-DETR is NMS-free, so cfg.iou does not apply. Threshold is applied by apply_filters.
        res = self._model.predict(rgb, threshold=self._cfg.conf, shape=(self._cfg.imgsz,) * 2)
        dets: list[Detection] = []
        for box, conf, cid in zip(res.xyxy, res.confidence, res.class_id, strict=True):
            name = self._coco.get(int(cid))
            if name is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            dets.append(Detection((x1, y1, x2, y2), float(conf), name))
        return apply_filters(dets, self._cfg)
