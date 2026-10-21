"""Ground-truth detector: replays boxes from a synthetic session's ``truth.json``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from speed_trap.config import ConfigError
from speed_trap.types import Box, Detection, FrameBundle


class GroundTruthDetector:
    name = "ground_truth"

    def __init__(self, truth_path: Path, noise_px: float = 2.0, seed: int = 0) -> None:
        try:
            doc = json.loads(Path(truth_path).read_text(encoding="utf-8"))
            self._frames: dict[str, list[dict[str, object]]] = doc["frames"]
        except (OSError, ValueError, KeyError) as exc:
            raise ConfigError(f"cannot read ground truth {str(truth_path)!r}: {exc}") from None
        self._noise = noise_px
        self._rng = np.random.default_rng(seed)

    def detect(self, bundle: FrameBundle) -> list[Detection]:
        out: list[Detection] = []
        for item in self._frames.get(f"{bundle.segment}:{bundle.index}", []):
            raw = item["xyxy"]
            assert isinstance(raw, list)
            jitter = self._rng.normal(0.0, self._noise, 4) if self._noise > 0 else np.zeros(4)
            box: Box = (
                float(raw[0] + jitter[0]),
                float(raw[1] + jitter[1]),
                float(raw[2] + jitter[2]),
                float(raw[3] + jitter[3]),
            )
            out.append(Detection(box, 0.9, "car"))
        return out
