"""Shared dataclasses passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Direction = Literal["coming", "going"]
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class RadarTarget:
    t_ns: int
    angle_deg: int
    distance_m: int
    approaching: bool
    speed_kmh: int
    snr: int
    index: int


@dataclass(frozen=True)
class RadarScan:
    t_ns: int
    targets: tuple[RadarTarget, ...]


@dataclass(frozen=True)
class FrameBundle:
    image: np.ndarray  # BGR uint8, H x W x 3
    ts_ns: int
    segment: int
    index: int
    wall_time_ns: int | None
    scans: tuple[RadarScan, ...]


@dataclass(frozen=True)
class Detection:
    xyxy: Box
    confidence: float
    class_name: str


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    xyxy: Box
    confidence: float
    class_name: str
    detected: bool


@dataclass(frozen=True)
class RadarMatch:
    t_ns: int
    target: RadarTarget
    road_xy: tuple[float, float]
    speed_kmh: float
    speed_raw_kmh: int
