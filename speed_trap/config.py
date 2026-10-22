"""YAML configuration: strict pydantic models, dotted-key overrides, stable hash."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_DISTORTION_LENGTHS = (4, 5, 8, 12, 14)


class ConfigError(ValueError):
    """Invalid configuration; the message names the offending dotted keys."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- camera ---------------------------------------------------------------------------------


class Intrinsics(_Strict):
    resolution: tuple[int, int]
    matrix: list[list[float]]
    distortion: list[float]

    @field_validator("resolution")
    @classmethod
    def _resolution_positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError("resolution must be positive")
        return v

    @field_validator("matrix")
    @classmethod
    def _matrix_3x3(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("matrix must be 3x3")
        return v

    @field_validator("distortion")
    @classmethod
    def _distortion_length(cls, v: list[float]) -> list[float]:
        if len(v) not in _DISTORTION_LENGTHS:
            raise ValueError(f"distortion must have length in {_DISTORTION_LENGTHS}, got {len(v)}")
        return v


class PoseConfig(_Strict):
    height_m: float = Field(3.5, gt=0)
    pitch_deg: float = 12.0
    roll_deg: float = 0.0
    yaw_deg: float = 0.0


class CameraConfig(_Strict):
    intrinsics: Intrinsics | None = None
    pose: PoseConfig = Field(default_factory=PoseConfig)


# --- radar ----------------------------------------------------------------------------------


class OffsetConfig(_Strict):
    x: float = 0.0
    y: float = 0.0
    z: float = -0.3


class BoresightConfig(_Strict):
    yaw_deg: float = -8.0
    pitch_deg: float = 0.0


class RangeConfig(_Strict):
    min: float = Field(1.0, gt=0)
    max: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _min_below_max(self) -> RangeConfig:
        if self.max is not None and self.min >= self.max:
            raise ValueError("min must be less than max")
        return self


class RadarConfig(_Strict):
    offset_m: OffsetConfig = Field(default_factory=OffsetConfig)
    boresight: BoresightConfig = Field(default_factory=BoresightConfig)
    fov_half_deg: float = Field(20.0, gt=0, le=90)
    range_m: RangeConfig = Field(default_factory=RangeConfig)
    angle_sign: Literal[1, -1] = 1
    approaching_byte: Literal[0, 1] = 0
    clock_offset_ms: float = 0.0
    report_hold_s: float = Field(0.3, gt=0)


class RoadConfig(_Strict):
    lane_heading_deg: float = 0.0


# --- detection, tracking, fusion ------------------------------------------------------------


class DetectorConfig(_Strict):
    backend: Literal["rfdetr", "ultralytics", "ground_truth"] = "rfdetr"
    weights: str = "rfdetr-small"
    device: str = "mps"
    imgsz: int = Field(960, ge=32)
    every_n_frames: int = Field(1, ge=1)
    conf: float = Field(0.35, ge=0, le=1)
    iou: float = Field(0.5, ge=0, le=1)
    classes: list[str] = Field(default_factory=lambda: ["car", "truck", "bus", "motorcycle"])
    roi: list[tuple[float, float]] | None = None

    @field_validator("roi")
    @classmethod
    def _roi_polygon(cls, v: list[tuple[float, float]] | None) -> list[tuple[float, float]] | None:
        if v is not None and len(v) < 3:
            raise ValueError("roi needs at least 3 points")
        return v


class TrackerConfig(_Strict):
    type: Literal["bytetrack"] = "bytetrack"
    track_activation_threshold: float = Field(0.35, ge=0, le=1)
    lost_track_s: float = Field(1.5, gt=0)
    match_threshold: float = Field(0.8, ge=0, le=1)
    minimum_consecutive_frames: int = Field(1, ge=1)


class GateConfig(_Strict):
    range_base_m: float = Field(1.5, gt=0)
    range_per_m2: float = Field(0.002, ge=0)
    angle_margin_deg: float = Field(1.0, ge=0)


class FusionConfig(_Strict):
    gate: GateConfig = Field(default_factory=GateConfig)
    ambiguity_ratio: float = Field(0.7, gt=0, le=1)
    max_range_m: float = Field(30.0, gt=0)
    lanes: list[float] | None = None
    direction_window_s: float = Field(0.5, gt=0)
    direction_min_travel_m: float = Field(0.5, gt=0)
    max_scan_age_s: float = Field(0.15, gt=0)
    speed_tolerance_kmh: float = Field(10.0, gt=0)
    speed_tolerance_ratio: float = Field(0.3, gt=0, le=1)
    reliable_min_matches: int = Field(2, ge=1)
    reliable_max_age_s: float = Field(1.5, gt=0)


class SpeedConfig(_Strict):
    limit_kmh: float = Field(40.0, gt=0)
    limit_kmh_coming: float | None = Field(None, gt=0)
    limit_kmh_going: float | None = Field(None, gt=0)
    cosine_correction: bool = False


# --- outputs --------------------------------------------------------------------------------


class CaptureConfig(_Strict):
    enabled: bool = False
    min_consecutive_over: int = Field(2, ge=1)
    crop_margin: float = Field(0.1, ge=0)
    save_full_frame: bool = False
    jpeg_quality: int = Field(95, ge=1, le=100)
    edge_margin_px: int = Field(2, ge=0)


class OverlayShowConfig(_Strict):
    wedge: bool = True
    range_rings: list[float] = Field(default_factory=lambda: [10.0, 20.0, 40.0, 70.0])
    hud: bool = True

    @field_validator("range_rings")
    @classmethod
    def _rings_positive(cls, v: list[float]) -> list[float]:
        if any(r <= 0 for r in v):
            raise ValueError("range rings must be positive")
        return v


class OverlayColors(_Strict):
    coming: str = "#35d0ba"
    going: str = "#ffb547"
    none: str = "#a7afb8"
    alert: str = "#ff4d5e"

    @field_validator("coming", "going", "none", "alert")
    @classmethod
    def _hex_color(cls, v: str) -> str:
        if len(v) != 7 or v[0] != "#":
            raise ValueError("color must look like #rrggbb")
        try:
            int(v[1:], 16)
        except ValueError:
            raise ValueError("color must look like #rrggbb") from None
        return v


class OverlayConfig(_Strict):
    style: Literal["tag"] = "tag"
    show: OverlayShowConfig = Field(default_factory=OverlayShowConfig)
    colors: OverlayColors = Field(default_factory=OverlayColors)


class LiteConfig(_Strict):
    pillar_height_m: float = Field(1.6, gt=0)
    lost_track_s: float = Field(1.5, gt=0)


class OutputConfig(_Strict):
    video_codec: Literal["h264"] = "h264"
    dir: Path = Path("out")


class SynthConfig(_Strict):
    report_interval_s: float = Field(0.1, gt=0)
    drop_probability: float = Field(0.05, ge=0, le=1)
    segment_s: float = Field(10.0, gt=0)
    gap_s: float = Field(3.0, ge=0)
    fps: float = Field(25.0, gt=0)
    resolution: tuple[int, int] = (1296, 972)
    frame_gaps: list[tuple[float, float]] = Field(default_factory=list)

    @field_validator("frame_gaps")
    @classmethod
    def _gaps_valid(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if any(start < 0 or length <= 0 for start, length in v):
            raise ValueError("frame gaps need start >= 0 and length > 0")
        return v


class Config(_Strict):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    radar: RadarConfig = Field(default_factory=RadarConfig)
    road: RoadConfig = Field(default_factory=RoadConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    speed: SpeedConfig = Field(default_factory=SpeedConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    lite: LiteConfig = Field(default_factory=LiteConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    synth: SynthConfig = Field(default_factory=SynthConfig)


# --- loading --------------------------------------------------------------------------------


def default_config_text() -> str:
    """Contents of the commented default configuration file."""
    return resources.files("speed_trap").joinpath("default_config.yaml").read_text("utf-8")


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return a copy of ``data`` with each ``"a.b=v"`` override applied (v parsed as YAML)."""
    out = copy.deepcopy(data)
    for item in overrides:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(f"invalid override {item!r}: expected KEY=VALUE")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid override {item!r}: bad value ({exc})") from None
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return out


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"{loc}: {err['msg']}")
    return "invalid configuration:\n  " + "\n  ".join(lines)


def load_config(path: Path, overrides: Sequence[str] = ()) -> Config:
    """Load and validate a YAML config file, applying ``KEY=VALUE`` overrides."""
    try:
        data = yaml.safe_load(Path(path).read_text("utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} must be a mapping at the top level")
    data = apply_overrides(data, overrides)
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_errors(exc)) from None


def config_hash(cfg: Config) -> str:
    """sha256 of the canonical (sorted-key) JSON form of the config."""
    import json

    canonical = json.dumps(json.loads(cfg.model_dump_json()), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
