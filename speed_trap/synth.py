"""Synthetic road session in the ``radar-cam-recorder`` format, plus ground truth."""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import av
import cv2
import ld2451
import numpy as np
from ld2451.protocol import encode_report

from speed_trap.config import Config
from speed_trap.geometry import CameraModel, RadarModel
from speed_trap.types import Direction

COMING_LANE_Y = 1.75
GOING_LANE_Y = -1.75
PARKED_Y = 4.0  # half on the kerb beside the coming lane, out of the moving traffic
VEHICLE_WIDTH_M = 1.8
VEHICLE_HEIGHT_M = 1.5
_NS = 1_000_000_000
_BOOT_START_NS = 100 * _NS
_SESSION_ID = "s000001"
_RADAR_RANGE_MAX_M = 100.0
_MAX_TARGETS = 5
_EMPTY_REPORT_S = 1.4
_SPLIT_DELAY_NS = 2_000_000
_MIN_VISIBLE_FRACTION = 0.3
_EXPECT_MARGIN_KMH = 1.0
_PALETTE = [
    (40, 40, 200),
    (200, 120, 40),
    (60, 160, 60),
    (190, 190, 190),
    (30, 180, 220),
    (160, 50, 140),
]


@dataclass(frozen=True)
class SynthVehicle:
    vehicle_id: int
    direction: Direction | None  # None = stopped
    lane_y_m: float
    speed_kmh: float
    start_x_m: float
    start_s: float

    def x_at(self, t: float) -> float | None:
        """Road X at timeline second ``t``, or None before the vehicle appears."""
        if t < self.start_s:
            return None
        if self.direction is None:
            return self.start_x_m
        sign = -1.0 if self.direction == "coming" else 1.0
        return self.start_x_m + sign * self.speed_kmh / 3.6 * (t - self.start_s)

    @property
    def vx_ms(self) -> float:
        if self.direction is None:
            return 0.0
        return (-1.0 if self.direction == "coming" else 1.0) * self.speed_kmh / 3.6


def default_scenario(duration_s: float, seed: int) -> list[SynthVehicle]:
    vehicles = [
        SynthVehicle(1, "coming", COMING_LANE_Y, 60.0, 60.0, 1.0),
        SynthVehicle(2, "going", GOING_LANE_Y, 30.0, 10.0, 0.5),
        SynthVehicle(3, "going", GOING_LANE_Y, 50.0, 8.0, 6.0),
        SynthVehicle(4, None, PARKED_Y, 0.0, 20.0, 0.0),
    ]
    rng = random.Random(seed)
    for i in range(max(int(duration_s / 3), 1)):
        coming = rng.random() < 0.5
        vehicles.append(
            SynthVehicle(
                5 + i,
                "coming" if coming else "going",
                COMING_LANE_Y if coming else GOING_LANE_Y,
                round(rng.uniform(10.0, 34.0), 1),
                round(rng.uniform(40.0, 90.0) if coming else rng.uniform(8.0, 20.0), 1),
                round(rng.uniform(0.0, duration_s), 2),
            )
        )
    return vehicles


# --- geometry helpers -----------------------------------------------------------------------


def _radar_view(
    radar: RadarModel, v: SynthVehicle, x: float
) -> tuple[float, float, float, float] | None:
    """(ground range, angle deg, slant range, radial km/h; + = receding) if inside the wedge."""
    ground, angle = radar.road_to_polar(x, v.lane_y_m)
    if abs(angle) > radar.fov_half_deg or not radar.min_range_m <= ground <= radar.max_range_m:
        return None
    slant = radar.slant_range(ground)
    dx = x - radar.position_xy[0]
    radial = v.vx_ms * dx / slant * 3.6
    return ground, angle, slant, radial


def _expect_capture(
    cfg: Config, radar: RadarModel, v: SynthVehicle, timeline_s: float, limit: float
) -> bool:
    if v.direction is None:
        return False
    worst = None
    for k in range(int(timeline_s * 20) + 1):
        x = v.x_at(k / 20)
        if x is None:
            continue
        view = _radar_view(radar, v, x)
        if view is None or view[0] > cfg.fusion.max_range_m:
            continue
        cos = abs(x - radar.position_xy[0]) / view[2]
        worst = cos if worst is None else min(worst, cos)
    return worst is not None and v.speed_kmh * worst > limit + _EXPECT_MARGIN_KMH


# --- rendering ------------------------------------------------------------------------------


def _background(cam: CameraModel) -> np.ndarray:
    w, h = cam.width, cam.height
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (70, 110, 80)  # ground (BGR)
    horizon = cam.K[1, 2] - cam.K[1, 1] * math.tan(math.radians(12.0))
    horizon_row = int(np.clip(horizon, 0, h))
    img[:horizon_row] = (235, 200, 160)  # sky

    def line(pts: list[tuple[float, float]], color: tuple[int, int, int], thick: int) -> None:
        uv, front = cam.project(np.array([(x, y, 0.0) for x, y in pts]))
        if front.all():
            cv2.polylines(img, [np.round(uv).astype(np.int32)], False, color, thick, cv2.LINE_AA)

    x0, x1, half = 3.0, 400.0, 3.5
    xs = np.linspace(x0, x1, 200)
    edge = [(float(x), s * half) for s in (1, -1) for x in xs]
    uv, front = cam.project(np.array([(x, y, 0.0) for x, y in edge]))
    if front.all():
        left, right = uv[:200], uv[200:]
        poly = np.round(np.vstack([left, right[::-1]])).astype(np.int32)
        cv2.fillPoly(img, [poly], (95, 95, 100), cv2.LINE_AA)
    for s in (1, -1):
        line([(float(x), s * half) for x in xs], (230, 230, 230), 2)
    for start in np.arange(x0, 200.0, 6.0):
        line([(float(start), 0.0), (float(start) + 3.0, 0.0)], (230, 230, 230), 2)
    return img


def _draw_vehicle(
    img: np.ndarray, cam: CameraModel, v: SynthVehicle, x: float
) -> tuple[float, float, float, float] | None:
    hw, y = VEHICLE_WIDTH_M / 2, v.lane_y_m
    corners = np.array(
        [
            (x, y + hw, 0.0),
            (x, y - hw, 0.0),
            (x, y - hw, VEHICLE_HEIGHT_M),
            (x, y + hw, VEHICLE_HEIGHT_M),
        ]
    )
    uv, front = cam.project(corners)
    if not front.all():
        return None
    x_min, y_min = uv.min(axis=0)
    x_max, y_max = uv.max(axis=0)
    area = (x_max - x_min) * (y_max - y_min)
    cx0, cy0 = max(x_min, 0.0), max(y_min, 0.0)
    cx1, cy1 = min(x_max, cam.width - 1.0), min(y_max, cam.height - 1.0)
    if cx1 <= cx0 or cy1 <= cy0 or area <= 0:
        return None
    if (cx1 - cx0) * (cy1 - cy0) < _MIN_VISIBLE_FRACTION * area:
        return None
    color = _PALETTE[v.vehicle_id % len(_PALETTE)]
    p0 = (round(x_min), round(y_min))
    p1 = (round(x_max), round(y_max))
    cv2.rectangle(img, p0, p1, color, -1)
    bw, bh = x_max - x_min, y_max - y_min
    window = (
        (round(x_min + 0.12 * bw), round(y_min + 0.1 * bh)),
        (round(x_max - 0.12 * bw), round(y_min + 0.45 * bh)),
    )
    cv2.rectangle(img, window[0], window[1], (50, 40, 30), -1)
    for fx in (0.15, 0.85):
        cv2.circle(
            img,
            (round(x_min + fx * bw), round(y_min + 0.65 * bh)),
            max(round(bh * 0.06), 1),
            (200, 240, 255),
            -1,
        )
    cv2.rectangle(img, p0, p1, (20, 20, 20), 1)
    return float(cx0), float(cy0), float(cx1), float(cy1)


# --- session --------------------------------------------------------------------------------


def _radar_report(
    radar: RadarModel, vehicles: list[SynthVehicle], t: float, rng: random.Random
) -> list[ld2451.Target]:
    seen = []
    for v in vehicles:
        x = v.x_at(t)
        if v.direction is None or x is None:
            continue
        view = _radar_view(radar, v, x)
        if view is None:
            continue
        _, angle, slant, radial = view
        speed, dist, ang = round(abs(radial)), round(slant), round(angle)
        if speed < 1 or speed > 255 or dist > 255 or not -128 <= ang <= 127:
            continue
        seen.append((slant, ld2451.Target(ang, dist, radial < 0, speed, rng.randint(20, 60))))
    seen.sort(key=lambda s: s[0])
    return [t_ for _, t_ in seen[:_MAX_TARGETS]]


def generate_session(
    out_dir: Path,
    cfg: Config,
    duration_s: float,
    seed: int,
    vehicles: list[SynthVehicle] | None = None,
) -> Path:
    syn = cfg.synth
    width, height = syn.resolution
    if vehicles is None:
        vehicles = default_scenario(duration_s, seed)
    cam = CameraModel.from_config(cfg.camera, width, height)
    radar = RadarModel.from_config(cfg.radar, cfg.camera.pose.height_m, _RADAR_RANGE_MAX_M)
    background = _background(cam)

    session = Path(out_dir) / _SESSION_ID
    (session / "video").mkdir(parents=True, exist_ok=True)
    (session / "radar").mkdir(parents=True, exist_ok=True)

    frame_rng = random.Random(seed * 2 + 1)
    radar_rng = random.Random(seed * 2 + 2)
    realtime0_ns = int(datetime(2026, 10, 21, 6, 0, 0, tzinfo=timezone.utc).timestamp()) * _NS
    dt = 1.0 / syn.fps
    n_segments = max(math.ceil(duration_s / syn.segment_s), 1)

    segments_meta: list[dict[str, Any]] = []
    truth_frames: dict[str, list[dict[str, Any]]] = {}
    timeline_end = 0.0

    for seg in range(n_segments):
        seg_len = min(syn.segment_s, duration_s - seg * syn.segment_s)
        t0 = seg * (syn.segment_s + syn.gap_s)
        timeline_end = t0 + seg_len
        stem = session / "video" / f"{seg:04d}"
        container = av.open(str(stem) + ".h264", "w", format="h264")
        stream = container.add_stream("libx264", rate=round(syn.fps))
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "23", "tune": "zerolatency", "bf": "0"}

        rows = ["index,sensor_ts_ns,frame_wall_clock_us,exposure_us,lux"]
        stamps: list[int] = []
        drops = 0
        for k in range(round(seg_len * syn.fps)):
            if frame_rng.random() < syn.drop_probability:
                drops += 1
                continue
            t = t0 + k * dt
            if any(start <= t < start + length for start, length in syn.frame_gaps):
                continue
            ts_ns = _BOOT_START_NS + round(t * _NS)
            index = len(stamps)
            stamps.append(ts_ns)
            img = background.copy()
            truth: list[dict[str, Any]] = []
            ordered = sorted(
                (v for v in vehicles if v.x_at(t) is not None),
                key=lambda v: -(v.x_at(t) or 0.0),
            )
            for v in ordered:
                box = _draw_vehicle(img, cam, v, v.x_at(t) or 0.0)
                if box is not None:
                    truth.append({"vehicle_id": v.vehicle_id, "xyxy": [round(c, 1) for c in box]})
            truth_frames[f"{seg}:{index}"] = truth
            wall_us = (realtime0_ns + ts_ns - _BOOT_START_NS) // 1000
            rows.append(f"{index},{ts_ns},{wall_us},20000,100.0")
            frame = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        Path(str(stem) + ".frames.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

        lines = ["t_rx_ns,hex"]
        last_emit = -math.inf
        for j in range(int(seg_len / syn.report_interval_s + 1e-9)):
            t = t0 + j * syn.report_interval_s
            targets = _radar_report(radar, vehicles, t, radar_rng)
            if not targets and t - last_emit < _EMPTY_REPORT_S - 1e-9:
                continue
            last_emit = t
            raw = encode_report(targets)
            t_rx = _BOOT_START_NS + round(t * _NS)
            lines.append(f"{t_rx},{raw[:1].hex()}")
            lines.append(f"{t_rx + _SPLIT_DELAY_NS},{raw[1:].hex()}")
        (session / "radar" / f"{seg:04d}.csv").write_text("\n".join(lines) + "\n", encoding="ascii")

        gaps = [b - a for a, b in itertools.pairwise(stamps)]
        segments_meta.append(
            {
                "drops": drops,
                "end_boottime_ns": _BOOT_START_NS + round(timeline_end * _NS),
                "frames": len(stamps),
                "index": seg,
                "max_gap_ns": max(gaps) if gaps else 0,
                "start_boottime_ns": _BOOT_START_NS + round(t0 * _NS),
            }
        )

    limit = cfg.speed.limit_kmh
    truth_doc = {
        "limit_kmh": limit,
        "vehicles": [
            {
                "vehicle_id": v.vehicle_id,
                "direction": v.direction,
                "speed_kmh": v.speed_kmh,
                "expect_capture": _expect_capture(cfg, radar, v, timeline_end, limit),
            }
            for v in vehicles
        ],
        "frames": truth_frames,
    }
    (session / "truth.json").write_text(json.dumps(truth_doc, indent=1), encoding="utf-8")

    meta = {
        "format_version": 1,
        "session_id": _SESSION_ID,
        "state": "complete",
        "ntp_synced": False,
        "radar_latency_ns": 0,
        "devices": {
            "camera": {"width": width, "height": height, "framerate": syn.fps},
            "radar": {
                "params": {
                    "direction": 2,
                    "max_distance_m": 100,
                    "min_speed_kmh": 1,
                    "no_target_delay_s": 1,
                }
            },
        },
        "segments": segments_meta,
        "clock_samples": [
            {
                "label": "start",
                "boottime_ns": _BOOT_START_NS,
                "monotonic_ns": _BOOT_START_NS,
                "realtime_ns": realtime0_ns,
            }
        ],
        "setup": {"notes": "synthetic", "geometry": {}},
    }
    (session / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return session
