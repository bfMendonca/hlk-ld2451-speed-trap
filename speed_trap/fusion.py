"""Radar-track fusion: attach radar readings to tracked vehicles (spec §5.4-5.5).

Safety first: a pair is only kept when it is inside a distance-aware gate, within
``fusion.max_range_m``, consistent in direction and lane, and unambiguous. When in doubt,
no match. Never reads ahead in time; no I/O.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from speed_trap.config import FusionConfig, RoadConfig, SpeedConfig
from speed_trap.geometry import CameraModel, RadarModel
from speed_trap.types import Box, Direction, RadarMatch, RadarScan, RadarTarget, TrackObservation

_INVALID = 1e6
_RANGE_STEP_HALF_M = 0.5  # radar range quantisation, ±0.5 m
_ANGLE_STEP_HALF_DEG = 0.5  # radar angle quantisation, ±0.5 deg
_MIN_COSINE = 0.5


@dataclass
class FusedTrack:
    track_id: int
    class_name: str
    xyxy: Box
    confidence: float
    first_ts_ns: int
    last_ts_ns: int
    road_xy: tuple[float, float] | None = None
    road_history: deque[tuple[int, float, float]] = field(default_factory=deque)
    boxes: list[tuple[int, Box]] = field(default_factory=list)
    matches: list[RadarMatch] = field(default_factory=list)
    reliable: bool = False
    reliable_ever: bool = False
    direction: Direction | None = None

    @property
    def subject_id(self) -> int:
        return self.track_id

    @property
    def last_match(self) -> RadarMatch | None:
        return self.matches[-1] if self.matches else None

    @property
    def max_speed_kmh(self) -> float | None:
        return max(m.speed_kmh for m in self.matches) if self.matches else None

    @property
    def mean_speed_kmh(self) -> float | None:
        if not self.matches:
            return None
        return sum(m.speed_kmh for m in self.matches) / len(self.matches)


def _target_ground_range(radar: RadarModel, target: RadarTarget) -> float:
    return math.sqrt(max(target.distance_m**2 - radar.height_m**2, 0.0))


def gate_cost(
    radar: RadarModel, cfg: FusionConfig, track_xy: tuple[float, float], target: RadarTarget
) -> float | None:
    """Normalised track-target cost, or ``None`` when outside the gate or the fusion range."""
    r_track, az_track = radar.road_to_polar(*track_xy)
    r_target = _target_ground_range(radar, target)
    if r_track > cfg.max_range_m or r_target > cfg.max_range_m:
        return None
    g = cfg.gate
    gate_r = g.range_base_m + g.range_per_m2 * r_track**2 + _RANGE_STEP_HALF_M
    gate_az = _ANGLE_STEP_HALF_DEG + g.angle_margin_deg
    ratio_r = abs(r_track - r_target) / gate_r
    ratio_az = abs(az_track - target.angle_deg) / gate_az
    if ratio_r > 1 or ratio_az > 1:
        return None
    return ratio_r**2 + ratio_az**2


def radar_match(
    radar: RadarModel, speed: SpeedConfig, road: RoadConfig, target: RadarTarget
) -> RadarMatch:
    """Radar target -> RadarMatch, applying the optional cosine correction to its speed."""
    target_xy = radar.target_to_road(target.angle_deg, target.distance_m)
    kmh = float(target.speed_kmh)
    if speed.cosine_correction:
        dx = target_xy[0] - radar.position_xy[0]
        dy = target_xy[1] - radar.position_xy[1]
        los = math.atan2(dy, dx)
        cosine = abs(math.cos(los - math.radians(road.lane_heading_deg)))
        kmh /= max(cosine, _MIN_COSINE)
    return RadarMatch(target.t_ns, target, target_xy, kmh, target.speed_kmh)


class Fusion:
    def __init__(
        self,
        camera: CameraModel,
        radar: RadarModel,
        fusion: FusionConfig,
        speed: SpeedConfig,
        road: RoadConfig,
    ) -> None:
        self._camera = camera
        self._radar = radar
        self._cfg = fusion
        self._speed = speed
        self._road = road
        self._lanes = sorted(fusion.lanes) if fusion.lanes is not None else None
        self._tracks: dict[int, FusedTrack] = {}

    @property
    def tracks(self) -> dict[int, FusedTrack]:
        """Active fused tracks by id (read-only view intended)."""
        return self._tracks

    def update(
        self, observations: list[TrackObservation], scans: Sequence[RadarScan], ts_ns: int
    ) -> list[FusedTrack]:
        current = [self._upsert(obs, ts_ns) for obs in observations]
        directions = [self._motion_direction(t, ts_ns) for t in current]
        for sc in scans:
            self._match_scan(current, directions, sc)
        for t in current:
            self._update_reliability(t, ts_ns)
        return current

    def finish(self, track_ids: Sequence[int]) -> list[FusedTrack]:
        return [self._tracks.pop(i) for i in track_ids if i in self._tracks]

    # --- per-track state ------------------------------------------------------------------

    def _upsert(self, obs: TrackObservation, ts_ns: int) -> FusedTrack:
        t = self._tracks.get(obs.track_id)
        if t is None:
            t = FusedTrack(obs.track_id, obs.class_name, obs.xyxy, obs.confidence, ts_ns, ts_ns)
            self._tracks[obs.track_id] = t
        t.class_name = obs.class_name
        t.xyxy = obs.xyxy
        t.confidence = obs.confidence
        t.last_ts_ns = ts_ns
        t.boxes.append((ts_ns, obs.xyxy))
        x0, _, x1, y1 = obs.xyxy
        t.road_xy = self._camera.backproject_ground((x0 + x1) / 2, y1)
        if t.road_xy is not None:
            t.road_history.append((ts_ns, *t.road_xy))
        horizon = ts_ns - round(4 * self._cfg.direction_window_s * 1e9)
        while t.road_history and t.road_history[0][0] < horizon:
            t.road_history.popleft()
        return t

    def _motion_direction(self, t: FusedTrack, ts_ns: int) -> Direction | None:
        if t.road_xy is None or not t.road_history:
            return None
        start = ts_ns - round(self._cfg.direction_window_s * 1e9)
        oldest = next((h for h in t.road_history if h[0] >= start), None)
        if oldest is None or oldest[0] == t.road_history[-1][0]:
            return None
        newest = t.road_history[-1]
        r0, _ = self._radar.road_to_polar(oldest[1], oldest[2])
        r1, _ = self._radar.road_to_polar(newest[1], newest[2])
        travel = self._cfg.direction_min_travel_m
        if r1 - r0 <= -travel:
            return "coming"
        if r1 - r0 >= travel:
            return "going"
        return None

    def _update_reliability(self, t: FusedTrack, ts_ns: int) -> None:
        last = t.last_match
        t.reliable = (
            last is not None
            and len(t.matches) >= self._cfg.reliable_min_matches
            and ts_ns - last.t_ns <= round(self._cfg.reliable_max_age_s * 1e9)
        )
        if t.reliable and last is not None:
            t.reliable_ever = True
            t.direction = "coming" if last.target.approaching else "going"

    # --- matching -------------------------------------------------------------------------

    def _lane(self, xy: tuple[float, float]) -> int:
        assert self._lanes is not None
        h = math.radians(self._road.lane_heading_deg)
        lateral = -xy[0] * math.sin(h) + xy[1] * math.cos(h)
        return bisect.bisect_right(self._lanes, lateral)

    def _pair_cost(
        self, t: FusedTrack, direction: Direction | None, target: RadarTarget
    ) -> float | None:
        if t.road_xy is None:
            return None
        if direction is not None and (direction == "coming") != target.approaching:
            return None
        cost = gate_cost(self._radar, self._cfg, t.road_xy, target)
        if cost is None:
            return None
        if self._lanes is not None:
            target_xy = self._radar.target_to_road(target.angle_deg, target.distance_m)
            if self._lane(t.road_xy) != self._lane(target_xy):
                return None
        return cost

    def _match_scan(
        self, tracks: list[FusedTrack], directions: list[Direction | None], sc: RadarScan
    ) -> None:
        if not tracks or not sc.targets:
            return
        cost = np.full((len(tracks), len(sc.targets)), _INVALID)
        for i, (t, d) in enumerate(zip(tracks, directions, strict=True)):
            for j, target in enumerate(sc.targets):
                c = self._pair_cost(t, d, target)
                if c is not None:
                    cost[i, j] = c
        valid = cost < _INVALID
        if not valid.any():
            return
        rows, cols = linear_sum_assignment(cost)
        ratio = self._cfg.ambiguity_ratio
        for i, j in zip(rows, cols, strict=True):
            if not valid[i, j]:
                continue
            best = cost[i, j]
            rivals = np.concatenate(
                [
                    np.delete(cost[i, :], j)[np.delete(valid[i, :], j)],
                    np.delete(cost[:, j], i)[np.delete(valid[:, j], i)],
                ]
            )
            # Ties (including zero costs) are ambiguous as well.
            if np.any((best > ratio * rivals) | (rivals <= best)):
                continue
            self._append_match(tracks[i], sc.targets[j])

    def _append_match(self, t: FusedTrack, target: RadarTarget) -> None:
        t.matches.append(radar_match(self._radar, self._speed, self._road, target))
