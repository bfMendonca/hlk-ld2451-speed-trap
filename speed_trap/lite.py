"""Lite mode: group radar targets across scans into one subject per passing vehicle (§5.6).

No camera detector. Never reads ahead in time; no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from speed_trap.config import FusionConfig, LiteConfig, RoadConfig, SpeedConfig
from speed_trap.fusion import gate_cost, radar_match
from speed_trap.geometry import RadarModel
from speed_trap.types import Direction, RadarMatch, RadarScan, RadarTarget

_INVALID = 1e6


@dataclass
class RadarSubject:
    subject_id: int
    matches: list[RadarMatch] = field(default_factory=list)
    direction: Direction = "coming"
    reliable: bool = True
    reliable_ever: bool = True
    xyxy: None = None
    confidence: float = 1.0

    @property
    def last_match(self) -> RadarMatch:
        return self.matches[-1]

    @property
    def max_speed_kmh(self) -> float:
        return max(m.speed_kmh for m in self.matches)

    @property
    def mean_speed_kmh(self) -> float:
        return sum(m.speed_kmh for m in self.matches) / len(self.matches)


class RadarTargetLinker:
    def __init__(
        self,
        radar: RadarModel,
        fusion: FusionConfig,
        lite: LiteConfig,
        speed: SpeedConfig,
        road: RoadConfig,
    ) -> None:
        self._radar = radar
        self._fusion = fusion
        self._lite = lite
        self._speed = speed
        self._road = road
        self._subjects: list[RadarSubject] = []
        self._next_id = 1

    def update(
        self, scans: Sequence[RadarScan], ts_ns: int
    ) -> tuple[list[RadarSubject], list[RadarSubject]]:
        """Link the scans' targets; return (active, finished) subjects."""
        for sc in scans:
            self._link_scan(sc)
        lost_ns = round(self._lite.lost_track_s * 1e9)
        active = [s for s in self._subjects if ts_ns - s.last_match.t_ns <= lost_ns]
        finished = [s for s in self._subjects if ts_ns - s.last_match.t_ns > lost_ns]
        self._subjects = active
        return list(active), finished

    def finish_all(self) -> list[RadarSubject]:
        out, self._subjects = self._subjects, []
        return out

    def _link_scan(self, sc: RadarScan) -> None:
        targets = sc.targets
        if not targets:
            return
        subjects = self._subjects
        used: set[int] = set()
        if subjects:
            cost = np.full((len(subjects), len(targets)), _INVALID)
            for i, s in enumerate(subjects):
                for j, tg in enumerate(targets):
                    if (s.direction == "coming") != tg.approaching:
                        continue
                    c = gate_cost(self._radar, self._fusion, self._predict(s, tg.t_ns), tg)
                    if c is not None:
                        cost[i, j] = c
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols, strict=True):
                if cost[i, j] < _INVALID:
                    subjects[i].matches.append(self._match(targets[j]))
                    used.add(int(j))
        for j, tg in enumerate(targets):
            if j not in used and self._in_range(tg):
                self._new_subject(tg)

    def _predict(self, s: RadarSubject, t_ns: int) -> tuple[float, float]:
        """Last road point moved along the lane at the last radial speed (under-predicts).

        Without it the angular gate splits a vehicle near the radar, where its azimuth
        changes by more than the gate between two reports.
        """
        last = s.last_match
        dist = last.speed_raw_kmh / 3.6 * max(t_ns - last.t_ns, 0) / 1e9
        heading = math.radians(self._road.lane_heading_deg)
        sign = -1.0 if s.direction == "coming" else 1.0
        x, y = last.road_xy
        return x + sign * dist * math.cos(heading), y + sign * dist * math.sin(heading)

    def _in_range(self, target: RadarTarget) -> bool:
        ground = math.sqrt(max(target.distance_m**2 - self._radar.height_m**2, 0.0))
        return ground <= self._fusion.max_range_m

    def _match(self, target: RadarTarget) -> RadarMatch:
        return radar_match(self._radar, self._speed, self._road, target)

    def _new_subject(self, target: RadarTarget) -> None:
        direction: Direction = "coming" if target.approaching else "going"
        s = RadarSubject(self._next_id, [self._match(target)], direction)
        self._next_id += 1
        self._subjects.append(s)
