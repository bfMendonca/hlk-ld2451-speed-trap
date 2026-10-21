"""Speeding decision and best-capture-frame selection (spec §5.7).

Never reads ahead in time; no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from speed_trap.config import CaptureConfig, SpeedConfig
from speed_trap.types import Direction, RadarMatch

Box = tuple[float, float, float, float]


class Subject(Protocol):
    """Read-only view of a fused track or radar subject (properties allow covariance)."""

    @property
    def subject_id(self) -> int: ...

    @property
    def matches(self) -> list[RadarMatch]: ...

    @property
    def direction(self) -> Direction | None: ...

    @property
    def reliable(self) -> bool: ...

    @property
    def xyxy(self) -> Box | None: ...

    @property
    def confidence(self) -> float: ...

    @property
    def max_speed_kmh(self) -> float | None: ...

    @property
    def mean_speed_kmh(self) -> float | None: ...


@dataclass(frozen=True)
class Candidate:
    subject_id: int
    segment: int
    index: int
    ts_ns: int
    xyxy: Box | None


@dataclass(frozen=True)
class CaptureEvent:
    subject_id: int
    direction: Direction
    limit_kmh: float
    speed_max_kmh: float
    speed_mean_kmh: float
    speed_at_capture_kmh: float
    capture: Candidate
    confidence: float
    history: tuple[RadarMatch, ...]


class SpeedingMonitor:
    def __init__(self, speed: SpeedConfig, capture: CaptureConfig, width: int, height: int) -> None:
        self._speed = speed
        self._capture = capture
        self._width = width
        self._height = height
        self._triggered: set[int] = set()
        self._best: dict[int, tuple[float, Candidate]] = {}

    @property
    def triggered_ids(self) -> set[int]:
        return self._triggered

    def limit_for(self, direction: Direction | None) -> float:
        if direction == "coming" and self._speed.limit_kmh_coming is not None:
            return self._speed.limit_kmh_coming
        if direction == "going" and self._speed.limit_kmh_going is not None:
            return self._speed.limit_kmh_going
        return self._speed.limit_kmh

    def update(
        self, active: Sequence[Subject], segment: int, index: int, ts_ns: int
    ) -> list[Candidate]:
        """Return the subjects whose best capture frame improved on this frame."""
        improved: list[Candidate] = []
        for s in active:
            limit = self.limit_for(s.direction)
            if not s.reliable or not s.matches:
                continue
            if s.subject_id not in self._triggered and self._over(s, limit):
                self._triggered.add(s.subject_id)
            if s.subject_id not in self._triggered or s.matches[-1].speed_kmh <= limit:
                continue
            score = self._score(s)
            if score is None:
                continue
            prev = self._best.get(s.subject_id)
            if prev is None or score > prev[0]:
                cand = Candidate(s.subject_id, segment, index, ts_ns, s.xyxy)
                self._best[s.subject_id] = (score, cand)
                improved.append(cand)
        return improved

    def finish(self, finished: Sequence[Subject]) -> list[CaptureEvent]:
        events: list[CaptureEvent] = []
        for s in finished:
            best = self._best.pop(s.subject_id, None)
            if s.subject_id not in self._triggered or best is None or s.direction is None:
                continue
            cand = best[1]
            at = [m for m in s.matches if m.t_ns <= cand.ts_ns]
            ref = at[-1] if at else s.matches[0]
            max_kmh, mean_kmh = s.max_speed_kmh, s.mean_speed_kmh
            assert max_kmh is not None and mean_kmh is not None
            events.append(
                CaptureEvent(
                    s.subject_id,
                    s.direction,
                    self.limit_for(s.direction),
                    max_kmh,
                    mean_kmh,
                    ref.speed_kmh,
                    cand,
                    s.confidence,
                    tuple(s.matches),
                )
            )
        return events

    def _over(self, s: Subject, limit: float) -> bool:
        """Last ``min_consecutive_over`` matches all exceed the limit (so the latest does)."""
        k = self._capture.min_consecutive_over
        if len(s.matches) < k:
            return False
        return all(m.speed_kmh > limit for m in s.matches[-k:])

    def _score(self, s: Subject) -> float | None:
        if s.xyxy is None:
            return -float(s.matches[-1].target.distance_m)
        x0, y0, x1, y1 = s.xyxy
        m = self._capture.edge_margin_px
        if x0 <= m or y0 <= m or x1 >= self._width - m or y1 >= self._height - m:
            return None
        return (x1 - x0) * (y1 - y0) * s.confidence
