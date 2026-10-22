"""Multi-object tracking: wraps ``trackers.ByteTrackTracker`` with run-unique ids.

Timeouts are expressed in seconds (video frame rate is irregular) and no
future frames are ever consulted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from speed_trap.config import TrackerConfig
from speed_trap.types import Box, Detection, TrackObservation


@dataclass(frozen=True)
class TrackerOutput:
    active: list[TrackObservation]
    finished: list[int]


@dataclass
class _Track:
    last_seen_ns: int
    xyxy: Box
    confidence: float
    votes: Counter[str]
    prev: tuple[int, Box] | None = None  # the detected box before ``xyxy``

    @property
    def class_name(self) -> str:
        return self.votes.most_common(1)[0][0]

    def predict(self, ts_ns: int) -> Box:
        """Constant-velocity box from the last two detected boxes (held if only one)."""
        if self.prev is None or self.last_seen_ns <= self.prev[0]:
            return self.xyxy
        w = (ts_ns - self.last_seen_ns) / (self.last_seen_ns - self.prev[0])
        a, b = self.prev[1], self.xyxy
        x0, y0, x1, y1 = (bv + w * (bv - av) for av, bv in zip(a, b, strict=True))
        return (x0, y0, x1, y1)


class Tracker:
    def __init__(self, cfg: TrackerConfig, fps: float) -> None:
        self._cfg = cfg
        self._lost_ns = round(cfg.lost_track_s * 1e9)
        self._bt = ByteTrackTracker(
            lost_track_buffer=max(1, round(cfg.lost_track_s * fps)),
            frame_rate=fps,
            track_activation_threshold=cfg.track_activation_threshold,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
            minimum_iou_threshold=1.0 - cfg.match_threshold,
        )
        self._epoch = 0
        self._next_id = 1
        self._ids: dict[tuple[int, int], int] = {}
        self._keys: dict[int, tuple[int, int]] = {}
        self._tracks: dict[int, _Track] = {}
        self._class_ids: dict[str, int] = {}
        self._class_names: list[str] = []
        self._prev_ts: int | None = None
        self._last_active: list[int] = []  # ids emitted on the last detected frame

    def update(self, detections: list[Detection] | None, ts_ns: int) -> TrackerOutput:
        finished: list[int] = []
        if self._prev_ts is not None and ts_ns - self._prev_ts > self._lost_ns:
            finished += self.finish_all()
        self._prev_ts = ts_ns

        active: list[TrackObservation] = []
        if detections is None:
            # Skipped detector frame: only tracks ByteTrack still held on the last detected
            # frame, at a predicted box (ByteTrack itself is not stepped).
            finished += self._expire(ts_ns)
            for tid in self._last_active:
                tr = self._tracks.get(tid)
                if tr is not None:
                    box = tr.predict(ts_ns)
                    active.append(TrackObservation(tid, box, tr.confidence, tr.class_name, False))
            return TrackerOutput(active, finished)

        self._last_active = []
        out = self._bt.update(self._to_sv(detections), timestamp=ts_ns / 1e9)
        if out.tracker_id is None or out.class_id is None or out.confidence is None:
            return TrackerOutput(active, finished + self._expire(ts_ns))
        ids, classes, confs = out.tracker_id, out.class_id, out.confidence
        for i, sv_id in enumerate(ids):
            if sv_id < 0:
                continue
            tid = self._public_id(int(sv_id))
            x1, y1, x2, y2 = (float(v) for v in out.xyxy[i])
            if tid not in self._tracks:
                self._tracks[tid] = _Track(ts_ns, (x1, y1, x2, y2), 0.0, Counter())
            tr = self._tracks[tid]
            if tr.last_seen_ns < ts_ns:
                tr.prev = (tr.last_seen_ns, tr.xyxy)
            tr.last_seen_ns = ts_ns
            tr.xyxy = (x1, y1, x2, y2)
            tr.confidence = float(confs[i])
            tr.votes[self._class_names[int(classes[i])]] += 1
            active.append(TrackObservation(tid, tr.xyxy, tr.confidence, tr.class_name, True))
            self._last_active.append(tid)
        finished += self._expire(ts_ns)
        return TrackerOutput(active, finished)

    def finish_all(self) -> list[int]:
        done = sorted(self._tracks)
        self._tracks.clear()
        self._last_active = []
        self._ids.clear()
        self._keys.clear()
        self._bt.reset()
        self._epoch += 1
        return done

    def _public_id(self, sv_id: int) -> int:
        key = (self._epoch, sv_id)
        if key not in self._ids:
            self._ids[key] = self._next_id
            self._keys[self._next_id] = key
            self._next_id += 1
        return self._ids[key]

    def _expire(self, ts_ns: int) -> list[int]:
        gone = [t for t, tr in self._tracks.items() if ts_ns - tr.last_seen_ns > self._lost_ns]
        for t in gone:
            del self._tracks[t]
            # A ByteTrack track that outlives our expiry gets a fresh public id.
            del self._ids[self._keys.pop(t)]
        return gone

    def _to_sv(self, detections: list[Detection]) -> sv.Detections:
        if not detections:
            return sv.Detections.empty()
        for d in detections:
            if d.class_name not in self._class_ids:
                self._class_ids[d.class_name] = len(self._class_names)
                self._class_names.append(d.class_name)
        return sv.Detections(
            xyxy=np.array([d.xyxy for d in detections], dtype=np.float32),
            confidence=np.array([d.confidence for d in detections], dtype=np.float32),
            class_id=np.array([self._class_ids[d.class_name] for d in detections], dtype=int),
        )
