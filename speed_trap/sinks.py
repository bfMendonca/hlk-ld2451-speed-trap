"""Output sinks: overlay video, tracks log, speeder captures and run.json (spec §5.7, §6.1)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from fractions import Fraction
from importlib import metadata
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from speed_trap.config import CaptureConfig, Config, config_hash
from speed_trap.fusion import FusedTrack
from speed_trap.lite import RadarSubject
from speed_trap.monitor import Candidate, CaptureEvent
from speed_trap.types import RadarMatch

_NS_PER_MS = 1_000_000
_ENCODERS = {"h264": "libx264"}  # output.video_codec -> FFmpeg encoder


class VideoSink:
    """One H.264 file per segment with variable frame rate (pts in ms since segment start)."""

    def __init__(self, out_dir: Path, width: int, height: int, codec: str = "h264") -> None:
        if width % 2 or height % 2:
            raise ValueError(f"video size must be even for yuv420p, got {width}x{height}")
        if codec not in _ENCODERS:
            raise ValueError(f"unsupported video codec {codec!r}; supported: {sorted(_ENCODERS)}")
        self._encoder = _ENCODERS[codec]
        self._dir = Path(out_dir)
        self._width = width
        self._height = height
        self._segment: int | None = None
        self._container: Any = None
        self._stream: Any = None
        self._t0_ns = 0
        self._last_pts = -1

    def write(self, segment: int, ts_ns: int, image: np.ndarray) -> None:
        if image.shape[:2] != (self._height, self._width):
            raise ValueError(
                f"frame shape {image.shape[:2]} does not match video size "
                f"(height, width) = ({self._height}, {self._width})"
            )
        if segment != self._segment:
            self._close_current()
            self._open(segment, ts_ns)
        pts = max((ts_ns - self._t0_ns) // _NS_PER_MS, self._last_pts + 1)  # strictly increasing
        self._last_pts = pts
        frame = av.VideoFrame.from_ndarray(image, format="bgr24").reformat(format="yuv420p")
        frame.pts = pts
        frame.time_base = Fraction(1, 1000)
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        self._close_current()

    def _open(self, segment: int, ts_ns: int) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self._dir / f"{segment:04d}.overlay.mp4"), mode="w")
        stream = self._container.add_stream(self._encoder, rate=25)
        stream.width = self._width
        stream.height = self._height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        self._stream = stream
        self._segment = segment
        self._t0_ns = ts_ns
        self._last_pts = -1

    def _close_current(self) -> None:
        if self._container is None:
            return
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._container = self._stream = None
        self._segment = None


def _imwrite(path: Path, image: np.ndarray, params: list[int]) -> None:
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"cannot write image {path}")


def _match_json(m: RadarMatch) -> dict[str, Any]:
    return {
        "t_ns": m.t_ns,
        "speed_kmh": m.speed_kmh,
        "speed_raw_kmh": m.speed_raw_kmh,
        "distance_m": m.target.distance_m,
        "angle_deg": m.target.angle_deg,
        "snr": m.target.snr,
        "approaching": m.target.approaching,
    }


class TracksSink:
    """tracks.jsonl: one line per finished subject.

    ``road`` is built from the matches' ``road_xy`` (the full radar-time history), not from
    ``FusedTrack.road_history``, which is trimmed to a short window for speed estimation.
    """

    def __init__(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")  # noqa: SIM115

    def write(self, finished: Sequence[FusedTrack | RadarSubject]) -> None:
        for s in finished:
            is_track = isinstance(s, FusedTrack)
            boxes = [[t, *b] for t, b in s.boxes] if isinstance(s, FusedTrack) else []
            if isinstance(s, FusedTrack):
                first, last = s.first_ts_ns, s.last_ts_ns
            else:
                first, last = s.matches[0].t_ns, s.matches[-1].t_ns
            rec = {
                "id": s.subject_id,
                "kind": "track" if is_track else "radar",
                "class": s.class_name if isinstance(s, FusedTrack) else None,
                "first_ts_ns": first,
                "last_ts_ns": last,
                "direction": s.direction,
                "reliable_ever": s.reliable_ever,
                "max_speed_kmh": s.max_speed_kmh,
                "mean_speed_kmh": s.mean_speed_kmh,
                "boxes": boxes,
                "road": [[m.t_ns, *m.road_xy] for m in s.matches],
                "matches": [_match_json(m) for m in s.matches],
            }
            self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class CaptureSink:
    """Holds one image copy per subject (the latest, i.e. best, candidate) until its event."""

    def __init__(
        self,
        out_dir: Path,
        session_id: str,
        cfg: CaptureConfig,
        ntp_synced: bool,
        config_hash: str,
    ) -> None:
        self._dir = Path(out_dir) / "captures"
        self._session_id = session_id
        self._cfg = cfg
        self._ntp = ntp_synced
        self._hash = config_hash
        # subject_id -> (candidate, crop, full raw frame or None, wall_time_ns)
        self._pending: dict[int, tuple[Candidate, np.ndarray, np.ndarray | None, int | None]] = {}

    def offer(
        self,
        candidates: Sequence[Candidate],
        raw: np.ndarray,
        rendered: np.ndarray,
        wall_time_ns: int | None,
    ) -> None:
        for c in candidates:
            crop = rendered.copy() if c.xyxy is None else self._crop(raw, c.xyxy)
            full = raw.copy() if self._cfg.save_full_frame else None
            self._pending[c.subject_id] = (c, crop, full, wall_time_ns)

    def discard(self, subject_ids: Iterable[int]) -> None:
        for i in subject_ids:
            self._pending.pop(i, None)

    def write(self, events: Sequence[CaptureEvent]) -> list[Path]:
        paths: list[Path] = []
        for ev in events:
            item = self._pending.pop(ev.subject_id, None)
            if item is None:
                continue
            _, crop, full, wall_ns = item
            if crop.size == 0:
                continue
            self._dir.mkdir(parents=True, exist_ok=True)
            stem = f"{ev.capture.segment:04d}_t{ev.subject_id}_{ev.speed_max_kmh:.0f}kmh"
            jpg, side = self._dir / f"{stem}.jpg", self._dir / f"{stem}.json"
            params = [cv2.IMWRITE_JPEG_QUALITY, self._cfg.jpeg_quality]
            _imwrite(jpg, crop, params)
            paths.append(jpg)
            if full is not None:
                full_path = self._dir / f"{stem}_full.jpg"
                _imwrite(full_path, full, params)
                paths.append(full_path)
            side.write_text(json.dumps(self._sidecar(ev, wall_ns), indent=2), encoding="utf-8")
            paths.append(side)
        return paths

    def _crop(self, raw: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray:
        h, w = raw.shape[:2]
        x0, y0, x1, y1 = xyxy
        mx, my = (x1 - x0) * self._cfg.crop_margin, (y1 - y0) * self._cfg.crop_margin
        ix0, iy0 = max(round(x0 - mx), 0), max(round(y0 - my), 0)
        ix1, iy1 = min(round(x1 + mx), w), min(round(y1 + my), h)
        crop: np.ndarray = raw[iy0:iy1, ix0:ix1].copy()
        return crop

    def _sidecar(self, ev: CaptureEvent, wall_ns: int | None) -> dict[str, Any]:
        cap = ev.capture
        wall = (
            None
            if wall_ns is None
            else datetime.fromtimestamp(wall_ns / 1e9, timezone.utc).isoformat()
        )
        return {
            "event": "speeding",
            "session": self._session_id,
            "segment": cap.segment,
            "track_id": ev.subject_id,
            "direction": ev.direction,
            "limit_kmh": ev.limit_kmh,
            "speed_kmh": {
                "max": ev.speed_max_kmh,
                "mean": ev.speed_mean_kmh,
                "at_capture": ev.speed_at_capture_kmh,
            },
            "capture": {
                "frame_index": cap.index,
                "sensor_ts_ns": cap.ts_ns,
                "wall_clock": wall,
                "wall_clock_reliable": self._ntp,
                "bbox_xyxy": None if cap.xyxy is None else list(cap.xyxy),
                "detector_conf": ev.confidence,
            },
            "radar_history": [_match_json(m) for m in ev.history],
            "config_sha256": self._hash,
        }


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def write_run_info(
    path: Path,
    cfg: Config,
    session_path: Path,
    counters: dict[str, int],
    started_ns: int,
    finished_ns: int,
    warnings: list[str],
    error: str | None = None,
) -> None:
    """Write run.json; ``error`` (only present when given) records a run that failed."""
    versions: dict[str, str] = {}
    for dist in ("speed_trap", "ld2451", "radar_cam_recorder", "rfdetr", "ultralytics"):
        v = _version(dist) or _version(dist.replace("_", "-"))
        if v is not None:
            versions[dist] = v
    info = {
        "config": json.loads(cfg.model_dump_json()),
        "config_hash": config_hash(cfg),
        "versions": versions,
        "model": cfg.detector.weights,
        "session_path": str(session_path),
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "counters": counters,
        "warnings": warnings,
    }
    if error is not None:
        info["error"] = error
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(info, indent=2), encoding="utf-8")
