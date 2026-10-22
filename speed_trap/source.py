"""Recorded session source: turns a session folder into a time-ordered stream of FrameBundles."""

from __future__ import annotations

import bisect
import csv
import itertools
import statistics
from collections.abc import Iterator, Sequence
from pathlib import Path

import av
from radar_cam_recorder.reader import Frame, open_session

from speed_trap.config import RadarConfig
from speed_trap.types import FrameBundle, RadarScan, RadarTarget

_NS = 1_000_000_000


class SourceError(Exception):
    """The session cannot be read as requested."""


class RecordedSessionSource:
    def __init__(
        self,
        path: Path,
        radar_cfg: RadarConfig,
        segments: Sequence[int] | None = None,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.warnings: list[str] = []
        try:
            self._session = open_session(self.path)
        except (OSError, ValueError) as e:
            raise SourceError(f"cannot open session {self.path}: {e}") from e
        meta = self._session.metadata
        camera = (meta.get("devices") or {}).get("camera") or {}
        radar_dev = (meta.get("devices") or {}).get("radar") or {}
        params = radar_dev.get("params") or {}

        self.session_id: str = str(meta.get("session_id", self.path.name))
        self.width: int = int(camera.get("width", 0))
        self.height: int = int(camera.get("height", 0))
        self.ntp_synced: bool = bool(meta.get("ntp_synced", False))
        self.radar_max_distance_m: float | None = params.get("max_distance_m")
        self.radar_direction_param: int | None = params.get("direction")
        if self.radar_direction_param in (0, 1):
            self.warnings.append("radar was set to report one direction only")

        all_frames = sorted(self._session.frames(), key=lambda f: (f.segment, f.index))
        if not all_frames:
            raise SourceError(f"{self.path}: session has no video frames")
        available = sorted({f.segment for f in all_frames})
        listing = ", ".join(map(str, available))
        if segments is not None:
            unknown = [s for s in segments if s not in available]
            if unknown:
                raise SourceError(
                    f"unknown segment(s) {', '.join(map(str, unknown))}; existing segments: "
                    f"{listing}"
                )
        self._counts = {s: sum(1 for f in all_frames if f.segment == s) for s in available}
        self.segments: list[int] = sorted(set(segments)) if segments is not None else available

        first_ts = min(f.sensor_ts_ns for f in all_frames)
        duration_s = (max(f.sensor_ts_ns for f in all_frames) - first_ts) / _NS
        for name, value in (("start", start_s), ("end", end_s)):
            if value is not None and not 0 <= value <= duration_s:
                raise SourceError(
                    f"--{name} {value:g} s is outside the session: valid range is "
                    f"0..{duration_s:.1f} s (segments: {listing})"
                )
        if start_s is not None and end_s is not None and end_s < start_s:
            raise SourceError(f"end ({end_s:g} s) is before start ({start_s:g} s)")

        lo = first_ts + int(start_s * _NS) if start_s is not None else None
        hi = first_ts + int(end_s * _NS) if end_s is not None else None
        self._frames_by_segment: dict[int, list[Frame]] = {s: [] for s in self.segments}
        for f in all_frames:
            if f.segment not in self._frames_by_segment:
                continue
            if (lo is not None and f.sensor_ts_ns < lo) or (hi is not None and f.sensor_ts_ns > hi):
                continue
            self._frames_by_segment[f.segment].append(f)

        missing = [
            str(fs[0].video)
            for fs in self._frames_by_segment.values()
            if fs and not Path(fs[0].video).is_file()
        ]
        if missing:
            raise SourceError(f"missing video file(s): {', '.join(missing)}")
        selected = [f.sensor_ts_ns for fs in self._frames_by_segment.values() for f in fs]
        if not selected:
            ranges = []
            for s in self.segments:
                stamps = [f.sensor_ts_ns for f in all_frames if f.segment == s]
                ranges.append(
                    f"segment {s}: {(min(stamps) - first_ts) / _NS:.1f}.."
                    f"{(max(stamps) - first_ts) / _NS:.1f} s"
                )
            raise SourceError(
                "the selection has no frames (--start/--end outside the selected segments); "
                + "; ".join(ranges)
            )
        deltas = [b - a for a, b in itertools.pairwise(selected) if b > a]
        self.fps: float = _NS / statistics.median(deltas) if deltas else 0.0

        self._clock_samples = sorted(
            (c["boottime_ns"], c["realtime_ns"])
            for c in meta.get("clock_samples") or []
            if "boottime_ns" in c and "realtime_ns" in c
        )
        self._scans_by_frame = self._assign_scans(all_frames, radar_cfg)

    @property
    def has_radar(self) -> bool:
        return self._has_radar

    @property
    def frame_counts(self) -> dict[int, int]:
        """Selected frames per selected segment (after --segment/--start/--end)."""
        return {s: len(fs) for s, fs in self._frames_by_segment.items()}

    def _assign_scans(
        self, all_frames: list[Frame], cfg: RadarConfig
    ) -> dict[tuple[int, int], list[RadarScan]]:
        """Group scans (t = t_rx - latency - clock_offset) by the first frame at or after them."""
        shift = int(self._session.metadata.get("radar_latency_ns") or 0) + int(
            cfg.clock_offset_ms * 1e6
        )
        scans: list[RadarScan] = []
        if (self.path / "radar").is_dir():
            for report in self._session.radar_reports():
                t_ns = report.t_rx_ns - shift
                targets = tuple(
                    RadarTarget(
                        t_ns=t_ns,
                        angle_deg=t.angle_deg,
                        distance_m=t.distance_m,
                        approaching=t.approaching
                        if cfg.approaching_byte == 0
                        else not t.approaching,
                        speed_kmh=t.speed_kmh,
                        snr=t.snr,
                        index=i,
                    )
                    for i, t in enumerate(report.targets)
                )
                scans.append(RadarScan(t_ns, targets))
        self._has_radar = bool(scans)
        if not self._has_radar:
            self.warnings.append("session has no radar reports; running video-only")
            return {}
        scans.sort(key=lambda s: s.t_ns)
        ordered = sorted(all_frames, key=lambda f: (f.sensor_ts_ns, f.segment, f.index))
        stamps = [f.sensor_ts_ns for f in ordered]
        out: dict[tuple[int, int], list[RadarScan]] = {}
        for scan in scans:
            pos = bisect.bisect_left(stamps, scan.t_ns)
            if pos >= len(ordered):
                continue  # after the last frame
            f = ordered[pos]
            out.setdefault((f.segment, f.index), []).append(scan)
        return out

    def _wall_times(self, segment: int) -> dict[int, int]:
        csv_path = self.path / "video" / f"{segment:04d}.frames.csv"
        out: dict[int, int] = {}
        if not csv_path.exists():
            return out
        with open(csv_path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    out[int(row["index"])] = int(row["frame_wall_clock_us"]) * 1000
                except (KeyError, TypeError, ValueError):
                    continue
        return out

    def _wall_from_clock(self, ts_ns: int) -> int | None:
        if not self._clock_samples:
            return None
        i = bisect.bisect_right(self._clock_samples, (ts_ns, float("inf"))) - 1
        boot, real = self._clock_samples[max(i, 0)]
        return int(real + ts_ns - boot)

    def __iter__(self) -> Iterator[FrameBundle]:
        for segment in self.segments:
            frames = self._frames_by_segment[segment]
            if not frames:
                continue
            walls = self._wall_times(segment)
            video = frames[0].video
            all_count = self._counts[segment]
            by_index = {f.index: f for f in frames}
            decoded = 0
            with av.open(str(video), format="h264") as container:
                for decoded, av_frame in enumerate(container.decode(video=0), 1):
                    f = by_index.get(decoded - 1)
                    if f is None:
                        continue
                    wall = walls.get(f.index)
                    if wall is None:
                        wall = self._wall_from_clock(f.sensor_ts_ns)
                    yield FrameBundle(
                        image=av_frame.to_ndarray(format="bgr24"),
                        ts_ns=f.sensor_ts_ns,
                        segment=segment,
                        index=f.index,
                        wall_time_ns=wall,
                        scans=tuple(self._scans_by_frame.get((segment, f.index), ())),
                    )
            if decoded != all_count:
                self.warnings.append(
                    f"segment {segment}: video has {decoded} frames, timestamps {all_count}"
                )


__all__ = ["RecordedSessionSource", "SourceError"]
