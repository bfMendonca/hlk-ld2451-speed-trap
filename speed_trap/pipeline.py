"""Frame-by-frame pipeline: source -> detector -> tracker -> fusion/linker -> monitor -> sinks.

The stages own no I/O; this module wires them to the recorded source and the output sinks.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from speed_trap.config import Config, config_hash
from speed_trap.detect import Detector
from speed_trap.fusion import FusedTrack, Fusion
from speed_trap.geometry import CameraModel, RadarModel
from speed_trap.lite import RadarSubject, RadarTargetLinker
from speed_trap.monitor import Candidate, SpeedingMonitor
from speed_trap.render.draw import HudInfo, Renderer
from speed_trap.sinks import CaptureSink, TracksSink, VideoSink, write_run_info
from speed_trap.source import RecordedSessionSource
from speed_trap.tracker import Tracker
from speed_trap.types import FrameBundle, RadarScan, RadarTarget

log = logging.getLogger(__name__)

NO_INTRINSICS_WARNING = "no camera intrinsics in config: using nominal OV5647 field of view"
_FALLBACK_FPS = 25.0
_FPS_SMOOTHING = 0.9

Mode = Literal["full", "lite"]
Progress = Callable[[int, int, float], None]  # (segment, frames done in segment, frames/s)
PROGRESS_INTERVAL_S = 5.0
Finished = Sequence[FusedTrack] | Sequence[RadarSubject]


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        source: RecordedSessionSource,
        out_dir: Path,
        mode: Mode,
        detector: Detector | None,
    ) -> None:
        if mode == "full" and detector is None:
            raise ValueError("full mode needs a detector")
        self._cfg = cfg
        self._source = source
        self._out = Path(out_dir)
        self._mode = mode
        self._detector = detector if mode == "full" else None
        self.warnings: list[str] = []
        if cfg.camera.intrinsics is None:
            self._warn(NO_INTRINSICS_WARNING)
        for message in source.warnings:
            self._warn(message)

        w, h = source.width, source.height
        self._camera = CameraModel.from_config(cfg.camera, w, h)
        self._radar = RadarModel.from_config(
            cfg.radar, cfg.camera.pose.height_m, max_range_default=source.radar_max_distance_m
        )
        self._renderer = Renderer(
            self._camera, self._radar, cfg.overlay, cfg.lite, far_range_m=cfg.fusion.max_range_m
        )
        self._monitor = SpeedingMonitor(cfg.speed, cfg.capture, w, h)
        fps = source.fps if source.fps > 0 else _FALLBACK_FPS
        self._tracker: Tracker | None = None
        self._fusion: Fusion | None = None
        self._linker: RadarTargetLinker | None = None
        if mode == "full":
            self._tracker = Tracker(cfg.tracker, fps)
            self._fusion = Fusion(self._camera, self._radar, cfg.fusion, cfg.speed, cfg.road)
        else:
            self._linker = RadarTargetLinker(self._radar, cfg.fusion, cfg.lite, cfg.speed, cfg.road)

        self._video: VideoSink | None = None
        self._tracks_sink: TracksSink | None = None
        self._captures: CaptureSink | None = None
        self._counters = {"frames": 0, "scans": 0, "tracks": 0, "matches": 0, "captures": 0}
        self._last_scan: RadarScan | None = None
        self._last_target_scan: RadarScan | None = None
        self._fps = 0.0

    # ------------------------------------------------------------------------------- run

    def run(
        self, progress: Progress | None = None, progress_interval_s: float = PROGRESS_INTERVAL_S
    ) -> dict[str, int]:
        """Process the whole source; ``progress`` is called at least every
        ``progress_interval_s`` of wall time within a segment and at each segment end."""
        started_ns = time.time_ns()
        try:
            self._run(progress, progress_interval_s)
        except BaseException as e:  # incl. KeyboardInterrupt: leave a record of the partial run
            self._write_run_info(started_ns, f"{type(e).__name__}: {e}")
            raise
        self._write_run_info(started_ns, None)
        return dict(self._counters)

    def _run(self, progress: Progress | None, interval_s: float) -> None:
        cfg, src = self._cfg, self._source
        n_warnings = len(src.warnings)
        seg: int | None = None
        seg_frames, reported, seg_t0 = 0, 0, time.perf_counter()
        t_prev = t_report = seg_t0

        def report() -> None:
            nonlocal reported, t_report
            if seg is not None and progress is not None and seg_frames != reported:
                progress(seg, seg_frames, _rate(seg_frames, seg_t0))
                reported, t_report = seg_frames, time.perf_counter()

        try:
            self._video = VideoSink(self._out, src.width, src.height, cfg.output.video_codec)
            self._tracks_sink = TracksSink(self._out / "tracks.jsonl")
            if cfg.capture.enabled:
                self._captures = CaptureSink(
                    self._out, src.session_id, cfg.capture, src.ntp_synced, config_hash(cfg)
                )
            for bundle in src:
                if bundle.segment != seg:
                    report()
                    seg, seg_frames, reported = bundle.segment, 0, 0
                    seg_t0 = t_report = time.perf_counter()
                self._step(bundle)
                now = time.perf_counter()
                self._tick(now - t_prev)  # decode + processing time of this frame
                t_prev = now
                seg_frames += 1
                if now - t_report >= interval_s:
                    report()
            report()
            self._finish(self._finish_all())
        finally:
            if self._video is not None:
                self._video.close()
            if self._tracks_sink is not None:
                self._tracks_sink.close()
            for message in src.warnings[n_warnings:]:
                self._warn(message)

    def _write_run_info(self, started_ns: int, error: str | None) -> None:
        write_run_info(
            self._out / "run.json",
            self._cfg,
            self._source.path,
            dict(self._counters),
            started_ns,
            time.time_ns(),
            self.warnings,
            error=error,
        )

    # ---------------------------------------------------------------------------- per frame

    def _step(self, b: FrameBundle) -> None:
        cfg = self._cfg
        frame_number = self._counters["frames"]
        self._counters["frames"] += 1
        self._counters["scans"] += len(b.scans)
        for sc in b.scans:
            self._last_scan = sc
            if sc.targets:
                self._last_target_scan = sc

        tracks: list[FusedTrack] = []
        subjects: list[RadarSubject] = []
        finished: Finished
        if self._mode == "full":
            assert self._tracker and self._fusion and self._detector
            dets = None
            if frame_number % cfg.detector.every_n_frames == 0:
                dets = self._detector.detect(b)
            out = self._tracker.update(dets, b.ts_ns)
            finished = self._fusion.finish(out.finished)
            tracks = self._fusion.update(out.active, b.scans, b.ts_ns)
            active: Sequence[FusedTrack] | Sequence[RadarSubject] = tracks
        else:
            assert self._linker
            subjects, finished = self._linker.update(b.scans, b.ts_ns)
            active = subjects

        candidates: list[Candidate] = []
        if self._captures is not None:
            candidates = self._monitor.update(active, b.segment, b.index, b.ts_ns)
        hold_ns = round(cfg.radar.report_hold_s * 1e9)
        drawn = [s for s in subjects if b.ts_ns - s.last_match.t_ns <= hold_ns]
        rendered = self._renderer.render(
            b.image,
            tracks=tracks,
            subjects=drawn,
            current_targets=self._current_targets(b.ts_ns),
            triggered=self._monitor.triggered_ids if self._captures is not None else set(),
            limit_for=self._monitor.limit_for,
            hud=self._hud(b),
        )
        if self._captures is not None and candidates:
            self._captures.offer(candidates, b.image, rendered, b.wall_time_ns)
        assert self._video is not None
        self._video.write(b.segment, b.ts_ns, rendered)
        self._finish(finished)

    def _current_targets(self, ts_ns: int) -> tuple[RadarTarget, ...]:
        """Targets of the latest scan with targets, if it is within ``radar.report_hold_s``."""
        sc = self._last_target_scan
        if sc is None or ts_ns - sc.t_ns > round(self._cfg.radar.report_hold_s * 1e9):
            return ()
        if self._last_scan is not None and self._last_scan.t_ns > sc.t_ns:
            return ()  # a newer, empty report says the targets are gone
        return sc.targets

    def _hud(self, b: FrameBundle) -> HudInfo:
        cfg = self._cfg
        full = self._mode == "full"
        last = self._last_scan
        return HudInfo(
            session_id=self._source.session_id,
            segment=b.segment,
            index=b.index,
            wall_time_ns=b.wall_time_ns,
            mode=self._mode,
            model=self._model_name() if full else None,
            tracker=cfg.tracker.type if full else None,
            fps=self._fps,
            radar_targets=len(self._current_targets(b.ts_ns)),
            radar_age_s=None if last is None else (b.ts_ns - last.t_ns) / 1e9,
            clock_offset_ms=cfg.radar.clock_offset_ms,
            limit_kmh=cfg.speed.limit_kmh,
        )

    def _model_name(self) -> str | None:
        if self._detector is None:
            return None
        if self._cfg.detector.backend == "ground_truth":
            return self._detector.name
        return self._cfg.detector.weights

    def _tick(self, dt: float) -> None:
        if dt <= 0:
            return
        rate = 1.0 / dt
        self._fps = (
            rate if self._fps == 0 else _FPS_SMOOTHING * self._fps + (1 - _FPS_SMOOTHING) * rate
        )

    # ------------------------------------------------------------------------- finishing

    def _finish_all(self) -> Finished:
        if self._mode == "full":
            assert self._tracker and self._fusion
            ids = self._tracker.finish_all()
            rest = [i for i in self._fusion.tracks if i not in ids]
            return self._fusion.finish([*ids, *rest])
        assert self._linker
        return self._linker.finish_all()

    def _finish(self, finished: Finished) -> None:
        if not finished:
            return
        if self._captures is not None:
            events = self._monitor.finish(finished)
            paths = self._captures.write(events)
            self._counters["captures"] += sum(1 for p in paths if p.suffix == ".json")
            self._captures.discard(s.subject_id for s in finished)
        assert self._tracks_sink is not None
        self._tracks_sink.write(finished)
        self._counters["tracks"] += len(finished)
        self._counters["matches"] += sum(len(s.matches) for s in finished)

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        log.warning(message)


def _rate(frames: int, t0: float) -> float:
    dt = time.perf_counter() - t0
    return frames / dt if dt > 0 else 0.0


__all__ = ["NO_INTRINSICS_WARNING", "Pipeline"]
