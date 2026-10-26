"""``speed-trap`` command line (spec §6.3).

Exit codes: 0 ok, 1 runtime error, 2 usage, config or source error.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from speed_trap import __version__
from speed_trap.config import Config, ConfigError, default_config_text, load_config

log = logging.getLogger("speed_trap")

EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130


class _UsageError(Exception):
    """A problem with the command line or its inputs (exit code 2)."""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="speed-trap",
        description="Offline speed trap: fuse HLK-LD2451 radar data with camera video.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    proc = sub.add_parser("process", help="process a recorded session")
    proc.add_argument("session", type=Path, metavar="SESSION_DIR")
    proc.add_argument("--config", type=Path, required=True, help="YAML config (required)")
    proc.add_argument(
        "--segment",
        type=int,
        nargs="+",
        action="extend",
        metavar="N",
        help="segment number(s) to process (default: all)",
    )
    proc.add_argument("--start", type=float, metavar="S", help="seconds from session start")
    proc.add_argument("--end", type=float, metavar="S", help="seconds from session start")
    proc.add_argument("--mode", choices=["full", "lite"], default="full")
    proc.add_argument("--device", choices=["cpu", "mps", "cuda"], help="detector device")
    proc.add_argument(
        "--out", type=Path, metavar="DIR", help="default: output.dir; writes DIR/<session>[-N]"
    )
    proc.add_argument(
        "--capture",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable speeder captures (default: capture.enabled)",
    )
    _add_set(proc)
    proc.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")

    init = sub.add_parser("init-config", help="write a commented default config file")
    init.add_argument("path", type=Path, metavar="PATH")

    check = sub.add_parser("check-config", help="print the effective config as JSON")
    check.add_argument("--config", type=Path, required=True)
    _add_set(check)

    syn = sub.add_parser("synth", help="generate a synthetic session with ground truth")
    syn.add_argument("out_dir", type=Path, metavar="OUT_DIR")
    syn.add_argument("--config", type=Path, help="YAML config (default: built-in defaults)")
    syn.add_argument("--duration", type=float, default=60.0, metavar="S")
    syn.add_argument("--seed", type=int, default=1, metavar="N")
    return p


def _add_set(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value by dotted key, e.g. speed.limit_kmh=50",
    )


def _setup_logging(verbose: int) -> None:
    for h in list(log.handlers):
        if getattr(h, "_speed_trap_cli", False):
            log.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("speed-trap: %(levelname)s: %(message)s"))
    handler._speed_trap_cli = True  # type: ignore[attr-defined]
    log.addHandler(handler)
    log.setLevel(logging.DEBUG if verbose >= 2 else logging.INFO if verbose else logging.WARNING)
    log.propagate = False


class ProgressPrinter:
    """Prints pipeline progress (frames/s and ETA) to stderr; calls may repeat a segment."""

    def __init__(self, frame_counts: dict[int, int]) -> None:
        self._total = sum(frame_counts.values())
        self._done: dict[int, int] = {}

    def __call__(self, segment: int, frames: int, fps: float) -> None:
        self._done[segment] = frames
        done = sum(self._done.values())
        eta = (self._total - done) / fps if fps > 0 else 0.0
        print(
            f"segment {segment:04d}: {frames} frames, {fps:.1f} frames/s, "
            f"{done}/{self._total} done, ETA {eta:.0f} s",
            file=sys.stderr,
        )


# --- commands -------------------------------------------------------------------------------


def _cmd_init_config(args: argparse.Namespace) -> int:
    path: Path = args.path
    if path.exists():
        raise _UsageError(f"{path} already exists; not overwriting")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_config_text(), encoding="utf-8")
    print(path)
    return EXIT_OK


def _cmd_check_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    print(cfg.model_dump_json(indent=2))
    return EXIT_OK


def _cmd_synth(args: argparse.Namespace) -> int:
    from speed_trap.synth import generate_session

    cfg = load_config(args.config) if args.config is not None else Config()
    if args.duration <= 0:
        raise _UsageError("--duration must be positive")
    print(generate_session(args.out_dir, cfg, args.duration, args.seed))
    return EXIT_OK


def _fresh_out_dir(parent: Path, session_id: str) -> Path:
    """`parent/<session_id>`, or `-2`, `-3`, ... after it, so a run never overwrites another."""
    out_dir = parent / session_id
    n = 2
    while out_dir.exists():
        out_dir = parent / f"{session_id}-{n}"
        n += 1
    return out_dir


def _cmd_process(args: argparse.Namespace) -> int:
    from speed_trap.pipeline import Pipeline
    from speed_trap.source import RecordedSessionSource, SourceError

    overrides = list(args.overrides)
    if args.device is not None:
        overrides.append(f"detector.device={args.device}")
    if args.capture is not None:
        overrides.append(f"capture.enabled={'true' if args.capture else 'false'}")
    cfg = load_config(args.config, overrides)

    try:
        source = RecordedSessionSource(
            args.session, cfg.radar, segments=args.segment, start_s=args.start, end_s=args.end
        )
    except SourceError as e:
        raise _UsageError(str(e)) from e
    if source.width <= 0 or source.height <= 0:
        raise _UsageError(f"{args.session}: session.json has no camera width/height")

    detector = None
    if args.mode == "full":
        from speed_trap.detect import make_detector

        detector = make_detector(cfg.detector)
    parent: Path = args.out if args.out is not None else cfg.output.dir
    out_dir = _fresh_out_dir(parent, source.session_id)
    pipeline = Pipeline(cfg, source, out_dir, args.mode, detector)

    counters = pipeline.run(ProgressPrinter(source.frame_counts))
    summary = ", ".join(f"{k} {v}" for k, v in counters.items())
    print(f"{out_dir}: {summary}", file=sys.stderr)
    print(out_dir)
    return EXIT_OK


_COMMANDS = {
    "process": _cmd_process,
    "init-config": _cmd_init_config,
    "check-config": _cmd_check_config,
    "synth": _cmd_synth,
}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:  # argparse usage errors exit 2, --help/--version exit 0
        return e.code if isinstance(e.code, int) else EXIT_USAGE
    verbose: int = getattr(args, "verbose", 0)
    _setup_logging(verbose)
    try:
        return _COMMANDS[args.command](args)
    except (_UsageError, ConfigError) as e:
        if verbose >= 2:
            traceback.print_exc()
        print(f"speed-trap: error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("speed-trap: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as e:
        if verbose >= 2:
            traceback.print_exc()
        print(f"speed-trap: error: {type(e).__name__}: {e} (use -vv for details)", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    sys.exit(main())
