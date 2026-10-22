from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from speed_trap.cli import main
from speed_trap.config import Config, load_config


def write_cfg(tmp_path: Path, text: str = "speed:\n  limit_kmh: 40\n") -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_init_config_writes_loadable_file(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "cfg.yaml"
    assert main(["init-config", str(p)]) == 0
    assert load_config(p) == Config()


def test_init_config_refuses_to_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = write_cfg(tmp_path, "{}\n")
    assert main(["init-config", str(p)]) == 2
    assert p.read_text(encoding="utf-8") == "{}\n"
    assert "exists" in capsys.readouterr().err


def test_check_config_prints_effective_json_with_override(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    p = write_cfg(tmp_path)
    assert main(["check-config", "--config", str(p), "--set", "speed.limit_kmh=55"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["speed"]["limit_kmh"] == 55
    assert doc["fusion"]["max_range_m"] == 30.0


def test_check_config_invalid_exit_2_names_key(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    p = write_cfg(tmp_path, "fusion:\n  max_rang_m: 10\n")
    assert main(["check-config", "--config", str(p)]) == 2
    err = capsys.readouterr().err
    assert "fusion.max_rang_m" in err
    assert "Traceback" not in err


def test_process_requires_config(synth_session: Path) -> None:
    assert main(["process", str(synth_session)]) == 2


def test_bad_start_exit_2_no_output_dir(
    tmp_path: Path, synth_session: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_file = write_cfg(tmp_path)
    out = tmp_path / "out"
    args = ["process", str(synth_session), "--config", str(cfg_file), "--start", "999"]
    assert main([*args, "--out", str(out)]) == 2
    assert not out.exists()
    assert "valid range" in capsys.readouterr().err


def test_unknown_segment_exit_2(
    tmp_path: Path, synth_session: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_file = write_cfg(tmp_path)
    out = tmp_path / "out"
    args = ["process", str(synth_session), "--config", str(cfg_file), "--segment", "7"]
    assert main([*args, "--out", str(out)]) == 2
    assert not out.exists()
    assert "existing segments: 0, 1" in capsys.readouterr().err


def test_bad_override_exit_2_no_output_dir(tmp_path: Path, synth_session: Path) -> None:
    cfg_file = write_cfg(tmp_path)
    out = tmp_path / "out"
    args = ["process", str(synth_session), "--config", str(cfg_file), "--set", "speed.limit_kmh=-1"]
    assert main([*args, "--out", str(out)]) == 2
    assert not out.exists()


def test_synth_prints_session_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["synth", str(tmp_path / "syn"), "--duration", "2", "--seed", "3"]) == 0
    path = Path(capsys.readouterr().out.strip())
    assert path == tmp_path / "syn" / "s000001"
    assert (path / "session.json").exists() and (path / "truth.json").exists()


def test_lite_mode_does_not_import_torch(tmp_path: Path, synth_session: Path) -> None:
    cfg_file = write_cfg(tmp_path)
    out = tmp_path / "out"
    argv = ["process", str(synth_session), "--config", str(cfg_file), "--mode", "lite"]
    argv += ["--segment", "1", "--out", str(out)]
    code = (
        "import sys\n"
        "from speed_trap.cli import main\n"
        f"rc = main({argv!r})\n"
        "assert rc == 0, rc\n"
        "bad = [m for m in ('torch', 'rfdetr', 'ultralytics') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "0001.overlay.mp4").exists()


def test_empty_selection_exit_2_no_output_dir(
    tmp_path: Path, synth_session: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_file = write_cfg(tmp_path)
    out = tmp_path / "out"
    args = ["process", str(synth_session), "--config", str(cfg_file), "--segment", "0"]
    assert main([*args, "--start", "15", "--out", str(out)]) == 2  # 15 s is in segment 1
    assert not out.exists()
    assert "no frames" in capsys.readouterr().err


class FailingDetector:
    name = "failing"

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    def detect(self, bundle: object) -> list[object]:
        self.calls += 1
        if self.calls > 3:
            raise self.exc
        return []


@pytest.mark.parametrize(
    ("exc", "code"),
    [(RuntimeError("boom"), 1), (KeyboardInterrupt(), 130)],
    ids=["runtime-error", "interrupt"],
)
def test_failed_run_exit_code_and_run_json_error(
    tmp_path: Path,
    synth_session: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exc: BaseException,
    code: int,
) -> None:
    import speed_trap.detect

    monkeypatch.setattr(speed_trap.detect, "make_detector", lambda _cfg: FailingDetector(exc))
    cfg_file = write_cfg(tmp_path)
    out = tmp_path / "out"
    args = ["process", str(synth_session), "--config", str(cfg_file), "--segment", "1"]
    assert main([*args, "--out", str(out)]) == code
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["error"].startswith(type(exc).__name__)
    assert run["counters"]["frames"] == 4
    assert "Traceback" not in capsys.readouterr().err


def test_progress_printer_counts_partial_segments(capsys: pytest.CaptureFixture[str]) -> None:
    from speed_trap.cli import ProgressPrinter

    p = ProgressPrinter({0: 40, 1: 10})
    p(0, 10, 5.0)  # mid-segment report
    p(0, 40, 5.0)  # segment end
    p(1, 5, 5.0)
    lines = capsys.readouterr().err.splitlines()
    assert lines == [
        "segment 0000: 10 frames, 5.0 frames/s, 10/50 done, ETA 8 s",
        "segment 0000: 40 frames, 5.0 frames/s, 40/50 done, ETA 2 s",
        "segment 0001: 5 frames, 5.0 frames/s, 45/50 done, ETA 1 s",
    ]


def test_process_missing_video_exits_2(
    tmp_path: Path, synth_session: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import shutil

    session = tmp_path / "s"
    shutil.copytree(synth_session, session)
    (session / "video" / "0000.h264").unlink()
    cfg = write_cfg(tmp_path)
    args = ["process", str(session), "--config", str(cfg), "--mode", "lite"]
    assert main([*args, "--out", str(tmp_path / "out")]) == 2
    assert "0000.h264" in capsys.readouterr().err
