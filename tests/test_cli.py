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
