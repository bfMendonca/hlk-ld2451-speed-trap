from pathlib import Path

import pytest
from pydantic import ValidationError

from speed_trap.config import (
    Config,
    ConfigError,
    OverlayConfig,
    apply_overrides,
    config_hash,
    default_config_text,
    load_config,
)


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


def test_default_file_loads_and_matches_model_defaults(tmp_path: Path) -> None:
    p = _write(tmp_path, default_config_text())
    assert load_config(p) == Config()


def test_unknown_key_names_dotted_path(tmp_path: Path) -> None:
    p = _write(tmp_path, "fusion:\n  max_rang_m: 10\n")
    with pytest.raises(ConfigError, match=r"fusion\.max_rang_m"):
        load_config(p)


def test_override_sets_nested_value_and_validates(tmp_path: Path) -> None:
    p = _write(tmp_path, "{}\n")
    cfg = load_config(p, ["speed.limit_kmh=50", "capture.enabled=true"])
    assert cfg.speed.limit_kmh == 50 and cfg.capture.enabled is True
    with pytest.raises(ConfigError, match=r"speed\.limit_kmh"):
        load_config(p, ["speed.limit_kmh=-3"])


def test_override_without_equals_is_error(tmp_path: Path) -> None:
    p = _write(tmp_path, "{}\n")
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        load_config(p, ["speed.limit_kmh"])


def test_apply_overrides_does_not_mutate_input() -> None:
    data: dict[str, object] = {"speed": {"limit_kmh": 40}}
    out = apply_overrides(data, ["speed.limit_kmh=50", "radar.range_m.min=2"])
    assert data == {"speed": {"limit_kmh": 40}}
    assert out == {"speed": {"limit_kmh": 50}, "radar": {"range_m": {"min": 2}}}


def test_approaching_byte_only_0_or_1(tmp_path: Path) -> None:
    p = _write(tmp_path, "radar:\n  approaching_byte: 2\n")
    with pytest.raises(ConfigError, match=r"radar\.approaching_byte"):
        load_config(p)


def test_matrix_must_be_3x3(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "camera:\n  intrinsics:\n    resolution: [1296, 972]\n"
        "    matrix: [[1, 0, 2], [0, 1, 3]]\n    distortion: [0, 0, 0, 0, 0]\n",
    )
    with pytest.raises(ConfigError, match=r"camera\.intrinsics\.matrix"):
        load_config(p)


def test_distortion_length_checked(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "camera:\n  intrinsics:\n    resolution: [1296, 972]\n"
        "    matrix: [[1, 0, 2], [0, 1, 3], [0, 0, 1]]\n    distortion: [0, 0, 0]\n",
    )
    with pytest.raises(ConfigError, match=r"camera\.intrinsics\.distortion"):
        load_config(p)


def test_range_min_must_be_below_max(tmp_path: Path) -> None:
    p = _write(tmp_path, "radar:\n  range_m: {min: 5.0, max: 4.0}\n")
    with pytest.raises(ConfigError, match=r"radar\.range_m"):
        load_config(p)


@pytest.mark.parametrize(
    "text",
    [
        "radar:\n  angle_sign: 2\n",
        "detector:\n  every_n_frames: 0\n",
        "fusion:\n  ambiguity_ratio: 0\n",
        "fusion:\n  ambiguity_ratio: 1.5\n",
        "fusion:\n  max_range_m: 0\n",
        "fusion:\n  direction_min_travel_m: 0\n",
        "fusion:\n  max_scan_age_s: 0\n",
        "synth:\n  frame_gaps: [[1.0, 0]]\n",
    ],
)
def test_out_of_range_values_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, text))


def test_non_mapping_file_is_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(_write(tmp_path, "- 1\n- 2\n"))


def test_config_hash_stable_and_sensitive() -> None:
    assert config_hash(Config()) == config_hash(Config())
    assert config_hash(Config()) != config_hash(Config(speed={"limit_kmh": 41}))


def test_video_codec_only_h264(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("output:\n  video_codec: h265\n")
    with pytest.raises(ConfigError, match=r"output\.video_codec"):
        load_config(p)


def test_radar_panel_defaults() -> None:
    cfg = OverlayConfig()
    panel = cfg.radar_panel
    assert (panel.enabled, panel.width_frac, panel.trail_s) == (True, 0.333, 3.5)
    assert (panel.min_speed_kmh, panel.max_range_m) == (15.0, 50.0)
    assert (panel.fov_half_deg, panel.max_rows) == (45.0, 6)
    assert cfg.show.wedge is False


@pytest.mark.parametrize(
    "bad",
    [
        {"width_frac": 0},
        {"width_frac": 1.5},
        {"trail_s": 0},
        {"max_rows": 0},
        {"fov_half_deg": 100},
        {"nope": 1},
    ],
)
def test_radar_panel_rejects_bad_values(bad: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        OverlayConfig.model_validate({"radar_panel": bad})


def test_fusion_match_on_default_and_validation() -> None:
    from speed_trap.config import FusionConfig

    assert FusionConfig().match_on == "range_azimuth"
    assert FusionConfig(match_on="azimuth").match_on == "azimuth"
    with pytest.raises(ValidationError):
        FusionConfig(match_on="nope")
