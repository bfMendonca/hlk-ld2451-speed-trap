from __future__ import annotations

import math

import numpy as np
import pytest

from speed_trap.config import CameraConfig, ConfigError, Intrinsics, PoseConfig, RadarConfig
from speed_trap.geometry import CameraModel, RadarModel, nominal_matrix, range_ring, wedge_polygon

K0 = [[1000.0, 0, 648], [0, 1000.0, 486], [0, 0, 1]]


def cfg_with(
    resolution: tuple[int, int] = (1296, 972),
    fx: float = 1000.0,
    cx: float = 648.0,
    dist: list[float] | None = None,
    **pose: float,
) -> CameraConfig:
    pose = {"height_m": 3.5, "pitch_deg": 0.0, "roll_deg": 0.0, "yaw_deg": 0.0, **pose}
    matrix = [[fx, 0, cx], [0, 1000.0, 486], [0, 0, 1]]
    return CameraConfig(
        intrinsics=Intrinsics(
            resolution=resolution, matrix=matrix, distortion=dist or [0, 0, 0, 0, 0]
        ),
        pose=PoseConfig(**pose),
    )


def make_cam(**kw: object) -> CameraModel:
    return CameraModel.from_config(cfg_with(**kw), 1296, 972)  # type: ignore[arg-type]


def make_radar(
    offset_z: float = -0.3,
    boresight_yaw: float = 0.0,
    angle_sign: int = 1,
    fov_half: float = 20.0,
    max_range: float | None = None,
    min_range: float = 1.0,
) -> RadarModel:
    cfg = RadarConfig.model_validate(
        {
            "offset_m": {"x": 0.0, "y": 0.0, "z": offset_z},
            "boresight": {"yaw_deg": boresight_yaw},
            "angle_sign": angle_sign,
            "fov_half_deg": fov_half,
            "range_m": {"min": min_range, "max": max_range},
        }
    )
    return RadarModel.from_config(cfg, 3.5, None)


def test_point_ahead_projects_below_centre() -> None:
    uv, ok = make_cam().project(np.array([[10.0, 0.0, 0.0]]))
    assert ok[0]
    assert uv[0] == pytest.approx([648, 486 + 1000 * 3.5 / 10])


def test_left_is_smaller_u() -> None:
    uv, _ = make_cam().project(np.array([[10.0, 2.0, 0.0]]))
    assert uv[0][0] == pytest.approx(448)


def test_pitch_down_moves_point_up() -> None:
    uv, _ = make_cam(pitch_deg=10).project(np.array([[10.0, 0, 0]]))
    assert uv[0][1] == pytest.approx(486 + 1000 * math.tan(math.atan(0.35) - math.radians(10)))


def test_yaw_left_moves_straight_ahead_point_right() -> None:
    uv, _ = make_cam(yaw_deg=5).project(np.array([[10.0, 0.0, 0.0]]))
    assert uv[0][0] > 648


def test_backproject_roundtrip_with_yaw_pitch_roll_and_distortion() -> None:
    cam = make_cam(pitch_deg=12, yaw_deg=-5, roll_deg=1.5, dist=[-0.2, 0.05, 0, 0, 0])
    for x, y in [(8, 0), (25, -3), (30, 4)]:
        uv, _ = cam.project(np.array([[x, y, 0.0]]))
        res = cam.backproject_ground(*uv[0])
        assert res == pytest.approx((x, y), abs=1e-3)


def test_backproject_above_horizon_is_none() -> None:
    assert make_cam(pitch_deg=0).backproject_ground(648, 400) is None


def test_point_behind_camera_flagged() -> None:
    _, ok = make_cam().project(np.array([[-5.0, 0, 0]]))
    assert not ok[0]


def test_intrinsics_scaled_to_video_resolution() -> None:
    cam = CameraModel.from_config(cfg_with(resolution=(2592, 1944), fx=2540, cx=1296), 1296, 972)
    assert cam.K[0, 0] == pytest.approx(1270)
    assert cam.K[0, 2] == pytest.approx(648)


def test_aspect_mismatch_is_config_error() -> None:
    with pytest.raises(ConfigError, match=r"camera\.intrinsics\.resolution"):
        CameraModel.from_config(cfg_with(resolution=(1920, 1080)), 1296, 972)


def test_missing_intrinsics_uses_nominal() -> None:
    cam = CameraModel.from_config(CameraConfig(), 1296, 972)
    assert pytest.approx(nominal_matrix(1296, 972)) == cam.K


def test_nominal_matrix() -> None:
    K = nominal_matrix(1296, 972)
    assert K[0, 0] == pytest.approx(648 / math.tan(math.radians(53.5 / 2)))
    assert K[0, 0] == K[1, 1]
    assert K[1, 2] == pytest.approx(486)


def test_radar_slant_to_ground() -> None:
    radar = make_radar(offset_z=-0.5)  # height 3.0
    assert radar.target_to_road(0, 5) == pytest.approx((4.0, 0.0))


def test_radar_angle_sign_and_boresight() -> None:
    x, y = make_radar(boresight_yaw=-8, angle_sign=1).target_to_road(10, 20)
    assert math.degrees(math.atan2(y, x)) == pytest.approx(2.0, abs=1e-6)
    _, y2 = make_radar(angle_sign=-1).target_to_road(10, 20)
    assert y2 < 0


def test_road_to_polar_inverts_target_to_road() -> None:
    r = make_radar(boresight_yaw=-8, angle_sign=-1)
    x, y = r.target_to_road(-6, 25)
    g, az = r.road_to_polar(x, y)
    assert az == pytest.approx(-6)
    assert r.slant_range(g) == pytest.approx(25)


def test_max_range_default_chain() -> None:
    cfg = RadarConfig()
    assert RadarModel.from_config(cfg, 3.5, None).max_range_m == 100.0
    assert RadarModel.from_config(cfg, 3.5, 80.0).max_range_m == 80.0
    cfg2 = RadarConfig.model_validate({"range_m": {"max": 60}})
    assert RadarModel.from_config(cfg2, 3.5, 80.0).max_range_m == 60.0


def test_wedge_stays_inside_fov_and_range() -> None:
    radar = make_radar(fov_half=20, max_range=100)
    pts = wedge_polygon(radar)
    g, az = zip(*(radar.road_to_polar(x, y) for x, y, _ in pts), strict=True)
    assert max(g) <= 100 + 1e-6
    assert min(g) >= radar.min_range_m - 1e-6
    assert max(abs(a) for a in az) <= 20 + 1e-6
    assert max(g) == pytest.approx(100)
    assert max(az) == pytest.approx(20)


def test_range_ring_is_constant_range() -> None:
    radar = make_radar(boresight_yaw=-8)
    pts = range_ring(radar, 20.0)
    gs = [radar.road_to_polar(x, y)[0] for x, y, _ in pts]
    assert gs == pytest.approx([20.0] * len(gs))


def test_ray_azimuth_needs_no_ground_hit() -> None:
    cam = CameraModel.from_config(cfg_with(pitch_deg=5.0), 1296, 972)
    f = 1000.0
    assert cam.ray_azimuth_deg(648.0, 100.0) == pytest.approx(0.0, abs=1e-6)  # above the horizon
    assert cam.backproject_ground(648.0, 100.0) is None
    right = cam.ray_azimuth_deg(648.0 + f * math.tan(math.radians(7)), 486.0)
    assert right < 0  # image right is negative azimuth (road Y is left)
    assert right == pytest.approx(
        -math.degrees(math.atan(math.tan(math.radians(7)) / math.cos(math.radians(5)))), abs=0.3
    )
    left_yawed = CameraModel.from_config(cfg_with(yaw_deg=10.0), 1296, 972)
    assert left_yawed.ray_azimuth_deg(648.0, 486.0) == pytest.approx(10.0, abs=1e-6)
