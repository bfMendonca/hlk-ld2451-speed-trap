"""Camera and radar geometry in the road frame.

Road frame: X forward (camera heading at yaw 0), Y left, Z up, origin on the ground below the
camera. Pitch positive = camera looks down, yaw positive = camera turns left.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from speed_trap.config import CameraConfig, ConfigError, RadarConfig

OV5647_HFOV_DEG = 53.5
_ASPECT_TOLERANCE = 0.01

# Road (X fwd, Y left, Z up) -> OpenCV camera (x right, y down, z fwd).
_R0 = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def nominal_matrix(width: int, height: int) -> np.ndarray:
    """Approximate pinhole K from the OV5647 nominal HFOV (square pixels, centred)."""
    f = (width / 2) / math.tan(math.radians(OV5647_HFOV_DEG / 2))
    return np.array([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class CameraModel:
    def __init__(
        self,
        K: np.ndarray,
        dist: np.ndarray,
        width: int,
        height: int,
        height_m: float,
        pitch_deg: float,
        roll_deg: float,
        yaw_deg: float,
    ) -> None:
        self.K = K
        self.dist = dist
        self.width = width
        self.height = height
        self.height_m = height_m
        self.center = np.array([0.0, 0.0, height_m])
        self.R = (
            _rot_z(math.radians(roll_deg))
            @ _rot_x(math.radians(pitch_deg))
            @ _R0
            @ _rot_z(-math.radians(yaw_deg))
        )
        self._rvec, _ = cv2.Rodrigues(self.R)
        self._tvec = -self.R @ self.center

    @classmethod
    def from_config(cls, cfg: CameraConfig, width: int, height: int) -> CameraModel:
        intr = cfg.intrinsics
        if intr is None:
            K = nominal_matrix(width, height)
            dist = np.zeros(5)
        else:
            w0, h0 = intr.resolution
            if abs((w0 / h0) / (width / height) - 1) > _ASPECT_TOLERANCE:
                raise ConfigError(
                    f"camera.intrinsics.resolution {w0}x{h0} has a different aspect ratio "
                    f"than the video {width}x{height}"
                )
            sx, sy = width / w0, height / h0
            K = np.array(intr.matrix, dtype=float)
            K[0, :] *= sx
            K[1, :] *= sy
            dist = np.array(intr.distortion, dtype=float)
        p = cfg.pose
        return cls(K, dist, width, height, p.height_m, p.pitch_deg, p.roll_deg, p.yaw_deg)

    def project(self, points_road: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """N x 3 road points -> (N x 2 pixels, N bool 'in front of camera')."""
        pts = np.asarray(points_road, dtype=float).reshape(-1, 3)
        in_front = (pts - self.center) @ self.R[2] > 1e-6
        uv, _ = cv2.projectPoints(pts, self._rvec, self._tvec, self.K, self.dist)
        return uv.reshape(-1, 2), in_front

    def backproject_ground(self, u: float, v: float) -> tuple[float, float] | None:
        """Pixel -> road (x, y) on the ground plane, or None if the ray misses it."""
        pt = np.array([[[u, v]]], dtype=np.float64)
        xn, yn = cv2.undistortPoints(pt, self.K, self.dist).reshape(2)
        ray = self.R.T @ np.array([xn, yn, 1.0])
        if ray[2] >= -1e-9:
            return None
        hit = self.center + (-self.center[2] / ray[2]) * ray
        return float(hit[0]), float(hit[1])


class RadarModel:
    def __init__(
        self,
        position_xy: tuple[float, float],
        height_m: float,
        boresight_yaw_deg: float,
        angle_sign: int,
        fov_half_deg: float,
        min_range_m: float,
        max_range_m: float,
    ) -> None:
        self.position_xy = position_xy
        self.height_m = height_m
        self.boresight_yaw_deg = boresight_yaw_deg
        self.angle_sign = angle_sign
        self.fov_half_deg = fov_half_deg
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m

    @classmethod
    def from_config(
        cls, cfg: RadarConfig, camera_height_m: float, max_range_default: float | None
    ) -> RadarModel:
        max_range = cfg.range_m.max
        if max_range is None:
            max_range = max_range_default if max_range_default is not None else 100.0
        return cls(
            (cfg.offset_m.x, cfg.offset_m.y),
            camera_height_m + cfg.offset_m.z,
            cfg.boresight.yaw_deg,
            cfg.angle_sign,
            cfg.fov_half_deg,
            cfg.range_m.min,
            max_range,
        )

    def slant_range(self, ground_range: float) -> float:
        return math.hypot(ground_range, self.height_m)

    def _road_xy(self, ground_range: float, angle_deg: float) -> tuple[float, float]:
        az = math.radians(self.boresight_yaw_deg + self.angle_sign * angle_deg)
        return (
            self.position_xy[0] + ground_range * math.cos(az),
            self.position_xy[1] + ground_range * math.sin(az),
        )

    def target_to_road(self, angle_deg: int, distance_m: int) -> tuple[float, float]:
        g = math.sqrt(max(distance_m**2 - self.height_m**2, 0.0))
        return self._road_xy(g, angle_deg)

    def road_to_polar(self, x: float, y: float) -> tuple[float, float]:
        """Road point -> (ground range m, azimuth deg relative to boresight, as reported)."""
        dx, dy = x - self.position_xy[0], y - self.position_xy[1]
        az = math.degrees(math.atan2(dy, dx)) - self.boresight_yaw_deg
        az = (az + 180.0) % 360.0 - 180.0
        return math.hypot(dx, dy), az / self.angle_sign


def range_ring(radar: RadarModel, r: float, step_deg: float = 0.5) -> np.ndarray:
    """Road points (M x 3, z = 0) of the ground-range arc ``r`` across the field of view."""
    n = max(math.ceil(2 * radar.fov_half_deg / step_deg), 1)
    angles = np.linspace(-radar.fov_half_deg, radar.fov_half_deg, n + 1)
    return np.array([(*radar._road_xy(r, float(a)), 0.0) for a in angles])


def wedge_polygon(radar: RadarModel, step_m: float = 1.0, step_deg: float = 1.0) -> np.ndarray:
    """Road points (M x 3, z = 0) outlining the sector between min and max ground range."""
    fov, r0, r1 = radar.fov_half_deg, radar.min_range_m, radar.max_range_m
    n_r = max(math.ceil((r1 - r0) / step_m), 1)
    radii = np.linspace(r0, r1, n_r + 1)
    pts: list[tuple[float, float]] = []
    pts += [radar._road_xy(r, -fov) for r in radii]
    pts += [p[:2] for p in range_ring(radar, r1, step_deg)][1:]
    pts += [radar._road_xy(r, fov) for r in radii[::-1]][1:]
    pts += [p[:2] for p in range_ring(radar, r0, step_deg)[::-1]][1:]
    return np.array([(x, y, 0.0) for x, y in pts])
