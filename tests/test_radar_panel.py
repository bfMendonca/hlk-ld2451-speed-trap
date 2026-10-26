from __future__ import annotations

import math

import numpy as np
import pytest

from speed_trap.config import Config, OverlayConfig, RadarPanelConfig
from speed_trap.geometry import RadarModel
from speed_trap.render.draw import _hex_to_bgr
from speed_trap.render.radar_panel import PanelState, RadarPanel, fade, fan_xy
from speed_trap.types import RadarScan, RadarTarget

NS = 1_000_000_000


def make_radar(angle_sign: int = 1) -> RadarModel:
    cfg = Config()
    radar = RadarModel.from_config(cfg.radar, cfg.camera.pose.height_m, 100.0)
    radar.angle_sign = angle_sign
    return radar


def tg(
    t_s: float, dist: int, angle: int = 0, speed: int = 40, approaching: bool = True
) -> RadarTarget:
    return RadarTarget(round(t_s * NS), angle, dist, approaching, speed, 100, 0)


def scan(t_s: float, *targets: RadarTarget) -> RadarScan:
    return RadarScan(round(t_s * NS), tuple(targets))


def state(**kw: float) -> PanelState:
    return PanelState(make_radar(), RadarPanelConfig(**kw))


def test_filters_slow_readings() -> None:
    s = state()
    s.update(NS, [scan(1.0, tg(1.0, 20, speed=3), tg(1.0, 21, speed=15))])
    assert [r.target.speed_kmh for r, _ in s.trail(NS)] == [15]


def test_filters_outside_fan_and_range() -> None:
    s = state()
    s.update(NS, [scan(1.0, tg(1.0, 20, angle=46), tg(1.0, 60), tg(1.0, 20, angle=45))])
    assert [(r.target.angle_deg, r.target.distance_m) for r, _ in s.trail(NS)] == [(45, 20)]


def test_ground_range_never_nan() -> None:
    radar = make_radar()
    s = PanelState(radar, RadarPanelConfig())
    s.update(NS, [scan(1.0, tg(1.0, 1))])  # slant range below the radar height
    ((r, _),) = s.trail(NS)
    assert r.ground_m == 0.0 and not math.isnan(r.ground_m)


def test_trail_expires_and_ignores_future() -> None:
    s = state()
    s.update(0, [scan(0.0, tg(0.0, 20))])
    assert len(s.trail(round(3.4 * NS))) == 1
    assert s.trail(round(3.6 * NS)) == []
    s2 = state()
    s2.update(NS, [scan(2.0, tg(2.0, 20))])  # scan stamped after "now"
    assert s2.trail(NS) == []


def test_table_window_order_and_cap() -> None:
    s = state()
    now = 10.0
    old = scan(now - 0.4, tg(now - 0.4, 12))
    new = scan(now - 0.1, *[tg(now - 0.1, 20 + 3 * i) for i in range(7)][::-1])
    s.update(round(now * NS), [old, new])
    rows = s.table(round(now * NS))
    assert [r.target.distance_m for r in rows] == [20, 23, 26, 29, 32, 35]


def test_fade_values() -> None:
    assert fade(0.0, 3.5) == 1.0
    assert fade(3.5, 3.5) == pytest.approx(0.15)
    assert fade(1.0, 3.5) > fade(2.0, 3.5) > fade(3.4, 3.5) >= 0.15


def test_fan_xy_matches_video_handedness() -> None:
    x, y = fan_xy(20.0, -10, 1, 100.0, 400.0, 5.0)
    assert x > 100.0 and y < 400.0
    assert fan_xy(20.0, -10, -1, 100.0, 400.0, 5.0)[0] < 100.0
    assert fan_xy(20.0, 0, 1, 100.0, 400.0, 5.0) == (100.0, 300.0)


def test_table_lists_each_vehicle_once_not_once_per_scan() -> None:
    s = state()
    now = 10.0
    scans = [
        scan(
            now - 0.2 + 0.1 * k,
            tg(now - 0.2 + 0.1 * k, 30 - k),
            tg(now - 0.2 + 0.1 * k, 40 - k, angle=5),
        )
        for k in range(3)
    ]
    s.update(round(now * NS), scans)
    rows = s.table(round(now * NS))
    assert [r.target.distance_m for r in rows] == [28, 38]  # only the newest scan, nearest first


# ---------------------------------------------------------------------------- drawing


def make_panel(
    frame_w: int = 1296, frame_h: int = 972, angle_sign: int = 1, **kw: float
) -> RadarPanel:
    overlay = OverlayConfig.model_validate({"radar_panel": kw})
    return RadarPanel(make_radar(angle_sign), overlay, frame_w, frame_h)


@pytest.mark.parametrize("frame_w", [1296, 1295, 64])
@pytest.mark.parametrize("frac", [0.333, 0.0001])
def test_width_is_even_and_at_least_two(frame_w: int, frac: float) -> None:
    w = make_panel(frame_w, 100, width_frac=frac).width_px
    assert w % 2 == 0 and w >= 2


def test_draw_shape_and_dtype() -> None:
    p = make_panel()
    img = p.draw(NS, [scan(1.0, tg(1.0, 20, angle=-10))])
    assert img.shape == (972, p.width_px, 3) and img.dtype == np.uint8


def test_empty_scan_draws_grid_only() -> None:
    p = make_panel()
    img = p.draw(NS, [])
    assert img.shape[0] == 972
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 1  # background plus grid lines


def _paints(img: np.ndarray, hex_color: str, tol: int = 8) -> np.ndarray:
    return np.abs(img.astype(int) - np.array(_hex_to_bgr(hex_color))).max(axis=2) <= tol


def test_dot_color_by_direction_and_side() -> None:
    p = make_panel()
    coming = p.draw(NS, [scan(1.0, tg(1.0, 20, angle=-10, approaching=True))])
    _, xs = np.nonzero(_paints(coming[: p.table_top_px], "#35d0ba"))  # fan only
    assert len(xs) > 0 and xs.mean() > p.width_px / 2  # negative angle: right, as in the video
    p2 = make_panel()
    going = p2.draw(NS, [scan(1.0, tg(1.0, 20, angle=-10, approaching=False))])
    assert _paints(going[: p2.table_top_px], "#ffb547").any()


def test_readings_outside_fan_leave_panel_unchanged() -> None:
    p, q = make_panel(), make_panel()
    assert (p.draw(NS, [scan(1.0, tg(1.0, 20, speed=3))]) == q.draw(NS, [])).all()


def test_table_text_changes_with_readings() -> None:
    p, q = make_panel(), make_panel()
    with_reading = p.draw(NS, [scan(1.0, tg(1.0, 20, angle=-10, speed=42))])
    without = q.draw(NS, [])
    table_area = slice(p.table_top_px, 972)
    assert 0 < p.table_top_px < 972
    assert (with_reading[table_area] != without[table_area]).any()
