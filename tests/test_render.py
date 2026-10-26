from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from speed_trap.config import Config, LiteConfig, OverlayConfig
from speed_trap.fusion import FusedTrack
from speed_trap.geometry import CameraModel, RadarModel
from speed_trap.lite import RadarSubject
from speed_trap.render.draw import HudInfo, Renderer, excess_label, limit_label
from speed_trap.source import RecordedSessionSource
from speed_trap.types import Box, Direction, RadarMatch, RadarTarget

W, H = 1296, 972
NS = 1_000_000_000


def bgr(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (4, 2, 0)])


def near_color(pixel: np.ndarray, hex_color: str, tol: int = 12) -> bool:
    return bool((np.abs(pixel.astype(int) - bgr(hex_color)) <= tol).all())


def make_renderer(overlay: OverlayConfig | None = None) -> Renderer:
    cfg = Config()
    camera = CameraModel.from_config(cfg.camera, W, H)
    radar = RadarModel.from_config(cfg.radar, cfg.camera.pose.height_m, 70.0)
    return Renderer(camera, radar, overlay or OverlayConfig(), LiteConfig())


def target(
    distance: int = 25, angle: int = -6, speed: int = 72, approaching: bool = True
) -> RadarTarget:
    return RadarTarget(NS, angle, distance, approaching, speed, 80, 0)


def match(speed: float = 72.0, distance: int = 25, angle: int = -6) -> RadarMatch:
    return RadarMatch(NS, target(distance, angle, round(speed)), (10.0, -2.0), speed, round(speed))


def track(
    box: Box,
    reliable: bool = True,
    direction: Direction | None = "coming",
    track_id: int = 12,
    matches: list[RadarMatch] | None = None,
) -> FusedTrack:
    if matches is None:
        matches = [match()] if reliable else []
    return FusedTrack(
        track_id, "car", box, 0.9, 0, NS, matches=matches, reliable=reliable, direction=direction
    )


def hud(mode: str = "full") -> HudInfo:
    return HudInfo(
        "s000011", 1, 3127, 1_700_000_000_340_000_000, mode,  # type: ignore[arg-type]
        "yolo11s", "ByteTrack", 21.4, 2, 0.08, 0.0, (40.0, 40.0),
    )  # fmt: skip


def render(
    r: Renderer,
    image: np.ndarray,
    tracks: list[FusedTrack] | None = None,
    subjects: list[RadarSubject] | None = None,
    targets: list[RadarTarget] | None = None,
    triggered: set[int] | None = None,
    mode: str = "full",
) -> np.ndarray:
    return r.render(
        image,
        tracks=tracks or [],
        subjects=subjects or [],
        current_targets=targets or [],
        triggered=triggered or set(),
        limit_for=lambda _d: 40.0,
        hud=hud(mode),
    )


def gray_image() -> np.ndarray:
    return np.full((H, W, 3), 90, np.uint8)


def render_one(
    t: FusedTrack, r: Renderer | None = None, triggered: set[int] | None = None
) -> np.ndarray:
    return render(r or make_renderer(), gray_image(), tracks=[t], triggered=triggered)


def test_output_shape_and_input_untouched() -> None:
    img = gray_image()
    before = img.copy()
    out = render(make_renderer(), img, tracks=[track((400, 300, 600, 450))])
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out is not img
    assert (img == before).all()


def test_reliable_coming_bracket_uses_coming_color() -> None:
    out = render_one(track((400, 300, 600, 450), reliable=True, direction="coming"))
    assert near_color(out[300 + 1, 400 + 10], "#35d0ba")


def test_reliable_going_bracket_uses_going_color() -> None:
    out = render_one(track((400, 300, 600, 450), reliable=True, direction="going"))
    assert near_color(out[300 + 1, 400 + 10], "#ffb547")


def test_unreliable_track_bracket_is_gray_and_dashed() -> None:
    out = render_one(track((400, 300, 600, 450), reliable=False, direction=None))
    row = out[300, 400:437].astype(int)
    on = [near_color(p, "#a7afb8") for p in row]
    assert any(on) and not all(on)  # dashed: gaps between dashes
    assert not any(near_color(p, "#35d0ba") for p in row)


def test_wedge_changes_ground_not_sky() -> None:
    img = gray_image()
    out = render(make_renderer(OverlayConfig.model_validate({"show": {"wedge": True}})), img)
    assert (out[900, 648] != img[900, 648]).any()
    assert (out[100, 648] == img[100, 648]).all()


def test_wedge_toggle_off_leaves_ground_untouched() -> None:
    cfg = OverlayConfig.model_validate({"show": {"wedge": False, "range_rings": [], "hud": False}})
    img = gray_image()
    out = render(make_renderer(cfg), img)
    assert (out == img).all()


def test_tag_sits_left_of_box_by_default() -> None:
    out = render_one(track((800, 300, 1000, 450)))
    # tag is 168 px wide, 32 px left of the box: dark fill inside it, box column untouched by it
    assert (out[365, 700:760] < 80).all()


def test_tag_flips_right_near_left_edge() -> None:
    img = gray_image()
    out = render_one(track((10, 300, 210, 450)))
    # right of the box: x2 + 32 .. x2 + 200 holds the dark tag fill
    assert (out[365, 260:330] < 80).all()
    assert (out[365, 0:8] == img[365, 0:8]).all()


def test_triggered_track_shows_alert_badge() -> None:
    plain = render_one(track((800, 300, 1000, 450)))
    out = render_one(track((800, 300, 1000, 450)), triggered={12})
    assert not (plain == out).all()
    assert any(near_color(p, "#ff4d5e") for p in out[300:340, 520:700].reshape(-1, 3))


def test_over_limit_speed_text_is_alert_colored() -> None:
    out = render_one(track((800, 300, 1000, 450), matches=[match(72.0)]))
    patch = out[300:370, 600:790].reshape(-1, 3)
    assert any(near_color(p, "#ff4d5e", 30) for p in patch)
    ok = render_one(track((800, 300, 1000, 450), matches=[match(30.0)]))
    patch = ok[300:370, 600:790].reshape(-1, 3)
    assert not any(near_color(p, "#ff4d5e", 30) for p in patch)


def test_lite_pillar_drawn_at_radar_point() -> None:
    r = make_renderer()
    m = match()
    subj = RadarSubject(1, [m], "coming")
    img = gray_image()
    out = render(r, img, subjects=[subj], mode="lite")
    uv, front = r._camera.project(np.array([[*m.road_xy, 0.0]]))
    assert front[0]
    u, v = int(uv[0, 0]), int(uv[0, 1])
    assert (out[v - 5, u - 6 : u + 7] != img[v - 5, u - 6 : u + 7]).any()
    # white core line of the pillar
    assert near_color(out[v - 10, u], "#ffffff", 60)


def _pillar_mid_changed(r: Renderer, distance: int, mode: str = "full") -> bool:
    tg = RadarTarget(NS, 0, distance, True, 50, 70, 0)
    img = gray_image()
    base = render(r, img, mode=mode)
    out = render(r, img, targets=[tg], mode=mode)
    xy = r._radar.target_to_road(0, distance)
    uv, front = r._camera.project(np.array([[*xy, 0.0], [*xy, 0.8]]))
    assert front.all()
    u, v = round(uv[0, 0]), round(uv[1, 1])
    assert 50 <= v < H - 50 and 0 <= u < W
    return bool((out[v, u] != base[v, u]).any())


def test_far_target_drawn_as_thin_pillar_without_label() -> None:
    r = make_renderer()
    assert _pillar_mid_changed(r, 40)  # beyond far_range_m (30), inside radar range (70)
    assert not _pillar_mid_changed(r, 20)  # inside fusion range: no thin pillar
    assert not _pillar_mid_changed(Renderer(r._camera, r._radar, OverlayConfig(), LiteConfig(),
                                            far_range_m=50.0), 40)  # fmt: skip


def test_lite_far_target_drawn_as_thin_pillar() -> None:
    r = make_renderer()
    assert _pillar_mid_changed(r, 40, mode="lite")
    assert not _pillar_mid_changed(r, 20, mode="lite")  # near targets are linked subjects


def test_hud_toggle_off_leaves_top_rows_untouched() -> None:
    cfg = OverlayConfig.model_validate({"show": {"hud": False, "wedge": False}})
    img = gray_image()
    out = render(make_renderer(cfg), img)
    assert (out[:44] == img[:44]).all() and (out[-40:] == img[-40:]).all()
    on = render(make_renderer(), img)
    assert (on[:44] != img[:44]).any() and (on[-40:] != img[-40:]).any()


def test_hud_without_wall_time_still_renders() -> None:
    info = HudInfo("s1", 0, 1, None, "lite", None, None, 0.0, 0, None, 0.0, (40.0, 40.0))
    out = make_renderer().render(
        gray_image(),
        tracks=[],
        subjects=[],
        current_targets=[],
        triggered=set(),
        limit_for=lambda _d: 40.0,
        hud=info,
    )
    assert out.shape == (H, W, 3)


def _scene_tracks(session: Path, camera: CameraModel) -> tuple[np.ndarray, list[RadarTarget]]:
    for b in RecordedSessionSource(session, Config().radar):
        targets = [t for sc in b.scans[-1:] for t in sc.targets]
        if len(targets) >= 2:
            return b.image, targets
    raise AssertionError("no frame with two radar targets")


def test_write_preview_png(synth_session: Path, tmp_path: Path) -> None:
    r = make_renderer()
    cam, radar = r._camera, r._radar
    image, targets = _scene_tracks(synth_session, cam)
    tracks: list[FusedTrack] = []
    subjects: list[RadarSubject] = []
    for i, tg in enumerate(targets, start=1):
        xy = radar.target_to_road(tg.angle_deg, tg.distance_m)
        uv, front = cam.project(np.array([[*xy, 0.0]]))
        if not front[0]:
            continue
        u, v = float(uv[0, 0]), float(uv[0, 1])
        depth = float((np.array([*xy, 0.0]) - cam.center) @ cam.R[2])
        w, h = cam.K[0, 0] * 1.8 / depth, cam.K[0, 0] * 1.5 / depth
        m = RadarMatch(tg.t_ns, tg, xy, float(tg.speed_kmh) + 30.0 * (i == 1), tg.speed_kmh)
        direction: Direction = "coming" if tg.approaching else "going"
        tracks.append(
            track(
                (u - w / 2, v - h, u + w / 2, v),
                direction=direction,
                track_id=i,
                matches=[m],
            )
        )
        subjects.append(RadarSubject(i, [m], direction))
    tracks.append(track((120, 520, 300, 640), reliable=False, direction=None, track_id=99))
    for mode, name in (("full", "render-preview-full.png"), ("lite", "render-preview-lite.png")):
        out = render(
            r,
            image,
            tracks=tracks if mode == "full" else [],
            subjects=subjects if mode == "lite" else [],
            targets=targets,
            triggered={1},
            mode=mode,
        )
        path = tmp_path / name
        assert cv2.imwrite(str(path), out)
        back = cv2.imread(str(path))
        assert back is not None and back.shape == (H, W, 3)
        keep = os.environ.get("SPEED_TRAP_PREVIEW_DIR")
        if keep:
            (Path(keep) / name).write_bytes(path.read_bytes())


def test_excess_label_is_the_max_excess_never_negative() -> None:
    assert excess_label([match(55.0), match(48.0)], 40.0) == "+15"
    assert excess_label([match(45.9), match(38.0)], 40.0) == "+5"  # floored
    assert excess_label([match(39.0)], 40.0) == "+0"


def test_limit_label_shows_both_directions_only_when_they_differ() -> None:
    assert limit_label(40.0, 40.0) == "LIMIT 40 km/h"
    assert limit_label(40.0, 60.0) == "LIMIT \u25bc40 \u25b260 km/h"


def test_hud_draws_differing_limits() -> None:
    import dataclasses

    r, img = make_renderer(), gray_image()

    def top_bar(info: HudInfo) -> np.ndarray:
        out = r.render(
            img,
            tracks=[],
            subjects=[],
            current_targets=[],
            triggered=set(),
            limit_for=lambda _d: 40.0,
            hud=info,
        )
        return out[:44]

    both = dataclasses.replace(hud(), limits_kmh=(40.0, 60.0))
    assert (top_bar(hud()) != top_bar(both)).any()
