"""Overlay renderer, label style B (spec §5.8).

Geometry is drawn with OpenCV (anti-aliased) and numpy alpha blending; all text goes through one
Pillow pass per frame. Sizes are specified for a 1296 px wide frame and scaled by ``width / 1296``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from speed_trap.config import LiteConfig, OverlayConfig
from speed_trap.fusion import FusedTrack
from speed_trap.geometry import CameraModel, RadarModel, range_ring, wedge_polygon
from speed_trap.lite import RadarSubject
from speed_trap.types import Direction, RadarMatch, RadarTarget

_FONT_DIR = Path(__file__).parent / "fonts"
_FONT_FILES = {
    "semibold": "BarlowSemiCondensed-SemiBold.ttf",
    "bold": "BarlowSemiCondensed-Bold.ttf",
    "mono": "IBMPlexMono-Regular.ttf",
}
_REF_WIDTH = 1296.0
_INK = (0x0D, 0x11, 0x13)  # RGB
_LIGHT = (0xDF, 0xE5, 0xE8)
_NONE_TEXT = (0xCF, 0xD5, 0xDA)
_WHITE = (255, 255, 255)
_PANEL_ALPHA = 0.86
_BAR_ALPHA = 0.72
_WEDGE_ALPHA_NEAR = 0.22
_WEDGE_ALPHA_FAR = 0.04
_WEDGE_OUTLINE_ALPHA = 0.55
_RING_ALPHA = 0.35
_Color = tuple[int, int, int]  # channel order of the frame (BGR)
_Dash = tuple[float, float]


class _Subject(Protocol):
    @property
    def subject_id(self) -> int: ...
    @property
    def direction(self) -> Direction | None: ...
    @property
    def last_match(self) -> RadarMatch | None: ...


@dataclass(frozen=True)
class HudInfo:
    session_id: str
    segment: int
    index: int
    wall_time_ns: int | None
    mode: Literal["full", "lite"]
    model: str | None
    tracker: str | None
    fps: float
    radar_targets: int
    radar_age_s: float | None
    clock_offset_ms: float
    limit_kmh: float


@dataclass(frozen=True)
class _Text:
    x: float
    y: float  # baseline
    text: str
    font: str
    size: float  # reference pixels (scaled by the font cache)
    color: _Color
    anchor: str = "ls"


def _hex_to_bgr(value: str) -> _Color:
    h = value.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def _wall_clock(ns: int | None) -> str:
    if ns is None:
        return "--:--:--"
    dt = datetime.fromtimestamp(ns // 1_000_000_000, tz=timezone.utc)
    return f"{dt:%H:%M:%S}.{(ns % 1_000_000_000) // 1_000_000:03d}"


class _Fonts:
    def __init__(self, scale: float) -> None:
        self._scale = scale
        self._cache: dict[tuple[str, float], ImageFont.FreeTypeFont] = {}

    def get(self, name: str, size: float) -> ImageFont.FreeTypeFont:
        key = (name, size)
        font = self._cache.get(key)
        if font is None:
            path = str(_FONT_DIR / _FONT_FILES[name])
            font = ImageFont.truetype(path, size * self._scale)
            self._cache[key] = font
        return font

    def width(self, name: str, size: float, text: str) -> float:
        return float(self.get(name, size).getlength(text))


def _blend_rect(
    img: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: _Color, alpha: float
) -> None:
    h, w = img.shape[:2]
    xa, xb = max(round(x0), 0), min(round(x1), w)
    ya, yb = max(round(y0), 0), min(round(y1), h)
    if xa >= xb or ya >= yb:
        return
    roi = img[ya:yb, xa:xb]
    col = np.array(color, np.float32)
    roi[:] = (roi * (1.0 - alpha) + col * alpha).astype(np.uint8)


def _blend_mask(
    img: np.ndarray,
    bbox: tuple[float, float, float, float],
    color: _Color,
    alpha: float,
    draw: Callable[[np.ndarray, tuple[int, int]], None],
) -> None:
    """Blend ``color`` through a mask that ``draw(mask, origin)`` paints inside ``bbox``."""
    h, w = img.shape[:2]
    xa, ya = max(math.floor(bbox[0]), 0), max(math.floor(bbox[1]), 0)
    xb, yb = min(math.ceil(bbox[2]), w), min(math.ceil(bbox[3]), h)
    if xa >= xb or ya >= yb:
        return
    mask = np.zeros((yb - ya, xb - xa), np.uint8)
    draw(mask, (xa, ya))
    a = (mask.astype(np.float32) * (alpha / 255.0))[..., None]
    roi = img[ya:yb, xa:xb]
    roi[:] = (roi * (1.0 - a) + np.array(color, np.float32) * a).astype(np.uint8)


def _dashed(
    img: np.ndarray, pts: np.ndarray, color: _Color | int, thickness: int, dash: _Dash
) -> None:
    """Draw a dashed anti-aliased polyline (pixel coordinates, ``dash`` = (on, off) px)."""
    on, off = dash
    period = on + off
    pos = 0.0
    for a, b in itertools.pairwise(pts):
        seg = float(np.hypot(*(b - a)))
        if seg == 0:
            continue
        d = (b - a) / seg
        t = 0.0
        while t < seg:
            phase = (pos + t) % period
            if phase < on:
                run = min(on - phase, seg - t)
                # cv2 thick lines have round caps: inset each end so a dash keeps its length
                inset = min(thickness / 2, run / 2)
                p0, p1 = a + d * (t + inset), a + d * (t + run - inset)
                cv2.line(
                    img,
                    (round(p0[0]), round(p0[1])),
                    (round(p1[0]), round(p1[1])),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                run = period - phase
            t += max(run, 1e-6)
        pos += seg


def _triangle(
    img: np.ndarray, cx: float, cy: float, size: float, down: bool, color: _Color
) -> None:
    half = size / 2
    sign = 1 if down else -1
    pts = np.array(
        [
            (cx - half, cy - sign * half * 0.8),
            (cx + half, cy - sign * half * 0.8),
            (cx, cy + sign * half * 0.8),
        ]
    )
    cv2.fillPoly(img, [np.round(pts * 16).astype(np.int32)], color, cv2.LINE_AA, shift=4)


class Renderer:
    def __init__(
        self,
        camera: CameraModel,
        radar: RadarModel,
        overlay: OverlayConfig,
        lite: LiteConfig,
        *,
        far_range_m: float = 30.0,
    ) -> None:
        self._camera = camera
        self._radar = radar
        self._overlay = overlay
        self._lite = lite
        self._far_range_m = far_range_m
        self._w, self._h = camera.width, camera.height
        self._k = camera.width / _REF_WIDTH
        self._fonts = _Fonts(self._k)
        c = overlay.colors
        self._col = {
            "coming": _hex_to_bgr(c.coming),
            "going": _hex_to_bgr(c.going),
            "none": _hex_to_bgr(c.none),
        }
        self._alert = _hex_to_bgr(c.alert)
        self._wedge_box: tuple[int, int, int, int] | None = None
        self._wedge_inv: np.ndarray | None = None
        self._wedge_add: np.ndarray | None = None
        self._ring_labels: list[_Text] = []
        if overlay.show.wedge:
            self._precompute_wedge()

    # ----------------------------------------------------------------- wedge (precomputed)

    def _px(self, x: float) -> int:
        return max(1, round(x * self._k))

    def _project(self, pts: np.ndarray) -> np.ndarray:
        """Road points -> pixel polyline, dropping points behind the camera."""
        uv, front = self._camera.project(pts)
        return np.asarray(np.clip(uv[front], -1e5, 1e5))

    def _precompute_wedge(self) -> None:
        w, h, k = self._w, self._h, self._k
        poly = self._project(wedge_polygon(self._radar))
        if len(poly) < 3:
            return
        alpha = np.zeros((h, w), np.float32)
        # Fill: alpha fades with distance, i.e. with the image row, as in the mockup gradient.
        fill = np.zeros((h, w), np.uint8)
        cv2.fillPoly(fill, [np.round(poly * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
        ys = np.where(fill.any(axis=1))[0]
        if len(ys) == 0:
            return
        top, bottom = float(ys[0]), float(ys[-1])
        t = np.clip((np.arange(h) - top) / max(bottom - top, 1.0), 0.0, 1.0)
        grad = (_WEDGE_ALPHA_FAR + (_WEDGE_ALPHA_NEAR - _WEDGE_ALPHA_FAR) * t).astype(np.float32)
        alpha = fill.astype(np.float32) / 255.0 * grad[:, None]
        # Outline.
        line = np.zeros((h, w), np.uint8)
        cv2.polylines(
            line,
            [np.round(poly * 16).astype(np.int32)],
            True,
            255,
            self._px(2),
            cv2.LINE_AA,
            shift=4,
        )
        alpha = np.maximum(alpha, line.astype(np.float32) / 255.0 * _WEDGE_OUTLINE_ALPHA)
        # Dashed range rings and their labels at the right edge.
        rings = np.zeros((h, w), np.uint8)
        for r in self._overlay.show.range_rings:
            arc = self._project(range_ring(self._radar, r))
            if len(arc) < 2:
                continue
            _dashed(rings, arc, 255, self._px(1.5), (6 * k, 6 * k))
            e = arc[-1]
            if 0 <= e[0] < w and 0 <= e[1] < h:
                size = 16.0 if r < 30 else 13.0
                label = f"{r:g} m"
                self._ring_labels.append(
                    _Text(e[0] + 8 * k, e[1] + 4 * k, label, "mono", size, (235, 235, 235))
                )
        alpha = np.maximum(alpha, rings.astype(np.float32) / 255.0 * _RING_ALPHA)
        rows = np.where(alpha.any(axis=1))[0]
        cols = np.where(alpha.any(axis=0))[0]
        if len(rows) == 0:
            return
        y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        a = alpha[y0:y1, x0:x1, None]
        self._wedge_box = (x0, y0, x1, y1)
        self._wedge_inv = 1.0 - a
        self._wedge_add = 255.0 * a

    def _draw_wedge(self, out: np.ndarray) -> None:
        if self._wedge_box is None or self._wedge_inv is None or self._wedge_add is None:
            return
        x0, y0, x1, y1 = self._wedge_box
        roi = out[y0:y1, x0:x1]
        roi[:] = roi * self._wedge_inv + self._wedge_add

    # ----------------------------------------------------------------------------- render

    def render(
        self,
        image: np.ndarray,
        *,
        tracks: Sequence[FusedTrack],
        subjects: Sequence[RadarSubject],
        current_targets: Sequence[RadarTarget],
        triggered: set[int],
        limit_for: Callable[[Direction], float],
        hud: HudInfo,
    ) -> np.ndarray:
        out = image.copy()
        texts: list[_Text] = []
        if self._overlay.show.wedge:
            self._draw_wedge(out)
            texts.extend(self._ring_labels)
        if hud.mode == "full":
            for tg in current_targets:
                if self._ground_range(tg) > self._far_range_m:
                    self._thin_pillar(out, tg)
            for tr in sorted(tracks, key=lambda t: t.xyxy[3]):
                self._draw_track(out, texts, tr, tr.subject_id in triggered, limit_for)
        else:
            for sj in sorted(subjects, key=lambda s: -s.last_match.road_xy[0]):
                self._draw_pillar(out, texts, sj, sj.subject_id in triggered, limit_for)
        if self._overlay.show.hud:
            self._draw_hud(out, texts, hud)
        return self._draw_text(out, texts)

    def _ground_range(self, tg: RadarTarget) -> float:
        return math.sqrt(max(tg.distance_m**2 - self._radar.height_m**2, 0.0))

    # ---------------------------------------------------------------------------- tracks

    def _color_for(self, direction: Direction | None, reliable: bool) -> _Color:
        if reliable and direction is not None:
            return self._col[direction]
        return self._col["none"]

    @staticmethod
    def _speed_parts(m: RadarMatch, limit: float) -> tuple[int, bool]:
        return round(m.speed_kmh), m.speed_kmh > limit

    def _brackets(
        self, out: np.ndarray, box: tuple[float, float, float, float], col: _Color, dashed: bool
    ) -> None:
        x1, y1, x2, y2 = box
        length = 0.25 * min(x2 - x1, y2 - y1)
        thick = self._px(3.5)
        corners = [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]
        for x, y, sx, sy in corners:
            pts = np.array([(x + sx * length, y), (x, y), (x, y + sy * length)])
            if dashed:
                _dashed(out, pts, col, thick, (4 * self._k, 4 * self._k))
            else:
                cv2.polylines(
                    out,
                    [np.round(pts * 16).astype(np.int32)],
                    False,
                    col,
                    thick,
                    cv2.LINE_AA,
                    shift=4,
                )

    def _draw_track(
        self,
        out: np.ndarray,
        texts: list[_Text],
        tr: FusedTrack,
        is_triggered: bool,
        limit_for: Callable[[Direction], float],
    ) -> None:
        k = self._k
        m = tr.last_match
        has_radar = tr.reliable and m is not None and tr.direction is not None
        col = self._color_for(tr.direction, has_radar)
        x1, y1, x2, _ = tr.xyxy
        self._brackets(out, tr.xyxy, col, dashed=not has_radar)

        tw, th = 168 * k, 64 * k
        left = x1 - 200 * k >= 0
        tx = x1 - 200 * k if left else x2 + 32 * k
        ty = min(max(y1 + 4 * k, 0.0), self._h - th)
        lead_from = (x1 if left else x2, y1 + 14 * k)
        lead_to = (tx + tw if left else tx, ty + 30 * k)
        cv2.line(
            out,
            (round(lead_from[0]), round(lead_from[1])),
            (round(lead_to[0]), round(lead_to[1])),
            col,
            self._px(2),
            cv2.LINE_AA,
        )
        _blend_rect(out, tx, ty, tx + tw, ty + th, _INK[::-1], _PANEL_ALPHA)
        cv2.rectangle(
            out,
            (round(tx), round(ty)),
            (round(tx + tw), round(ty + th)),
            col,
            self._px(1.5),
            cv2.LINE_AA,
        )

        if not has_radar or m is None or tr.direction is None:
            texts.append(
                _Text(
                    tx + 10 * k,
                    ty + 26 * k,
                    f"#{tr.track_id} NO RADAR",
                    "bold",
                    17,
                    _NONE_TEXT[::-1],
                )
            )
            if m is None:
                last = "no reading"
            else:
                age = (tr.last_ts_ns - m.t_ns) / 1e9
                last = f"last {round(m.speed_kmh)} km/h · {age:.1f} s"
            texts.append(_Text(tx + 10 * k, ty + 48 * k, last, "mono", 11.5, col))
            return

        limit = limit_for(tr.direction)
        speed, over = self._speed_parts(m, limit)
        speed_col = self._alert if over else col
        self._speed_text(texts, tx + 10 * k, ty + 32 * k, speed, speed_col, _LIGHT[::-1])
        tgt = m.target
        id_text = f"#{tr.track_id}"
        right = tx + 158 * k
        texts.append(_Text(right, ty + 28 * k, id_text, "bold", 16, col, "rs"))
        id_w = self._fonts.width("bold", 16, id_text)
        _triangle(out, right - id_w - 9 * k, ty + 22 * k, 11 * k, tr.direction == "coming", col)
        texts.append(
            _Text(tx + 10 * k, ty + 53 * k, self._details(tgt), "mono", 11.5, _LIGHT[::-1])
        )

        if is_triggered:
            bw, bh = 86 * k, 40 * k
            bx = tx - 6 * k - bw if left else tx + tw + 6 * k
            by = ty
            if bx < 0 or bx + bw > self._w:
                bx, by = tx, max(ty - bh - 4 * k, 0.0)
            self._badge(out, texts, bx, by, bw, bh, f"+{round(m.speed_kmh - limit)}")

    def _badge(
        self,
        out: np.ndarray,
        texts: list[_Text],
        x: float,
        y: float,
        w: float,
        h: float,
        top: str,
    ) -> None:
        k = self._k
        _blend_rect(out, x, y, x + w, y + h, self._alert, 1.0)
        white = _WHITE
        texts.append(_Text(x + w / 2, y + 17 * k, top, "bold", 17, white, "ms"))
        texts.append(
            _Text(x + w / 2 + 6 * k, y + 33 * k, "CAPTURED", "semibold", 11.5, white, "ms")
        )
        cv2.circle(
            out,
            (round(x + w / 2 - 29 * k), round(y + 29.5 * k)),
            max(2, round(3 * k)),
            white,
            -1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _details(tgt: RadarTarget) -> str:
        sign = "\u2212" if tgt.angle_deg < 0 else "+"
        return f"{tgt.distance_m} m · {sign}{abs(tgt.angle_deg)}° · SNR {tgt.snr}"

    def _speed_text(
        self, texts: list[_Text], x: float, y: float, speed: int, col: _Color, unit_col: _Color
    ) -> None:
        texts.append(_Text(x, y, str(speed), "bold", 30, col))
        w = self._fonts.width("bold", 30, str(speed))
        texts.append(_Text(x + w + 4 * self._k, y, "km/h", "semibold", 15, unit_col))

    # ------------------------------------------------------------------------ lite pillars

    def _ground_px(self, x: float, y: float, z: float = 0.0) -> tuple[np.ndarray, bool]:
        uv, front = self._camera.project(np.array([[x, y, z]]))
        return uv[0], bool(front[0])

    def _pillar_geometry(
        self, xy: tuple[float, float], height_m: float
    ) -> tuple[float, float, float, float, float] | None:
        base, ok = self._ground_px(*xy)
        top, ok_top = self._ground_px(*xy, height_m)
        if not (ok and ok_top) or not np.isfinite([*base, *top]).all():
            return None
        return float(base[0]), float(base[1]), float(top[1]), *self._depth_info(xy)

    def _depth_info(self, xy: tuple[float, float]) -> tuple[float, float]:
        cam = self._camera
        depth = float((np.array([xy[0], xy[1], 0.0]) - cam.center) @ cam.R[2])
        return depth, float(cam.K[0, 0])

    def _thin_pillar(self, out: np.ndarray, tg: RadarTarget) -> None:
        xy = self._radar.target_to_road(tg.angle_deg, tg.distance_m)
        g = self._pillar_geometry(xy, self._lite.pillar_height_m)
        if g is None:
            return
        u, v, vtop, _depth, _f = g
        col = self._color_for("coming" if tg.approaching else "going", True)
        w = self._px(3)
        _blend_rect(out, u - w / 2, vtop, u + w / 2, v, col, 0.55)

    def _draw_pillar(
        self,
        out: np.ndarray,
        texts: list[_Text],
        sj: RadarSubject,
        is_triggered: bool,
        limit_for: Callable[[Direction], float],
    ) -> None:
        k = self._k
        m = sj.last_match
        g = self._pillar_geometry(m.road_xy, self._lite.pillar_height_m)
        if g is None:
            return
        u, v, vtop, depth, f = g
        col = self._color_for(sj.direction, True)
        wid = max(6 * k, f * m.target.distance_m * math.tan(math.radians(1)) / depth)
        x, y = m.road_xy
        left, _ = self._ground_px(x, y + 0.5)
        right, _ = self._ground_px(x, y - 0.5)
        near, _ = self._ground_px(x - 0.5, y)
        far, _ = self._ground_px(x + 0.5, y)
        rx = max(abs(float(left[0] - right[0])) / 2 * 1.2, wid)
        ry = max(3 * k, abs(float(near[1] - far[1])) / 2)

        def ellipse(mask: np.ndarray, o: tuple[int, int]) -> None:
            cv2.ellipse(mask, (round(u - o[0]), round(v - o[1])), (round(rx), round(ry)),
                        0, 0, 360, 255, -1, cv2.LINE_AA)  # fmt: skip

        _blend_mask(out, (u - rx - 2, v - ry - 2, u + rx + 2, v + ry + 2), col, 0.35, ellipse)
        cv2.ellipse(out, (round(u), round(v)), (round(rx), round(ry)), 0, 0, 360, col,
                    self._px(2), cv2.LINE_AA)  # fmt: skip
        _blend_rect(out, u - wid / 2, vtop, u + wid / 2, v, col, 0.55)
        _blend_rect(out, u - 1.5 * k, vtop, u + 1.5 * k, v, _WHITE, 0.9)
        _blend_rect(
            out, u - wid / 2 - 3 * k, vtop - 3 * k, u + wid / 2 + 3 * k, vtop + 2 * k, col, 1.0
        )

        lw, lh, strip = 160 * k, 40 * k, 22 * k
        lx, ly = u - lw / 2, vtop - 72 * k
        _blend_rect(out, lx, ly, lx + lw, ly + lh, col, 1.0)
        _blend_rect(out, lx, ly + lh, lx + lw, ly + lh + strip, _INK[::-1], _PANEL_ALPHA)
        limit = limit_for(sj.direction)
        speed, over = self._speed_parts(m, limit)
        self._speed_text(texts, lx + 10 * k, ly + 29 * k, speed,
                         _WHITE if over else _INK[::-1], _INK[::-1])  # fmt: skip
        _triangle(out, lx + lw - 14 * k, ly + 20 * k, 12 * k, sj.direction == "coming", _INK[::-1])
        texts.append(
            _Text(
                lx + 10 * k, ly + lh + 15.5 * k, self._details(m.target), "mono", 11.5, _LIGHT[::-1]
            )
        )
        if is_triggered:
            bw, bh = 86 * k, 40 * k
            bx = lx + lw + 6 * k
            if bx + bw > self._w:
                bx = lx - 6 * k - bw
            self._badge(out, texts, bx, ly, bw, bh, f"+{round(m.speed_kmh - limit)}")

    # ------------------------------------------------------------------------------- HUD

    def _draw_hud(self, out: np.ndarray, texts: list[_Text], hud: HudInfo) -> None:
        k, w, h = self._k, self._w, self._h
        _blend_rect(out, 0, 0, w, 44 * k, _INK[::-1], _BAR_ALPHA)
        _blend_rect(out, 0, h - 40 * k, w, h, _INK[::-1], _BAR_ALPHA)
        light = _LIGHT[::-1]
        grey = self._col["none"]
        head = (
            f"{hud.session_id} / seg {hud.segment:04d} · frame {hud.index} · "
            f"{_wall_clock(hud.wall_time_ns)}"
        )
        texts.append(_Text(16 * k, 29 * k, head, "mono", 15, light))

        legend = [(w - 520 * k, "coming", "COMING", False), (w - 410 * k, "going", "GOING", False)]
        if hud.mode == "full":
            legend.append((w - 305 * k, "none", "NO RADAR", True))
        for x, key, label, dashed in legend:
            col = self._col[key]
            box = np.array([(x, 14 * k), (x + 20 * k, 14 * k), (x + 20 * k, 30 * k), (x, 30 * k)])
            if dashed:
                _dashed(out, np.vstack([box, box[:1]]), col, self._px(3), (4 * k, 3 * k))
            else:
                cv2.polylines(
                    out, [np.round(box).astype(np.int32)], True, col, self._px(3), cv2.LINE_AA
                )
            texts.append(_Text(x + 28 * k, 28 * k, label, "semibold", 17, light))

        limit_text = f"LIMIT {hud.limit_kmh:g} km/h"
        pill_w = 32 * k + self._fonts.width("bold", 18, limit_text) + 14 * k
        px = w - 12 * k - pill_w
        self._pill(out, px, 7 * k, pill_w, 30 * k)
        cv2.circle(out, (round(px + 15 * k), round(22 * k)), round(11 * k), self._alert,
                   self._px(3), cv2.LINE_AA)  # fmt: skip
        texts.append(_Text(px + 32 * k, 29 * k, limit_text, "bold", 18, _INK[::-1]))

        age = (
            "no report yet"
            if hud.radar_age_s is None
            else f"last frame {hud.radar_age_s:.2f} s ago"
        )
        radar = (
            f"RADAR {hud.radar_targets} targets · {age} · "
            f"clock offset {hud.clock_offset_ms:+.0f} ms"
        )
        texts.append(_Text(16 * k, h - 14 * k, radar, "mono", 14, grey))
        if hud.mode == "full":
            mode = f"FULL · {hud.model or '-'} · {hud.tracker or '-'} · {hud.fps:.1f} fps"
        else:
            mode = "LITE · radar only"
        texts.append(_Text(w - 16 * k, h - 14 * k, mode, "mono", 14, grey, "rs"))

    @staticmethod
    def _pill(out: np.ndarray, x: float, y: float, w: float, h: float) -> None:
        r = h / 2
        pts = [
            (round(x + r), round(y + r)),
            (round(x + w - r), round(y + r)),
        ]
        cv2.line(out, pts[0], pts[1], _WHITE, round(h), cv2.LINE_AA)
        cv2.circle(out, pts[0], round(r), _WHITE, -1, cv2.LINE_AA)
        cv2.circle(out, pts[1], round(r), _WHITE, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------------------ text

    def _draw_text(self, out: np.ndarray, texts: list[_Text]) -> np.ndarray:
        if not texts:
            return out
        # The frame is BGR; colors are already BGR, so treat the buffer as "RGB" for Pillow.
        pil = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        for t in texts:
            draw.text(
                (t.x, t.y),
                t.text,
                font=self._fonts.get(t.font, t.size),
                fill=t.color,
                anchor=t.anchor,
            )
        return np.asarray(pil)
