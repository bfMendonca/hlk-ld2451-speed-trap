"""Radar side panel: a top-down fan of the radar's own readings plus a table of the current ones.

Radar only: no camera pose, intrinsics or boresight is used, and nothing depends on fusion. Fading
follows video time (frame and scan timestamps), so reprocessing a session gives the same video.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from speed_trap.config import OverlayConfig, RadarPanelConfig
from speed_trap.geometry import RadarModel
from speed_trap.render.draw import _REF_WIDTH, _Fonts, _hex_to_bgr, _Text
from speed_trap.types import RadarScan, RadarTarget

_NS = 1_000_000_000
_TABLE_HOLD_S = 0.3  # readings this recent count as "current"
_FADE_FLOOR = 0.15


@dataclass(frozen=True)
class Reading:
    t_ns: int
    target: RadarTarget
    ground_m: float


def fade(age_s: float, trail_s: float) -> float:
    """Opacity of a reading of the given age: 1 when new, linear down to a floor at ``trail_s``."""
    return max(_FADE_FLOOR, 1.0 - (1.0 - _FADE_FLOOR) * age_s / trail_s)


def fan_xy(
    ground_m: float, angle_deg: float, angle_sign: int, cx: float, cy: float, scale: float
) -> tuple[float, float]:
    """Panel pixel of a reading; x is mirrored so negative angles land on the right."""
    a = math.radians(angle_deg)
    return cx - angle_sign * ground_m * math.sin(a) * scale, cy - ground_m * math.cos(a) * scale


class PanelState:
    def __init__(self, radar: RadarModel, cfg: RadarPanelConfig) -> None:
        self._radar = radar
        self._cfg = cfg
        self._history: deque[Reading] = deque()

    def _ground(self, target: RadarTarget) -> float:
        return math.sqrt(max(target.distance_m**2 - self._radar.height_m**2, 0.0))

    def _shown(self, target: RadarTarget, ground_m: float) -> bool:
        c = self._cfg
        return (
            target.speed_kmh >= c.min_speed_kmh
            and abs(target.angle_deg) <= c.fov_half_deg
            and ground_m <= c.max_range_m
        )

    def update(self, ts_ns: int, scans: Sequence[RadarScan]) -> None:
        """Add the readings of ``scans`` that pass the filters and drop the expired ones."""
        for sc in scans:
            for target in sc.targets:
                ground = self._ground(target)
                if self._shown(target, ground):
                    self._history.append(Reading(sc.t_ns, target, ground))
        horizon = ts_ns - round(self._cfg.trail_s * _NS)
        while self._history and self._history[0].t_ns < horizon:
            self._history.popleft()

    def trail(self, ts_ns: int) -> list[tuple[Reading, float]]:
        """(reading, age in s) for every reading no newer than ``ts_ns``, oldest first."""
        horizon = ts_ns - round(self._cfg.trail_s * _NS)
        return [(r, (ts_ns - r.t_ns) / _NS) for r in self._history if horizon <= r.t_ns <= ts_ns]

    def table(self, ts_ns: int) -> list[Reading]:
        """Readings of the newest scan within 0.3 s, nearest first, capped at ``max_rows``.

        Only one scan: every scan repeats the same vehicles, and listing several would fill the
        table with duplicates."""
        since = ts_ns - round(_TABLE_HOLD_S * _NS)
        recent = [r for r in self._history if since <= r.t_ns <= ts_ns]
        if not recent:
            return []
        newest = max(r.t_ns for r in recent)
        rows = sorted((r for r in recent if r.t_ns == newest), key=lambda r: r.ground_m)
        return rows[: self._cfg.max_rows]


# ------------------------------------------------------------------------------------ drawing

_BG = (0x13, 0x0F, 0x0C)  # BGR
_GRID = (0x3C, 0x32, 0x2A)
_AXIS = (0x4D, 0x40, 0x34)
_LABEL = (0x86, 0x78, 0x6C)
_TEXT = (0xDD, 0xD6, 0xCF)
_RINGS_M = (10.0, 20.0, 40.0)
_MARGIN = 14  # reference px around the fan


def _blend(color: tuple[int, int, int], opacity: float) -> tuple[int, int, int]:
    b, g, r = (round(c * opacity + bg * (1 - opacity)) for c, bg in zip(color, _BG, strict=True))
    return b, g, r


class RadarPanel:
    """Draws the strip: fan view of the trail on top, table of the current readings below."""

    def __init__(
        self, radar: RadarModel, overlay: OverlayConfig, frame_width: int, frame_height: int
    ) -> None:
        cfg = overlay.radar_panel
        self._radar = radar
        self._cfg = cfg
        self._h = frame_height
        self._w = max(2, round(frame_width * cfg.width_frac / 2) * 2)
        self._k = frame_width / _REF_WIDTH
        self._fonts = _Fonts(self._k)
        self._state = PanelState(radar, cfg)
        c = overlay.colors
        self._col = {True: _hex_to_bgr(c.coming), False: _hex_to_bgr(c.going)}
        margin = _MARGIN * self._k
        side = math.sin(math.radians(cfg.fov_half_deg))
        self._scale = max((self._w / 2 - margin) / (cfg.max_range_m * side), 0.01)
        self._cx = self._w / 2
        self._cy = margin + cfg.max_range_m * self._scale + 18 * self._k
        self._table_top = min(round(self._cy + margin + 8 * self._k), self._h)

    @property
    def width_px(self) -> int:
        return self._w

    @property
    def table_top_px(self) -> int:
        return self._table_top

    def draw(self, ts_ns: int, scans: Sequence[RadarScan]) -> np.ndarray:
        self._state.update(ts_ns, scans)
        img = np.empty((self._h, self._w, 3), np.uint8)
        img[:] = _BG
        texts: list[_Text] = []
        self._draw_grid(img, texts)
        self._draw_trail(img, ts_ns)
        self._draw_table(img, texts, ts_ns)
        return self._text_pass(img, texts)

    def _draw_grid(self, img: np.ndarray, texts: list[_Text]) -> None:
        cfg, k, s = self._cfg, self._k, self._scale
        cx, cy, fov = self._cx, self._cy, cfg.fov_half_deg
        sign = self._radar.angle_sign
        for r in _RINGS_M:
            if r > cfg.max_range_m:
                continue
            pts = [fan_xy(r, a, sign, cx, cy, s) for a in np.linspace(-fov, fov, 40)]
            arr = np.round(np.array(pts) * 16).astype(np.int32)
            cv2.polylines(img, [arr], False, _GRID, max(1, round(k)), cv2.LINE_AA, shift=4)
            texts.append(_Text(cx + 4 * k, cy - r * s - 3 * k, f"{r:g} m", "mono", 11, _LABEL))
        for a in range(-45, 46, 15):
            if abs(a) > fov:
                continue
            x, y = fan_xy(cfg.max_range_m, a, sign, cx, cy, s)
            color = _AXIS if a == 0 else _GRID
            line = (round(cx), round(cy)), (round(x), round(y))
            cv2.line(img, *line, color, max(1, round(k)), cv2.LINE_AA)
        cv2.circle(img, (round(cx), round(cy)), max(3, round(5 * k)), _TEXT, -1, cv2.LINE_AA)

    def _draw_trail(self, img: np.ndarray, ts_ns: int) -> None:
        k, s, trail = self._k, self._scale, self._cfg.trail_s
        sign = self._radar.angle_sign
        for r, age in self._state.trail(ts_ns):  # oldest first, so new dots end up on top
            x, y = fan_xy(r.ground_m, r.target.angle_deg, sign, self._cx, self._cy, s)
            color = _blend(self._col[r.target.approaching], fade(age, trail))
            radius = max(2, round((5 if age <= _TABLE_HOLD_S else 3) * k))
            cv2.circle(img, (round(x), round(y)), radius, color, -1, cv2.LINE_AA)

    def _draw_table(self, img: np.ndarray, texts: list[_Text], ts_ns: int) -> None:
        k = self._k
        y = self._table_top + 22 * k
        x_dot, x_v, x_d, x_a, x_snr = 12 * k, 80 * k, 150 * k, 215 * k, 275 * k
        for label, x in (("km/h", x_v), ("dist", x_d), ("ang", x_a), ("snr", x_snr)):
            texts.append(_Text(x, y, label, "mono", 13, _LABEL, "rs"))
        texts.append(_Text(x_snr + 12 * k, y, "dir", "mono", 13, _LABEL, "ls"))
        for r in self._state.table(ts_ns):
            y += 24 * k
            if y > self._h:
                break
            t = r.target
            dot = (round(x_dot), round(y - 5 * k))
            cv2.circle(img, dot, max(3, round(5 * k)), self._col[t.approaching], -1, cv2.LINE_AA)
            texts.append(_Text(x_v, y, f"{t.speed_kmh}", "mono", 15, _TEXT, "rs"))
            texts.append(_Text(x_d, y, f"{t.distance_m} m", "mono", 15, _TEXT, "rs"))
            texts.append(_Text(x_a, y, f"{t.angle_deg}\u00b0", "mono", 15, _TEXT, "rs"))
            texts.append(_Text(x_snr, y, f"{t.snr}", "mono", 15, _TEXT, "rs"))
            word = "aprox." if t.approaching else "afasta"
            texts.append(_Text(x_snr + 12 * k, y, word, "mono", 15, self._col[t.approaching], "ls"))

    def _text_pass(self, img: np.ndarray, texts: list[_Text]) -> np.ndarray:
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        for t in texts:
            font = self._fonts.get(t.font, t.size)
            draw.text((t.x, t.y), t.text, font=font, fill=t.color, anchor=t.anchor)
        return np.asarray(pil)
