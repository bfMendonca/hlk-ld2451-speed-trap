# Radar Side Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the radar cone from the camera image and add a right-hand strip to the overlay video with a top-down fan view (fading 3.5 s trail) and a table of the current radar readings.

**Architecture:** A new `RadarPanel` class (`speed_trap/render/radar_panel.py`) keeps its own reading history, fed with each frame's `scans`, and draws an `H x panel_width` image. The pipeline stacks it to the right of the rendered frame after the capture sink has received the frame, and opens `VideoSink` with the wider size. The `Renderer` is untouched except for the wedge default.

**Tech Stack:** Python, numpy, OpenCV, Pillow (already used by `render/draw.py`), pydantic config, pytest (`uv run pytest`).

**Spec:** `docs/superpowers/specs/2026-10-25-radar-panel-design.md`

## Global Constraints

- Panel height equals frame height; panel width = `round(frame_width * width_frac / 2) * 2` (even); total output width `frame_width + panel_width` stays even (yuv420p).
- Defaults under `overlay.radar_panel`: `enabled: true`, `width_frac: 0.333`, `trail_s: 3.5`, `min_speed_kmh: 15`, `max_range_m: 50`, `fov_half_deg: 45`, `max_rows: 6`.
- `overlay.show.wedge` default becomes `false`; the option stays.
- Table "current" window = readings from the last 0.3 s of video time, nearest ground range first, at most `max_rows`.
- Ground range = `sqrt(max(distance_m**2 - radar.height_m**2, 0))`; the table prints the radar's own `distance_m`.
- Handedness follows the video: screen x = `cx - radar.angle_sign * ground_range * sin(angle)` so negative angles (`angle_sign: 1`) are on the right.
- The panel never uses camera pose, intrinsics or boresight; it only reads `RadarModel.height_m` and `RadarModel.angle_sign`.
- Fading uses video time (`ts_ns` of frames/scans), never wall-clock.
- Capture images and crops stay panel-free; no change to detection, tracking, fusion, `tracks.jsonl`.
- Reuse `_Fonts`, `_Text`, `_hex_to_bgr` and the colours from `speed_trap/render/draw.py`.
- `README.md`, `CHANGELOG.md`, `speed_trap/cli.py`, `tests/test_cli.py`, `tests/test_e2e.py` already have uncommitted user edits: append, never overwrite; stage only the files a task names; do not commit without the user's go-ahead.

## Review Focus

- Frame with no scans at all: panel is drawn (grid only), no exception.
- Reading with `distance_m` smaller than the radar height: ground range 0, no NaN, still drawn at the origin.
- Readings outside the fan (angle beyond `fov_half_deg`, range beyond `max_range_m`): neither drawn nor listed.
- More simultaneous readings than `max_rows`: nearest kept, none overlap beyond the cap.
- Timestamp jump (segment gap) or a reading newer than "now": old readings expire, future ones are ignored.
- Odd frame width or tiny `width_frac`: panel width still even and at least 2 px; total width even.

---

### Task 1: Config and wedge default

**Files:**
- Modify: `speed_trap/config.py` (`OverlayShowConfig.wedge`, new `RadarPanelConfig`, `OverlayConfig.radar_panel`)
- Modify: `speed_trap/default_config.yaml` (overlay section)
- Test: `tests/test_config.py`, `tests/test_render.py::test_wedge_changes_ground_not_sky`

**Interfaces:**
- Produces: `RadarPanelConfig(enabled: bool = True, width_frac: float = 0.333, trail_s: float = 3.5, min_speed_kmh: float = 15.0, max_range_m: float = 50.0, fov_half_deg: float = 45.0, max_rows: int = 6)`; `OverlayConfig.radar_panel: RadarPanelConfig`; `OverlayShowConfig.wedge` default `False`.

- [ ] **Step 1: Write the failing tests** in `tests/test_config.py`: `test_radar_panel_defaults` (all seven defaults above, `OverlayConfig().show.wedge is False`), `test_radar_panel_rejects_bad_values` (`width_frac` 0 and 1.5, `trail_s` 0, `max_rows` 0, `fov_half_deg` 100 each raise `ValidationError`; unknown key raises, as for other sections).
- [ ] **Step 2: Run** `uv run pytest tests/test_config.py -k radar_panel -v` — Expected: FAIL (`radar_panel` missing).
- [ ] **Step 3: Implement** `RadarPanelConfig` (strict model like its neighbours; `width_frac` in `(0, 1]`, `trail_s > 0`, `min_speed_kmh >= 0`, `max_range_m > 0`, `fov_half_deg` in `(0, 90]`, `max_rows >= 1`), wire it into `OverlayConfig`, flip the wedge default, and mirror the new defaults with comments in `default_config.yaml`. In `tests/test_render.py` make `test_wedge_changes_ground_not_sky` pass `{"show": {"wedge": True}}` explicitly.
- [ ] **Step 4: Run** `uv run pytest tests/test_config.py tests/test_render.py -v` — Expected: PASS. Also run `uv run pytest tests/test_cli.py -q` (a test may compare against the default config file) — Expected: PASS.
- [ ] **Step 5: Commit** `speed_trap/config.py speed_trap/default_config.yaml tests/test_config.py tests/test_render.py` as `feat: radar panel config, wedge off by default` (only with the user's go-ahead).

### Task 2: Panel logic (history, filter, fade, mapping, table)

**Files:**
- Create: `speed_trap/render/radar_panel.py`
- Test: `tests/test_radar_panel.py`

**Interfaces:**
- Consumes: `RadarPanelConfig` (Task 1), `RadarModel.height_m`/`angle_sign`, `RadarScan`/`RadarTarget` from `speed_trap/types.py`.
- Produces, all in `radar_panel.py`:
  - `class Reading` (frozen dataclass: `t_ns: int`, `target: RadarTarget`, `ground_m: float`).
  - `class PanelState(radar: RadarModel, cfg: RadarPanelConfig)` with `update(ts_ns: int, scans: Sequence[RadarScan]) -> None`, `trail(ts_ns: int) -> list[tuple[Reading, float]]` (reading, age in seconds, oldest first), `table(ts_ns: int) -> list[Reading]` (current window, nearest first, capped).
  - `fade(age_s: float, trail_s: float) -> float` (1.0 at age 0, linear to a floor of 0.15 at `trail_s`).
  - `fan_xy(ground_m: float, angle_deg: float, angle_sign: int, cx: float, cy: float, scale: float) -> tuple[float, float]` (mirrored x as in Global Constraints; y = `cy - ground_m * cos(angle) * scale`).

- [ ] **Step 1: Write the failing tests** in `tests/test_radar_panel.py` with a `RadarModel` built like `tests/test_render.py::make_renderer` (height `3.5 + radar z offset` is fine; read `radar.height_m`): `test_filters_slow_readings` (3 km/h dropped, 15 kept), `test_filters_outside_fan_and_range` (angle 46 and ground range 51 dropped), `test_ground_range_never_nan` (`distance_m` below `radar.height_m` gives `ground_m == 0.0`), `test_trail_expires_and_ignores_future` (reading at 0 s: age 3.4 s kept, 3.6 s gone; a reading stamped after `ts_ns` is ignored), `test_table_window_order_and_cap` (7 readings in the last 0.3 s with distinct ranges → 6 rows, nearest first; a 0.4 s old reading excluded), `test_fade_values` (1.0, ~0.5-ish monotone, floor 0.15, never below), `test_fan_xy_matches_video_handedness` (angle -10 with `angle_sign=1` has x > cx; with `angle_sign=-1` x < cx; angle 0 → x == cx).
- [ ] **Step 2: Run** `uv run pytest tests/test_radar_panel.py -v` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** the names above in `radar_panel.py`. `update` stores readings that pass the filters and prunes anything older than `trail_s` (plus a small margin) relative to `ts_ns`; history is a `collections.deque`.
- [ ] **Step 4: Run** `uv run pytest tests/test_radar_panel.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `speed_trap/render/radar_panel.py tests/test_radar_panel.py` as `feat: radar panel history and layout logic`.

### Task 3: Panel drawing

**Files:**
- Modify: `speed_trap/render/radar_panel.py`
- Test: `tests/test_radar_panel.py`

**Interfaces:**
- Consumes: `PanelState`, `fade`, `fan_xy` (Task 2); `_Fonts`, `_Text`, `_hex_to_bgr`, `_REF_WIDTH` and the colour constants from `speed_trap/render/draw.py`; `OverlayColors` for coming/going colours.
- Produces: `class RadarPanel(radar: RadarModel, overlay: OverlayConfig, frame_width: int, frame_height: int)` with property `width_px: int` and `draw(ts_ns: int, scans: Sequence[RadarScan]) -> np.ndarray` (BGR uint8, shape `(frame_height, width_px, 3)`); it calls `PanelState.update` itself.

- [ ] **Step 1: Write the failing tests**: `test_width_is_even_and_at_least_two` (frame widths 1296, 1295, 2 with `width_frac` 0.333 and 0.0001 → even, `>= 2`, and `frame_width + width_px` even when `frame_width` is even), `test_draw_shape_and_dtype`, `test_empty_scan_draws_grid_only` (no exception; some non-background pixels from the grid), `test_dot_color_by_direction` (one approaching reading at `angle_sign=1` paints a pixel near `_hex_to_bgr(colors.coming)` at `fan_xy(...)` position; going uses the `going` colour), `test_readings_outside_fan_leave_panel_unchanged` (draw with only a 3 km/h reading equals draw with none), `test_table_text_changes_with_readings` (frames differ when a 42 km/h reading is present in the table area).
- [ ] **Step 2: Run** `uv run pytest tests/test_radar_panel.py -v` — Expected: new tests FAIL.
- [ ] **Step 3: Implement `RadarPanel.draw`**: top part (~62% of height) fan with range arcs at 10/20/40 m (≤ `max_range_m`), angle lines every 15°, radar marker at the bottom centre, trail dots with `fade` opacity (newest dot larger), direction colours from `overlay.colors`; bottom part table with header `km/h  dist  ang  snr  dir` and one row per `PanelState.table` reading, using the mono font through one Pillow text pass as `Renderer._draw_text` does. Sizes scale with `frame_width / _REF_WIDTH`.
- [ ] **Step 4: Run** `uv run pytest tests/test_radar_panel.py -v` — Expected: PASS.
- [ ] **Step 5: Render a sample for a human check:** run a short Python snippet from the scratchpad (not committed) that draws the panel with the s000064 readings from t = 15.0 s and save a PNG; open it and compare with `.superpowers/brainstorm/radar-strip-mockup.html` option B (mirrored: car on the right).
- [ ] **Step 6: Commit** the module and tests as `feat: radar panel drawing`.

### Task 4: Pipeline integration

**Files:**
- Modify: `speed_trap/pipeline.py` (constructor near line 64, `_run` near line 116, `_step` near lines 189-203)
- Test: `tests/test_e2e.py`, `tests/test_sinks.py` if it constructs the pipeline output size

**Interfaces:**
- Consumes: `RadarPanel` (Task 3), `cfg.overlay.radar_panel.enabled`.
- Produces: output video of size `(src.height, src.width + panel.width_px)` when enabled, unchanged size when disabled.

- [ ] **Step 1: Write the failing test** in `tests/test_e2e.py` (follow how that file already runs a synthetic session): `test_overlay_video_includes_radar_panel` reads the output video and asserts width `== frame_width + RadarPanel(...).width_px` with the panel enabled, and `== frame_width` with `overlay.radar_panel.enabled: false`; `test_captures_stay_panel_free` asserts a capture crop's width is not larger than the frame width and that the saved full frame (when `save_full_frame`) has the frame width.
- [ ] **Step 2: Run** `uv run pytest tests/test_e2e.py -k "panel or captures_stay" -v` — Expected: FAIL.
- [ ] **Step 3: Implement**: build `self._panel: RadarPanel | None` in the constructor from `self._radar`, `cfg.overlay`, `w`, `h` (only when enabled); open `VideoSink` with `src.width + (self._panel.width_px if self._panel else 0)`; in `_step`, after `self._captures.offer(...)` and before `self._video.write`, compute `panel = self._panel.draw(b.ts_ns, b.scans)` and `rendered = np.hstack((rendered, panel))`. Call `draw` for every frame so the history sees every scan.
- [ ] **Step 4: Run** `uv run pytest -q` — Expected: PASS (full suite; fix any test that assumed the old video width).
- [ ] **Step 5: Commit** `speed_trap/pipeline.py tests/test_e2e.py` (plus any test file the width change touched) as `feat: add radar panel to the overlay video`.

### Task 5: Docs and real-session check

**Files:**
- Modify (append): `README.md`, `CHANGELOG.md`

- [ ] **Step 1: Document** the panel (what it shows, the `overlay.radar_panel` options, wedge now off by default, output width grows by `width_frac`) in README's overlay section and a CHANGELOG entry under the unreleased heading, appending to the existing uncommitted text.
- [ ] **Step 2: Run on a real session:** `uv run speed-trap process data/s000064 --config data/cfg.yaml --device cuda --capture --out out/s000064` and extract frames at 14.3 s and 15.0 s with ffmpeg; Expected: video width 1296 + panel, car-A trail visible on the right of the panel, no cone on the image.
- [ ] **Step 3: Run** `uv run pytest -q` and `uv run ruff check .` if the repo uses ruff (check `pyproject.toml`) — Expected: clean.
- [ ] **Step 4: Commit** `README.md CHANGELOG.md` as `docs: radar panel` (only with the user's go-ahead; the files hold other uncommitted edits, so stage hunks deliberately).
