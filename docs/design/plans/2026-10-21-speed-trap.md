# speed-trap v1 Implementation Plan

> **Historical design record.** Written before implementation; the code is the source of truth. Since then the `ld2451` and `radar-cam-recorder` dependencies moved from sibling checkouts to their GitHub repositories, and later review rounds added safeguards (scan-time matching, speed-consistency gate, verified matches) described in the README.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An offline CLI that turns a `radar-cam-recorder` session into an annotated overlay video, a per-vehicle track log and optional speeder captures, by fusing a vehicle detector + tracker with HLK-LD2451 radar readings.

**Architecture:** A frame-by-frame pipeline `Source → Detector → Tracker → Fusion → SpeedingMonitor → Renderer → Sinks`. The core (types, geometry, tracker, fusion, monitor) does no file I/O, decoding or drawing, and never reads ahead in time, so a live source can replace the recorded one later. Lite mode is the same pipeline with the detector and tracker off, and a radar-target linker in place of fusion.

**Tech Stack:** Python ≥ 3.10 (dev pin 3.12), uv, pydantic v2, PyYAML, numpy, scipy, OpenCV, Pillow, PyAV, supervision (ByteTrack); optional extras `rfdetr` (default detector) and `yolo` (ultralytics); pytest, ruff, mypy strict.

**Spec:** `docs/design/specs/2026-10-21-speed-trap-design.md` (read it before starting any task).

## Global Constraints

- `requires-python = ">=3.10"`; `.python-version` = `3.12`.
- Package `speed_trap` (flat layout like the sibling repos), console script `speed-trap`.
- Licence MIT (`LICENSE`, `pyproject.toml`). README states that the `yolo` extra brings AGPL-3.0 terms.
- Path dependencies, editable: `ld2451 = { path = "../driver-hlk-ld2451", editable = true }`, `radar-cam-recorder = { path = "../radar-cam-recorder", editable = true }`.
- Base install must not import torch, rfdetr or ultralytics. Detector backends import their library lazily inside the backend.
- The YAML config is always required for `process` and `check-config`. Unknown keys and out-of-range values are errors that name the dotted key.
- All times are boot-clock nanoseconds (`int`). Never assume a constant frame rate.
- Colors: coming `#35d0ba`, going `#ffb547`, no radar `#a7afb8`, alert `#ff4d5e`.
- Fusion and captures only within `fusion.max_range_m` (default 30.0 m).
- A capture must never overstate speed: cosine correction defaults to off; ambiguous matches are dropped.
- ruff (`E,F,W,I,B,UP,SIM,RUF`, line length 100), mypy strict on `speed_trap` and `tests`.
- Commit after each task with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Video shorter or longer than its timestamps** (killed segment: `.h264` frame count ≠ `frames.csv` rows): process up to the shorter, log one warning per segment, never crash. Test in Task 4.
2. **Session without radar data** (radar dir missing, or radar disabled): process the video, every track gray, `run.json` counts zero scans, one warning. Test in Task 4 and Task 11.
3. **Box whose bottom is at or above the horizon, or a radar target behind the camera / outside the image:** back-projection returns `None`, the track simply can't match, drawing skips off-image points. Test in Task 2 and Task 7.
4. **Intrinsics calibrated at another resolution** (e.g. 2592×1944 while the video is 1296×972): the matrix is scaled; aspect-ratio mismatch is a config error. Test in Task 2.
5. **`--start`/`--end` outside the session, or `--segment` that doesn't exist:** clear error message naming the valid range, exit code 2, no output folder created. Test in Task 11.

---

## File Structure

```
pyproject.toml, .python-version, LICENSE, README.md
speed_trap/
  __init__.py            version
  config.py              pydantic schema, load_config, overrides, init-config text
  default_config.yaml    commented default config (package data)
  types.py               core dataclasses shared by all stages
  geometry.py            CameraModel, RadarModel, wedge
  synth.py               synthetic session generator + ground truth
  source.py              RecordedSessionSource → FrameBundle stream
  detect/__init__.py     Detector protocol, make_detector()
  detect/truth.py        ground-truth detector (tests/demos)
  detect/rfdetr.py       RF-DETR backend
  detect/yolo.py         Ultralytics backend
  tracker.py             ByteTrack wrapper with run-unique ids
  fusion.py              radar ↔ track association, reliability, speed
  lite.py                radar-target linker for lite mode
  monitor.py             speeding trigger + best-frame candidates
  render/__init__.py     Renderer
  render/draw.py         primitives (alpha fill, AA lines, Pillow text)
  render/fonts/          BarlowSemiCondensed-SemiBold.ttf, BarlowSemiCondensed-Bold.ttf,
                         IBMPlexMono-Regular.ttf, OFL.txt
  sinks.py               VideoSink, TracksSink, CaptureSink, RunInfo
  pipeline.py            Pipeline.run
  cli.py                 argparse entry point
tests/ (one test file per module, plus test_e2e.py)
```

---

### Task 1: Project scaffold and configuration

**Files:**
- Create: `pyproject.toml`, `.python-version`, `LICENSE`, `speed_trap/__init__.py`, `speed_trap/config.py`, `speed_trap/default_config.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `class Config(BaseModel)` with sections `camera, radar, road, detector, tracker, fusion, speed, capture, overlay, lite, output, synth` (all `extra="forbid"`), keys and defaults exactly as spec §6.2 plus the additions below.
  - `load_config(path: Path, overrides: Sequence[str] = ()) -> Config`
  - `apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]` (`"a.b=v"`, value parsed with `yaml.safe_load`)
  - `default_config_text() -> str` (contents of `default_config.yaml`)
  - `config_hash(cfg: Config) -> str` (sha256 of `cfg.model_dump_json()` with sorted keys)
  - `class ConfigError(ValueError)`

Additions to the §6.2 sketch (all also in `default_config.yaml`):
- `radar.approaching_byte: int = 0` (0 or 1), `radar.report_hold_s: float = 0.3`.
- `camera.intrinsics: Intrinsics | None = None`, `Intrinsics(resolution: tuple[int,int], matrix: 3×3, distortion: list[float])` (length 4, 5, 8, 12 or 14).
- `tracker.minimum_consecutive_frames: int = 1`.
- `fusion.direction_min_travel_m: float = 0.5`.
- `capture.edge_margin_px: int = 2`.
- `lite.lost_track_s: float = 1.5`.
- `synth.report_interval_s: float = 0.1`, `synth.drop_probability: float = 0.05`, `synth.segment_s: float = 10.0`, `synth.gap_s: float = 3.0`, `synth.fps: float = 25.0`, `synth.resolution: [1296, 972]`.
- `overlay.style: Literal["tag"]` (only style B exists in v1).
- `output.dir: Path = Path("out")`.

- [ ] **Step 1: Write `pyproject.toml`** (setuptools backend like `../radar-cam-recorder/pyproject.toml`; dependencies `numpy>=1.26, scipy>=1.11, opencv-python>=4.9, pillow>=10, av>=12, pydantic>=2.6, pyyaml>=6, supervision>=0.25, ld2451, radar-cam-recorder`; `[project.optional-dependencies] rfdetr = ["rfdetr>=1.2"]`, `yolo = ["ultralytics>=8.3"]`; `[dependency-groups] dev = ["pytest>=8", "ruff>=0.5", "mypy>=1.10", "types-PyYAML"]`; `[tool.uv] default-groups = ["dev"]`; package data `default_config.yaml`, `render/fonts/*`, `py.typed`; ruff and mypy sections as Global Constraints). `LICENSE` MIT, copyright 2026 the repo owner.

- [ ] **Step 2: Verify the environment resolves**

Run: `uv sync && uv run python -c "import ld2451, radar_cam_recorder, supervision, av, cv2"`
Expected: exits 0.

- [ ] **Step 3: Write failing tests in `tests/test_config.py`**

```python
def test_default_file_loads_and_matches_model_defaults(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text(default_config_text())
    assert load_config(p) == Config()

def test_unknown_key_names_dotted_path(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text("fusion:\n  max_rang_m: 10\n")
    with pytest.raises(ConfigError, match=r"fusion\.max_rang_m"):
        load_config(p)

def test_override_sets_nested_value_and_validates(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text("{}\n")
    cfg = load_config(p, ["speed.limit_kmh=50", "capture.enabled=true"])
    assert cfg.speed.limit_kmh == 50 and cfg.capture.enabled is True
    with pytest.raises(ConfigError, match=r"speed\.limit_kmh"):
        load_config(p, ["speed.limit_kmh=-3"])

def test_override_without_equals_is_error(tmp_path): ...  # ConfigError mentioning "KEY=VALUE"

def test_approaching_byte_only_0_or_1(tmp_path): ...       # 2 → ConfigError "radar.approaching_byte"

def test_matrix_must_be_3x3(tmp_path): ...                  # 2x3 → ConfigError "camera.intrinsics.matrix"

def test_config_hash_stable_and_sensitive():
    assert config_hash(Config()) == config_hash(Config())
    assert config_hash(Config()) != config_hash(Config(speed={"limit_kmh": 41}))
```

- [ ] **Step 4: Run** `uv run pytest tests/test_config.py -v` — Expected: FAIL (import error).

- [ ] **Step 5: Implement `speed_trap/config.py` and `default_config.yaml`.** Every key in the YAML has an inline comment with unit and range, like `../radar-cam-recorder/deploy/config.example.toml`. Validation: `limit_kmh > 0`, gates and ranges `> 0`, `every_n_frames >= 1`, `ambiguity_ratio` in (0, 1], `range_m.min < range_m.max` when max given, `angle_sign in (1, -1)`. Convert pydantic `ValidationError` to `ConfigError` whose message lists `dotted.loc: msg` per error.

- [ ] **Step 6: Run** `uv run pytest tests/test_config.py -v && uv run ruff check . && uv run mypy` — Expected: all PASS, no lint or type errors.

- [ ] **Step 7: Commit** `feat: project scaffold and YAML configuration`

---

### Task 2: Core types and geometry

**Files:**
- Create: `speed_trap/types.py`, `speed_trap/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: `Config.camera`, `Config.radar` (Task 1).
- Produces (`types.py`, all `@dataclass(frozen=True)` unless noted):
  - `RadarTarget(t_ns: int, angle_deg: int, distance_m: int, approaching: bool, speed_kmh: int, snr: int, index: int)`
  - `RadarScan(t_ns: int, targets: tuple[RadarTarget, ...])`
  - `FrameBundle(image: np.ndarray, ts_ns: int, segment: int, index: int, wall_time_ns: int | None, scans: tuple[RadarScan, ...])` (image BGR uint8 H×W×3)
  - `Detection(xyxy: tuple[float, float, float, float], confidence: float, class_name: str)`
  - `TrackObservation(track_id: int, xyxy: tuple[float, float, float, float], confidence: float, class_name: str, detected: bool)`
  - `RadarMatch(t_ns: int, target: RadarTarget, road_xy: tuple[float, float], speed_kmh: float, speed_raw_kmh: int)`
  - `Direction = Literal["coming", "going"]`
- Produces (`geometry.py`). Road frame: X forward (camera heading at yaw 0), Y left, Z up, origin on the ground below the camera.
  - `class CameraModel`: `from_config(cfg: CameraConfig, width: int, height: int) -> CameraModel`; `project(points_road: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (N×3 → N×2 pixels and N bool "in front of camera"); `backproject_ground(u: float, v: float) -> tuple[float, float] | None`; attributes `K`, `dist`, `width`, `height`.
  - `nominal_matrix(width: int, height: int) -> np.ndarray` (OV5647 HFOV 53.5°, square pixels, principal point at centre).
  - `class RadarModel`: `from_config(cfg: RadarConfig, camera_height_m: float, max_range_default: float | None) -> RadarModel`; `target_to_road(angle_deg: int, distance_m: int) -> tuple[float, float]`; `road_to_polar(x: float, y: float) -> tuple[float, float]` (ground range m, azimuth deg relative to boresight, sign as reported, i.e. divided by `angle_sign`); `slant_range(ground_range: float) -> float`; attributes `position_xy`, `height_m`, `max_range_m`, `min_range_m`, `fov_half_deg`.
  - `wedge_polygon(radar: RadarModel, step_m: float = 1.0, step_deg: float = 1.0) -> np.ndarray` (M×3 road points outlining the sector) and `range_ring(radar: RadarModel, r: float, step_deg: float = 0.5) -> np.ndarray`.

Conventions pinned by tests: pitch positive = camera looks down; yaw positive = camera turns left; radar azimuth = `boresight.yaw_deg + angle_sign * angle_deg`, positive = left; radar position = `(offset.x, offset.y)`, height = `camera.pose.height_m + offset.z`; ground range = `sqrt(max(d² − h², 0))`.

- [ ] **Step 1: Write failing tests** (`K = [[1000,0,648],[0,1000,486],[0,0,1]]`, no distortion, height 3.5, all angles 0 unless stated):

```python
def test_point_ahead_projects_below_centre():
    uv, ok = cam.project(np.array([[10.0, 0.0, 0.0]]))
    assert ok[0] and uv[0] == pytest.approx([648, 486 + 1000 * 3.5 / 10])

def test_left_is_smaller_u():
    uv, _ = cam.project(np.array([[10.0, 2.0, 0.0]])); assert uv[0][0] == pytest.approx(448)

def test_pitch_down_moves_point_up():
    cam = make_cam(pitch_deg=10)
    uv, _ = cam.project(np.array([[10.0, 0, 0]]))
    assert uv[0][1] == pytest.approx(486 + 1000 * math.tan(math.atan(0.35) - math.radians(10)))

def test_backproject_roundtrip_with_yaw_pitch_roll_and_distortion():
    cam = make_cam(pitch_deg=12, yaw_deg=-5, roll_deg=1.5, dist=[-0.2, 0.05, 0, 0, 0])
    for x, y in [(8, 0), (25, -3), (30, 4)]:
        uv, _ = cam.project(np.array([[x, y, 0.0]]))
        assert cam.backproject_ground(*uv[0]) == pytest.approx((x, y), abs=1e-3)

def test_backproject_above_horizon_is_none():
    assert make_cam(pitch_deg=0).backproject_ground(648, 400) is None

def test_point_behind_camera_flagged():
    _, ok = cam.project(np.array([[-5.0, 0, 0]])); assert not ok[0]

def test_intrinsics_scaled_to_video_resolution():
    cam = CameraModel.from_config(cfg_with(resolution=(2592, 1944), fx=2540, cx=1296), 1296, 972)
    assert cam.K[0, 0] == pytest.approx(1270) and cam.K[0, 2] == pytest.approx(648)

def test_aspect_mismatch_is_config_error():  # resolution (1920,1080) for a 1296x972 video
    with pytest.raises(ConfigError, match="camera.intrinsics.resolution"): ...

def test_nominal_matrix():
    K = nominal_matrix(1296, 972)
    assert K[0, 0] == pytest.approx(648 / math.tan(math.radians(53.5 / 2)))

def test_radar_slant_to_ground():
    radar = make_radar(offset_z=-0.5)                  # height 3.0
    assert radar.target_to_road(0, 5) == pytest.approx((4.0, 0.0))

def test_radar_angle_sign_and_boresight():
    x, y = make_radar(boresight_yaw=-8, angle_sign=1).target_to_road(10, 20)
    assert math.degrees(math.atan2(y, x)) == pytest.approx(2.0, abs=1e-6)
    _, y2 = make_radar(angle_sign=-1).target_to_road(10, 20); assert y2 < 0

def test_road_to_polar_inverts_target_to_road():
    r = make_radar(boresight_yaw=-8, angle_sign=-1)
    x, y = r.target_to_road(-6, 25)
    g, az = r.road_to_polar(x, y)
    assert az == pytest.approx(-6) and r.slant_range(g) == pytest.approx(25)

def test_wedge_stays_inside_fov_and_range():
    pts = wedge_polygon(make_radar(fov_half=20, max_range=100))
    g, az = zip(*(make_radar(fov_half=20, max_range=100).road_to_polar(x, y) for x, y, _ in pts))
    assert max(g) <= 100 + 1e-6 and max(abs(a) for a in az) <= 20 + 1e-6
```

- [ ] **Step 2: Run** `uv run pytest tests/test_geometry.py -v` — Expected: FAIL (import error).

- [ ] **Step 3: Implement `types.py` and `geometry.py`.** Use `cv2.projectPoints` and `cv2.undistortPoints`. Rotation road→camera: `R = Rz(roll) @ Rx(-pitch) @ R0 @ Rz_road(-yaw)`-style composition is the implementer's to get right; the tests above fix the result. `R0` maps road (X fwd, Y left, Z up) to OpenCV camera (x right, y down, z fwd). Camera centre at `(0, 0, height_m)`. `backproject_ground` returns `None` when the ray's road-Z component is ≥ −1e-9 or the hit point is behind the camera. Max range: config `radar.range_m.max`, else `max_range_default`, else 100.0.

- [ ] **Step 4: Run** `uv run pytest tests/test_geometry.py -v` — Expected: PASS.

- [ ] **Step 5: Commit** `feat: core types and camera/radar geometry`

---

### Task 3: Synthetic session generator

**Files:**
- Create: `speed_trap/synth.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Consumes: `CameraModel`, `RadarModel`, `Config` (Tasks 1–2); `ld2451.Target`, `ld2451.protocol.encode_report`.
- Produces:
  - `@dataclass(frozen=True) SynthVehicle(vehicle_id: int, direction: Direction | None, lane_y_m: float, speed_kmh: float, start_x_m: float, start_s: float)` (`direction None` = stopped).
  - `default_scenario(duration_s: float, seed: int) -> list[SynthVehicle]`: at least one coming vehicle at 60 km/h, one going at 30 km/h, one going at 50 km/h, one stopped vehicle at x = 20 m in the coming lane; others random under 35 km/h. Lanes from the default road: coming lane centre y = 1.75, going y = −1.75 (camera on the road's centre line; fine for synthetic).
  - `generate_session(out_dir: Path, cfg: Config, duration_s: float, seed: int, vehicles: list[SynthVehicle] | None = None) -> Path` returns the session path `out_dir/s000001/`.
  - Ground truth file `truth.json` in the session folder: `{"limit_kmh", "vehicles": [{"vehicle_id", "direction", "speed_kmh", "expect_capture"}], "frames": {"<segment>:<index>": [{"vehicle_id", "xyxy"}]}}`. `expect_capture` = direction not None and `speed_kmh` × cos(worst line-of-sight angle while within `fusion.max_range_m`) > `limit_kmh` + 1 (clear margin so radial under-reading still triggers).

Behaviour: frames at `synth.fps` with each frame dropped with `synth.drop_probability` (seeded), split into segments of `synth.segment_s` separated by `synth.gap_s` with no frames and no radar. Each frame renders sky, ground, two-lane road, lane markings and each visible vehicle as a filled 1.8 × 1.5 m box face (front/back) through `CameraModel`, encoded with PyAV `libx264` into `video/NNNN.h264` (raw H.264, `format="h264"`), and `frames.csv` with `index,sensor_ts_ns,frame_wall_clock_us,exposure_us,lux`. Radar every `synth.report_interval_s`: moving vehicles inside the radar FOV and range, at most 5 (nearest first), quantised with `round()` (angle from `road_to_polar`, distance from `slant_range`, speed = radial speed), `snr` 20–60; empty report every 1.4 s when none. Write each report as two `t_rx_ns,hex` lines (first byte, then the rest 2 ms later) to `radar/NNNN.csv` with the header `t_rx_ns,hex`. `session.json`: `format_version 1`, `session_id "s000001"`, `state "complete"`, `ntp_synced false`, `radar_latency_ns 0`, `devices.camera {width,height,framerate}`, `devices.radar.params {direction 2, max_distance_m 100, min_speed_kmh 1, no_target_delay_s 1}`, `segments [...]`, `clock_samples` with a start sample (`realtime_ns` = 2026-10-21T06:00:00Z), `setup {"notes": "synthetic", "geometry": {}}`. Boot clock starts at 100 s.

- [ ] **Step 1: Write failing tests** (20 s, seed 1, default config):

```python
def test_session_opens_with_recorder_reader(synth_session):
    s = open_session(synth_session).summary()
    assert s["state"] == "complete" and 15 < s["mean_fps"] < 25 and s["radar_reports"] > 100

def test_video_frame_count_matches_csv(synth_session):
    for seg in (0, 1):
        with av.open(str(synth_session / f"video/{seg:04d}.h264"), format="h264") as c:
            n = sum(1 for _ in c.decode(video=0))
        assert n == len(read_csv_rows(synth_session / f"video/{seg:04d}.frames.csv"))

def test_segment_gap_present(synth_session):
    segs = json.loads((synth_session / "session.json").read_text())["segments"]
    assert segs[1]["start_boottime_ns"] - segs[0]["end_boottime_ns"] >= 2.9e9

def test_radar_values_quantised_and_capped(synth_session):
    reports = list(open_session(synth_session).radar_reports())
    assert all(len(r.targets) <= 5 for r in reports)
    assert any(t.approaching for r in reports for t in r.targets)
    assert any(not t.approaching for r in reports for t in r.targets)

def test_truth_marks_only_the_fast_vehicles(synth_session):
    truth = json.loads((synth_session / "truth.json").read_text())
    assert {v["speed_kmh"] for v in truth["vehicles"] if v["expect_capture"]} >= {60, 50}
    assert not any(v["expect_capture"] for v in truth["vehicles"] if v["direction"] is None)

def test_deterministic_with_seed(tmp_path): ...  # same seed → identical radar CSV bytes
```

`synth_session` is a session-scoped fixture in `tests/conftest.py` calling `generate_session(tmp_path_factory.mktemp("synth"), Config(), 20.0, 1)`.

- [ ] **Step 2: Run** `uv run pytest tests/test_synth.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement `speed_trap/synth.py`.**
- [ ] **Step 4: Run** `uv run pytest tests/test_synth.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: synthetic session generator with ground truth`

---

### Task 4: Recorded session source

**Files:**
- Create: `speed_trap/source.py`
- Test: `tests/test_source.py`

**Interfaces:**
- Consumes: `radar_cam_recorder.reader.open_session`, `RecordedSession.frames()`, `.radar_reports()`, `.metadata`; types from Task 2; `Config.radar`.
- Produces:
  - `class SourceError(Exception)`
  - `class RecordedSessionSource`: `__init__(self, path: Path, radar_cfg: RadarConfig, segments: Sequence[int] | None = None, start_s: float | None = None, end_s: float | None = None)`; properties `session_id: str`, `width: int`, `height: int`, `fps: float` (median measured), `segments: list[int]`, `has_radar: bool`, `ntp_synced: bool`, `radar_max_distance_m: float | None`, `radar_direction_param: int | None`, `warnings: list[str]`; `__iter__() -> Iterator[FrameBundle]`.

Behaviour: frames come from `open_session(path).frames()` grouped by segment; each segment's `.h264` is decoded with PyAV (`format="h264"`, BGR via `frame.to_ndarray(format="bgr24")`) and zipped with that segment's frames in index order. On a count mismatch, stop at the shorter and add a warning `"segment N: video has A frames, timestamps B"`. Radar reports are shifted by `radar_latency_ns + clock_offset_ms × 1e6`, converted to `RadarScan` (`approaching = t.approaching if approaching_byte == 0 else not t.approaching`), and attached to the first frame whose `ts_ns >=` the scan time (scans before the first frame go to the first frame; scans after the last frame are dropped). `wall_time_ns` comes from `frame_wall_clock_us` in `frames.csv` when present, else from the latest clock sample before the frame (`realtime_ns + ts − boottime_ns`). `start_s`/`end_s` are seconds from the first frame of the session; validation raises `SourceError` naming the valid range (`0..<duration>` s) and existing segments. No radar dir or no reports → `has_radar False` and a warning. `radar_direction_param` in (0, 1) → warning "radar was set to report one direction only".

- [ ] **Step 1: Write failing tests** using `synth_session` and small copies of it:

```python
def test_frames_in_order_with_images(synth_session):
    b = list(RecordedSessionSource(synth_session, RadarConfig()))
    assert [x.ts_ns for x in b] == sorted(x.ts_ns for x in b)
    assert b[0].image.shape == (972, 1296, 3) and b[0].image.dtype == np.uint8

def test_every_scan_delivered_once_after_its_time(synth_session):
    src = RecordedSessionSource(synth_session, RadarConfig())
    bundles = list(src); scans = [s for b in bundles for s in b.scans]
    assert len(scans) == count_reports_within_video_span(synth_session)
    assert all(s.t_ns <= b.ts_ns for b in bundles for s in b.scans)

def test_clock_offset_shifts_scans(synth_session): ...      # +50 ms → every t_ns 50e6 smaller

def test_approaching_byte_flips_direction(synth_session): ...

def test_truncated_video_warns_and_stops(tmp_path, synth_session):
    s = copy_session(synth_session, tmp_path); append_csv_rows(s / "video/0000.frames.csv", 5)
    src = RecordedSessionSource(s, RadarConfig()); list(src)
    assert any("segment 0" in w for w in src.warnings)

def test_missing_radar_dir(tmp_path, synth_session):
    s = copy_session(synth_session, tmp_path); shutil.rmtree(s / "radar")
    src = RecordedSessionSource(s, RadarConfig())
    assert not src.has_radar and all(b.scans == () for b in src)

def test_bad_time_range_names_valid_range(synth_session):
    with pytest.raises(SourceError, match=r"0\.\.\d+"):
        RecordedSessionSource(synth_session, RadarConfig(), start_s=999)

def test_unknown_segment(synth_session):
    with pytest.raises(SourceError, match="segments: 0, 1"):
        RecordedSessionSource(synth_session, RadarConfig(), segments=[7])

def test_wall_time_from_frames_csv(synth_session): ...      # first bundle wall_time_ns == csv µs × 1000
```

- [ ] **Step 2: Run** `uv run pytest tests/test_source.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement `speed_trap/source.py`.** Decode lazily one segment at a time (memory must not grow with session length).
- [ ] **Step 4: Run** `uv run pytest tests/test_source.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: recorded session source`

---

### Task 5: Detectors

**Files:**
- Create: `speed_trap/detect/__init__.py`, `speed_trap/detect/truth.py`, `speed_trap/detect/rfdetr.py`, `speed_trap/detect/yolo.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Consumes: `Detection`, `FrameBundle` (Task 2), `Config.detector`.
- Produces:
  - `class Detector(Protocol)`: `name: str`; `detect(self, bundle: FrameBundle) -> list[Detection]`.
  - `make_detector(cfg: DetectorConfig) -> Detector`; backends `"rfdetr"`, `"ultralytics"`, `"ground_truth"`. Missing extra → `ConfigError("detector.backend 'rfdetr' needs: uv sync --extra rfdetr")`.
  - `GroundTruthDetector(truth_path: Path, noise_px: float = 2.0, seed: int = 0)` reads `truth.json` from Task 3 (`detector.weights` is the truth path); class name `"car"`, confidence 0.9.

Behaviour shared by real backends: keep only `cfg.classes` (COCO names), `confidence >= cfg.conf`, apply `cfg.roi` polygon (keep boxes whose bottom-centre is inside), pass `cfg.device`, `cfg.imgsz`, `cfg.iou` to the library. RF-DETR: weights names `rfdetr-nano|small|medium|base|large` map to the library's model classes, or a path to a checkpoint; class ids mapped to names with the library's COCO table (check its indexing). Ultralytics: `YOLO(weights).predict(image, imgsz, conf, iou, device, verbose=False)`, names from `model.names`.

- [ ] **Step 1: Write failing tests:**

```python
def test_ground_truth_detector_returns_truth_boxes(synth_session):
    det = make_detector(DetectorConfig(backend="ground_truth", weights=str(synth_session / "truth.json")))
    bundle = next(iter(RecordedSessionSource(synth_session, RadarConfig())))
    assert len(det.detect(bundle)) == len(truth_boxes(synth_session, bundle.segment, bundle.index))

def test_roi_filters_by_bottom_centre(): ...   # stub detector output through apply_filters()

def test_class_and_conf_filter(): ...          # apply_filters keeps car 0.5, drops person 0.9 and car 0.2

def test_missing_extra_message(monkeypatch):    # make import fail → ConfigError mentions "uv sync --extra rfdetr"

@pytest.mark.skipif(importlib.util.find_spec("rfdetr") is None, reason="rfdetr extra not installed")
def test_rfdetr_smoke(): ...                    # detect on a black 1296x972 frame → list (possibly empty)

@pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None, reason="yolo extra not installed")
def test_yolo_smoke(): ...
```

`apply_filters(dets: list[Detection], cfg: DetectorConfig) -> list[Detection]` lives in `detect/__init__.py` and is used by both real backends.

- [ ] **Step 2: Run** `uv run pytest tests/test_detect.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement the four files.**
- [ ] **Step 4: Run** `uv run pytest tests/test_detect.py -v` — Expected: PASS (smoke tests skipped).
- [ ] **Step 5: Install one extra and run its smoke test:** `uv sync --extra rfdetr && uv run pytest tests/test_detect.py -k rfdetr -v` — Expected: PASS. If RF-DETR fails on MPS, note it in the commit message and the README; do not change the default.
- [ ] **Step 6: Commit** `feat: detector backends (RF-DETR, Ultralytics, ground truth)`

---

### Task 6: Tracker

**Files:**
- Create: `speed_trap/tracker.py`
- Test: `tests/test_tracker.py`

**Interfaces:**
- Consumes: `Detection`, `TrackObservation` (Task 2), `Config.tracker`.
- Produces:
  - `@dataclass(frozen=True) TrackerOutput(active: list[TrackObservation], finished: list[int])`
  - `class Tracker`: `__init__(self, cfg: TrackerConfig, fps: float)`; `update(self, detections: list[Detection] | None, ts_ns: int) -> TrackerOutput`; `finish_all(self) -> list[int]`.

Behaviour: wraps `supervision.ByteTrack(track_activation_threshold, lost_track_buffer=max(1, round(lost_track_s × fps)), minimum_matching_threshold=match_threshold, frame_rate=round(fps), minimum_consecutive_frames)`. `detections=None` (a frame skipped by `every_n_frames`) does not call ByteTrack; active tracks are returned unchanged with `detected=False`. Run-unique ids: own counter mapped from `(epoch, sv_tracker_id)`. When `ts_ns − previous ts_ns > lost_track_s`, all tracks finish, ByteTrack is reset and the epoch increments. A track also finishes when it has not been returned by ByteTrack for `lost_track_s` (by timestamp). Class name: majority class of the track's detections.

- [ ] **Step 1: Write failing tests:**

```python
def test_moving_box_keeps_one_id(): ...                 # 30 frames, box shifts 5 px/frame → one id throughout
def test_two_boxes_two_ids(): ...
def test_time_gap_finishes_and_new_ids_are_unique():
    out1 = run_frames(tracker, n=10, t0=0)               # id A
    out2 = tracker.update([box_at(0)], ts_ns=10 * 40_000_000 + 3_000_000_000)
    assert A in out2.finished and out2.active[0].track_id != A
def test_none_detections_hold_last_box(): ...          # update(None) → same xyxy, detected False
def test_lost_track_finishes_after_timeout(): ...      # box disappears; finished within lost_track_s + 1 frame
```

- [ ] **Step 2: Run** `uv run pytest tests/test_tracker.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement `speed_trap/tracker.py`.**
- [ ] **Step 4: Run** `uv run pytest tests/test_tracker.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: ByteTrack wrapper with run-unique ids`

---

### Task 7: Radar–track fusion

**Files:**
- Create: `speed_trap/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: `CameraModel`, `RadarModel`, `TrackObservation`, `RadarScan`, `RadarMatch`, `Direction` (Task 2); `Config.fusion`, `Config.speed`, `Config.road`.
- Produces:
  - `@dataclass class FusedTrack` (mutable): `track_id: int`, `class_name: str`, `xyxy`, `confidence: float`, `first_ts_ns: int`, `last_ts_ns: int`, `road_xy: tuple[float, float] | None`, `road_history: deque[tuple[int, float, float]]` (bounded to `direction_window_s` × 4), `boxes: list[tuple[int, tuple[float, float, float, float]]]`, `matches: list[RadarMatch]`, `reliable: bool`, `direction: Direction | None`, properties `last_match: RadarMatch | None`, `max_speed_kmh: float | None`, `mean_speed_kmh: float | None`, `subject_id` (= `track_id`).
  - `gate_cost(radar: RadarModel, cfg: FusionConfig, track_xy: tuple[float, float], target: RadarTarget) -> float | None` (normalised cost; `None` = outside gate). Range gate = `range_base_m + range_per_m2 × r² + 0.5`; angle gate = `0.5 + angle_margin_deg` (degrees); cost = `(Δr/gate_r)² + (Δaz/gate_az)²`, `None` if either ratio > 1, or ground range of either side > `max_range_m`.
  - `class Fusion`: `__init__(self, camera: CameraModel, radar: RadarModel, fusion: FusionConfig, speed: SpeedConfig, road: RoadConfig)`; `update(self, observations: list[TrackObservation], scans: Sequence[RadarScan], ts_ns: int) -> list[FusedTrack]`; `finish(self, track_ids: Sequence[int]) -> list[FusedTrack]`.

Behaviour per `update`:
1. Upsert tracks; `road_xy` = `backproject_ground` of the box bottom-centre (None allowed).
2. Track direction from `road_history` over `direction_window_s`: change in radar ground range ≤ −`direction_min_travel_m` → `"coming"`, ≥ +that → `"going"`, else unknown.
3. For each scan: cost matrix tracks × targets via `gate_cost`; also invalid when the track's motion direction is known and disagrees with `target.approaching`, or when `fusion.lanes` is set and the two fall in different lanes. `scipy.optimize.linear_sum_assignment` on valid entries (invalid = 1e6). Ambiguity: drop an assigned pair if, in its row or its column, another valid cost `c2` satisfies `c_best > ambiguity_ratio × c2`.
4. A kept pair appends `RadarMatch`; `speed_kmh` = raw, or with `speed.cosine_correction` raw / max(cos θ, 0.5) where θ = angle between radar line of sight and the lane heading (`road.lane_heading_deg`).
5. `reliable = len(matches) >= reliable_min_matches and ts_ns − last_match.t_ns <= reliable_max_age_s`; `direction` = radar direction of the last match while reliable, else unchanged (the renderer draws gray when not reliable).

- [ ] **Step 1: Write failing tests** (radar at camera, height 3.5, boresight 0, default fusion config):

```python
def test_single_track_matches():
    f = fusion(); t = obs_at_road(20, 0)
    tracks = f.update([t], [scan(target(angle=0, dist=20, approaching=True, speed=50))], ts(0))
    assert tracks[0].last_match.speed_raw_kmh == 50

def test_two_nearby_tracks_are_ambiguous():
    tracks = fusion().update([obs_at_road(20, 0.15), obs_at_road(20, -0.15)], [scan(target(0, 20))], ts(0))
    assert all(t.last_match is None for t in tracks)

def test_other_lane_track_does_not_steal_match():
    tracks = fusion().update([obs_at_road(20, 0), obs_at_road(20, -3.5)], [scan(target(0, 20))], ts(0))
    assert [t.last_match is not None for t in tracks] == [True, False]

def test_beyond_max_range_never_matches():
    assert fusion().update([obs_at_road(40, 0)], [scan(target(0, 40))], ts(0))[0].last_match is None

def test_direction_disagreement_rejected():  # track moves 30 m → 25 m in 0.5 s, target approaching=False
    ...

def test_reliability_needs_two_matches_and_expires():
    f = fusion(); f.update([o], [scan(target(0, 20))], ts(0)); assert not track(f).reliable
    f.update([o], [scan(target(0, 20))], ts(0.1)); assert track(f).reliable
    f.update([o], [], ts(1.7)); assert not track(f).reliable and track(f).last_match is not None

def test_box_above_horizon_has_no_road_point_and_no_match(): ...

def test_cosine_correction_off_by_default_and_capped(): ...
    # off: speed_kmh == raw; on with θ = 70°: speed_kmh == raw / 0.5

def test_gate_widens_with_distance():
    assert gate_cost(radar, cfg, (28, 0), target(0, 30)) is not None   # Δr ≈ 1.8 m, gate ≈ 3.6 m at 28 m
    assert gate_cost(radar, cfg, (8, 0), target(0, 11)) is None         # Δr ≈ 2.4 m, gate ≈ 2.1 m at 8 m
```

- [ ] **Step 2: Run** `uv run pytest tests/test_fusion.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement `speed_trap/fusion.py`.**
- [ ] **Step 4: Run** `uv run pytest tests/test_fusion.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: radar-track fusion with distance-aware gates`

---

### Task 8: Lite linker and speeding monitor

**Files:**
- Create: `speed_trap/lite.py`, `speed_trap/monitor.py`
- Test: `tests/test_lite.py`, `tests/test_monitor.py`

**Interfaces:**
- Consumes: `RadarModel`, `RadarScan`, `RadarMatch`, `gate_cost` (Tasks 2, 7); `FusedTrack`; `Config.lite`, `Config.speed`, `Config.capture`, `Config.fusion`.
- Produces (`lite.py`):
  - `@dataclass class RadarSubject` (mutable): `subject_id: int`, `matches: list[RadarMatch]`, `direction: Direction`, `reliable: bool` (always True), `xyxy: None`, `confidence: float` (1.0), properties `last_match`, `max_speed_kmh`, `mean_speed_kmh`.
  - `class RadarTargetLinker`: `__init__(self, radar: RadarModel, fusion: FusionConfig, lite: LiteConfig, speed: SpeedConfig, road: RoadConfig)`; `update(self, scans: Sequence[RadarScan], ts_ns: int) -> tuple[list[RadarSubject], list[RadarSubject]]` (active, finished). Linking: Hungarian on `gate_cost` between a subject's last road point and each target, same direction required; unmatched target → new subject (only within `max_range_m`); a subject not updated for `lite.lost_track_s` finishes.
- Produces (`monitor.py`):
  - `class Subject(Protocol)`: `subject_id: int`, `matches: list[RadarMatch]`, `direction: Direction | None`, `reliable: bool`, `xyxy: tuple[float, float, float, float] | None`, `confidence: float`, `max_speed_kmh`, `mean_speed_kmh`.
  - `@dataclass(frozen=True) Candidate(subject_id: int, segment: int, index: int, ts_ns: int, xyxy: tuple[float, float, float, float] | None)` (a better frame for this subject's capture).
  - `@dataclass(frozen=True) CaptureEvent(subject_id: int, direction: Direction, limit_kmh: float, speed_max_kmh: float, speed_mean_kmh: float, speed_at_capture_kmh: float, capture: Candidate, confidence: float, history: tuple[RadarMatch, ...])`
  - `class SpeedingMonitor`: `__init__(self, speed: SpeedConfig, capture: CaptureConfig, width: int, height: int)`; `update(self, active: Sequence[Subject], segment: int, index: int, ts_ns: int) -> list[Candidate]`; `finish(self, finished: Sequence[Subject]) -> list[CaptureEvent]`; property `triggered_ids: set[int]`; `limit_for(direction: Direction) -> float`.

Monitor behaviour: limit = `limit_kmh_<direction>` or `limit_kmh`. A subject is triggered once it is reliable and its last `min_consecutive_over` matches all exceed the limit. While its latest match is over the limit, a frame is a candidate when the box does not touch the frame edge (within `edge_margin_px`) and `area × confidence` beats its previous best; for `xyxy None` (lite), the candidate score is `−last_match.target.distance_m` (closest reading wins). `finish` emits a `CaptureEvent` only for triggered subjects that have a candidate; `speed_at_capture` = speed of the last match at or before the candidate time.

- [ ] **Step 1: Write failing tests:**

```python
# test_monitor.py — limit 40, K = 2
def test_two_consecutive_over_triggers_and_event_on_finish():
    m = monitor(); s = subject(speeds=[38, 45, 47], box=(100, 100, 300, 250))
    m.update([s], 0, 10, ts(1)); ev = m.finish([s])
    assert ev[0].speed_max_kmh == 47 and ev[0].speed_mean_kmh == pytest.approx(130 / 3)

def test_single_spike_does_not_trigger():
    m = monitor(); s = subject(speeds=[45, 30]); m.update([s], 0, 1, ts(1)); assert m.finish([s]) == []

def test_unreliable_subject_never_triggers(): ...
def test_per_direction_limit_override(): ...         # limit_kmh_going=60 → going at 50 not triggered
def test_best_frame_is_largest_box_not_touching_edge():
    # three updates: box A area 1e4, box B area 4e4 touching right edge, box C area 2e4 → capture index of C
def test_lite_subject_best_is_closest_reading(): ...

# test_lite.py
def test_one_subject_per_passing_target(): ...       # 10 scans, distance 30→21, approaching → 1 subject
def test_opposite_directions_two_subjects(): ...
def test_subject_finishes_after_lost_track_s(): ...
def test_targets_beyond_max_range_ignored(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/test_lite.py tests/test_monitor.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement `speed_trap/lite.py` and `speed_trap/monitor.py`.**
- [ ] **Step 4: Run** the same command — Expected: PASS.
- [ ] **Step 5: Commit** `feat: lite radar linker and speeding monitor`

---

### Task 9: Overlay renderer

**Files:**
- Create: `speed_trap/render/__init__.py`, `speed_trap/render/draw.py`, `speed_trap/render/fonts/` (download `BarlowSemiCondensed-SemiBold.ttf`, `BarlowSemiCondensed-Bold.ttf` from github.com/google/fonts `ofl/barlowsemicondensed`, `IBMPlexMono-Regular.ttf` from `ofl/ibmplexmono`, plus their `OFL.txt`)
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `CameraModel`, `RadarModel`, `wedge_polygon`, `range_ring` (Task 2); `FusedTrack` (Task 7); `RadarSubject`, `SpeedingMonitor.triggered_ids`, `limit_for` (Task 8); `Config.overlay`, `Config.lite`.
- Produces:
  - `@dataclass(frozen=True) HudInfo(session_id: str, segment: int, index: int, wall_time_ns: int | None, mode: Literal["full", "lite"], model: str | None, tracker: str | None, fps: float, radar_targets: int, radar_age_s: float | None, clock_offset_ms: float, limit_kmh: float)`
  - `class Renderer`: `__init__(self, camera: CameraModel, radar: RadarModel, overlay: OverlayConfig, lite: LiteConfig)`; `render(self, image: np.ndarray, *, tracks: Sequence[FusedTrack], subjects: Sequence[RadarSubject], current_targets: Sequence[RadarTarget], triggered: set[int], limit_for: Callable[[Direction], float], hud: HudInfo) -> np.ndarray` (new array, same shape). Wedge polygon and ring pixel coordinates are computed once in `__init__`.

Drawing (spec §5.8, approved mockup, style B), scaled from the 1296-px-wide mockup by `width / 1296`:
- Wedge: translucent white fill fading with distance (alpha 0.22 near → 0.04 far), outline alpha 0.55, dashed range rings at `overlay.show.range_rings` with `"N m"` labels at the left/right edge.
- Full mode, per track: corner brackets (length 25 % of the shorter box side, 3.5 px) in the direction color when reliable, dashed gray otherwise; side tag 168 × 64 px left of the box (right if it would leave the frame), dark fill alpha 0.86 with a 1.5 px border in the track color, leader line from the box top-left to the tag. Tag text: speed (30 px bold, red if over `limit_for(direction)`) + `km/h`, `▼ #id` (coming) or `▲ #id` (going) right-aligned, `"{d} m · {±a}° · SNR {s}"` in mono 11.5 px. Gray tag: `#id NO RADAR` and `last N km/h · X.X s` (or `no reading` if never matched). Triggered tracks: red badge `+{speed − limit}` / `● CAPTURED` next to the tag.
- Lite mode, per subject's latest target: pillar at the radar road point, `lite.pillar_height_m` tall, width = 1° at that distance (min 6 px), ground ellipse ±0.5 m, label chip above (speed, arrow, details strip).
- Radar targets in `current_targets` beyond `max_range_m` (full mode) are drawn as thin pillars without a label.
- HUD: top bar 44 px dark alpha 0.72 with `"{session} / seg {segment:04d} · frame {index} · HH:MM:SS.mmm"` (UTC wall time; `--:--:--` when None), legend chips COMING/GOING/NO RADAR (lite: no NO RADAR), limit sign `LIMIT {limit} km/h`; bottom bar 40 px with `RADAR {n} targets · last frame {age:.2f} s ago · clock offset {±ms} ms` and `FULL · {model} · {tracker} · {fps:.1f} fps` or `LITE · radar only`.
- Every element toggled by `overlay.show.*`; colors from `overlay.colors`.

- [ ] **Step 1: Write failing tests:**

```python
def test_output_shape_and_input_untouched(): ...
def test_reliable_coming_bracket_uses_coming_color():
    out = render_one(track(reliable=True, direction="coming", box=(400, 300, 600, 450)))
    assert near_color(out[300 + 1, 400 + 10], "#35d0ba")
def test_unreliable_track_bracket_is_gray(): ...
def test_wedge_changes_ground_not_sky():
    out = Renderer(...).render(gray_image, tracks=[], ...)
    assert (out[900, 648] != gray_image[900, 648]).any() and (out[100, 648] == gray_image[100, 648]).all()
def test_tag_flips_right_near_left_edge(): ...     # box at x=10 → tag pixels to the right of the box
def test_lite_pillar_drawn_at_radar_point(): ...
def test_hud_toggle_off_leaves_top_rows_untouched(): ...
def test_write_preview_png(tmp_path): ...           # renders one full and one lite frame to PNG for manual review
```

- [ ] **Step 2: Run** `uv run pytest tests/test_render.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement the renderer** (OpenCV `LINE_AA` for lines/polygons, numpy alpha blending for fills, one Pillow `ImageDraw` pass per frame for all text).
- [ ] **Step 4: Run** `uv run pytest tests/test_render.py -v` — Expected: PASS. Open the preview PNGs and compare with the mockup page.
- [ ] **Step 5: Commit** `feat: overlay renderer (style B, wedge, HUD, lite pillars)`

---

### Task 10: Output sinks

**Files:**
- Create: `speed_trap/sinks.py`
- Test: `tests/test_sinks.py`

**Interfaces:**
- Consumes: `FrameBundle` (Task 2), `FusedTrack` (Task 7), `RadarSubject`, `Candidate`, `CaptureEvent` (Task 8), `Config`, `config_hash` (Task 1).
- Produces:
  - `class VideoSink`: `__init__(self, out_dir: Path, width: int, height: int)`; `write(self, segment: int, ts_ns: int, image: np.ndarray) -> None` (opens `NNNN.overlay.mp4` per segment, closes the previous one); `close() -> None`. H.264 via PyAV, stream `time_base = Fraction(1, 1000)`, `pts` = ms since the segment's first frame (variable frame rate, so playback matches real time), `pix_fmt yuv420p`.
  - `class TracksSink`: `__init__(self, path: Path)`; `write(self, finished: Sequence[FusedTrack | RadarSubject]) -> None`; `close()`. One JSON line per finished subject: `id, kind ("track"|"radar"), class, first_ts_ns, last_ts_ns, direction, reliable_ever, max_speed_kmh, mean_speed_kmh, boxes [[ts_ns, x1,y1,x2,y2]...], road [[ts_ns,x,y]...], matches [{t_ns, speed_kmh, speed_raw_kmh, distance_m, angle_deg, snr, approaching}]`.
  - `class CaptureSink`: `__init__(self, out_dir: Path, session_id: str, cfg: CaptureConfig, ntp_synced: bool, config_hash: str)`; `offer(self, candidates: Sequence[Candidate], raw: np.ndarray, rendered: np.ndarray, wall_time_ns: int | None) -> None` (stores a copy of the crop — box ± `crop_margin` of its size, clipped — or, for `xyxy None`, the rendered frame; plus the full raw frame only if `save_full_frame`); `write(self, events: Sequence[CaptureEvent]) -> list[Path]`; `discard(self, subject_ids: Iterable[int])` (finished without event). Files: `captures/{segment:04d}_t{id}_{max:.0f}kmh.jpg` (+ `_full.jpg`), sidecar `.json` with the keys in spec §5.7 (`wall_clock` ISO-8601 UTC and `wall_clock_reliable: ntp_synced`), JPEG quality `jpeg_quality`.
  - `write_run_info(path: Path, cfg: Config, session_path: Path, counters: dict[str, int], started_ns: int, finished_ns: int, warnings: list[str]) -> None` (`run.json` with effective config, hash, versions of speed_trap/ld2451/radar_cam_recorder/detector library, model name, counters, warnings).

- [ ] **Step 1: Write failing tests:**

```python
def test_video_sink_vfr_timing(tmp_path):
    ts = [0, 40, 80, 200, 240]                       # ms, one 120 ms drop
    sink = VideoSink(tmp_path, 320, 240)
    for t in ts: sink.write(0, t * 1_000_000, np.zeros((240, 320, 3), np.uint8))
    sink.close()
    assert decoded_pts_ms(tmp_path / "0000.overlay.mp4") == pytest.approx(ts, abs=1)

def test_video_sink_one_file_per_segment(tmp_path): ...
def test_tracks_sink_one_line_per_subject(tmp_path): ...
def test_capture_sink_crop_and_sidecar(tmp_path):
    # offer candidate box (100,100,300,200) on a 640x480 frame, write event → jpg 240x120 (10 % margin), json speed_max 47
def test_capture_keeps_only_best_candidate_memory(tmp_path): ...   # two offers → one stored crop
def test_capture_wall_clock_reliability_flag(tmp_path): ...
def test_run_info_contains_hash_and_counters(tmp_path): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/test_sinks.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement `speed_trap/sinks.py`.**
- [ ] **Step 4: Run** `uv run pytest tests/test_sinks.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: video, tracks, capture and run-info outputs`

---

### Task 11: Pipeline, CLI, end-to-end tests and README

**Files:**
- Create: `speed_trap/pipeline.py`, `speed_trap/cli.py`, `README.md`
- Modify: `pyproject.toml` (`[project.scripts] speed-trap = "speed_trap.cli:main"`)
- Test: `tests/test_cli.py`, `tests/test_e2e.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `class Pipeline`: `__init__(self, cfg: Config, source: RecordedSessionSource, out_dir: Path, mode: Literal["full", "lite"], detector: Detector | None)`; `run(self, progress: Callable[[int, int, float], None] | None = None) -> dict[str, int]` (counters: `frames, scans, tracks, matches, captures`).
  - `main(argv: Sequence[str] | None = None) -> int` with subcommands exactly as spec §6.3: `process`, `init-config`, `check-config`, `synth`. Exit codes: 0 ok, 1 runtime error, 2 usage/config/source error (message on stderr, no traceback unless `-vv`).

Pipeline per bundle: detect only when `frame_number % every_n_frames == 0` (else `None` to the tracker); tracker → fusion (full) or linker (lite) → monitor.update → renderer → `CaptureSink.offer(candidates, raw, rendered)` → `VideoSink.write`; finished subjects → `monitor.finish` → `CaptureSink.write(events)` / `discard`, `TracksSink.write`. At the end: `tracker.finish_all()`, flush everything, `write_run_info`. Current targets for drawing are those from scans within `radar.report_hold_s` of the bundle time. `RadarModel.from_config` gets `max_range_default=source.radar_max_distance_m`. Source warnings go to stderr and `run.json`. Capture sinks exist only when `capture.enabled`. `--mode lite` never constructs a detector. The output folder is created only after the source and config validated. Progress: one line per segment on stderr with frames/s and ETA.

`synth` subcommand: `speed-trap synth OUT_DIR [--config cfg.yaml] [--duration 60] [--seed 1]` (config optional here; default `Config()`), prints the session path.

- [ ] **Step 1: Write failing tests:**

```python
# test_e2e.py — synth 20 s, seed 1, capture enabled, limit 40
def test_lite_end_to_end(tmp_path, synth_session):
    assert main(["process", str(synth_session), "--config", cfg_file, "--mode", "lite", "--out", str(out), "--capture"]) == 0
    assert decoded_frame_count(out / "0000.overlay.mp4") == csv_rows(synth_session / "video/0000.frames.csv")
    assert captured_speeds(out) covers every truth vehicle with expect_capture (±2 km/h radial)

def test_full_end_to_end_with_truth_detector(tmp_path, synth_session):
    rc = main(["process", str(synth_session), "--config", cfg_file, "--out", str(out), "--capture",
               "--set", "detector.backend=ground_truth", "--set", f"detector.weights={synth_session/'truth.json'}"])
    assert rc == 0
    events = load_sidecars(out / "captures")
    assert {e["track_id"] for e in events} maps one-to-one to truth vehicles with expect_capture
    assert all(e["speed_kmh"]["max"] <= truth_speed(e) + 1 for e in events)   # never overstated
    stopped = truth_stopped_vehicle(synth_session)
    assert not any(line["reliable_ever"] for line in tracks_matching(out / "tracks.jsonl", stopped))
    assert json.loads((out / "run.json").read_text())["counters"]["captures"] == len(events)

def test_session_without_radar_all_gray(tmp_path, synth_session): ...  # rmtree radar → rc 0, run.json scans 0, warning

# test_cli.py
def test_init_config_writes_loadable_file(tmp_path): ...
def test_check_config_prints_effective_json_with_override(capsys, tmp_path): ...
def test_process_requires_config(): ...                               # argparse exit 2
def test_bad_start_exit_2_no_output_dir(tmp_path, synth_session):
    assert main(["process", str(synth_session), "--config", cfg_file, "--start", "999", "--out", str(out)]) == 2
    assert not out.exists()
def test_unknown_segment_exit_2(tmp_path, synth_session): ...
def test_lite_mode_does_not_import_torch(tmp_path, synth_session): ...  # subprocess; "torch" not in sys.modules
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py tests/test_e2e.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement `pipeline.py`, `cli.py`; add the console script.**
- [ ] **Step 4: Run** `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy` — Expected: all pass.
- [ ] **Step 5: Write `README.md`**: what it is, install (`uv sync`, `uv sync --extra rfdetr`, `--extra yolo` with the AGPL note), quick start with `synth` → `process --mode lite` → `process`, config overview pointing at `init-config`, outputs, calibration notes (OpenCV matrix, measuring pose, checking the angle sign and direction byte in lite mode on the first real recording), known limits (30 m fusion range, flat road, radar quantisation).
- [ ] **Step 6: Manual check:** `uv run speed-trap synth /tmp/st && uv run speed-trap init-config /tmp/st/cfg.yaml && uv run speed-trap process /tmp/st/s000001 --config /tmp/st/cfg.yaml --mode lite --capture --out /tmp/st/out-lite`, open the overlay MP4 and a capture. Expected: wedge on the road, pillars on the moving vehicles, captures for the fast ones.
- [ ] **Step 7: Commit** `feat: pipeline, CLI, end-to-end tests and README`
