# speed-trap: offline radar + camera processing, design

Date: 2026-10-21. Status: awaiting the user's approval.

## 1. Intent

A Raspberry Pi Zero at the entrance of a private area records video (OV5647 camera via
`rpicam-vid`) and HLK-LD2451 24 GHz radar data. The scene is a road with two lanes, one
coming toward the installation and one going away. Sessions are copied to a Mac (Apple M2,
8 GB RAM) for processing.

`speed-trap` is the offline companion that turns a recorded session into:

1. an annotated video: every vehicle boxed and tracked, fused with the radar so its speed and
   other radar readings sit next to the box, plus the radar's coverage drawn on the road;
2. a per-vehicle history (track log) of position and radar readings;
3. optionally, one capture per speeding vehicle: a crop of the vehicle and a JSON sidecar.

The end goal (not v1) is a live system that reliably stores a photo of every vehicle over
the speed limit, later with the licence plate read. v1 is offline, but the code is
structured so the core can move to real-time processing on a Jetson or on a computer
receiving the stream (not the Pi Zero; the target is undecided).

## 2. Goals, non-goals, anti-goals

Goals (v1):

- CLI that processes a recorded session (or one segment of it) into the outputs above.
- Two modes:
  - **full**: object detector + tracker + radar fusion;
  - **lite**: no detector; each radar target is drawn as a "pillar" on the road at its
    reported angle and distance, with its speed. For quick previews and for checking the
    camera–radar calibration on machines that can't run a detector comfortably.
- Detector selectable by configuration: YOLO (Ultralytics) and RF-DETR at minimum, behind a
  common interface so others can be added.
- Tracking good enough to keep one identity per vehicle through the radar's range, so each
  vehicle accumulates a speed history while it comes and goes.
- Radar coverage (the field-of-view wedge) drawn on the ground plane of the image.
- Optional speeder capture (crop + JSON), both directions, one speed limit.
- All parameters in a YAML file; the CLI carries only what changes run to run.
- A synthetic session generator so the whole pipeline is testable before real road data
  exists.

Non-goals (v1):

- Live/real-time processing (but see the architecture constraints in section 7).
- Automatic calibration of the camera–radar geometry from data (v2 idea).
- Licence-plate recognition (future; v1 crops the whole vehicle).
- Camera intrinsic calibration tooling (checkerboard). The user produces the matrix with any
  OpenCV calibration script and pastes it into the YAML.
- A GUI.
- Running on the Pi Zero.

Anti-goals (would make it a failure even if it "works"):

- A speeding capture that overstates a vehicle's speed or attaches a radar reading to the
  wrong vehicle. When in doubt, the track stays gray and no capture is made.
- Assuming a constant frame rate. Real sessions run at about 20–22 fps with hundreds of
  dropped frames per 5-minute segment and about 3 s gaps between segments. All timing uses
  per-frame timestamps.
- A core that is entangled with file reading, video decoding or drawing, so that it has to be
  rewritten for live use.
- Parameters hard-coded in the code instead of the config.

## 3. What already exists (inputs)

Two sibling repositories on this machine:

- `~/workspace/driver-hlk-ld2451`: the `ld2451` Python driver (decoder, models, raw
  record/replay). Python 3.9+, only depends on pyserial.
- `~/workspace/radar-cam-recorder`: the Pi recorder and a session reader
  (`radar_cam_recorder.reader.open_session`). Uses uv; depends on `ld2451` by path.

Session layout (`recordings_dir/sNNNNNN/`, format_version 1):

- `session.json`: metadata, devices (camera resolution/fps, radar firmware and detection
  params), `radar_latency_ns`, clock samples (boottime/monotonic/realtime read together),
  segments, `setup.geometry` (currently always empty), `setup.notes`.
- `events.jsonl`: incidents (frame drops, segment gaps, camera restarts).
- `video/NNNN.h264` (raw H.264, no container timestamps), `video/NNNN.frames.csv`
  (`index,sensor_ts_ns,frame_wall_clock_us,exposure_us,lux`), `video/NNNN.meta.json`,
  `video/NNNN.pts.txt`. Some sessions also have a `video/NNNN.mp4` converted on the Mac; it
  must not be required.
- `radar/NNNN.csv`: raw radar reads, `t_rx_ns,hex` lines.

Facts that shape the design:

- Camera `sensor_ts_ns` and radar `t_rx_ns` share `CLOCK_BOOTTIME`; alignment is about
  50 ms. The reader's `timeline()` merges frames and radar reports in time order and already
  shifts radar by `radar_latency_ns`. `frame_at()` does not.
- Frame `index` in `frames.csv` is the decode order of the matching `.h264`.
- Video: nominally 1296×972 at 25 fps; real mean 20–22 fps; drops and 3 s segment gaps.
- Radar report (`ld2451.RadarReport`): `t_rx_ns` and up to 5 targets, each with
  `angle_deg` (integer degrees), `distance_m` (integer metres), `approaching` (bool),
  `speed_kmh` (integer), `snr`. With no target the radar sends an empty report about every
  1.4 s. The report rate with targets present is not yet measured.
- Radar quirks: the direction byte meaning (`00` = approaching) is from the manual and not yet
  confirmed on hardware (driver `docs/hardware-checks.md` item 7). The radar does not report
  stationary targets (minimum speed parameter, 1 km/h in the recorded sessions).
- There is no real road recording yet. The two sessions on this Mac
  (`radar-cam-recorder/recordings/s000011`, `s000012`) are indoor acceptance runs with zero
  radar targets.

## 4. Decisions made with the user

- Python, uv, `pyproject.toml`.
- YAML config file is **always required**; the CLI keeps only frequently changed options.
- Camera intrinsics come from the config as an OpenCV camera matrix + distortion
  coefficients. Camera pose and radar pose are measured and entered in the config.
- Colors: coming and going have distinct colors; vehicles without a reliable radar reading
  get a gray box that keeps the last measured values (if any).
- Label style **B** from the mockups: corner brackets instead of a full box, and a side tag
  joined to the box by a leader line (speed large, then track id + direction arrow, then
  distance · angle · SNR). Mockup page: https://claude.ai/artifact/JZsmwtN8qDbNDdmHxHXnpP
  (private to the user).
- Mockup colors and HUD accepted: teal (coming), amber (going), gray dashed (no radar),
  red for the over-limit speed and the capture marker; top bar with session/segment/frame/
  time and a legend and limit sign; bottom bar with radar status and mode/model/fps.
- Lite mode exists (pillars, no detector).
- Speeder capture is in v1, optional (off unless enabled). Both directions, one limit.
  Output per event: crop of the whole vehicle at full resolution + JSON sidecar.
- Target hardware for the future live system: Jetson or a computer receiving the stream;
  not the Pi Zero.
- Automatic calibration is v2. Plate reading is future.
- Fusion and capture only within about 30 m ("near but never wrong").
- Public GitHub portfolio project.

## 5. Behaviour

### 5.1 Geometry

Coordinate frame: a road frame with the ground at z = 0 and the road assumed flat. The camera
is described by intrinsics (camera matrix K at a stated calibration resolution, distortion
coefficients) and a pose (height above the road, pitch, roll, yaw). The radar is described
by its position relative to the camera (x, y, z in metres), its boresight yaw and pitch,
its horizontal field-of-view half-angle, and its usable range (min/max).

- If the video resolution differs from the calibration resolution, K is scaled accordingly.
- If no intrinsics are given, use an approximate pinhole from the OV5647's nominal field of
  view for the recorded resolution and log a warning (so lite mode works before calibration).
  *Claude's call.*
- Radar target → road point: `distance_m` is slant range from the radar; convert to ground
  range using the radar height, then place it at `angle_deg` relative to the radar boresight.
  The sign convention of `angle_deg` (positive = left or right) is a config parameter,
  because it is not documented clearly enough to trust.
- Image box → road point: the bottom-centre of the (undistorted) box, back-projected onto the
  road plane.
- Radar wedge: the sector between min and max range and ± half-angle on the road, projected
  into the image, clipped at the horizon and at the frame. Range rings at configurable
  distances (default 10, 20, 40, 70 m), labelled.
- Max range defaults to the radar's `max_distance_m` from `session.json` when present.

### 5.2 Time alignment

All times are on the session's boot clock. Radar times are `t_rx_ns − radar_latency_ns −
config offset`, where the config offset is a fine-tune (default 0). Each video frame is
processed with the radar reports that arrived since the previous frame. A radar report is
"current" for drawing and matching for a configurable hold time (default: a little over the
typical report interval; the builder measures it on synthetic and later real data).

### 5.3 Detection and tracking (full mode)

- Detector interface: image in, list of (box, class, confidence) out. Implementations:
  Ultralytics (any YOLO weights by name or path) and RF-DETR. Each lives behind an optional
  dependency group, so lite mode installs without torch.
- Config: backend, weights, device (`cpu`/`mps`/`cuda`), input size, confidence threshold,
  NMS IoU, vehicle classes kept (default car, truck, bus, motorcycle), optional region-of-
  interest mask polygon (to drop parked cars or the far road; optional).
- Tracker: ByteTrack via the `supervision` library (detector-agnostic). Timeouts in the config
  are in seconds and converted using the measured frame rate. *Claude's call.*
- Track IDs are unique per run, including across segment gaps; a track that disappears for
  longer than the timeout is finished. Tracks do not try to bridge the ~3 s segment gap
  (a new ID afterwards is acceptable).

### 5.4 Radar–track fusion

- For each radar report, build a cost between every current track and every radar target from
  their distance on the road plane. Gates widen with distance (*Claude's call*): the range gate
  is a base margin plus a term proportional to the square of the distance (back-projection error
  of the box bottom grows that way), and the lateral gate is mostly angular (radar 1° step plus
  a margin, times distance). The range gate also allows for the radar reflecting off a different
  part of the vehicle than the box bottom. Assign with the Hungarian algorithm, rejecting pairs
  outside the gate.
- Fusion range (decided with the user: "near but never wrong"): matching, reliable color and
  captures only happen within `fusion.max_range_m` (default 30 m) of the radar. Beyond it,
  tracks are gray and the radar target is still drawn (in lite style) but never attached to a
  track. Gates stay strict inside that range.
- Ambiguity guard (*Claude's call*): a match is discarded if a second track or a second radar
  target also falls inside the gate with a cost close to the best one (configurable ratio), and
  if the track and the radar target are in different lanes (lane boundaries from config, when
  given). Ambiguous frames leave the track's state unchanged.
- Direction consistency: a pair is only valid if the track's own motion agrees with the radar's
  `approaching` flag (track moving toward the radar vs. away, estimated from its road-plane
  positions over a short window). A track too new to have a direction can still match.
- A track is **reliable** once it has at least N matches (config, default 2) and its last
  match is no older than T seconds (config). Reliable tracks are drawn in the coming/going
  color. Unreliable tracks are gray and show the last matched values with their age, or no
  values if never matched.
- The track's displayed direction is the radar's direction once reliable; before that, gray.
- Each match is appended to the track's history.

### 5.5 Speed

- Displayed speed is the latest matched radar `speed_kmh`.
- Optional cosine correction for the angle between the radar line of sight and the vehicle's
  path (lane direction from config), **off by default**: the raw radial speed reads slightly
  low, so captures never overstate speed. Both raw and corrected values are logged when the
  correction is enabled.

### 5.6 Lite mode

- No detector or tracker. Each radar target in the current report is a pillar: a vertical bar
  standing on its road point, about 1.6 m tall (config), with width equal to the radar's 1°
  angle step at that distance and a ground disc showing the ±0.5 m range step.
- Pillar color by direction; label as in style B (speed large, distance · angle · SNR).
- Lite-mode capture: if enabled, the same over-limit rule applies to radar targets; the output
  is the full frame with the pillar overlay (no crop) and the JSON sidecar. Radar targets are
  associated across reports by simple nearest-neighbour on the road plane so each passing
  vehicle produces one event. *Claude's call.*

### 5.7 Speeder capture (optional)

- Trigger: a reliable track whose matched speed exceeds the limit for at least K consecutive
  radar matches (config, default 2). Limit: one value for both directions; per-direction
  override optional (config). *Override is Claude's call.*
- One event per track. While the track is over the limit, keep the best frame candidate:
  largest box area weighted by detector confidence, and not touching the frame edge.
  Write the event when the track ends (or at end of input).
- Output per event: `captures/<session>/<segment>_t<track>_<maxspeed>kmh.jpg` (full-resolution
  crop of the whole vehicle with a configurable margin, no overlay) and a `.json` sidecar
  with: event type, session, segment, track id, direction, limit, speed max/mean/at capture,
  capture frame index, `sensor_ts_ns`, wall-clock time (from the frame's
  `frame_wall_clock_us`, flagged unreliable when `session.json` says NTP was not synced),
  box, detector confidence, the full radar match history of the track, and a hash of the
  effective config. Optional (off by default): the full clean frame as well.

### 5.8 Overlay drawing

Matches the accepted mockup, style B:

- Corner brackets around each tracked vehicle; dashed gray brackets for unreliable tracks.
- Side tag with leader line: large speed (red when over the limit), `▼`/`▲` + track id,
  `distance m · angle° · SNR`. For gray tracks: `#id NO RADAR`, `last N km/h · X.X s`.
- When a capture was triggered for a track: an excess badge (`+32`) and a `CAPTURED` marker.
- Radar wedge on the road (translucent fill fading with distance, outline, labelled range
  rings).
- Top bar: session / segment, frame index, time of day; legend (coming / going / no radar);
  speed-limit sign. Bottom bar: radar status (targets in the current report, age of the last
  report, clock offset) and mode / model / tracker / processing fps.
- Every element can be toggled and colors set in the config (`overlay` section). Tag
  placement avoids leaving the frame; overlapping tags are acceptable in v1.

## 6. Outputs, config and CLI

### 6.1 Outputs

Per run, under `--out` (default `./out/<session_id>/`):

- `NNNN.overlay.mp4` per processed segment (H.264). Playback must match real time despite
  dropped frames (variable frame rate from the timestamps, or constant rate with repeated
  frames: builder's choice).
- `tracks.jsonl`: one line per finished track: id, class, first/last time, direction,
  per-frame boxes (or a decimated list), road-plane positions, radar match history, reliable
  flag, max/mean speed.
- `captures/` when capture is enabled.
- `run.json`: effective config (after CLI overrides), config hash, package versions, model
  name, input session path, start/end time, counters (frames, tracks, matches, captures).

### 6.2 Config (YAML)

Always required. `speed-trap init-config PATH` writes a complete, commented default file.
Sections (names indicative; keys exhaustive in the builder's schema):

```yaml
camera:
  intrinsics:            # OpenCV; optional (falls back to nominal OV5647 FOV with a warning)
    resolution: [1296, 972]
    matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    distortion: [k1, k2, p1, p2, k3]
  pose: { height_m: 3.5, pitch_deg: 12.0, roll_deg: 0.0, yaw_deg: 0.0 }
radar:
  offset_m: { x: 0.0, y: 0.0, z: -0.3 }    # relative to the camera
  boresight: { yaw_deg: -8.0, pitch_deg: 0.0 }
  fov_half_deg: 20.0                         # check against the datasheet
  range_m: { min: 1.0, max: null }           # null = session.json max_distance_m
  angle_sign: 1                              # +1 or -1
  approaching_byte: 0                        # direction byte value meaning "approaching" (manual: 0)
  clock_offset_ms: 0.0
  report_hold_s: 0.3
road:
  lane_heading_deg: 0.0                      # used by the optional cosine correction
detector:
  backend: rfdetr                            # rfdetr (default, Apache-2.0) | ultralytics (AGPL-3.0, optional)
  weights: rfdetr-small
  device: mps
  imgsz: 960
  every_n_frames: 1                          # run the detector on every Nth frame; tracker fills in
  conf: 0.35
  iou: 0.5
  classes: [car, truck, bus, motorcycle]
  roi: null                                  # optional polygon in image pixels
tracker:
  type: bytetrack
  track_activation_threshold: 0.35
  lost_track_s: 1.5
  match_threshold: 0.8
fusion:
  gate: { range_base_m: 1.5, range_per_m2: 0.002, angle_margin_deg: 1.0 }
  ambiguity_ratio: 0.7
  max_range_m: 30.0                          # no matching or capture beyond this
  lanes: null                                # optional lateral lane boundaries, metres
  direction_window_s: 0.5
  reliable_min_matches: 2
  reliable_max_age_s: 1.5
speed:
  limit_kmh: 40
  limit_kmh_coming: null
  limit_kmh_going: null
  cosine_correction: false
capture:
  enabled: false
  min_consecutive_over: 2
  crop_margin: 0.1
  save_full_frame: false
  jpeg_quality: 95
overlay:
  style: tag
  show: { wedge: true, range_rings: [10, 20, 40, 70], hud: true }
  colors: { coming: "#35d0ba", going: "#ffb547", none: "#a7afb8", alert: "#ff4d5e" }
lite:
  pillar_height_m: 1.6
output:
  video_codec: h264
  dir: ./out
```

Validation is strict: unknown keys and out-of-range values are errors that name the key.
`speed-trap check-config --config cfg.yaml` prints the effective config.

### 6.3 CLI

```
speed-trap process SESSION_DIR --config cfg.yaml
           [--segment N ...] [--start S] [--end S]        # seconds from session start
           [--mode full|lite] [--device cpu|mps|cuda]
           [--out DIR] [--capture/--no-capture]
           [--set KEY=VALUE ...]                          # dotted path, e.g. speed.limit_kmh=50
           [-v/-vv]
speed-trap init-config PATH
speed-trap check-config --config cfg.yaml [--set ...]
speed-trap synth OUT_DIR [--config synth.yaml] [--duration S] [--seed N]
```

`--set` is *Claude's call*; values are parsed as YAML scalars and validated like the file.
Progress is shown per segment with frames/s and an ETA.

## 7. Architecture

The core is a frame-by-frame pipeline with no file I/O, no video decoding and no drawing:

```
Source ──► Detector ──► Tracker ──► Fusion ──► Events ──► Sinks
(frames +   (optional)  (optional)  (radar ↔     (speeder   (overlay video,
 radar)                              tracks)      trigger)    tracks log,
                                                              captures, run.json)
```

- **Source** yields, in time order, `FrameBundle(image, ts_ns, segment, index, radar_reports)`.
  v1 implementation: recorded session (uses `open_session` for timestamps and radar, decodes
  the `.h264` itself). Later: a live source.
- **Detector**, **Tracker**, **Fusion**, **Events**: small interfaces with plain dataclass
  inputs/outputs; each testable alone. Fusion and events take the clock from the bundle,
  never from the wall clock.
- **Sinks** receive the per-frame result (`FrameResult`: tracks with fused state, current radar
  targets, events) and the image; the renderer is a sink-side component.
- Geometry (projection, back-projection, wedge) is its own module used by fusion and drawing.
- Config: typed schema (e.g. pydantic) loaded from YAML, with `--set` overrides applied before
  validation.
- Lite mode is the same pipeline with the detector and tracker disabled and the pillar
  renderer.

Constraints for the later live migration (v1 must respect them, not implement them):

- No stage may read ahead in time (only past and current bundles), so the same code runs on a
  live stream. The capture writer's "write when the track ends" fits this.
- Per-frame work must not grow with session length (bounded track history kept in memory for
  active tracks; finished tracks flushed to sinks).

Packaging:

- Python ≥ 3.10 (Jetson JetPack ships 3.10). uv, `pyproject.toml`, `uv.lock`.
- Path dependencies on `../driver-hlk-ld2451` and `../radar-cam-recorder`, as the recorder
  already does for the driver.
- Optional dependency groups: `yolo` (ultralytics), `rfdetr` (rfdetr). Base install covers lite
  mode, synth and everything else.
- ruff, mypy (strict), pytest, matching the sibling repos.

## 8. Testing without real data

- `speed-trap synth` writes a valid format_version 1 session: rendered frames of a simple road
  scene with vehicles moving in both lanes at chosen speeds (including one over the limit and
  one stopped vehicle), irregular frame timestamps with drops and a segment gap, and radar
  reports encoded with the real LD2451 byte format and quantisation (1 m, 1°, 1 km/h, max 5
  targets, empty frames when idle), using the configured geometry. It also writes the ground
  truth (vehicle positions, speeds, which vehicle should be captured).
- Unit tests: geometry round-trips, time alignment, fusion assignment and gating, reliability
  state, direction consistency, capture trigger and best-frame choice, config validation and
  `--set`.
- End-to-end test: synth → `process` (lite, and full with a fake detector that returns the
  ground-truth boxes with noise) → check tracks, matches and captures against the ground truth.
- The rendered synth vehicles need not fool a real detector; a real-detector smoke test runs on
  any video with cars when the `yolo` extra is installed and is skipped otherwise.

## 9. Left to the builder

- Video decoding library (PyAV or OpenCV) and how output timing is produced.
- Exact config key names beyond the sections above, and defaults not stated here.
- Best-frame scoring details, tag placement rules, font (the mockups use Barlow Semi
  Condensed and IBM Plex Mono; any clear condensed font is fine).
- How the synth scene is rendered.
- Whether `tracks.jsonl` stores every frame's box or a decimated list.

## 10. Risks and open items

- Radar angle sign and direction byte are unconfirmed on hardware: both must stay
  configurable, and the first real recording should be checked in lite mode before trusting
  full-mode captures.
- The LD2451 horizontal field of view used as the default must be checked against the vendor
  datasheet.
- Radar report rate with targets present is unknown; the hold time and reliability defaults
  may need retuning on the first real recording.
- The M2 with 8 GB RAM limits model size; default weights should be a small model.
- Processing speed: offline, slower than real time is acceptable (*Claude's call*, based on the
  user's remark that heavy models are hard on this laptop). v1 is a simple sequential loop;
  `detector.every_n_frames` is the only speed knob. Parallel decode/detect is left for the live
  version.
- If `session.json` shows the radar was set to report only one direction, `process` warns.
- Licence (Claude's recommendation): the project is MIT, like the sibling repos, for a public
  GitHub portfolio. RF-DETR (Apache-2.0) is the default detector, so the default install is
  permissive end to end. Ultralytics is an optional `yolo` extra; the README states that
  installing it brings AGPL-3.0 terms. If RF-DETR turns out impractical on MPS, YOLO is a
  one-line config change.
- Matching beyond 30 m is out of scope for v1; widening it needs the camera pose to be
  verified on real data first.
