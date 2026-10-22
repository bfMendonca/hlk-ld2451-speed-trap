# speed-trap

Offline speed trap that fuses an HLK-LD2451 24 GHz radar with camera video.

A Raspberry Pi Zero at the entrance of a private area records video and radar data with
[`radar-cam-recorder`](../radar-cam-recorder). `speed-trap` processes those sessions
on a laptop and produces:

- **an annotated video** for each segment: every vehicle tracked, its radar speed, distance,
  angle and SNR in a tag beside the box, and the radar's coverage drawn on the road;
- **a track log** (`tracks.jsonl`): one line per vehicle with its boxes, road positions and
  radar readings;
- **speeder captures** (optional): a crop of each vehicle over the limit and a JSON sidecar
  with the full radar history behind the decision.

The guiding rule is *near but never wrong*. A capture must never overstate a speed or attach
a radar reading to the wrong vehicle. Radar readings are attached only within 30 m, only
when the match is unambiguous, and never when the track's own direction or speed disagrees
with the radar. A track too young to show either can still be matched by position, but a
capture also needs at least one *verified* reading (direction and speed both checked
against the track's own motion) among the consecutive over-limit readings. The raw radial
speed (which reads slightly low) is used by default. When in doubt, the vehicle stays gray
and no capture is made.

## How it works

```
Source ──► Detector ──► Tracker ──► Fusion ──► Speeding monitor ──► Sinks
(frames +   (RF-DETR /   (ByteTrack)  (radar ↔     (limit, best      (overlay video,
 radar)      YOLO)                     tracks)      frame)            tracks log,
                                                                      captures, run.json)
```

- **Source** reads a recorded session: decodes the raw H.264, takes each frame's timestamp
  from `frames.csv` (the real frame rate is irregular, with dropped frames and gaps between
  segments) and hands each frame the radar reports that arrived since the previous one.
- **Detector** finds vehicles. RF-DETR (Apache-2.0) is the default; Ultralytics YOLO is
  optional. Detectors are loaded only when used.
- **Tracker** keeps one identity per vehicle (ByteTrack, timeouts in seconds).
- **Fusion** places each box on the road plane (bottom-centre, back-projected through the
  camera model) and each radar target on the road from its range and angle, then assigns
  them with the Hungarian algorithm inside distance-aware gates. Once a track has been seen
  long enough, its own radial speed must also agree with the radar's (so a parked car never
  inherits the speed of a car passing in front of it). A track becomes *reliable*
  (colored) after enough recent matches.
- **Speeding monitor** triggers when a reliable track exceeds the limit for several
  consecutive radar matches, at least one of them verified, keeps the best frame (largest box, not cut by the frame edge)
  and writes the capture when the track ends.

The core stages do no file I/O, decoding or drawing and never read ahead in time, so the
same code can later run on a live stream.

**Lite mode** skips the detector and tracker: each radar target within the fusion range is
drawn as a labelled pillar standing on the road at its reported range and angle, linked across
reports into one subject per vehicle; targets farther away are drawn as thin unlabelled pillars. It runs fast on any machine, and it is the way to
check the camera–radar calibration.

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). The radar driver and the session
reader are path dependencies, so clone the three repositories side by side:

```
workspace/
  driver-hlk-ld2451/
  radar-cam-recorder/
  hlk-ld2451-speed-trap/
```

```sh
cd hlk-ld2451-speed-trap
uv sync                   # base install: lite mode, synth, tests (no torch)
uv sync --extra rfdetr    # + RF-DETR detector for full mode (default backend)
uv sync --extra yolo      # + Ultralytics YOLO detector (see licence note below)
```

`uv sync` installs exactly the extras you name, so run `uv sync --extra rfdetr` again
(not plain `uv sync`) after changing dependencies if you want to keep the detector.

## Quick start

No road recording yet? Generate a synthetic session: a two-lane road with vehicles in both
directions, one of them parked, irregular frame timestamps, a gap between segments, and
radar reports in the real LD2451 byte format and quantisation. The generator also writes the
ground truth.

```sh
uv run speed-trap synth /tmp/st                    # prints /tmp/st/s000001
uv run speed-trap init-config /tmp/st/cfg.yaml     # commented default config

# Lite mode: radar pillars only (no detector)
uv run speed-trap process /tmp/st/s000001 --config /tmp/st/cfg.yaml \
    --mode lite --capture --out /tmp/st/out-lite

# Full mode with the detector from the config (RF-DETR by default)
uv run speed-trap process /tmp/st/s000001 --config /tmp/st/cfg.yaml --capture --out /tmp/st/out

# Full mode on synthetic data with the ground-truth "detector" (no model needed)
uv run speed-trap process /tmp/st/s000001 --config /tmp/st/cfg.yaml --capture \
    --set detector.backend=ground_truth --set detector.weights=/tmp/st/s000001/truth.json
```

The synthetic vehicles are flat-shaded boxes. They are not meant to fool a real detector, so
use the ground-truth backend (or a real recording) to see full mode on synthetic data.

## Command line

```
speed-trap process SESSION_DIR --config cfg.yaml
           [--segment N ...] [--start S] [--end S]     # seconds from session start
           [--mode full|lite] [--device cpu|mps|cuda]
           [--out DIR] [--capture/--no-capture]
           [--set KEY=VALUE ...]                       # dotted key, e.g. speed.limit_kmh=50
           [-v/-vv]
speed-trap init-config PATH
speed-trap check-config --config cfg.yaml [--set ...]
speed-trap synth OUT_DIR [--config cfg.yaml] [--duration S] [--seed N]
```

Exit codes: `0` success, `1` runtime error, `2` usage, config or session error, `130`
interrupted. A failed run still writes `run.json`, with an `error` field. Errors are one
line on stderr; `-vv` adds the traceback. Progress (frames/s and an ETA) is printed at least every
5 s and at the end of each segment. Nothing is written until the config and the session have been validated.

## Configuration

All parameters live in a YAML file, which `process` always requires. Create a complete,
commented one with `speed-trap init-config cfg.yaml`; every key states its unit and valid range.
Unknown keys and out-of-range values are errors that name the dotted key.
`speed-trap check-config --config cfg.yaml` prints the effective config, including `--set`
overrides (values parsed as YAML).

| Section | What it sets |
| --- | --- |
| `camera` | intrinsics (optional OpenCV matrix + distortion) and pose (height, pitch, roll, yaw) |
| `radar` | position relative to the camera, boresight, field of view, range, angle sign, direction byte, clock offset, report hold time |
| `road` | lane heading (for the optional cosine correction) |
| `detector` | backend (`rfdetr`, `ultralytics`, `ground_truth`), weights, device, thresholds, classes, region of interest, `every_n_frames` |
| `tracker` | ByteTrack thresholds and the lost-track timeout in seconds |
| `fusion` | gates, ambiguity ratio, `max_range_m` (30 m), optional lane boundaries, reliability rules |
| `speed` | limit (with optional per-direction overrides), cosine correction (off) |
| `capture` | on/off, consecutive over-limit reports, crop margin, full frame, JPEG quality |
| `overlay`, `lite` | what to draw, colors, pillar height |
| `output`, `synth` | output directory, synthetic session parameters |

## Outputs

Under `--out` (default `out/<session_id>/`):

| File | Content |
| --- | --- |
| `NNNN.overlay.mp4` | annotated H.264 video per segment, variable frame rate from the frame timestamps, so playback runs in real time despite dropped frames |
| `tracks.jsonl` | one line per finished track (or radar subject in lite mode): id, class, first/last time, direction, detected boxes, `road` (per-box back-projected road position, null where it fails), `radar_road` (radar road positions of the matches), radar matches, `reliable_ever`, max/mean speed |
| `captures/NNNN_t<id>_<speed>kmh.jpg` | full-resolution crop of the vehicle, without overlay (lite mode: the overlay frame) |
| `captures/NNNN_t<id>_<speed>kmh.json` | sidecar: direction, limit, speed max/mean/at capture, frame index, sensor timestamp, wall-clock time (flagged unreliable without NTP), box, detector confidence, the full radar history, config hash |
| `run.json` | effective config and its hash, package versions, model, session path, start/end time, counters (frames, scans, tracks, matches, captures), warnings, and `error` if the run failed |

## Calibration

Fusion is only as good as the geometry, so set it up before trusting full-mode captures.

1. **Camera intrinsics.** Calibrate the OV5647 with any OpenCV checkerboard script and paste
   the camera matrix and distortion coefficients into `camera.intrinsics`, with the resolution
   you calibrated at (the matrix is scaled to the video size; the aspect ratio must match).
   Without intrinsics, a pinhole from the sensor's nominal field of view is used and a warning
   is logged; good enough for lite mode, not for captures.
2. **Pose.** Measure the camera height above the road with a tape. Get pitch from a phone
   inclinometer on the camera housing, or adjust it until the drawn range rings fall on road
   markings at known distances. Measure the radar's offset from the camera (x forward, y left,
   z up) and its yaw relative to the camera.
3. **Check the radar conventions in lite mode on the first real recording.** The sign of the
   reported angle and the meaning of the direction byte come from the datasheet and have not
   been confirmed on hardware. Drive past once and process the session with `--mode lite`:
   - pillars on the wrong side of the road mean `radar.angle_sign` should be flipped;
   - approaching vehicles drawn as going (▲ instead of ▼) mean `radar.approaching_byte`
     should be flipped;
   - pillars running consistently ahead of or behind the vehicles point to
     `radar.clock_offset_ms`.

## Known limits

- **30 m fusion range.** Matching, colored tracks and captures stop at `fusion.max_range_m`.
  Farther radar targets are drawn as thin unlabelled pillars (in both modes) but never
  attached to a vehicle or captured. Widening it needs a
  well-verified camera pose.
- **Flat road.** Back-projection assumes the road is a plane; slopes and crowns shift the
  road positions and with them the gates.
- **Radar quantisation.** The LD2451 reports integer metres, degrees and km/h, and up to five
  targets per report. Gates allow for it, but two vehicles within about a metre and a degree
  of each other cannot be told apart; such matches are dropped as ambiguous.
- **Radial speed.** The radar measures the speed component along its line of sight, which
  reads slightly low. The optional cosine correction is off so captures never overstate.
- **Stationary vehicles** are not reported by the radar and stay gray.
- **Offline only.** v1 is a sequential loop; `detector.every_n_frames` is the speed knob.
  Live processing (Jetson or a stream receiver) and licence-plate reading are future work.

## Development

```sh
uv sync --extra rfdetr     # or plain `uv sync` without detector tests
uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

The end-to-end tests generate a synthetic session and check captures, tracks and the
stationary vehicle against its ground truth, in both lite and full mode.

## Licence

MIT, see [LICENSE](LICENSE). The default install is permissive end to end (RF-DETR is
Apache-2.0).

The optional `yolo` extra installs [Ultralytics](https://github.com/ultralytics/ultralytics),
which is licensed under **AGPL-3.0**. If you install and use that extra, its AGPL-3.0 terms
apply to the combined work. Fonts in `speed_trap/render/fonts` are under the SIL Open Font
License (`OFL-*.txt` beside them).
