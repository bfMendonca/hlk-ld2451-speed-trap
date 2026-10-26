# Radar side panel (bird's-eye summary) — design

## Intent

The radar cone and range rings drawn over the camera image are hard to read and say nothing
the numbers don't. Replace them with a strip to the right of the output video: a top-down
view of what the radar itself reports, plus a table of the current readings. It is a radar
summary only — it does not depend on camera tracks or on fusion matches, so it still works
when matching fails (as on far or receding vehicles).

## Goals

- A panel on the right of the overlay video, same height as the frame, about 1/3 of its width
  (432 px for a 1296 px frame), added to the output video, not drawn over the image.
- Top half: fan-shaped top-down view. Radar at the bottom, range upward, true angle. Each
  reading is a dot; dots stay 3.5 s and fade with age. Colour by direction: coming / going
  (existing overlay colours).
- Bottom half: table of the current readings (speed, ground range, angle, SNR, direction),
  nearest first.
- Readings below a speed threshold are hidden, on by default (15 km/h), configurable. This
  removes the 3 km/h clutter without touching what the pipeline or fusion see.
- The camera-side cone and rings are removed from the picture (`overlay.show.wedge` default
  becomes false; the option stays).

## Non-goals / anti-goals

- No change to detection, tracking, fusion, captures or `tracks.jsonl`. Captures and crops
  keep coming from the frame without the panel.
- The panel must not use camera calibration (pose, intrinsics, boresight): it shows the radar
  as reported. A bad calibration must not move a dot.
- Not a failure: a panel that looks good but lies about time. Fading follows video time
  (frame timestamps), so reprocessing a session gives the same video.

## Constraints

- `VideoSink` writes frames of exactly the size it was opened with, even width and height
  (yuv420p). The sink is opened with `frame width + panel width`; the panel width is rounded
  to an even number.
- `Renderer` keeps assuming its image is the camera frame. It is not changed beyond the
  wedge default.
- Reuse the renderer's fonts and colours (`render/draw.py`) instead of a second text stack.

## Approaches considered

1. **New `render/radar_panel.py`, composed next to the rendered frame in the pipeline** — chosen.
2. Draw the panel inside `Renderer` on a wider canvas — rejected: the renderer's projection,
   wedge, tags and HUD all assume camera size.
3. Compose the panel in a second pass after processing and join the videos with ffmpeg —
   rejected: extra tool, extra step, two videos to keep in sync.

Chosen with the user on 2026-10-25.

## Decisions made with the user

- Rolling trail of 3-4 s that fades (3.5 s default).
- Option B: fan view without labels on the dots, plus a table of all properties of the
  current readings.
- Speed filter on by default, configurable.

## Decisions made by me (to check)

- **Handedness matches the video.** The earlier HTML mockup drew negative angles to the left;
  in the camera image negative angles are on the right (`angle_sign: 1`, negative = right).
  The panel mirrors accordingly so a car on the right of the video is on the right of the
  panel; the sign follows `radar.angle_sign`.
- Coverage 0-50 m and ±45°; readings outside are not drawn. Config: `max_range_m`, `fov_half_deg`.
- "Current" for the table = readings from the last 0.3 s of video time; the newest dot of
  each reading is drawn larger. Up to 6 table rows; extra rows are cut, nearest kept.
- Range shown is ground range (`sqrt(distance² − radar_height²)` with the config's radar
  height), as the rest of the code does; the table prints the radar's own slant distance as
  reported. The fan uses ground range.
- New config under `overlay.radar_panel`: `enabled: true`, `width_frac: 0.333`, `trail_s: 3.5`,
  `min_speed_kmh: 15`, `max_range_m: 50`, `fov_half_deg: 45`, `max_rows: 6`.
- Radar readings come from the `scans` already attached to each frame bundle.
- Panel appears in every overlay video the pipeline writes (not only `--capture` runs).

## Found while reading the code

- `CaptureSink.offer` receives the rendered frame (`pipeline.py`, right before
  `self._video.write`). Attach the panel after `offer`, just before `VideoSink.write`, so
  capture images stay panel-free.
- `test_render.py::test_wedge_changes_ground_not_sky` relies on the wedge being on by default.
  It must set `show.wedge: true` explicitly once the default flips to false.
- The pipeline's `_current_targets` only keeps the latest scan; the panel keeps its own
  history fed by each frame bundle's `scans`.

## Left to the builder

- Exact layout, fonts, dot sizes, grid line spacing; keep it consistent with the existing
  overlay look.
- Tests: panel unit tests (fading, filter, mirrored x, table order and cut), a pipeline test
  that the output video is `W + panel` wide, and the config defaults.
- README and CHANGELOG entries (both are already modified in the working tree — append, don't
  overwrite).
