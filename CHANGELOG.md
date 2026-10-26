# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- `overlay.show.wedge` now defaults to `false`: the radar cone and range rings no longer cover
  the image unless asked for.
- `speed-trap process --out DIR` is now a parent folder: each run writes
  `DIR/<session_id>/`, or `<session_id>-2/`, `-3/`, ... if that exists, so processing several
  sessions, or the same one twice, never overwrites earlier results.

### Added

- Radar side panel in the overlay video: a top-down fan with a 3.5 s fading trail of the radar
  readings and a table of the current ones (`overlay.radar_panel`: `enabled`, `width_frac`,
  `trail_s`, `min_speed_kmh`, `max_range_m`, `fov_half_deg`, `max_rows`). The output video is
  `width_frac` wider; captures and crops are unchanged.
- `fusion.match_on: azimuth`: a radar reading is matched to the vehicle whose box spans its
  bearing (plus `fusion.gate.angle_margin_deg`); camera range, speed and direction are not used
  and no ground plane is needed, which suits low cameras where monocular range is unreliable.
  Two vehicles on one bearing are still dropped as ambiguous. Default stays `range_azimuth`.
- `speed-trap process`: overlay video, `tracks.jsonl`, speeder captures and `run.json` from a
  `radar-cam-recorder` session, in full mode (detector + ByteTrack + radar fusion) or lite
  mode (radar pillars, no detector).
- Detector backends: RF-DETR (default), Ultralytics YOLO (optional, AGPL-3.0) and a
  ground-truth backend for synthetic sessions.
- Radar fusion on the road plane: distance-aware gates, Hungarian assignment, ambiguity guard,
  30 m trusted range, direction and speed consistency, matching at the radar report's own
  time, and a verified reading required before any capture.
- Overlay: corner brackets and speed tags, radar coverage wedge with range rings, HUD, lite
  pillars.
- YAML configuration with strict validation, `--set` overrides, `init-config` and
  `check-config`.
- `speed-trap synth`: synthetic sessions in the recorder's format with LD2451 radar bytes and
  ground truth, used by the end-to-end tests.
- GitHub CI (lint, `mypy --strict`, tests on Python 3.10-3.13), issue and pull request
  templates, `CONTRIBUTING.md`, `SECURITY.md`.
