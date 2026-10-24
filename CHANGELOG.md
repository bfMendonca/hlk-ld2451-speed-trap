# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
