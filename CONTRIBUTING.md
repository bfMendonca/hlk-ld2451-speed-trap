# Contributing

Thanks for your interest in `speed-trap`! Bug reports, results from real installations and
pull requests are all welcome.

## Setup

The project uses [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/bfMendonca/hlk-ld2451-speed-trap.git
cd hlk-ld2451-speed-trap
uv sync --extra rfdetr      # or plain `uv sync` to skip the detector
```

The [`ld2451`](https://github.com/bfMendonca/ld2451) driver and the
[`radar-cam-recorder`](https://github.com/bfMendonca/radar-cam-recorder) session reader are
installed from GitHub. To work on them at the same time, clone them next to this repo and
install them over the locked copies (the next `uv sync` puts the GitHub versions back):

```sh
uv pip install -e ../ld2451 -e ../radar-cam-recorder
```

## Before opening a pull request

```sh
uv run pytest                      # no hardware and no model needed
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

- **Never wrong beats always right.** Changes to fusion, the speeding monitor or captures must
  not make it possible to overstate a speed or attach a reading to the wrong vehicle. Add an
  end-to-end case on a synthetic session (`speed_trap.synth.generate_session`, with custom
  vehicles or frame gaps if needed) that checks the captures against the ground truth.
- Keep the core stages (`types`, `geometry`, `tracker`, `fusion`, `lite`, `monitor`) free of
  file I/O, decoding and drawing, and causal (no reading ahead in time).
- New config keys go in `speed_trap/config.py` with bounds and in
  `speed_trap/default_config.yaml` with a comment giving the unit and range.
- Detector backends import their library lazily: the base install must not import torch.
- Add an entry under *Unreleased* in `CHANGELOG.md`.

## Real recordings

Recordings come from [`radar-cam-recorder`](https://github.com/bfMendonca/radar-cam-recorder).
If you report a problem seen on real data, the session's `session.json`, the `run.json` of the
failing run and, if you can share it, a short segment help a lot. Remember that recordings may
show people, vehicles and number plates: share only footage you are allowed to share.
