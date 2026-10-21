from __future__ import annotations

from pathlib import Path

import pytest

from speed_trap.config import Config
from speed_trap.synth import generate_session


@pytest.fixture(scope="session")
def synth_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_session(tmp_path_factory.mktemp("synth"), Config(), 20.0, 1)
