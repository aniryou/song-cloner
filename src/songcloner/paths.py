"""Where song-cloner keeps its files, and how the engine is told about them."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HOME_ENV = "SONGCLONER_HOME"
OUTPUTS_ENV = "SONGCLONER_OUTPUTS"


@dataclass(frozen=True)
class Paths:
    """All directories song-cloner reads or writes.

    ``home`` holds engine data (base models, voice models, intermediate audio).
    ``outputs`` is where finished songs are copied for the user.
    """

    home: Path
    outputs: Path

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def voices(self) -> Path:
        # Layout expected by the engine: <models>/rvc/voice_models/<name>/
        return self.models / "rvc" / "voice_models"

    @property
    def audio(self) -> Path:
        return self.home / "audio"

    @property
    def temp(self) -> Path:
        return self.home / "temp"

    @property
    def config(self) -> Path:
        return self.home / "config"

    @property
    def logs(self) -> Path:
        return self.home / "logs"


def get_paths() -> Paths:
    """Resolve directories from the environment, defaulting to the current directory."""
    home = Path(os.environ.get(HOME_ENV) or Path.cwd() / "data").resolve()
    outputs = Path(os.environ.get(OUTPUTS_ENV) or Path.cwd() / "outputs").resolve()
    return Paths(home=home, outputs=outputs)


def engine_env(paths: Paths) -> dict[str, str]:
    """Environment variables that point the engine (ultimate-rvc) at our directories.

    The engine reads these once, when it is first imported.
    """
    return {
        "URVC_MODELS_DIR": str(paths.models),
        "URVC_VOICE_MODELS_DIR": str(paths.voices),
        "URVC_AUDIO_DIR": str(paths.audio),
        "URVC_TEMP_DIR": str(paths.temp),
        "URVC_CONFIG_DIR": str(paths.config),
        "URVC_LOGS_DIR": str(paths.logs),
    }
