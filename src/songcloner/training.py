"""Train a voice model from recordings the user provides."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from songcloner import engine, voices
from songcloner.paths import Paths

AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"})
SAMPLE_RATES = (32000, 40000, 48000)


class TrainingError(Exception):
    """A problem with a training request that the user can act on."""


def find_recordings(dataset_dir: Path) -> list[Path]:
    """All audio files under ``dataset_dir``, in a stable order."""
    if not dataset_dir.is_dir():
        raise TrainingError(f"Dataset folder not found: {dataset_dir}")
    recordings = sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not recordings:
        raise TrainingError(
            f"No audio files ({', '.join(sorted(AUDIO_SUFFIXES))}) in {dataset_dir}"
        )
    return recordings


def train(
    name: str,
    dataset_dir: Path,
    *,
    paths: Paths,
    epochs: int = 200,
    batch_size: int = 8,
    sample_rate: int = 40000,
    isolate_vocals: bool = False,
    save_interval: int = 10,
    detect_overtraining: bool = True,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> voices.Voice:
    """Train voice ``name`` from the recordings in ``dataset_dir``.

    Recordings should be clean solo vocals of one singer. Pass
    ``isolate_vocals=True`` when they are full songs: instruments, backing
    vocals and reverb are then stripped first.
    """
    voices.validate_name(name)
    if sample_rate not in SAMPLE_RATES:
        raise TrainingError(f"Sample rate must be one of {SAMPLE_RATES}, got {sample_rate}.")
    if epochs < 1 or batch_size < 1:
        raise TrainingError("Epochs and batch size must be at least 1.")
    exists = (paths.voices / name).exists()
    if exists and not force:
        raise TrainingError(f"Voice {name!r} already exists. Use --force to retrain it.")
    recordings = find_recordings(dataset_dir.expanduser().resolve())

    engine.prepare(paths)
    engine.ensure_base_models(training_sample_rate=sample_rate, log=log)
    if isolate_vocals:
        engine.ensure_separators(log=log)
        isolated = []
        for number, recording in enumerate(recordings, start=1):
            log(f"Isolating lead vocal {number}/{len(recordings)}: {recording.name}")
            work_dir = paths.temp / "isolate" / name / f"{number:03d}"
            isolated.append(engine.isolate_lead_vocal(recording, work_dir))
        recordings = isolated

    # Move an existing voice aside rather than deleting it, so a failed
    # training run does not cost the user the model they already had.
    backup: Path | None = None
    if exists:
        backup = paths.temp / "replaced-voices" / name
        shutil.rmtree(backup, ignore_errors=True)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(paths.voices / name, backup)
    try:
        engine.train_voice(
            name,
            recordings,
            epochs=epochs,
            batch_size=batch_size,
            sample_rate=sample_rate,
            save_interval=save_interval,
            detect_overtraining=detect_overtraining,
            log=log,
        )
        voice = voices.get_voice(paths, name)
    except BaseException:
        if backup is not None:
            shutil.rmtree(paths.voices / name, ignore_errors=True)
            shutil.move(backup, paths.voices / name)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
    return voice
