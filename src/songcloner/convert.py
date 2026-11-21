"""Convert a song: separate the lead vocal, re-voice it, and mix it back."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from songcloner import engine, voices
from songcloner.paths import Paths


class ConvertError(Exception):
    """A problem with a conversion request that the user can act on."""


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def resolve_source(source: str) -> str:
    """Return a URL unchanged, or the absolute path of an existing local file."""
    if is_url(source):
        return source
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ConvertError(f"Song file not found: {path}")
    return str(path)


def unique_destination(directory: Path, filename: str) -> Path:
    """A path in ``directory`` for ``filename`` that does not overwrite an existing file."""
    candidate = directory / filename
    stem, suffix = candidate.stem, candidate.suffix
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem} ({counter}){suffix}"
        counter += 1
    return candidate


def convert_song(
    source: str,
    voice: str,
    *,
    paths: Paths,
    quality: str = engine.DEFAULT_QUALITY,
    output_dir: Path | None = None,
    log: Callable[[str], None] = print,
    **options: Any,
) -> Path:
    """Convert ``source`` (file or YouTube URL) to ``voice``; return the finished song.

    ``quality`` picks the vocal separator ("high" or "standard"). ``options``
    are passed to :func:`songcloner.engine.run_cover`.
    """
    voices.get_voice(paths, voice)  # fail before any heavy work
    engine.vocal_separator(quality)
    resolved = resolve_source(source)
    engine.prepare(paths)
    engine.ensure_base_models(log=log)
    engine.ensure_separators(quality, log=log)

    log(f"Converting with voice {voice!r} on {engine.device_summary()}...")
    cover = engine.run_cover(resolved, voice, quality=quality, log=log, **options)

    destination_dir = output_dir or paths.outputs
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = unique_destination(destination_dir, cover.name)
    shutil.copy2(cover, destination)
    return destination
