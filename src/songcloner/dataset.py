"""Build a training dataset: the singer's dry lead vocal, isolated from each song.

The dataset decides most of a voice model's quality, so this is a separate step
from training: build it, read the report, drop anything that is not the singer
alone (duets, chorus-heavy tracks), then train.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from songcloner import engine, voices
from songcloner.convert import is_url
from songcloner.paths import Paths
from songcloner.training import AUDIO_SUFFIXES

# Rough guide for RVC voice models: below the minimum the voice is unstable;
# from about twenty minutes of singing it is usually good.
MIN_SUNG_MINUTES = 10.0
GOOD_SUNG_MINUTES = 20.0


class DatasetError(Exception):
    """A problem with a dataset request that the user can act on."""


@dataclass(frozen=True)
class Clip:
    source: str
    path: Path
    seconds: float
    sung_seconds: float


@dataclass
class DatasetReport:
    directory: Path
    clips: list[Clip] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def sung_minutes(self) -> float:
        """Minutes of singing in the clips built by this run."""
        return sum(clip.sung_seconds for clip in self.clips) / 60

    def verdict(self) -> str:
        minutes = self.sung_minutes
        if not self.clips:
            return "No new clips were built."
        if minutes < MIN_SUNG_MINUTES:
            return (
                f"{minutes:.1f} minutes of singing: too little for a good voice. "
                f"Aim for {GOOD_SUNG_MINUTES:.0f} or more."
            )
        if minutes < GOOD_SUNG_MINUTES:
            return (
                f"{minutes:.1f} minutes of singing: usable, but "
                f"{GOOD_SUNG_MINUTES:.0f} or more gives a noticeably better voice."
            )
        return f"{minutes:.1f} minutes of singing: enough for a good voice."


# Characters that are unsafe or reserved in file names. Everything else is
# kept: titles are often in Devanagari, whose vowel signs are not "word
# characters", so an allow-list of \w would silently mangle them.
_UNSAFE_IN_FILE_NAMES = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def slug(title: str) -> str:
    """A safe file name derived from a song title."""
    cleaned = _UNSAFE_IN_FILE_NAMES.sub("", title).strip().lstrip(".").strip()
    return re.sub(r"\s+", "_", cleaned)[:80] or "song"


def expand_sources(sources: Sequence[str]) -> list[str]:
    """Turn files, folders and URLs into a flat list of files and URLs."""
    expanded: list[str] = []
    for source in sources:
        if is_url(source):
            expanded.append(source)
            continue
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            found = sorted(
                str(item)
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES
            )
            if not found:
                raise DatasetError(f"No audio files in {path}")
            expanded.extend(found)
        elif path.is_file():
            expanded.append(str(path))
        else:
            raise DatasetError(f"Not found: {path}")
    if not expanded:
        raise DatasetError("Give at least one song: a file, a folder or a YouTube URL.")
    return expanded


def build_dataset(
    name: str,
    sources: Sequence[str],
    *,
    paths: Paths,
    datasets_dir: Path | None = None,
    quality: str = engine.DEFAULT_QUALITY,
    log: Callable[[str], None] = print,
) -> DatasetReport:
    """Isolate the lead vocal of every song into ``<datasets_dir>/<name>/``.

    Songs already in the dataset are skipped, so the command can be repeated to
    add more. One song failing (for example a blocked download) does not stop
    the rest; failures are listed in the report.
    """
    voices.validate_name(name)
    engine.vocal_separator(quality)
    songs = expand_sources(sources)
    directory = (datasets_dir or Path.cwd() / "datasets") / name
    directory.mkdir(parents=True, exist_ok=True)

    engine.prepare(paths)
    engine.ensure_separators(quality, log=log)
    report = DatasetReport(directory=directory)
    for number, source in enumerate(songs, start=1):
        log(f"[{number}/{len(songs)}] {source}")
        # For a local file the clip's name is known up front, so a song that is
        # already in the dataset costs nothing. A URL's title is only known
        # once it has been fetched; that case is caught after isolation.
        if not is_url(source) and (directory / f"{slug(Path(source).stem)}.flac").exists():
            report.skipped.append(source)
            log("  already in the dataset")
            continue
        try:
            lead, title = engine.isolate_lead_vocal(source, quality, log)
            destination = directory / f"{slug(title)}.flac"
            if destination.exists():
                report.skipped.append(source)
                log(f"  already in the dataset: {destination.name}")
                continue
            seconds, sung_seconds = engine.export_vocal(lead, destination)
        except Exception as error:  # noqa: BLE001 - one bad song must not lose the others
            report.failures.append((source, f"{type(error).__name__}: {error}"))
            log(f"  failed: {type(error).__name__}: {error}")
            continue
        report.clips.append(Clip(source, destination, seconds, sung_seconds))
        log(f"  {destination.name}: {sung_seconds / 60:.1f} min of singing")
    return report
