"""Voice models: a local registry, plus optional sync with a private Hugging Face repo.

A voice is a directory ``<voices>/<name>/`` holding one ``.pth`` model file and,
optionally, one ``.index`` file. That is the layout the engine expects.

Security note: a ``.pth`` file is a Python pickle. Loading one runs code from
it, so only add voice models you trained yourself or fully trust.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from songcloner.paths import Paths

HF_REPO_ENV = "SONGCLONER_HF_REPO"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class VoiceError(Exception):
    """A problem with a voice model that the user can act on."""


@dataclass(frozen=True)
class Voice:
    name: str
    model: Path
    index: Path | None


def validate_name(name: str) -> str:
    if not _NAME_PATTERN.match(name):
        raise VoiceError(
            f"Invalid voice name {name!r}: use letters, digits, '-', '_' and '.', "
            "starting with a letter or digit."
        )
    return name


def _read_voice(directory: Path) -> Voice | None:
    models = sorted(directory.glob("*.pth"))
    if not models:
        return None
    indexes = sorted(directory.glob("*.index"))
    return Voice(name=directory.name, model=models[0], index=indexes[0] if indexes else None)


def list_voices(paths: Paths) -> list[Voice]:
    if not paths.voices.is_dir():
        return []
    found = (_read_voice(entry) for entry in sorted(paths.voices.iterdir()) if entry.is_dir())
    return [voice for voice in found if voice is not None]


def get_voice(paths: Paths, name: str) -> Voice:
    validate_name(name)
    voice = _read_voice(paths.voices / name) if (paths.voices / name).is_dir() else None
    if voice is None:
        available = ", ".join(v.name for v in list_voices(paths)) or "none"
        raise VoiceError(
            f"No voice named {name!r} (available: {available}). "
            "Add one with `songcloner voices add` or `songcloner voices pull`."
        )
    return voice


def _split_files(files: Sequence[Path]) -> tuple[Path, Path | None]:
    models = [f for f in files if f.suffix == ".pth"]
    indexes = [f for f in files if f.suffix == ".index"]
    others = [f for f in files if f.suffix not in {".pth", ".index"}]
    if others:
        raise VoiceError(f"Unsupported file(s): {', '.join(f.name for f in others)}.")
    if len(models) != 1 or len(indexes) > 1:
        raise VoiceError("Provide exactly one .pth file and at most one .index file.")
    for path in files:
        if not path.is_file():
            raise VoiceError(f"File not found: {path}")
    return models[0], (indexes[0] if indexes else None)


def add_voice(paths: Paths, name: str, files: Sequence[Path], *, force: bool = False) -> Voice:
    """Copy a model (.pth) and optional index (.index) into the voice registry."""
    validate_name(name)
    model, index = _split_files(files)
    target = paths.voices / name
    if target.exists():
        if not force:
            raise VoiceError(f"Voice {name!r} already exists. Use --force to replace it.")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(model, target / model.name)
    if index is not None:
        shutil.copy2(index, target / index.name)
    return get_voice(paths, name)


def remove_voice(paths: Paths, name: str) -> None:
    voice = get_voice(paths, name)
    shutil.rmtree(voice.model.parent)


def resolve_repo(repo: str | None) -> str:
    repo = repo or os.environ.get(HF_REPO_ENV)
    if not repo:
        raise VoiceError(
            f"No Hugging Face repository given. Pass --repo <user>/<name> or set {HF_REPO_ENV}."
        )
    if repo.count("/") != 1:
        raise VoiceError(f"Repository must look like <user>/<name>, got {repo!r}.")
    return repo


def pull_voice(paths: Paths, name: str, repo: str | None = None, *, force: bool = False) -> Voice:
    """Download ``<name>/*.pth`` and ``<name>/*.index`` from a Hugging Face model repo."""
    from huggingface_hub import snapshot_download

    validate_name(name)
    repo_id = resolve_repo(repo)
    if (paths.voices / name).exists() and not force:
        raise VoiceError(f"Voice {name!r} already exists. Use --force to replace it.")
    with tempfile.TemporaryDirectory(prefix="songcloner-pull-") as scratch:
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{name}/*.pth", f"{name}/*.index"],
            local_dir=scratch,
        )
        downloaded = sorted(
            p for p in (Path(scratch) / name).glob("*") if p.suffix in {".pth", ".index"}
        )
        if not downloaded:
            raise VoiceError(f"Repository {repo_id} has no voice files under {name}/.")
        return add_voice(paths, name, downloaded, force=force)


def push_voice(paths: Paths, name: str, repo: str | None = None) -> str:
    """Upload a voice to a *private* Hugging Face model repo, creating it if needed."""
    from huggingface_hub import HfApi

    voice = get_voice(paths, name)
    repo_id = resolve_repo(repo)
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
    if not api.repo_info(repo_id, repo_type="model").private:
        raise VoiceError(
            f"Repository {repo_id} is public. Refusing to upload a voice model to it; "
            "make it private on huggingface.co first."
        )
    for path in (voice.model, voice.index):
        if path is not None:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"{name}/{path.name}",
                repo_id=repo_id,
                repo_type="model",
            )
    return repo_id
