"""Bridge to the engine, ultimate-rvc.

This is the only module that imports ``ultimate_rvc``. Every import of it is
lazy and happens after :func:`prepare`, because the engine reads its directory
settings from the environment the moment it is first imported.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from songcloner.paths import Paths, engine_env, get_paths

INSTALL_HINT = (
    "The conversion engine is not installed. Install it with:\n"
    '    uv sync --extra engine        (or: pip install -e ".[engine]")'
)

# Smallest plausible size for a model file. The upstream downloader writes
# whatever the server returns, so an error page can masquerade as a model.
_MIN_MODEL_BYTES = 1024


class EngineNotInstalledError(RuntimeError):
    """Raised when ultimate-rvc is not importable."""


class EngineError(RuntimeError):
    """Raised when the engine cannot be set up as expected."""


_prepared_for: Paths | None = None


def prepare(paths: Paths | None = None) -> Paths:
    """Point the engine at our directories. Call before anything else in this module."""
    global _prepared_for
    paths = paths or get_paths()
    if _prepared_for is not None:
        if _prepared_for != paths:
            raise EngineError("The engine was already prepared with different directories.")
        return paths
    if "ultimate_rvc" in sys.modules:
        raise EngineError("ultimate_rvc was imported before songcloner.engine.prepare().")

    for key, value in engine_env(paths).items():
        os.environ[key] = value
    # The engine's own log files are not useful to us and add startup cost.
    os.environ.setdefault("URVC_NO_LOGGING", "1")
    for directory in (paths.models, paths.voices, paths.audio, paths.temp, paths.config):
        directory.mkdir(parents=True, exist_ok=True)
    # The engine downloads sample celebrity voices on first start unless this
    # marker exists. song-cloner only uses voices the user adds explicitly.
    (paths.models / ".initialized").touch()

    _require_engine()
    _use_system_sox()
    _prepared_for = paths
    return paths


def _require_engine() -> None:
    import importlib.util

    if importlib.util.find_spec("ultimate_rvc") is None:
        raise EngineNotInstalledError(INSTALL_HINT)


def _use_system_sox() -> None:
    """Make the engine use an installed ``sox`` instead of downloading its own copy.

    ultimate-rvc fetches a private sox binary the first time it needs one. On
    machines that already have sox (``apt install sox``, ``brew install sox``)
    that download is unnecessary, and on restricted networks it fails.
    """
    sox = shutil.which("sox")
    if sox is None:
        return
    try:
        import static_sox.run as static_sox_run
    except ImportError:
        return
    static_sox_run.get_or_fetch_platform_executables_else_raise = lambda *_a, **_k: sox


def _download(url: str, destination: Path) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    if partial.stat().st_size < _MIN_MODEL_BYTES and destination.suffix != ".json":
        partial.unlink()
        raise EngineError(f"Download of {url} returned an implausibly small file.")
    partial.replace(destination)


def ensure_base_models(
    training_sample_rate: int | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Download the base models conversion needs, skipping files already present.

    Conversion needs the pitch predictors and the ContentVec embedder (about
    0.6 GB). Training additionally needs the pretrained generator and
    discriminator for the chosen sample rate; pass ``training_sample_rate``
    (32000, 40000 or 48000) to fetch those too.

    The engine's own downloader fetches every embedder and every pretrained
    model (about 3.3 GB), which is wasteful for a session that only converts.
    """
    prepare()
    from ultimate_rvc.rvc.lib.tools import prerequisites_download as upstream

    try:
        wanted: list[tuple[str, list[str]]] = [
            (folder, list(files)) for folder, files in upstream.models_list
        ]
        wanted += [
            (folder, list(files))
            for folder, files in upstream.embedders_list
            if folder == "embedders/contentvec/"
        ]
        if training_sample_rate is not None:
            tag = f"{training_sample_rate // 1000}k"
            for folder, files in upstream.pretraineds_hifigan_list:
                matching = [name for name in files if tag in name]
                if matching:
                    wanted.append((folder, matching))
        url_base = upstream.url_base
        folders = upstream.folder_mapping_list
    except AttributeError:
        # The engine's internals moved; fall back to its full download.
        log("Downloading the engine's full model set...")
        upstream.prequisites_download_pipeline(
            pretraineds_hifigan=training_sample_rate is not None, models=True, exe=False
        )
        return

    for folder, files in wanted:
        local_folder = folders.get(folder)
        if not local_folder:
            raise EngineError(f"The engine has no local folder for {folder!r}.")
        for name in files:
            destination = Path(local_folder) / name
            if destination.exists() and destination.stat().st_size > 0:
                continue
            log(f"Downloading {folder}{name}...")
            _download(f"{url_base}/{folder}{name}", destination)


def ensure_separators(log: Callable[[str], None] = print) -> None:
    """Download the vocal-separation models (about 180 MB), if missing."""
    prepare()
    from ultimate_rvc.core.generate.song_cover import initialize_audio_separator

    log("Checking vocal-separation models...")
    initialize_audio_separator()


def device_summary() -> str:
    """Describe the compute device PyTorch will use."""
    prepare()
    import torch

    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "Apple GPU (MPS)"
    return f"CPU ({os.cpu_count()} cores)"


def run_cover(
    source: str,
    voice: str,
    *,
    octaves: int = 0,
    semitones: int = 0,
    auto_pitch: bool = False,
    f0_method: str = "rmvpe",
    index_rate: float = 0.3,
    protect_rate: float = 0.33,
    clean_vocals: bool = False,
    main_gain: int = 0,
    inst_gain: int = 0,
    backup_gain: int = 0,
    output_format: str = "mp3",
    output_name: str | None = None,
) -> Path:
    """Run the full cover pipeline and return the path of the mixed song.

    ``source`` is a local audio file or a YouTube URL. ``octaves`` shifts only
    the converted vocal; ``semitones`` shifts vocal, instrumental and backing
    vocals together (a key change).
    """
    prepare()
    from ultimate_rvc.core.generate.song_cover import run_pipeline
    from ultimate_rvc.typing_extra import AudioExt, F0Method

    result = run_pipeline(
        source=source,
        model_name=voice,
        n_octaves=octaves,
        n_semitones=semitones,
        f0_method=F0Method(f0_method),
        index_rate=index_rate,
        protect_rate=protect_rate,
        proposed_pitch=auto_pitch,
        clean_vocals=clean_vocals,
        main_gain=main_gain,
        inst_gain=inst_gain,
        backup_gain=backup_gain,
        output_format=AudioExt(output_format),
        output_name=output_name,
    )
    return Path(result[0])


def isolate_lead_vocal(audio_file: Path, work_dir: Path) -> Path:
    """Strip instruments, backing vocals and reverb from a song; return the lead vocal."""
    prepare()
    from ultimate_rvc.core.generate.song_cover import separate_audio
    from ultimate_rvc.typing_extra import SegmentSize, SeparationModel

    work_dir.mkdir(parents=True, exist_ok=True)
    vocals, _instrumentals = separate_audio(
        audio_file, work_dir, SeparationModel.UVR_MDX_NET_VOC_FT, SegmentSize.SEG_512
    )
    _backing, lead = separate_audio(
        vocals, work_dir, SeparationModel.UVR_MDX_NET_KARA_2, SegmentSize.SEG_512
    )
    _reverb, dry_lead = separate_audio(
        lead, work_dir, SeparationModel.REVERB_HQ_BY_FOXJOY, SegmentSize.SEG_256
    )
    return Path(dry_lead)


def train_voice(
    name: str,
    audio_files: Sequence[Path],
    *,
    epochs: int,
    batch_size: int,
    sample_rate: int,
    save_interval: int,
    detect_overtraining: bool,
    log: Callable[[str], None] = print,
) -> None:
    """Train a voice model from clean vocal recordings and register it as a voice."""
    prepare()
    from ultimate_rvc.core.train.extract import extract_features
    from ultimate_rvc.core.train.prepare import populate_dataset, preprocess_dataset
    from ultimate_rvc.core.train.train import run_training
    from ultimate_rvc.typing_extra import TrainingSampleRate

    log(f"[1/4] Collecting {len(audio_files)} recording(s)...")
    dataset_path = populate_dataset(name, [str(path) for path in audio_files])
    log("[2/4] Slicing and normalising the recordings...")
    preprocess_dataset(name, dataset_path, sample_rate=TrainingSampleRate(sample_rate))
    log("[3/4] Extracting pitch and voice features...")
    extract_features(name)
    log(f"[4/4] Training for {epochs} epoch(s) on {device_summary()}...")
    run_training(
        name,
        num_epochs=epochs,
        batch_size=batch_size,
        save_interval=save_interval,
        detect_overtraining=detect_overtraining,
        upload_model=True,
        upload_name=name,
    )
