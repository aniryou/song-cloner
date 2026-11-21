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
    """Point the engine at our directories. Call before anything else in this module.

    The first call decides the directories (``paths``, or the defaults for the
    current directory). Later calls without ``paths`` just return that choice,
    whatever the current directory has become since; asking for different
    directories later is an error, because the engine cannot be re-pointed.
    """
    global _prepared_for
    if _prepared_for is not None:
        if paths is not None and paths != _prepared_for:
            raise EngineError("The engine was already prepared with different directories.")
        return _prepared_for
    paths = paths or get_paths()
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


# Models for the first and most audible separation step: vocals from music.
# "high" was measured against "standard" on a MUSDB18 track with ground-truth
# stems: about 1 dB better on both the vocal and the instrumental (less of the
# original singer left in the backing track), at about twice the CPU time.
VOCAL_SEPARATORS = {
    "standard": "UVR-MDX-NET-Voc_FT.onnx",
    "high": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
}
DEFAULT_QUALITY = "high"


def vocal_separator(quality: str) -> str:
    try:
        return VOCAL_SEPARATORS[quality]
    except KeyError:
        choices = ", ".join(sorted(VOCAL_SEPARATORS))
        raise EngineError(f"Unknown quality {quality!r}; choose one of: {choices}.") from None


def ensure_separators(quality: str = DEFAULT_QUALITY, log: Callable[[str], None] = print) -> None:
    """Download the vocal-separation models for ``quality``, if missing.

    The three models every quality level uses total about 180 MB; the "high"
    vocal separator adds about 640 MB.
    """
    model = vocal_separator(quality)
    paths = prepare()
    from ultimate_rvc.core.generate.song_cover import initialize_audio_separator

    log("Checking vocal-separation models...")
    initialize_audio_separator()
    model_dir = paths.models / "audio_separator"
    if not (model_dir / model).is_file():
        from audio_separator.separator import Separator

        log(f"Downloading {model}...")
        Separator(model_file_dir=str(model_dir), log_level=40).download_model_files(model)


def device_summary() -> str:
    """Describe the compute device PyTorch will use."""
    prepare()
    import torch

    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "Apple GPU (MPS)"
    return f"CPU ({os.cpu_count()} cores)"


def _split_song(source: str, quality: str, log: Callable[[str], None]) -> dict[str, Path]:
    """Fetch a song and split it into instrumental, backing vocals and dry lead vocal.

    Mirrors the first half of the engine's own ``run_pipeline``, with the vocal
    separator chosen by ``quality``. The engine caches every stem in the song's
    directory, so repeating a call for the same song is cheap.
    """
    from ultimate_rvc.core.generate import song_cover
    from ultimate_rvc.typing_extra import SegmentSize, SeparationModel

    model = vocal_separator(quality)
    log("Retrieving song...")
    song, song_dir = song_cover.retrieve_song(source)
    log(f"Separating vocals from music ({model})...")
    vocals, instrumental = song_cover.separate_audio(song, song_dir, model, SegmentSize.SEG_512)
    log("Separating lead vocal from backing vocals...")
    backing, lead = song_cover.separate_audio(
        vocals, song_dir, SeparationModel.UVR_MDX_NET_KARA_2, SegmentSize.SEG_512
    )
    log("Removing reverb from the lead vocal...")
    _reverb, dry_lead = song_cover.separate_audio(
        lead, song_dir, SeparationModel.REVERB_HQ_BY_FOXJOY, SegmentSize.SEG_256
    )
    return {
        "song": Path(song),
        "song_dir": Path(song_dir),
        "instrumental": Path(instrumental),
        "backing": Path(backing),
        "lead": Path(dry_lead),
    }


def run_cover(
    source: str,
    voice: str,
    *,
    quality: str = DEFAULT_QUALITY,
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
    log: Callable[[str], None] = print,
) -> Path:
    """Run the full cover pipeline and return the path of the mixed song.

    ``source`` is a local audio file or a YouTube URL. ``octaves`` shifts only
    the converted vocal; ``semitones`` shifts vocal, instrumental and backing
    vocals together (a key change).
    """
    prepare()
    from ultimate_rvc.core.generate import song_cover
    from ultimate_rvc.core.generate.common import convert
    from ultimate_rvc.typing_extra import AudioExt, F0Method, RVCContentType

    stems = _split_song(source, quality, log)
    song_dir = stems["song_dir"]
    log("Converting the lead vocal...")
    converted = convert(
        audio_track=stems["lead"],
        directory=song_dir,
        model_name=voice,
        n_octaves=octaves,
        n_semitones=semitones,
        f0_method=F0Method(f0_method),
        index_rate=index_rate,
        protect_rate=protect_rate,
        proposed_pitch=auto_pitch,
        clean_audio=clean_vocals,
        content_type=RVCContentType.VOCALS,
    )
    log("Mixing...")
    effected = song_cover.postprocess(converted, song_dir)
    instrumental = song_cover.pitch_shift(stems["instrumental"], song_dir, semitones)
    backing = song_cover.pitch_shift(stems["backing"], song_dir, semitones)
    cover = song_cover.mix_song(
        [(effected, main_gain), (instrumental, inst_gain), (backing, backup_gain)],
        song_dir,
        44100,
        AudioExt(output_format),
        output_name,
    )
    return Path(cover)


def isolate_lead_vocal(
    source: str, quality: str = DEFAULT_QUALITY, log: Callable[[str], None] = print
) -> tuple[Path, str]:
    """Strip music, backing vocals and reverb from a song.

    ``source`` is a local audio file or a YouTube URL. Returns the dry lead
    vocal and the song's title (its file name without the extension).
    """
    prepare()
    stems = _split_song(source, quality, log)
    # The engine stores the original as "00_<title>.wav".
    return stems["lead"], stems["song"].stem.removeprefix("00_")


# A frame counts as singing when it is louder than this, relative to full scale,
# and also louder than a tenth of the clip's typical loud level. Isolated vocals
# are never perfectly silent between phrases, so an absolute floor is not enough.
_SUNG_FLOOR_DB = -45.0


def measure_singing(vocal: Path) -> tuple[float, float]:
    """Return a vocal clip's length and how much of it is singing, in seconds."""
    prepare()
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(str(vocal), always_2d=True)
    mono = audio.mean(axis=1)
    frame = max(1, int(sample_rate * 0.05))
    count = len(mono) // frame
    if count == 0:
        return len(mono) / sample_rate, 0.0
    rms = np.sqrt((mono[: count * frame].reshape(count, frame) ** 2).mean(axis=1))
    threshold = max(10 ** (_SUNG_FLOOR_DB / 20), 0.1 * float(np.percentile(rms, 95)))
    return len(mono) / sample_rate, float((rms > threshold).sum() * frame / sample_rate)


def export_vocal(vocal: Path, destination: Path) -> tuple[float, float]:
    """Save ``vocal`` as FLAC; return its length and how much of it is singing, in seconds."""
    prepare()
    import soundfile as sf

    audio, sample_rate = sf.read(str(vocal), always_2d=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), audio, sample_rate, format="FLAC")
    return measure_singing(destination)


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
