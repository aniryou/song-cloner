"""Command-line interface: ``songcloner``."""

from __future__ import annotations

import platform
import shutil
from pathlib import Path
from typing import Annotated

import typer

from songcloner import __version__, convert, engine, training, voices
from songcloner.paths import get_paths

app = typer.Typer(
    help="Re-voice the lead vocal of a song with an RVC voice model (personal use).",
    no_args_is_help=True,
    add_completion=False,
)
voices_app = typer.Typer(help="Manage voice models.", no_args_is_help=True)
app.add_typer(voices_app, name="voices")

USER_ERRORS = (
    voices.VoiceError,
    convert.ConvertError,
    training.TrainingError,
    engine.EngineNotInstalledError,
    engine.EngineError,
)


def _fail(error: Exception) -> typer.Exit:
    typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


@app.command()
def setup(
    training_sample_rate: Annotated[
        int | None,
        typer.Option(help="Also fetch the pretrained models for training at this rate (Hz)."),
    ] = None,
) -> None:
    """Download the base models (about 0.8 GB). Safe to run again."""
    try:
        engine.prepare()
        engine.ensure_base_models(training_sample_rate=training_sample_rate, log=typer.echo)
        engine.ensure_separators(log=typer.echo)
    except USER_ERRORS as error:
        raise _fail(error) from error
    typer.echo("Setup complete.")


@app.command()
def doctor() -> None:
    """Show what is installed and which voices are available."""
    paths = get_paths()
    typer.echo(f"song-cloner {__version__} on Python {platform.python_version()}")
    typer.echo(f"Data directory:    {paths.home}")
    typer.echo(f"Output directory:  {paths.outputs}")
    for tool in ("ffmpeg", "sox"):
        typer.echo(
            f"{tool + ':':<18} {shutil.which(tool) or 'not on PATH (the engine fetches its own)'}"
        )
    try:
        engine.prepare(paths)
        typer.echo(f"Engine:            installed, using {engine.device_summary()}")
    except USER_ERRORS as error:
        typer.echo(f"Engine:            NOT READY - {error}")
    found = voices.list_voices(paths)
    typer.echo(f"Voices:            {', '.join(v.name for v in found) or 'none yet'}")


@app.command("convert")
def convert_command(
    source: Annotated[str, typer.Argument(help="A local audio file or a YouTube URL.")],
    voice: Annotated[
        str, typer.Option("--voice", "-v", envvar="SONGCLONER_VOICE", help="Voice model name.")
    ],
    octaves: Annotated[
        int, typer.Option(help="Shift only the converted vocal by this many octaves.")
    ] = 0,
    semitones: Annotated[
        int, typer.Option(help="Change the key of the whole song by this many semitones.")
    ] = 0,
    auto_pitch: Annotated[
        bool, typer.Option(help="Let the engine move the vocal into the voice's natural range.")
    ] = False,
    f0_method: Annotated[
        str, typer.Option(help="Pitch tracker: rmvpe, fcpe, crepe or crepe-tiny.")
    ] = "rmvpe",
    index_rate: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="How strongly the .index file shapes timbre.")
    ] = 0.3,
    protect: Annotated[
        float, typer.Option(min=0.0, max=0.5, help="Protect consonants and breaths (0.5 = off).")
    ] = 0.33,
    clean: Annotated[bool, typer.Option(help="Apply noise reduction to the vocal.")] = False,
    vocal_gain: Annotated[int, typer.Option(help="Lead vocal volume change in dB.")] = 0,
    inst_gain: Annotated[int, typer.Option(help="Instrumental volume change in dB.")] = 0,
    backing_gain: Annotated[int, typer.Option(help="Backing vocals volume change in dB.")] = 0,
    output_format: Annotated[
        str, typer.Option("--format", help="mp3, wav, flac, ogg, m4a or aac.")
    ] = "mp3",
    output_dir: Annotated[
        Path | None, typer.Option(help="Where to put the result (default: ./outputs).")
    ] = None,
) -> None:
    """Convert a song's lead vocal to VOICE and write the remixed song."""
    try:
        result = convert.convert_song(
            source,
            voice,
            paths=get_paths(),
            output_dir=output_dir,
            log=typer.echo,
            octaves=octaves,
            semitones=semitones,
            auto_pitch=auto_pitch,
            f0_method=f0_method,
            index_rate=index_rate,
            protect_rate=protect,
            clean_vocals=clean,
            main_gain=vocal_gain,
            inst_gain=inst_gain,
            backup_gain=backing_gain,
            output_format=output_format,
        )
    except USER_ERRORS as error:
        raise _fail(error) from error
    typer.echo(f"Done: {result}")


@app.command("train")
def train_command(
    name: Annotated[str, typer.Argument(help="Name for the new voice.")],
    dataset: Annotated[Path, typer.Argument(help="Folder of recordings of one singer.")],
    epochs: Annotated[int, typer.Option(min=1, help="Training epochs.")] = 200,
    batch_size: Annotated[
        int, typer.Option(min=1, help="Lower this if the GPU runs out of memory.")
    ] = 8,
    sample_rate: Annotated[int, typer.Option(help="32000, 40000 or 48000.")] = 40000,
    isolate_vocals: Annotated[
        bool, typer.Option(help="Recordings are full songs: strip music and backing vocals first.")
    ] = False,
    save_interval: Annotated[int, typer.Option(min=1, help="Checkpoint every N epochs.")] = 10,
    detect_overtraining: Annotated[
        bool, typer.Option(help="Stop early when the model stops improving.")
    ] = True,
    force: Annotated[
        bool, typer.Option(help="Replace an existing voice of the same name.")
    ] = False,
) -> None:
    """Train a voice model from recordings you provide (needs a GPU to be practical)."""
    try:
        voice = training.train(
            name,
            dataset,
            paths=get_paths(),
            epochs=epochs,
            batch_size=batch_size,
            sample_rate=sample_rate,
            isolate_vocals=isolate_vocals,
            save_interval=save_interval,
            detect_overtraining=detect_overtraining,
            force=force,
            log=typer.echo,
        )
    except USER_ERRORS as error:
        raise _fail(error) from error
    typer.echo(f"Trained voice {voice.name!r}: {voice.model}")


@voices_app.command("list")
def voices_list() -> None:
    """List the voices available locally."""
    found = voices.list_voices(get_paths())
    if not found:
        typer.echo("No voices yet. Add one with `songcloner voices add` or `voices pull`.")
        return
    for voice in found:
        index = voice.index.name if voice.index else "no index"
        typer.echo(f"{voice.name}\t{voice.model.name}\t{index}")


@voices_app.command("add")
def voices_add(
    name: Annotated[str, typer.Argument(help="Name to register the voice under.")],
    files: Annotated[list[Path], typer.Argument(help="One .pth file and, optionally, one .index.")],
    force: Annotated[bool, typer.Option(help="Replace an existing voice.")] = False,
) -> None:
    """Register a voice model from local files. Only add models you trust: .pth files run code."""
    try:
        voice = voices.add_voice(get_paths(), name, files, force=force)
    except USER_ERRORS as error:
        raise _fail(error) from error
    typer.echo(f"Added voice {voice.name!r}.")


@voices_app.command("remove")
def voices_remove(name: Annotated[str, typer.Argument(help="Voice to delete.")]) -> None:
    """Delete a local voice."""
    try:
        voices.remove_voice(get_paths(), name)
    except USER_ERRORS as error:
        raise _fail(error) from error
    typer.echo(f"Removed voice {name!r}.")


RepoOption = Annotated[
    str | None,
    typer.Option(envvar=voices.HF_REPO_ENV, help="Private Hugging Face model repo, <user>/<name>."),
]


@voices_app.command("pull")
def voices_pull(
    name: Annotated[str, typer.Argument(help="Voice to download.")],
    repo: RepoOption = None,
    force: Annotated[bool, typer.Option(help="Replace an existing voice.")] = False,
) -> None:
    """Download a voice from your private Hugging Face repo."""
    try:
        voice = voices.pull_voice(get_paths(), name, repo, force=force)
    except USER_ERRORS as error:
        raise _fail(error) from error
    typer.echo(f"Pulled voice {voice.name!r}.")


@voices_app.command("push")
def voices_push(
    name: Annotated[str, typer.Argument(help="Voice to upload.")],
    repo: RepoOption = None,
) -> None:
    """Upload a voice to your private Hugging Face repo (created private if missing)."""
    try:
        repo_id = voices.push_voice(get_paths(), name, repo)
    except USER_ERRORS as error:
        raise _fail(error) from error
    typer.echo(f"Pushed voice {name!r} to {repo_id} (private).")


if __name__ == "__main__":
    app()
