"""What runs inside a rented GPU job: ``python -m songcloner.job <action> ...``.

The job sees the owner's private storage bucket as a directory. Songs and voices
are read from it and results are written back to it, so the job needs no access
token. Everything heavy happens on the job's local disk; the bucket only
receives finished files.

Bucket layout::

    songs/<voice>/       training songs uploaded by the owner
    datasets/<voice>/    isolated lead vocals (kept, so retraining skips separation)
    voices/<voice>/      the trained voice (.pth and .index)
    inbox/               songs waiting to be converted
    covers/              converted songs
    reports/             what each job did
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

from songcloner import convert, dataset, engine, training, voices
from songcloner.paths import Paths

BUCKET_ENV = "SONGCLONER_BUCKET_DIR"
WORK_ENV = "SONGCLONER_WORK_DIR"

# Refuse to spend GPU time training on less singing than this. Far below the
# amount needed for a good voice; it only catches "the songs did not arrive".
DEFAULT_MIN_MINUTES = 5.0


def bucket_dir() -> Path:
    return Path(os.environ.get(BUCKET_ENV) or "/bucket")


def work_dir() -> Path:
    return Path(os.environ.get(WORK_ENV) or "/tmp/songcloner")


def work_paths() -> Paths:
    # Resolved, like the defaults in songcloner.paths, so that the two agree
    # when the working directory is reached through a symlink.
    root = work_dir().resolve()
    return Paths(home=root / "data", outputs=root / "outputs")


def copy_file(source: Path, destination: Path) -> None:
    """Copy file contents only: a mounted bucket has no permissions or timestamps to set."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def audio_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in training.AUDIO_SUFFIXES
    )


def install_system_tools(log: Callable[[str], None]) -> None:
    """Install sox and ffmpeg when the image lacks them and we are allowed to."""
    missing = [tool for tool in ("sox", "ffmpeg") if shutil.which(tool) is None]
    if not missing:
        return
    if shutil.which("apt-get") is None or os.geteuid() != 0:
        log(f"{', '.join(missing)} not installed; the engine will fetch its own copy.")
        return
    log(f"Installing {', '.join(missing)}...")
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "--no-install-recommends", "sox", "ffmpeg"],
        check=True,
    )


def start(log: Callable[[str], None]) -> tuple[Path, Paths]:
    """Common start-up: tools, working directory, engine. Returns (bucket, paths)."""
    bucket = bucket_dir()
    if not bucket.is_dir():
        raise SystemExit(f"Storage bucket is not mounted at {bucket}.")
    install_system_tools(log)
    work_dir().mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir())
    paths = work_paths()
    engine.prepare(paths)
    log(f"Running on {engine.device_summary()}")
    return bucket, paths


class BestModelMirror:
    """Copies the best model so far into the bucket while training runs.

    A job that hits its time limit is killed without warning. With this, the
    bucket still holds the best model reached, instead of nothing.
    """

    def __init__(self, source: Path, destination: Path, interval: float = 120.0) -> None:
        self.source = source
        self.destination = destination
        self.interval = interval
        self._copied_mtime = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def sync(self) -> bool:
        """Copy the model if it changed since the last copy. Returns whether it copied."""
        try:
            mtime = self.source.stat().st_mtime
        except FileNotFoundError:
            return False
        if mtime == self._copied_mtime:
            return False
        copy_file(self.source, self.destination)
        self._copied_mtime = mtime
        return True

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sync()
            except OSError as error:  # the next round will try again
                print(f"Could not mirror the model: {error}", flush=True)

    def __enter__(self) -> BestModelMirror:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=30)
        # One last copy, so that a run that crashes between two rounds still
        # leaves its best model behind.
        try:
            self.sync()
        except OSError as error:
            print(f"Could not mirror the model: {error}", flush=True)


def run_smoke(_args: argparse.Namespace, log: Callable[[str], None]) -> int:
    """Check that the job can see a GPU and write to the bucket."""
    bucket, _paths = start(log)
    marker = bucket / "reports" / "smoke.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{engine.device_summary()}\n")
    log(f"Wrote {marker}")
    log(f"Bucket contains: {', '.join(sorted(p.name for p in bucket.iterdir())) or 'nothing'}")
    return 0


def run_train(args: argparse.Namespace, log: Callable[[str], None]) -> int:
    """Build the dataset from the owner's songs, then train the voice."""
    name = voices.validate_name(args.voice)
    bucket, paths = start(log)
    datasets_dir = work_dir() / "datasets"
    local_dataset = datasets_dir / name
    bucket_dataset = bucket / "datasets" / name

    # Vocals isolated by an earlier job are reused rather than separated again.
    for clip in sorted(bucket_dataset.glob("*.flac")) if bucket_dataset.is_dir() else []:
        copy_file(clip, local_dataset / clip.name)

    sources: list[str] = []
    local_songs = work_dir() / "songs" / name
    for song in audio_files(bucket / "songs" / name):
        copy_file(song, local_songs / song.name)
    if audio_files(local_songs):
        sources.append(str(local_songs))
    sources.extend(args.url or [])

    failures: list[tuple[str, str]] = []
    if sources:
        report = dataset.build_dataset(
            name, sources, paths=paths, datasets_dir=datasets_dir, quality=args.quality, log=log
        )
        failures = report.failures
        for clip in report.clips:
            copy_file(clip.path, bucket_dataset / clip.path.name)

    clips = []
    for clip in sorted(local_dataset.glob("*.flac")) if local_dataset.is_dir() else []:
        _seconds, sung_seconds = engine.measure_singing(clip)
        clips.append({"clip": clip.name, "sung_minutes": round(sung_seconds / 60, 2)})
    sung_minutes = round(sum(item["sung_minutes"] for item in clips), 2)
    summary = {
        "voice": name,
        "sung_minutes": sung_minutes,
        "clips": clips,
        "failures": [{"source": source, "reason": reason} for source, reason in failures],
        "trained": False,
    }
    report_path = bucket / "reports" / f"train-{name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"DATASET {len(clips)} clip(s), {sung_minutes} min of singing, {len(failures)} failure(s)")

    if args.dataset_only:
        return 0
    if sung_minutes < args.min_minutes:
        log(
            f"Only {sung_minutes} min of singing (minimum {args.min_minutes}). "
            "Not training: add songs and run again."
        )
        return 2

    best_model = paths.models / "rvc" / "training" / name / f"{name}_best.pth"
    with BestModelMirror(best_model, bucket / "voices" / name / best_model.name):
        voice = training.train(
            name,
            local_dataset,
            paths=paths,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            force=True,
            log=log,
        )
    for path in (voice.model, voice.index):
        if path is not None:
            copy_file(path, bucket / "voices" / name / path.name)
    summary["trained"] = True
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"VOICE {name} saved to the bucket")
    return 0


def run_convert(args: argparse.Namespace, log: Callable[[str], None]) -> int:
    """Convert one song with a voice stored in the bucket."""
    name = voices.validate_name(args.voice)
    bucket, paths = start(log)
    voice_dir = bucket / "voices" / name
    files = (
        sorted(p for p in voice_dir.iterdir() if p.suffix in {".pth", ".index"})
        if voice_dir.is_dir()
        else []
    )
    if not any(p.suffix == ".pth" for p in files):
        log(f"No trained voice named {name!r} in the bucket.")
        return 2
    local_voice = work_dir() / "voice-files"
    local_files = []
    for path in files:
        copy_file(path, local_voice / path.name)
        local_files.append(local_voice / path.name)
    voices.add_voice(paths, name, local_files, force=True)

    if convert.is_url(args.song):
        source = args.song
    else:
        remote_song = bucket / args.song
        if not remote_song.is_file():
            log(f"Song not found in the bucket: {args.song}")
            return 2
        local_song = work_dir() / "inbox" / remote_song.name
        copy_file(remote_song, local_song)
        source = str(local_song)

    result = convert.convert_song(
        source,
        name,
        paths=paths,
        quality=args.quality,
        log=log,
        octaves=args.octaves,
        semitones=args.semitones,
        auto_pitch=args.auto_pitch,
        index_rate=args.index_rate,
        protect_rate=args.protect,
        output_format=args.format,
    )
    copy_file(result, bucket / "covers" / result.name)
    log(f"COVER {result.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m songcloner.job", description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    actions.add_parser("smoke", help="Check the GPU and the bucket.").set_defaults(run=run_smoke)

    train = actions.add_parser("train", help="Build the dataset and train a voice.")
    train.add_argument("voice")
    train.add_argument("--url", action="append", help="A song URL; repeat for more.")
    train.add_argument("--quality", default=engine.DEFAULT_QUALITY)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--sample-rate", type=int, default=40000)
    train.add_argument("--min-minutes", type=float, default=DEFAULT_MIN_MINUTES)
    train.add_argument(
        "--dataset-only", action="store_true", help="Isolate the vocals and stop before training."
    )
    train.set_defaults(run=run_train)

    cover = actions.add_parser("convert", help="Convert a song with a trained voice.")
    cover.add_argument("song", help="A path inside the bucket (inbox/...) or a URL.")
    cover.add_argument("--voice", required=True)
    cover.add_argument("--quality", default=engine.DEFAULT_QUALITY)
    cover.add_argument("--octaves", type=int, default=0)
    cover.add_argument("--semitones", type=int, default=0)
    cover.add_argument("--auto-pitch", action="store_true")
    cover.add_argument("--index-rate", type=float, default=0.3)
    cover.add_argument("--protect", type=float, default=0.33)
    cover.add_argument("--format", default="mp3")
    cover.set_defaults(run=run_convert)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    try:
        return args.run(args, log)
    except Exception:  # noqa: BLE001 - the job log is the only place the owner can see this
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
