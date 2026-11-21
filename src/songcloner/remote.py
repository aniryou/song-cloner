"""Run training and conversion on rented Hugging Face GPUs: ``scripts/remote ...``.

Each command starts a Hugging Face Job that mounts the owner's private storage
bucket and runs :mod:`songcloner.job` there. Nothing here needs a GPU, or the
engine, locally.

This module needs ``huggingface_hub`` 2.4 or newer, which the engine's own
dependencies do not allow in the same environment. ``scripts/remote`` runs it
in a separate, lightweight environment.

Costs are bounded before anything starts: every job gets a hard time limit, and
a job whose time limit multiplied by the hardware's hourly price exceeds
``--max-cost`` is refused.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from songcloner import restore, voices
from songcloner.convert import is_url, unique_destination
from songcloner.training import AUDIO_SUFFIXES

# US dollars per hour, from https://huggingface.co/docs/hub/jobs-pricing
# (read November 2026). Jobs are billed by the minute while starting or running.
HOURLY_PRICE = {
    "cpu-basic": 0.01,
    "cpu-upgrade": 0.03,
    "t4-small": 0.40,
    "t4-medium": 0.60,
    "l4x1": 0.80,
    "a10g-small": 1.00,
}
DEFAULT_GPU = "t4-small"
DEFAULT_MAX_COST = 2.00
BUCKET_NAME = "song-cloner"
MOUNT_PATH = "/bucket"
FINISHED_STAGES = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}


class RemoteError(Exception):
    """A problem with a remote request that the user can act on."""


def worst_case_cost(flavor: str, hours: float) -> float:
    """The most a job can cost: it is stopped, and billing ends, at its time limit."""
    if flavor not in HOURLY_PRICE:
        known = ", ".join(sorted(HOURLY_PRICE))
        raise RemoteError(f"No price known for hardware {flavor!r}; choose one of: {known}.")
    if hours <= 0:
        raise RemoteError("The time limit must be positive.")
    return HOURLY_PRICE[flavor] * hours


def check_budget(flavor: str, hours: float, max_cost: float) -> float:
    cost = worst_case_cost(flavor, hours)
    if cost > max_cost + 1e-9:
        raise RemoteError(
            f"{hours:g} h on {flavor} can cost up to ${cost:.2f}, above the ${max_cost:.2f} limit. "
            "Lower --hours, or raise --max-cost if you mean it."
        )
    return cost


@dataclass(frozen=True)
class CodeReference:
    """The exact, already-pushed commit a job will install."""

    repository: str  # "owner/name" on github.com
    commit: str

    @property
    def dependency(self) -> str:
        # An archive URL installs without git in the job's image.
        archive = f"https://github.com/{self.repository}/archive/{self.commit}.zip"
        return f"song-cloner[engine] @ {archive}"


def _git(repo_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RemoteError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def parse_repository(remote_url: str) -> str:
    """Extract "owner/name" from a GitHub remote URL (https, ssh, or a local git proxy)."""
    cleaned = remote_url.strip().removesuffix(".git").rstrip("/")
    parts = cleaned.replace(":", "/").split("/")
    if len(parts) < 2 or not parts[-1] or not parts[-2]:
        raise RemoteError(f"Cannot read a GitHub repository from remote URL {remote_url!r}.")
    return f"{parts[-2]}/{parts[-1]}"


def code_reference(repo_dir: Path) -> CodeReference:
    """The current commit, which must be on GitHub: the job downloads the code from there."""
    commit = _git(repo_dir, "rev-parse", "HEAD")
    if _git(repo_dir, "status", "--porcelain", "--untracked-files=no"):
        raise RemoteError("There are uncommitted changes; the job would not run this code.")
    if not _git(repo_dir, "branch", "--remotes", "--contains", commit):
        raise RemoteError(f"Commit {commit[:7]} is not pushed; the job could not download it.")
    return CodeReference(parse_repository(_git(repo_dir, "remote", "get-url", "origin")), commit)


def _hub() -> Any:
    try:
        import huggingface_hub
        from huggingface_hub import Volume  # noqa: F401 - probes for a new enough client
    except ImportError as error:
        raise RemoteError(
            "This command needs huggingface_hub 2.4 or newer. Run it through scripts/remote."
        ) from error
    return huggingface_hub


# Sent when no token is configured on this machine. In managed environments
# (Claude cloud sessions) a network proxy holds the real credential and puts it
# on each request to huggingface.co; the client only needs something to send.
# Anywhere else, Hugging Face rejects it and the error says to log in.
PLACEHOLDER_TOKEN = "hf_placeholder_replaced_by_the_network_proxy"


def make_api() -> Any:
    hub = _hub()
    return hub.HfApi(token=hub.get_token() or PLACEHOLDER_TOKEN)


@contextlib.contextmanager
def _empty_working_directory() -> Iterator[None]:
    """Run a block from an empty directory.

    The Hugging Face client uploads any job argument that names an existing
    local file. Our arguments are never local files, but a voice called
    "train", say, could collide with a file of that name here.
    """
    previous = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="songcloner-launch-") as scratch:
        os.chdir(scratch)
        try:
            yield
        finally:
            os.chdir(previous)


def ensure_bucket(api: Any) -> str:
    """Create the owner's private bucket if needed; refuse to use a public one."""
    bucket_id = f"{api.whoami()['name']}/{BUCKET_NAME}"
    api.create_bucket(bucket_id, private=True, exist_ok=True)
    if not api.bucket_info(bucket_id).private:
        raise RemoteError(
            f"Bucket {bucket_id} is public. Songs and voice models must stay private; "
            "make it private on huggingface.co first."
        )
    return bucket_id


def collect_songs(sources: Sequence[str]) -> tuple[list[Path], list[str]]:
    """Split sources into local audio files (folders are searched) and URLs."""
    files: list[Path] = []
    urls: list[str] = []
    for source in sources:
        if is_url(source):
            urls.append(source)
            continue
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            found = sorted(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES
            )
            if not found:
                raise RemoteError(f"No audio files in {path}")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise RemoteError(f"Not found: {path}")
    names = [path.name for path in files]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RemoteError(f"Two songs share a file name: {', '.join(duplicates)}. Rename one.")
    return files, urls


def launch(
    api: Any,
    job_args: Sequence[str],
    *,
    code: CodeReference,
    bucket_id: str,
    flavor: str,
    hours: float,
    max_cost: float,
    log: Callable[[str], None],
) -> Any:
    """Start a job running ``python -m songcloner.job <job_args>``; return its JobInfo."""
    cost = check_budget(flavor, hours, max_cost)
    hub = _hub()
    if not flavor.startswith("cpu"):
        # Paying for a GPU that the job then cannot use is the expensive failure:
        # make the job stop at once instead of working on the CPU for hours.
        job_args = [*job_args, "--require-gpu"]
    with _empty_working_directory():
        job = api.run_uv_job(
            "python",
            script_args=["-m", "songcloner.job", *job_args],
            dependencies=[code.dependency],
            python="3.12",
            flavor=flavor,
            timeout=int(hours * 3600),
            volumes=[hub.Volume(type="bucket", source=bucket_id, mount_path=MOUNT_PATH)],
            env={"SONGCLONER_BUCKET_DIR": MOUNT_PATH},
            labels={"app": "song-cloner", "action": job_args[0]},
        )
    log(f"Started job {job.id} on {flavor}: {job.url}")
    log(f"Time limit {hours:g} h, so it can cost at most ${cost:.2f}.")
    return job


def stage_of(job: Any) -> str:
    """A job's stage as a plain string such as "RUNNING" or "COMPLETED"."""
    stage = job.status.stage
    return str(getattr(stage, "value", stage))


def follow(api: Any, job_id: str, log: Callable[[str], None]) -> str:
    """Print a job's log until it ends; return its final stage."""
    for line in api.fetch_job_logs(job_id=job_id, follow=True):
        log(line.rstrip("\n"))
    stage = stage_of(api.inspect_job(job_id=job_id))
    if stage not in FINISHED_STAGES:
        stage = stage_of(api.wait_for_job(job_id, poll_interval=10))
    return stage


def fetch(api: Any, bucket_id: str, prefix: str, destination: Path) -> list[Path]:
    """Download every file under ``prefix`` in the bucket that is not already here."""
    destination.mkdir(parents=True, exist_ok=True)
    wanted = []
    for item in api.list_bucket_tree(bucket_id, prefix=prefix, recursive=True):
        if getattr(item, "type", "file") != "file":
            continue
        local = destination / Path(item.path).name
        if not local.exists():
            wanted.append((item.path, local))
    if wanted:
        api.download_bucket_files(
            bucket_id, files=[(remote, str(local)) for remote, local in wanted]
        )
    return [local for _remote, local in wanted]


def _finish(stage: str) -> int:
    if stage != "COMPLETED":
        print(f"Job ended as {stage}.", file=sys.stderr)
        return 1
    return 0


def _start(args: argparse.Namespace) -> tuple[Any, str, CodeReference]:
    api = make_api()
    return api, ensure_bucket(api), code_reference(Path(args.repo))


def cmd_smoke(args: argparse.Namespace) -> int:
    api, bucket_id, code = _start(args)
    job = launch(
        api,
        ["smoke"],
        code=code,
        bucket_id=bucket_id,
        flavor=args.flavor,
        hours=args.hours,
        max_cost=args.max_cost,
        log=print,
    )
    return _finish(follow(api, job.id, print))


def cmd_train(args: argparse.Namespace) -> int:
    voices.validate_name(args.voice)
    files, urls = collect_songs(args.songs)
    check_budget(args.flavor, args.hours, args.max_cost)  # before uploading anything
    api, bucket_id, code = _start(args)
    if files:
        print(f"Uploading {len(files)} song(s) to the private bucket {bucket_id}...")
        api.batch_bucket_files(
            bucket_id, add=[(str(path), f"songs/{args.voice}/{path.name}") for path in files]
        )
    job_args = ["train", args.voice, "--quality", args.quality]
    job_args += ["--epochs", str(args.epochs), "--batch-size", str(args.batch_size)]
    for url in urls:
        job_args += ["--url", url]
    if args.dataset_only:
        job_args.append("--dataset-only")
    job = launch(
        api,
        job_args,
        code=code,
        bucket_id=bucket_id,
        flavor=args.flavor,
        hours=args.hours,
        max_cost=args.max_cost,
        log=print,
    )
    if args.detach:
        print(f"Not waiting. Follow it with: scripts/remote logs {job.id}")
        return 0
    return _finish(follow(api, job.id, print))


def cmd_convert(args: argparse.Namespace) -> int:
    voices.validate_name(args.voice)
    check_budget(args.flavor, args.hours, args.max_cost)
    api, bucket_id, code = _start(args)
    song = args.song
    if not is_url(song):
        path = Path(song).expanduser().resolve()
        if not path.is_file():
            raise RemoteError(f"Song file not found: {path}")
        song = f"inbox/{path.name}"
        print(f"Uploading {path.name} to the private bucket {bucket_id}...")
        api.batch_bucket_files(bucket_id, add=[(str(path), song)])
    job_args = ["convert", song, "--voice", args.voice, "--quality", args.quality]
    job_args += ["--octaves", str(args.octaves), "--semitones", str(args.semitones)]
    job_args += ["--index-rate", str(args.index_rate), "--format", args.format]
    if args.auto_pitch:
        job_args.append("--auto-pitch")
    for variant in args.restore or []:
        restore.parse(variant)  # a typo should not cost a job
        job_args += ["--restore", variant]
    job = launch(
        api,
        job_args,
        code=code,
        bucket_id=bucket_id,
        flavor=args.flavor,
        hours=args.hours,
        max_cost=args.max_cost,
        log=print,
    )
    cover_names: list[str] = []

    def log(line: str) -> None:
        if line.startswith("COVER "):
            cover_names.append(line.removeprefix("COVER ").strip())
        print(line)

    status = _finish(follow(api, job.id, log))
    if status != 0 or not cover_names:
        return status or 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for cover_name in cover_names:
        destination = unique_destination(output_dir, cover_name)
        api.download_bucket_files(bucket_id, files=[(f"covers/{cover_name}", str(destination))])
        print(f"Done: {destination}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    api = make_api()
    return _finish(follow(api, args.job_id, print))


def cmd_status(args: argparse.Namespace) -> int:
    api = make_api()
    job = api.inspect_job(job_id=args.job_id)
    print(f"{job.id}: {stage_of(job)} ({job.flavor}) {job.url}")
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    api = make_api()
    api.cancel_job(job_id=args.job_id)
    print(f"Cancelled {args.job_id}; billing for it has stopped.")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    api = make_api()
    bucket_id = ensure_bucket(api)
    for path in fetch(api, bucket_id, args.prefix, Path(args.output_dir)):
        print(f"Downloaded {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/remote",
        description="Run training and conversion on rented Hugging Face GPUs.",
    )
    parser.add_argument("--repo", default=".", help="The song-cloner checkout (default: here).")
    commands = parser.add_subparsers(dest="command", required=True)

    def budget(sub: argparse.ArgumentParser, flavor: str, hours: float) -> None:
        sub.add_argument("--flavor", default=flavor, help=f"Hardware (default {flavor}).")
        sub.add_argument("--hours", type=float, default=hours, help="Hard time limit.")
        sub.add_argument(
            "--max-cost",
            type=float,
            default=DEFAULT_MAX_COST,
            help="Refuse a job that could cost more than this many dollars.",
        )

    smoke = commands.add_parser("smoke", help="Cheap check that jobs and the bucket work.")
    budget(smoke, "cpu-basic", 0.5)
    smoke.set_defaults(run=cmd_smoke)

    train = commands.add_parser("train", help="Build the dataset and train a voice on a GPU.")
    train.add_argument("voice")
    train.add_argument("songs", nargs="*", help="Song files, folders or URLs to add.")
    train.add_argument("--quality", default="high")
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--dataset-only", action="store_true", help="Stop before training.")
    train.add_argument("--detach", action="store_true", help="Start the job and return.")
    budget(train, DEFAULT_GPU, 4.0)
    train.set_defaults(run=cmd_train)

    cover = commands.add_parser("convert", help="Convert a song on a GPU and download it.")
    cover.add_argument("song", help="A local audio file or a URL.")
    cover.add_argument("--voice", required=True)
    cover.add_argument("--quality", default="high")
    cover.add_argument("--octaves", type=int, default=0)
    cover.add_argument("--semitones", type=int, default=0)
    cover.add_argument("--auto-pitch", action="store_true")
    cover.add_argument("--index-rate", type=float, default=0.3)
    cover.add_argument("--format", default="mp3")
    cover.add_argument("--output-dir", default="outputs")
    cover.add_argument(
        "--restore",
        action="append",
        help="Backing-track restore steps, e.g. denoise,tone. Repeat to render several variants.",
    )
    budget(cover, DEFAULT_GPU, 0.5)
    cover.set_defaults(run=cmd_convert)

    for name, handler, text in (
        ("logs", cmd_logs, "Follow a job's log until it ends."),
        ("status", cmd_status, "Show a job's state."),
        ("cancel", cmd_cancel, "Stop a job and its billing."),
    ):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("job_id")
        sub.set_defaults(run=handler)

    get = commands.add_parser("fetch", help="Download files from the bucket.")
    get.add_argument("prefix", help="For example covers/ or reports/.")
    get.add_argument("--output-dir", default="outputs")
    get.set_defaults(run=cmd_fetch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (RemoteError, voices.VoiceError, restore.RestoreError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        # Hugging Face's own errors carry the useful part in their message
        # (bad token, no credit, unknown hardware); a traceback buries it.
        if type(error).__module__.startswith("huggingface_hub"):
            print(f"Hugging Face error: {error}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
