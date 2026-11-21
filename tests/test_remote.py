from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from songcloner import remote
from songcloner.remote import CodeReference, RemoteError

CODE = CodeReference("aniryou/song-cloner", "abc1234def")


def test_worst_case_cost_is_time_limit_times_hourly_price() -> None:
    assert remote.worst_case_cost("t4-small", 4) == pytest.approx(1.60)
    assert remote.worst_case_cost("cpu-basic", 0.5) == pytest.approx(0.005)
    with pytest.raises(RemoteError, match="No price known for hardware 'h200'"):
        remote.worst_case_cost("h200", 1)
    with pytest.raises(RemoteError, match="must be positive"):
        remote.worst_case_cost("t4-small", 0)


def test_check_budget_refuses_jobs_that_could_exceed_the_limit() -> None:
    assert remote.check_budget("t4-small", 5, max_cost=2.00) == pytest.approx(2.00)
    with pytest.raises(RemoteError, match=r"can cost up to \$2\.40, above the \$2\.00 limit"):
        remote.check_budget("t4-small", 6, max_cost=2.00)
    # An hour of the larger GPU costs what 2.5 hours of the default one does.
    with pytest.raises(RemoteError):
        remote.check_budget("a10g-small", 2.5, max_cost=2.00)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/aniryou/song-cloner",
        "https://github.com/aniryou/song-cloner.git",
        "git@github.com:aniryou/song-cloner.git",
        "http://local_proxy@127.0.0.1:38291/git/aniryou/song-cloner",
    ],
)
def test_parse_repository(url: str) -> None:
    assert remote.parse_repository(url) == "aniryou/song-cloner"


def test_parse_repository_rejects_nonsense() -> None:
    with pytest.raises(RemoteError, match="Cannot read a GitHub repository"):
        remote.parse_repository("song-cloner")


def test_dependency_pins_the_exact_commit_and_needs_no_git() -> None:
    assert CODE.dependency == (
        "song-cloner[engine] @ https://github.com/aniryou/song-cloner/archive/abc1234def.zip"
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def pushed_repo(tmp_path: Path) -> Path:
    """A clone whose HEAD exists on its 'origin', like a pushed checkout."""
    origin = tmp_path / "aniryou" / "song-cloner"
    origin.mkdir(parents=True)
    _git(origin, "init", "-q", "-b", "main")
    (origin / "file.txt").write_text("one\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "first")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    return clone


def test_code_reference_of_a_pushed_checkout(pushed_repo: Path) -> None:
    reference = remote.code_reference(pushed_repo)

    assert reference.repository == "aniryou/song-cloner"
    assert reference.commit == _git(pushed_repo, "rev-parse", "HEAD")


def test_code_reference_refuses_unpushed_commits(pushed_repo: Path) -> None:
    (pushed_repo / "file.txt").write_text("two\n")
    _git(pushed_repo, "commit", "-q", "-am", "local only")

    # The job downloads the code from GitHub; an unpushed commit is not there.
    with pytest.raises(RemoteError, match="is not pushed"):
        remote.code_reference(pushed_repo)


def test_code_reference_refuses_uncommitted_changes(pushed_repo: Path) -> None:
    (pushed_repo / "file.txt").write_text("edited\n")

    with pytest.raises(RemoteError, match="uncommitted changes"):
        remote.code_reference(pushed_repo)


def test_collect_songs_splits_files_and_urls(tmp_path: Path) -> None:
    folder = tmp_path / "songs"
    folder.mkdir()
    (folder / "b.mp3").write_bytes(b"b")
    (folder / "a.flac").write_bytes(b"a")
    (folder / "notes.txt").write_text("x")
    single = tmp_path / "single.wav"
    single.write_bytes(b"s")

    files, urls = remote.collect_songs([str(folder), "https://youtu.be/abc", str(single)])

    assert [p.name for p in files] == ["a.flac", "b.mp3", "single.wav"]
    assert urls == ["https://youtu.be/abc"]


def test_collect_songs_errors(tmp_path: Path) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    (tmp_path / "one" / "song.mp3").write_bytes(b"1")
    (tmp_path / "two" / "song.mp3").write_bytes(b"2")
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(RemoteError, match="Not found"):
        remote.collect_songs([str(tmp_path / "absent.mp3")])
    with pytest.raises(RemoteError, match="No audio files"):
        remote.collect_songs([str(empty)])
    # Both would be uploaded as songs/<voice>/song.mp3; one would silently replace the other.
    with pytest.raises(RemoteError, match="share a file name: song.mp3"):
        remote.collect_songs([str(tmp_path / "one"), str(tmp_path / "two")])


@dataclass
class FakeVolume:
    type: str
    source: str
    mount_path: str


class FakeApi:
    """Records what would be sent to Hugging Face."""

    def __init__(self, *, private: bool = True, final_stage: str = "COMPLETED") -> None:
        self.private = private
        self.final_stage = final_stage
        self.created: list[tuple[str, bool]] = []
        self.jobs: list[dict] = []
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []
        self.tree: list[SimpleNamespace] = []
        self.log_lines: list[str] = []

    def whoami(self) -> dict:
        return {"name": "owner"}

    def create_bucket(self, bucket_id: str, *, private: bool, exist_ok: bool) -> None:
        self.created.append((bucket_id, private))

    def bucket_info(self, bucket_id: str) -> SimpleNamespace:
        return SimpleNamespace(private=self.private)

    def run_uv_job(self, script: str, **kwargs) -> SimpleNamespace:
        self.jobs.append({"script": script, "cwd_files": sorted(Path.cwd().iterdir()), **kwargs})
        return SimpleNamespace(id="job123", url="https://huggingface.co/jobs/owner/job123")

    def fetch_job_logs(self, *, job_id: str, follow: bool):
        yield from self.log_lines

    def inspect_job(self, *, job_id: str) -> SimpleNamespace:
        return SimpleNamespace(status=SimpleNamespace(stage=self.final_stage))

    def batch_bucket_files(self, bucket_id: str, *, add) -> None:
        self.uploads.extend((Path(local).name, remote_path) for local, remote_path in add)

    def list_bucket_tree(self, bucket_id: str, *, prefix: str, recursive: bool):
        return [item for item in self.tree if item.path.startswith(prefix)]

    def download_bucket_files(self, bucket_id: str, *, files) -> None:
        for remote_path, local in files:
            self.downloads.append((remote_path, local))
            Path(local).write_bytes(b"downloaded")


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    fake = FakeApi()
    monkeypatch.setattr(remote, "_hub", lambda: SimpleNamespace(Volume=FakeVolume))
    monkeypatch.setattr(remote, "make_api", lambda: fake)
    monkeypatch.setattr(remote, "code_reference", lambda _repo: CODE)
    return fake


def test_ensure_bucket_creates_it_private(api: FakeApi) -> None:
    assert remote.ensure_bucket(api) == "owner/song-cloner"
    assert api.created == [("owner/song-cloner", True)]


def test_ensure_bucket_refuses_a_public_bucket(api: FakeApi) -> None:
    api.private = False  # it already existed, and is public

    with pytest.raises(RemoteError, match="is public"):
        remote.ensure_bucket(api)


def test_launch_sends_a_bounded_job_that_mounts_the_bucket(
    api: FakeApi, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A local file named like one of the arguments must not leak into the job:
    # the Hugging Face client uploads arguments that name existing files.
    (tmp_path / "train").write_text("unrelated local file")
    monkeypatch.chdir(tmp_path)
    lines: list[str] = []

    remote.launch(
        api,
        ["train", "arijit", "--epochs", "300"],
        code=CODE,
        bucket_id="owner/song-cloner",
        flavor="t4-small",
        hours=4,
        max_cost=2.0,
        log=lines.append,
    )

    job = api.jobs[0]
    assert job["script"] == "python"
    assert job["script_args"] == [
        "-m", "songcloner.job", "train", "arijit", "--epochs", "300", "--require-gpu",
    ]  # fmt: skip
    assert job["dependencies"] == [CODE.dependency]
    assert job["flavor"] == "t4-small"
    assert job["timeout"] == 4 * 3600
    assert job["volumes"] == [FakeVolume("bucket", "owner/song-cloner", "/bucket")]
    assert job["env"] == {"SONGCLONER_BUCKET_DIR": "/bucket"}
    assert "secrets" not in job  # the job needs no token: the bucket is mounted
    assert job["cwd_files"] == []
    assert Path.cwd() == tmp_path  # the working directory is restored
    assert any("at most $1.60" in line for line in lines)


def test_launch_starts_nothing_when_over_budget(api: FakeApi) -> None:
    with pytest.raises(RemoteError, match="above the"):
        remote.launch(
            api,
            ["train", "arijit"],
            code=CODE,
            bucket_id="owner/song-cloner",
            flavor="a10g-small",
            hours=4,
            max_cost=2.0,
            log=lambda _m: None,
        )

    assert api.jobs == []


def test_follow_relays_the_log_and_reports_the_final_stage(api: FakeApi) -> None:
    api.log_lines = ["Running on GPU: Tesla T4\n", "VOICE arijit saved to the bucket\n"]
    seen: list[str] = []

    stage = remote.follow(api, "job123", seen.append)

    assert seen == ["Running on GPU: Tesla T4", "VOICE arijit saved to the bucket"]
    assert stage == "COMPLETED"


def test_stage_of_reads_enum_values_and_plain_strings() -> None:
    enum_like = SimpleNamespace(status=SimpleNamespace(stage=SimpleNamespace(value="RUNNING")))
    plain = SimpleNamespace(status=SimpleNamespace(stage="ERROR"))

    assert remote.stage_of(enum_like) == "RUNNING"
    assert remote.stage_of(plain) == "ERROR"


def test_fetch_downloads_only_files_not_already_here(api: FakeApi, tmp_path: Path) -> None:
    api.tree = [
        SimpleNamespace(type="file", path="covers/Song A (arijit Ver).mp3"),
        SimpleNamespace(type="file", path="covers/Song B (arijit Ver).mp3"),
        SimpleNamespace(type="directory", path="covers/old"),
        SimpleNamespace(type="file", path="reports/train-arijit.json"),
    ]
    (tmp_path / "Song A (arijit Ver).mp3").write_bytes(b"already here")

    fetched = remote.fetch(api, "owner/song-cloner", "covers/", tmp_path)

    assert [p.name for p in fetched] == ["Song B (arijit Ver).mp3"]
    assert api.downloads == [
        ("covers/Song B (arijit Ver).mp3", str(tmp_path / "Song B (arijit Ver).mp3"))
    ]
    assert (tmp_path / "Song A (arijit Ver).mp3").read_bytes() == b"already here"


def test_train_command_uploads_songs_then_starts_the_job(
    api: FakeApi, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    song = tmp_path / "Tum Hi Ho.mp3"
    song.write_bytes(b"audio")

    status = remote.main(
        ["train", "arijit", str(song), "https://youtu.be/abc", "--epochs", "250", "--hours", "3"]
    )

    assert status == 0
    assert api.uploads == [("Tum Hi Ho.mp3", "songs/arijit/Tum Hi Ho.mp3")]
    assert api.jobs[0]["script_args"][2:] == [
        "train", "arijit", "--quality", "high", "--epochs", "250", "--batch-size", "8",
        "--url", "https://youtu.be/abc", "--require-gpu",
    ]  # fmt: skip
    assert api.jobs[0]["flavor"] == "t4-small"
    assert "at most $1.20" in capsys.readouterr().out


def test_train_command_checks_the_budget_before_uploading(
    api: FakeApi, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    song = tmp_path / "song.mp3"
    song.write_bytes(b"audio")

    status = remote.main(["train", "arijit", str(song), "--hours", "10"])

    assert status == 1
    assert "above the $2.00 limit" in capsys.readouterr().err
    assert api.uploads == [] and api.jobs == [] and api.created == []


def test_convert_command_uploads_runs_and_downloads_the_cover(
    api: FakeApi, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    song = tmp_path / "Kesariya.mp3"
    song.write_bytes(b"audio")
    api.log_lines = ["Converting...\n", "COVER Kesariya (arijit Ver).mp3\n"]
    out = tmp_path / "outputs"

    status = remote.main(
        ["convert", str(song), "--voice", "arijit", "--octaves", "-1", "--output-dir", str(out)]
    )

    assert status == 0
    assert api.uploads == [("Kesariya.mp3", "inbox/Kesariya.mp3")]
    args = api.jobs[0]["script_args"][2:]
    assert args[:4] == ["convert", "inbox/Kesariya.mp3", "--voice", "arijit"]
    assert args[args.index("--octaves") + 1] == "-1"
    assert api.jobs[0]["timeout"] == 1800  # half an hour: at most $0.20 on the default GPU
    assert api.downloads == [
        ("covers/Kesariya (arijit Ver).mp3", str(out / "Kesariya (arijit Ver).mp3"))
    ]
    assert "Done:" in capsys.readouterr().out


def test_convert_command_fails_when_the_job_fails(
    api: FakeApi, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    song = tmp_path / "Kesariya.mp3"
    song.write_bytes(b"audio")
    api.final_stage = "ERROR"

    status = remote.main(["convert", str(song), "--voice", "arijit"])

    assert status == 1
    assert "Job ended as ERROR" in capsys.readouterr().err
    assert api.downloads == []


def test_convert_command_passes_urls_through_without_uploading(
    api: FakeApi, tmp_path: Path
) -> None:
    api.log_lines = ["COVER Song (arijit Ver).mp3\n"]

    remote.main(
        ["convert", "https://youtu.be/xyz", "--voice", "arijit", "--output-dir", str(tmp_path)]
    )

    assert api.uploads == []
    assert api.jobs[0]["script_args"][3] == "https://youtu.be/xyz"


def test_cpu_jobs_are_not_told_to_require_a_gpu(api: FakeApi) -> None:
    remote.launch(
        api,
        ["smoke"],
        code=CODE,
        bucket_id="owner/song-cloner",
        flavor="cpu-basic",
        hours=0.5,
        max_cost=2.0,
        log=lambda _m: None,
    )

    assert api.jobs[0]["script_args"] == ["-m", "songcloner.job", "smoke"]
