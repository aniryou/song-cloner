from __future__ import annotations

import json
from pathlib import Path

import pytest

from songcloner import convert, dataset, engine, job, training, voices
from songcloner.dataset import Clip, DatasetReport
from songcloner.voices import Voice


@pytest.fixture
def bucket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A local folder standing in for the mounted storage bucket, plus stubbed heavy parts."""
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    monkeypatch.setenv(job.BUCKET_ENV, str(bucket))
    monkeypatch.setenv(job.WORK_ENV, str(tmp_path / "work"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(job, "install_system_tools", lambda log: None)
    monkeypatch.setattr(engine, "prepare", lambda paths=None: paths)
    monkeypatch.setattr(engine, "device_summary", lambda: "GPU: Test T4")
    # Every clip counts as three minutes of singing.
    monkeypatch.setattr(engine, "measure_singing", lambda clip: (240.0, 180.0))
    return bucket


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    calls: dict = {"dataset": [], "train": [], "convert": [], "fail_training": False}

    def build_dataset(name, sources, *, paths, datasets_dir, quality, log) -> DatasetReport:
        calls["dataset"].append({"name": name, "sources": list(sources), "quality": quality})
        directory = datasets_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        report = DatasetReport(directory=directory)
        for source in sources:
            if source.startswith("https://"):
                # In these tests every URL is one whose download fails.
                report.failures.append((source, "RuntimeError: download blocked"))
                continue
            for song in sorted(Path(source).iterdir()):
                clip = directory / f"{song.stem}.flac"
                clip.write_bytes(b"vocal")
                report.clips.append(Clip(str(song), clip, 240.0, 180.0))
        return report

    def train(name, dataset_dir, *, paths, epochs, batch_size, sample_rate, force, log) -> Voice:
        calls["train"].append(
            {
                "name": name,
                "clips": sorted(p.name for p in Path(dataset_dir).glob("*.flac")),
                "epochs": epochs,
                "batch_size": batch_size,
            }
        )
        training_dir = paths.models / "rvc" / "training" / name
        training_dir.mkdir(parents=True, exist_ok=True)
        (training_dir / f"{name}_best.pth").write_bytes(b"best-so-far")
        if calls["fail_training"]:
            raise RuntimeError("CUDA out of memory")
        voice_dir = paths.voices / name
        voice_dir.mkdir(parents=True, exist_ok=True)
        model, index = voice_dir / f"{name}_best.pth", voice_dir / f"{name}.index"
        model.write_bytes(b"final-model")
        index.write_bytes(b"final-index")
        return Voice(name, model, index)

    def convert_song(source, voice, *, paths, quality, log, **options) -> Path:
        calls["convert"].append(
            {"source": source, "voice": voice, "quality": quality, "options": options}
        )
        suffix = f" [{options['restore']}]" if options.get("restore") else ""
        result = tmp_path / "work" / "outputs" / f"{Path(source).stem} ({voice} Ver){suffix}.mp3"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_bytes(b"mixed-song")
        return result

    monkeypatch.setattr(dataset, "build_dataset", build_dataset)
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(convert, "convert_song", convert_song)
    return calls


def _put_songs(bucket: Path, voice: str, names: list[str]) -> None:
    folder = bucket / "songs" / voice
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_bytes(b"song")


def test_smoke_proves_the_bucket_is_writable(bucket: Path, capsys: pytest.CaptureFixture) -> None:
    assert job.main(["smoke"]) == 0

    assert (bucket / "reports" / "smoke.txt").read_text() == "GPU: Test T4\n"
    assert "Running on GPU: Test T4" in capsys.readouterr().out


def test_train_builds_the_dataset_trains_and_saves_everything_to_the_bucket(
    bucket: Path, stubs: dict
) -> None:
    _put_songs(bucket, "arijit", ["Tum Hi Ho.mp3", "Kesariya.mp3"])

    status = job.main(["train", "arijit", "--epochs", "250", "--quality", "standard"])

    assert status == 0
    assert stubs["dataset"][0]["quality"] == "standard"
    assert stubs["train"] == [
        {
            "name": "arijit",
            "clips": ["Kesariya.flac", "Tum Hi Ho.flac"],
            "epochs": 250,
            "batch_size": 8,
        }
    ]
    assert sorted(p.name for p in (bucket / "datasets" / "arijit").iterdir()) == [
        "Kesariya.flac",
        "Tum Hi Ho.flac",
    ]
    assert (bucket / "voices" / "arijit" / "arijit_best.pth").read_bytes() == b"final-model"
    assert (bucket / "voices" / "arijit" / "arijit.index").read_bytes() == b"final-index"
    report = json.loads((bucket / "reports" / "train-arijit.json").read_text())
    assert report["sung_minutes"] == 6.0
    assert report["trained"] is True
    assert report["failures"] == []


def test_train_refuses_to_train_on_too_little_singing(
    bucket: Path, stubs: dict, capsys: pytest.CaptureFixture
) -> None:
    _put_songs(bucket, "arijit", ["Only One.mp3"])  # 3 minutes; the default minimum is 5

    status = job.main(["train", "arijit"])

    assert status == 2
    assert stubs["train"] == []
    assert "Not training" in capsys.readouterr().out
    report = json.loads((bucket / "reports" / "train-arijit.json").read_text())
    assert report["trained"] is False and report["sung_minutes"] == 3.0
    # The isolated vocal is kept, so adding songs later does not repeat this work.
    assert (bucket / "datasets" / "arijit" / "Only One.flac").exists()


def test_dataset_only_stops_before_training(bucket: Path, stubs: dict) -> None:
    _put_songs(bucket, "arijit", ["A.mp3", "B.mp3"])

    assert job.main(["train", "arijit", "--dataset-only"]) == 0

    assert stubs["train"] == []
    assert len(list((bucket / "datasets" / "arijit").iterdir())) == 2


def test_train_reuses_vocals_isolated_by_an_earlier_job(bucket: Path, stubs: dict) -> None:
    earlier = bucket / "datasets" / "arijit"
    earlier.mkdir(parents=True)
    for name in ("Old One.flac", "Old Two.flac"):
        (earlier / name).write_bytes(b"vocal")

    status = job.main(["train", "arijit"])  # no new songs at all

    assert status == 0
    assert stubs["dataset"] == []  # nothing to separate
    assert stubs["train"][0]["clips"] == ["Old One.flac", "Old Two.flac"]


def test_train_reports_songs_that_failed_but_still_trains_on_the_rest(
    bucket: Path, stubs: dict
) -> None:
    _put_songs(bucket, "arijit", ["A.mp3", "B.mp3"])

    status = job.main(["train", "arijit", "--url", "https://youtu.be/blocked"])

    assert status == 0
    report = json.loads((bucket / "reports" / "train-arijit.json").read_text())
    assert report["failures"] == [
        {"source": "https://youtu.be/blocked", "reason": "RuntimeError: download blocked"}
    ]
    assert stubs["train"][0]["clips"] == ["A.flac", "B.flac"]


def test_failed_training_exits_nonzero_and_keeps_the_best_model_so_far(
    bucket: Path, stubs: dict, capsys: pytest.CaptureFixture
) -> None:
    _put_songs(bucket, "arijit", ["A.mp3", "B.mp3"])
    stubs["fail_training"] = True

    status = job.main(["train", "arijit"])

    assert status == 1
    assert "CUDA out of memory" in capsys.readouterr().err
    # The mirror copies on exit too, so a crash late in training loses nothing.
    assert (bucket / "voices" / "arijit" / "arijit_best.pth").read_bytes() == b"best-so-far"


def test_best_model_mirror_copies_only_when_the_model_changes(tmp_path: Path) -> None:
    source, destination = tmp_path / "best.pth", tmp_path / "bucket" / "best.pth"
    mirror = job.BestModelMirror(source, destination)

    assert mirror.sync() is False  # nothing to copy yet
    source.write_bytes(b"epoch-10")
    assert mirror.sync() is True
    assert mirror.sync() is False  # unchanged
    assert destination.read_bytes() == b"epoch-10"


def _put_voice(bucket: Path, name: str) -> None:
    folder = bucket / "voices" / name
    folder.mkdir(parents=True)
    (folder / f"{name}_best.pth").write_bytes(b"model")
    (folder / f"{name}.index").write_bytes(b"index")


def test_convert_uses_the_bucket_voice_and_saves_the_cover(
    bucket: Path, stubs: dict, capsys: pytest.CaptureFixture
) -> None:
    _put_voice(bucket, "arijit")
    (bucket / "inbox").mkdir()
    (bucket / "inbox" / "Kesariya.mp3").write_bytes(b"song")

    status = job.main(
        ["convert", "inbox/Kesariya.mp3", "--voice", "arijit", "--octaves", "-1", "--auto-pitch"]
    )

    assert status == 0
    call = stubs["convert"][0]
    assert Path(call["source"]).name == "Kesariya.mp3"
    assert call["voice"] == "arijit" and call["quality"] == "high"
    assert call["options"]["octaves"] == -1 and call["options"]["auto_pitch"] is True
    assert (bucket / "covers" / "Kesariya (arijit Ver).mp3").read_bytes() == b"mixed-song"
    # The launcher finds the file to download from this line.
    assert "COVER Kesariya (arijit Ver).mp3" in capsys.readouterr().out
    registered = voices.get_voice(job.work_paths(), "arijit")
    assert registered.model.read_bytes() == b"model" and registered.index is not None


def test_convert_passes_urls_straight_through(bucket: Path, stubs: dict) -> None:
    _put_voice(bucket, "arijit")

    assert job.main(["convert", "https://youtu.be/xyz", "--voice", "arijit"]) == 0

    assert stubs["convert"][0]["source"] == "https://youtu.be/xyz"


@pytest.mark.parametrize(
    ("song", "message"),
    [("inbox/missing.mp3", "Song not found in the bucket"), ("https://youtu.be/x", None)],
)
def test_convert_fails_clearly_when_inputs_are_missing(
    bucket: Path, stubs: dict, capsys: pytest.CaptureFixture, song: str, message: str | None
) -> None:
    if message is None:
        # No voice in the bucket at all.
        assert job.main(["convert", song, "--voice", "arijit"]) == 2
        assert "No trained voice named 'arijit'" in capsys.readouterr().out
    else:
        _put_voice(bucket, "arijit")
        assert job.main(["convert", song, "--voice", "arijit"]) == 2
        assert message in capsys.readouterr().out
    assert stubs["convert"] == []


def test_job_refuses_to_run_without_the_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(job.BUCKET_ENV, str(tmp_path / "not-mounted"))
    monkeypatch.setattr(job, "install_system_tools", lambda log: None)

    with pytest.raises(SystemExit, match="Storage folder not found"):
        job.main(["smoke"])


def test_require_gpu_stops_before_any_work_when_there_is_none(
    bucket: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine, "device_summary", lambda: "CPU (96 cores)")
    _put_songs(bucket, "arijit", ["A.mp3", "B.mp3"])

    # A TPU runtime on Colab looks exactly like this: plenty of CPU, no GPU.
    with pytest.raises(SystemExit, match="(?s)No GPU found.*T4 GPU.*TPU does not work"):
        job.main(["train", "arijit", "--require-gpu"])

    assert stubs["dataset"] == [] and stubs["train"] == []


def test_without_require_gpu_a_cpu_machine_may_run(
    bucket: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine, "device_summary", lambda: "CPU (4 cores)")

    assert job.main(["smoke"]) == 0


def test_require_gpu_passes_on_a_gpu(bucket: Path) -> None:
    assert job.main(["smoke", "--require-gpu"]) == 0


def test_convert_renders_one_cover_per_restore_variant(
    bucket: Path, stubs: dict, capsys: pytest.CaptureFixture
) -> None:
    _put_voice(bucket, "arijit")
    (bucket / "inbox").mkdir()
    (bucket / "inbox" / "Old Song.mp3").write_bytes(b"song")

    status = job.main(
        ["convert", "inbox/Old Song.mp3", "--voice", "arijit",
         "--restore", "none", "--restore", "denoise,tone", "--restore", "all"]
    )  # fmt: skip

    assert status == 0
    assert [call["options"]["restore"] for call in stubs["convert"]] == [
        "none",
        "denoise,tone",
        "all",
    ]
    assert sorted(p.name for p in (bucket / "covers").iterdir()) == [
        "Old Song (arijit Ver) [all].mp3",
        "Old Song (arijit Ver) [denoise,tone].mp3",
        "Old Song (arijit Ver) [none].mp3",
    ]
    assert capsys.readouterr().out.count("COVER ") == 3


def test_convert_without_restore_makes_a_single_plain_cover(bucket: Path, stubs: dict) -> None:
    _put_voice(bucket, "arijit")

    assert job.main(["convert", "https://youtu.be/xyz", "--voice", "arijit"]) == 0

    assert [call["options"]["restore"] for call in stubs["convert"]] == [""]
