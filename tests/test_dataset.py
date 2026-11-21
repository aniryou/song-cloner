from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from songcloner import dataset, engine
from songcloner.cli import app
from songcloner.dataset import Clip, DatasetError, DatasetReport
from songcloner.paths import HOME_ENV, OUTPUTS_ENV


def _songs(tmp_path: Path) -> Path:
    folder = tmp_path / "songs"
    (folder / "album").mkdir(parents=True)
    (folder / "Tum Hi Ho.mp3").write_bytes(b"1")
    (folder / "album" / "Channa Mereya.flac").write_bytes(b"2")
    (folder / "cover.jpg").write_bytes(b"not audio")
    return folder


def test_expand_sources_mixes_files_folders_and_urls(tmp_path: Path) -> None:
    folder = _songs(tmp_path)
    single = tmp_path / "single.wav"
    single.write_bytes(b"3")

    expanded = dataset.expand_sources([str(folder), "https://youtu.be/abc", str(single)])

    assert expanded == [
        str(folder / "Tum Hi Ho.mp3"),
        str(folder / "album" / "Channa Mereya.flac"),
        "https://youtu.be/abc",
        str(single),
    ]


def test_expand_sources_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetError, match="Not found"):
        dataset.expand_sources([str(tmp_path / "absent.mp3")])
    with pytest.raises(DatasetError, match="No audio files"):
        dataset.expand_sources([str(empty)])
    with pytest.raises(DatasetError, match="at least one song"):
        dataset.expand_sources([])


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Tum Hi Ho", "Tum_Hi_Ho"),
        ("Kesariya (Full Video) | Brahmastra", "Kesariya_(Full_Video)_Brahmastra"),
        ("../../etc/passwd", "etcpasswd"),
        ("...hidden", "hidden"),
        ("तुम ही हो", "तुम_ही_हो"),
        ("???", "song"),
    ],
)
def test_slug_is_a_safe_file_name(title: str, expected: str) -> None:
    result = dataset.slug(title)

    assert result == expected
    assert "/" not in result and result not in {"", ".", ".."}


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """An engine stub: 'isolates' instantly and reports 3 of every 4 minutes as singing."""
    calls: dict = {"isolated": [], "separators": [], "fail_on": set()}

    def isolate_lead_vocal(source: str, quality: str, log) -> tuple[Path, str]:
        calls["isolated"].append((source, quality))
        if source in calls["fail_on"]:
            raise RuntimeError("download blocked")
        title = "Fetched Title" if source.startswith("https://") else Path(source).stem
        return tmp_path / "engine" / f"{title}_lead.wav", title

    def export_vocal(vocal: Path, destination: Path) -> tuple[float, float]:
        destination.write_bytes(b"flac")
        return 240.0, 180.0

    monkeypatch.setattr(engine, "prepare", lambda paths=None: paths)
    monkeypatch.setattr(
        engine, "ensure_separators", lambda quality, log: calls["separators"].append(quality)
    )
    monkeypatch.setattr(engine, "isolate_lead_vocal", isolate_lead_vocal)
    monkeypatch.setattr(engine, "export_vocal", export_vocal)
    return calls


def test_build_dataset_writes_one_clip_per_song(paths, fake_engine, tmp_path: Path) -> None:
    report = dataset.build_dataset(
        "arijit",
        [str(_songs(tmp_path))],
        paths=paths,
        datasets_dir=tmp_path / "datasets",
        quality="standard",
        log=lambda _m: None,
    )

    directory = tmp_path / "datasets" / "arijit"
    assert report.directory == directory
    assert sorted(p.name for p in directory.iterdir()) == ["Channa_Mereya.flac", "Tum_Hi_Ho.flac"]
    assert [clip.path.name for clip in report.clips] == ["Tum_Hi_Ho.flac", "Channa_Mereya.flac"]
    assert report.sung_minutes == pytest.approx(6.0)
    assert report.failures == [] and report.skipped == []
    assert fake_engine["separators"] == ["standard"]
    assert {quality for _, quality in fake_engine["isolated"]} == {"standard"}


def test_songs_already_in_the_dataset_are_not_processed_again(
    paths, fake_engine, tmp_path: Path
) -> None:
    folder = _songs(tmp_path)
    kwargs = {"paths": paths, "datasets_dir": tmp_path / "datasets", "log": lambda _m: None}
    dataset.build_dataset("arijit", [str(folder)], **kwargs)
    fake_engine["isolated"].clear()
    (folder / "Kesariya.mp3").write_bytes(b"new")

    report = dataset.build_dataset("arijit", [str(folder)], **kwargs)

    # Separation is the expensive step: only the new song may reach it.
    assert [Path(source).name for source, _ in fake_engine["isolated"]] == ["Kesariya.mp3"]
    assert [clip.path.name for clip in report.clips] == ["Kesariya.flac"]
    assert len(report.skipped) == 2


def test_a_url_fetched_twice_is_skipped_once_its_title_is_known(
    paths, fake_engine, tmp_path: Path
) -> None:
    kwargs = {"paths": paths, "datasets_dir": tmp_path / "datasets", "log": lambda _m: None}
    dataset.build_dataset("arijit", ["https://youtu.be/abc"], **kwargs)

    report = dataset.build_dataset("arijit", ["https://youtu.be/abc"], **kwargs)

    assert report.clips == []
    assert report.skipped == ["https://youtu.be/abc"]
    assert [p.name for p in (tmp_path / "datasets" / "arijit").iterdir()] == ["Fetched_Title.flac"]


def test_one_failing_song_does_not_lose_the_others(paths, fake_engine, tmp_path: Path) -> None:
    folder = _songs(tmp_path)
    fake_engine["fail_on"] = {"https://youtu.be/blocked"}

    report = dataset.build_dataset(
        "arijit",
        ["https://youtu.be/blocked", str(folder)],
        paths=paths,
        datasets_dir=tmp_path / "datasets",
        log=lambda _m: None,
    )

    assert report.failures == [("https://youtu.be/blocked", "RuntimeError: download blocked")]
    assert len(report.clips) == 2


def test_build_dataset_validates_before_any_work(paths, fake_engine, tmp_path: Path) -> None:
    folder = _songs(tmp_path)
    with pytest.raises(engine.EngineError, match="Unknown quality"):
        dataset.build_dataset("arijit", [str(folder)], paths=paths, quality="ultra")
    with pytest.raises(Exception, match="Invalid voice name"):
        dataset.build_dataset("../escape", [str(folder)], paths=paths)

    assert fake_engine["isolated"] == [] and fake_engine["separators"] == []


def _report(minutes: float) -> DatasetReport:
    clip = Clip("song.mp3", Path("song.flac"), seconds=minutes * 80, sung_seconds=minutes * 60)
    return DatasetReport(directory=Path("datasets/x"), clips=[clip])


def test_verdict_tells_the_user_whether_there_is_enough_singing() -> None:
    assert "too little" in _report(4).verdict()
    assert "usable" in _report(12).verdict()
    assert "enough for a good voice" in _report(25).verdict()
    assert DatasetReport(directory=Path("x")).verdict() == "No new clips were built."


def test_cli_dataset_reports_and_fails_when_a_song_failed(
    fake_engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(HOME_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(OUTPUTS_ENV, str(tmp_path / "outputs"))
    monkeypatch.chdir(tmp_path)
    folder = _songs(tmp_path)
    fake_engine["fail_on"] = {"https://youtu.be/blocked"}

    result = CliRunner().invoke(app, ["dataset", "arijit", str(folder), "https://youtu.be/blocked"])

    assert result.exit_code == 1
    assert "Tum_Hi_Ho.flac" in result.output
    assert "6.0 minutes of singing: too little" in result.output
    assert "Failed: https://youtu.be/blocked: RuntimeError: download blocked" in result.output
    assert (tmp_path / "datasets" / "arijit" / "Tum_Hi_Ho.flac").exists()


def test_vocal_separator_presets() -> None:
    assert engine.vocal_separator("standard") == "UVR-MDX-NET-Voc_FT.onnx"
    assert engine.vocal_separator("high") == "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    assert engine.DEFAULT_QUALITY == "high"
    with pytest.raises(engine.EngineError, match="choose one of: high, standard"):
        engine.vocal_separator("ultra")
