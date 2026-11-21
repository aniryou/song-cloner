from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from songcloner import engine
from songcloner.cli import app
from songcloner.paths import HOME_ENV, OUTPUTS_ENV


def plain(text: str) -> str:
    """Strip terminal styling. On GitHub Actions the argument parser colours its
    error panels, which splits option names such as ``--voice`` across escape codes."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    monkeypatch.setenv(HOME_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(OUTPUTS_ENV, str(tmp_path / "outputs"))
    monkeypatch.delenv("SONGCLONER_VOICE", raising=False)
    monkeypatch.delenv("SONGCLONER_HF_REPO", raising=False)
    return CliRunner()


def test_help_lists_the_commands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("setup", "doctor", "convert", "train", "voices"):
        assert command in result.output


def test_voices_add_list_remove_round_trip(runner: CliRunner, model_files) -> None:
    model, index = model_files

    assert "No voices yet" in runner.invoke(app, ["voices", "list"]).output
    added = runner.invoke(app, ["voices", "add", "singer", str(model), str(index)])
    listed = runner.invoke(app, ["voices", "list"])
    removed = runner.invoke(app, ["voices", "remove", "singer"])

    assert added.exit_code == 0 and "Added voice 'singer'" in added.output
    assert listed.output.split() == ["singer", "singer.pth", "singer.index"]
    assert removed.exit_code == 0
    assert "No voices yet" in runner.invoke(app, ["voices", "list"]).output


def test_user_errors_exit_1_with_a_message_not_a_traceback(
    runner: CliRunner, tmp_path: Path
) -> None:
    song = tmp_path / "track.wav"
    song.write_bytes(b"audio")

    result = runner.invoke(app, ["convert", str(song), "--voice", "nobody"])

    assert result.exit_code == 1
    assert "Error: No voice named 'nobody'" in result.output
    assert "Traceback" not in result.output


def test_convert_requires_a_voice(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["convert", str(tmp_path / "track.wav")])

    assert result.exit_code == 2  # usage error from the argument parser
    assert "--voice" in plain(result.output)


def test_convert_passes_options_through(
    runner: CliRunner, model_files, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}

    def run_cover(source: str, voice: str, *, log, **options) -> Path:
        seen.update(source=source, voice=voice, **options)
        cover = tmp_path / "cover.flac"
        cover.write_bytes(b"mixed")
        return cover

    monkeypatch.setattr(engine, "prepare", lambda paths=None: paths)
    monkeypatch.setattr(engine, "ensure_base_models", lambda **_k: None)
    monkeypatch.setattr(engine, "ensure_separators", lambda quality, log: None)
    monkeypatch.setattr(engine, "device_summary", lambda: "CPU (test)")
    monkeypatch.setattr(engine, "run_cover", run_cover)
    song = tmp_path / "track.wav"
    song.write_bytes(b"audio")
    runner.invoke(app, ["voices", "add", "singer", str(model_files[0])])

    result = runner.invoke(
        app,
        [
            "convert",
            str(song),
            "-v",
            "singer",
            "--octaves",
            "-1",
            "--semitones",
            "2",
            "--auto-pitch",
            "--index-rate",
            "0.5",
            "--protect",
            "0.2",
            "--vocal-gain",
            "3",
            "--format",
            "flac",
            "--quality",
            "standard",
            "--restore",
            "tone:3,denoise",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == {
        "source": str(song.resolve()),
        "voice": "singer",
        "quality": "standard",
        "octaves": -1,
        "semitones": 2,
        "auto_pitch": True,
        "f0_method": "rmvpe",
        "index_rate": 0.5,
        "protect_rate": 0.2,
        "clean_vocals": False,
        "main_gain": 3,
        "inst_gain": 0,
        "backup_gain": 0,
        "output_format": "flac",
        "restore": "tone:3,denoise",
    }
    assert (tmp_path / "outputs" / "cover.flac").read_bytes() == b"mixed"


def test_voice_can_come_from_the_environment(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SONGCLONER_VOICE", "from-env")
    song = tmp_path / "track.wav"
    song.write_bytes(b"audio")

    result = runner.invoke(app, ["convert", str(song)])

    assert result.exit_code == 1
    assert "No voice named 'from-env'" in result.output


def test_train_reports_a_missing_dataset(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", "singer", str(tmp_path / "absent")])

    assert result.exit_code == 1
    assert "Dataset folder not found" in result.output


def test_convert_rejects_a_bad_restore_spec_before_doing_any_work(
    runner: CliRunner, model_files, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[str] = []
    monkeypatch.setattr(engine, "prepare", lambda paths=None: started.append("prepare"))
    song = tmp_path / "track.wav"
    song.write_bytes(b"audio")
    runner.invoke(app, ["voices", "add", "singer", str(model_files[0])])

    result = runner.invoke(app, ["convert", str(song), "-v", "singer", "--restore", "sparkle"])

    assert result.exit_code == 1
    assert "Unknown restore step 'sparkle'" in result.output
    assert started == []
