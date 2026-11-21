from __future__ import annotations

from pathlib import Path

import pytest

from songcloner import convert, engine, voices
from songcloner.convert import ConvertError


def test_is_url() -> None:
    assert convert.is_url("https://www.youtube.com/watch?v=abc")
    assert convert.is_url("http://example.com/song.mp3")
    assert not convert.is_url("songs/track.mp3")
    assert not convert.is_url("/abs/https.mp3")


def test_resolve_source_keeps_urls_and_absolutises_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    song = tmp_path / "track.mp3"
    song.write_bytes(b"audio")
    monkeypatch.chdir(tmp_path)

    assert convert.resolve_source("https://youtu.be/abc") == "https://youtu.be/abc"
    assert convert.resolve_source("track.mp3") == str(song.resolve())
    with pytest.raises(ConvertError, match="Song file not found"):
        convert.resolve_source("missing.mp3")


def test_unique_destination_never_overwrites(tmp_path: Path) -> None:
    first = convert.unique_destination(tmp_path, "Song (singer Ver).mp3")
    assert first == tmp_path / "Song (singer Ver).mp3"
    first.write_bytes(b"1")

    second = convert.unique_destination(tmp_path, "Song (singer Ver).mp3")
    assert second == tmp_path / "Song (singer Ver) (2).mp3"
    second.write_bytes(b"2")

    assert convert.unique_destination(tmp_path, "Song (singer Ver).mp3") == (
        tmp_path / "Song (singer Ver) (3).mp3"
    )


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Replace the engine with a stub that writes a small 'cover' file."""
    calls: dict = {"run_cover": [], "separators": []}

    def run_cover(source: str, voice: str, *, log, **options) -> Path:
        calls["run_cover"].append((source, voice, options))
        cover = tmp_path / "engine" / f"Track ({voice} Ver).mp3"
        cover.parent.mkdir(parents=True, exist_ok=True)
        cover.write_bytes(b"mixed-song")
        return cover

    monkeypatch.setattr(engine, "prepare", lambda paths=None: paths)
    monkeypatch.setattr(engine, "ensure_base_models", lambda **_k: None)
    monkeypatch.setattr(
        engine, "ensure_separators", lambda quality, log: calls["separators"].append(quality)
    )
    monkeypatch.setattr(engine, "device_summary", lambda: "CPU (test)")
    monkeypatch.setattr(engine, "run_cover", run_cover)
    return calls


def test_convert_song_copies_the_cover_into_outputs(
    paths, model_files, fake_engine, tmp_path: Path
) -> None:
    voices.add_voice(paths, "singer", list(model_files))
    song = tmp_path / "track.wav"
    song.write_bytes(b"audio")

    result = convert.convert_song(
        str(song), "singer", paths=paths, log=lambda _m: None, octaves=-1, output_format="mp3"
    )

    assert result == paths.outputs / "Track (singer Ver).mp3"
    assert result.read_bytes() == b"mixed-song"
    assert fake_engine["run_cover"] == [
        (
            str(song.resolve()),
            "singer",
            {"quality": "high", "octaves": -1, "output_format": "mp3"},
        )
    ]
    assert fake_engine["separators"] == ["high"]


def test_convert_song_uses_the_requested_quality(
    paths, model_files, fake_engine, tmp_path: Path
) -> None:
    voices.add_voice(paths, "singer", list(model_files))
    song = tmp_path / "track.wav"
    song.write_bytes(b"audio")

    convert.convert_song(str(song), "singer", paths=paths, quality="standard", log=lambda _m: None)

    # The same quality must reach both the model download and the pipeline;
    # otherwise the pipeline would stall downloading a model mid-conversion.
    assert fake_engine["separators"] == ["standard"]
    assert fake_engine["run_cover"][0][2] == {"quality": "standard"}


def test_convert_song_rejects_an_unknown_quality_before_heavy_work(
    paths, model_files, fake_engine, tmp_path: Path
) -> None:
    voices.add_voice(paths, "singer", list(model_files))
    song = tmp_path / "track.wav"
    song.write_bytes(b"audio")

    with pytest.raises(engine.EngineError, match="Unknown quality 'ultra'"):
        convert.convert_song(str(song), "singer", paths=paths, quality="ultra", log=lambda _m: None)

    assert fake_engine["separators"] == [] and fake_engine["run_cover"] == []


def test_convert_song_keeps_earlier_results(
    paths, model_files, fake_engine, tmp_path: Path
) -> None:
    voices.add_voice(paths, "singer", list(model_files))
    song = tmp_path / "track.wav"
    song.write_bytes(b"audio")

    first = convert.convert_song(str(song), "singer", paths=paths, log=lambda _m: None)
    second = convert.convert_song(str(song), "singer", paths=paths, log=lambda _m: None)

    assert first != second
    assert first.exists() and second.exists()


def test_convert_song_fails_before_heavy_work_when_the_voice_is_missing(
    paths, fake_engine, tmp_path: Path
) -> None:
    song = tmp_path / "track.wav"
    song.write_bytes(b"audio")

    with pytest.raises(voices.VoiceError, match="No voice named 'singer'"):
        convert.convert_song(str(song), "singer", paths=paths, log=lambda _m: None)

    assert fake_engine["run_cover"] == []


def test_convert_song_honours_output_dir(paths, model_files, fake_engine, tmp_path: Path) -> None:
    voices.add_voice(paths, "singer", list(model_files))
    song = tmp_path / "track.wav"
    song.write_bytes(b"audio")
    elsewhere = tmp_path / "elsewhere"

    result = convert.convert_song(
        str(song), "singer", paths=paths, output_dir=elsewhere, log=lambda _m: None
    )

    assert result.parent == elsewhere


def test_convert_song_rejects_a_bad_restore_spec_before_heavy_work(
    paths, model_files, fake_engine, tmp_path: Path
) -> None:
    from songcloner.restore import RestoreError

    voices.add_voice(paths, "singer", list(model_files))
    song = tmp_path / "track.wav"
    song.write_bytes(b"audio")

    with pytest.raises(RestoreError, match="Unknown restore step"):
        convert.convert_song(
            str(song), "singer", paths=paths, log=lambda _m: None, restore="sparkle"
        )

    assert fake_engine["separators"] == [] and fake_engine["run_cover"] == []
