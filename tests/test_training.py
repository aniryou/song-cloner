from __future__ import annotations

from pathlib import Path

import pytest

from songcloner import engine, training, voices
from songcloner.training import TrainingError


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    (dataset / "album").mkdir(parents=True)
    (dataset / "b.wav").write_bytes(b"b")
    (dataset / "a.MP3").write_bytes(b"a")
    (dataset / "album" / "c.flac").write_bytes(b"c")
    (dataset / "notes.txt").write_text("not audio")
    return dataset


def test_find_recordings_is_recursive_sorted_and_audio_only(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    found = training.find_recordings(dataset)

    assert [p.relative_to(dataset).as_posix() for p in found] == [
        "a.MP3",
        "album/c.flac",
        "b.wav",
    ]


def test_find_recordings_errors(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="Dataset folder not found"):
        training.find_recordings(tmp_path / "absent")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("x")
    with pytest.raises(TrainingError, match="No audio files"):
        training.find_recordings(empty)


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch, paths) -> dict:
    """An engine stub whose training 'succeeds' by writing a model file."""
    calls: dict = {"base_models": [], "separators": 0, "isolated": [], "trained": [], "fail": False}

    def ensure_base_models(*, training_sample_rate=None, log=print) -> None:
        calls["base_models"].append(training_sample_rate)

    def ensure_separators(*, log=print) -> None:
        calls["separators"] += 1

    def isolate_lead_vocal(audio_file: Path, work_dir: Path) -> Path:
        calls["isolated"].append(audio_file.name)
        return work_dir / f"{audio_file.stem}_lead.wav"

    def train_voice(name: str, audio_files, **options) -> None:
        calls["trained"].append((name, [Path(p).name for p in audio_files], options))
        if calls["fail"]:
            raise RuntimeError("out of memory")
        target = paths.voices / name
        target.mkdir(parents=True)
        (target / f"{name}.pth").write_bytes(b"freshly-trained")

    monkeypatch.setattr(engine, "prepare", lambda paths=None: paths)
    monkeypatch.setattr(engine, "ensure_base_models", ensure_base_models)
    monkeypatch.setattr(engine, "ensure_separators", ensure_separators)
    monkeypatch.setattr(engine, "isolate_lead_vocal", isolate_lead_vocal)
    monkeypatch.setattr(engine, "train_voice", train_voice)
    return calls


def test_train_passes_recordings_and_settings_to_the_engine(
    paths, fake_engine, tmp_path: Path
) -> None:
    voice = training.train(
        "singer",
        _dataset(tmp_path),
        paths=paths,
        epochs=3,
        batch_size=2,
        sample_rate=48000,
        log=lambda _m: None,
    )

    assert voice.model.read_bytes() == b"freshly-trained"
    assert fake_engine["base_models"] == [48000]
    assert fake_engine["separators"] == 0 and fake_engine["isolated"] == []
    name, files, options = fake_engine["trained"][0]
    assert name == "singer"
    assert files == ["a.MP3", "c.flac", "b.wav"]
    assert options["epochs"] == 3
    assert options["batch_size"] == 2
    assert options["sample_rate"] == 48000


def test_train_isolates_vocals_first_when_asked(paths, fake_engine, tmp_path: Path) -> None:
    training.train(
        "singer", _dataset(tmp_path), paths=paths, isolate_vocals=True, log=lambda _m: None
    )

    assert fake_engine["separators"] == 1
    assert fake_engine["isolated"] == ["a.MP3", "c.flac", "b.wav"]
    _, files, _ = fake_engine["trained"][0]
    assert files == ["a_lead.wav", "c_lead.wav", "b_lead.wav"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_rate": 44100}, "Sample rate must be one of"),
        ({"epochs": 0}, "at least 1"),
        ({"batch_size": 0}, "at least 1"),
    ],
)
def test_train_rejects_bad_settings(
    paths, fake_engine, tmp_path: Path, kwargs: dict, message: str
) -> None:
    with pytest.raises(TrainingError, match=message):
        training.train("singer", _dataset(tmp_path), paths=paths, log=lambda _m: None, **kwargs)
    assert fake_engine["trained"] == []


def test_train_refuses_to_replace_a_voice_without_force(
    paths, model_files, fake_engine, tmp_path: Path
) -> None:
    voices.add_voice(paths, "singer", list(model_files))

    with pytest.raises(TrainingError, match="already exists"):
        training.train("singer", _dataset(tmp_path), paths=paths, log=lambda _m: None)

    assert fake_engine["trained"] == []
    assert voices.get_voice(paths, "singer").model.read_bytes() == b"model-bytes"


def test_force_replaces_the_voice_on_success(
    paths, model_files, fake_engine, tmp_path: Path
) -> None:
    voices.add_voice(paths, "singer", list(model_files))

    voice = training.train(
        "singer", _dataset(tmp_path), paths=paths, force=True, log=lambda _m: None
    )

    assert voice.model.read_bytes() == b"freshly-trained"
    assert voice.index is None  # nothing left over from the replaced voice


def test_force_restores_the_old_voice_when_training_fails(
    paths, model_files, fake_engine, tmp_path: Path
) -> None:
    voices.add_voice(paths, "singer", list(model_files))
    fake_engine["fail"] = True

    with pytest.raises(RuntimeError, match="out of memory"):
        training.train("singer", _dataset(tmp_path), paths=paths, force=True, log=lambda _m: None)

    restored = voices.get_voice(paths, "singer")
    assert restored.model.read_bytes() == b"model-bytes"
    assert restored.index is not None and restored.index.read_bytes() == b"index-bytes"
