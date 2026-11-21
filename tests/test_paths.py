from __future__ import annotations

from pathlib import Path

import pytest

from songcloner.paths import HOME_ENV, OUTPUTS_ENV, engine_env, get_paths


def test_defaults_are_relative_to_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(HOME_ENV, raising=False)
    monkeypatch.delenv(OUTPUTS_ENV, raising=False)

    paths = get_paths()

    assert paths.home == (tmp_path / "data").resolve()
    assert paths.outputs == (tmp_path / "outputs").resolve()


def test_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HOME_ENV, str(tmp_path / "elsewhere"))
    monkeypatch.setenv(OUTPUTS_ENV, str(tmp_path / "songs-out"))

    paths = get_paths()

    assert paths.home == (tmp_path / "elsewhere").resolve()
    assert paths.outputs == (tmp_path / "songs-out").resolve()


def test_engine_env_points_the_engine_at_our_directories(paths) -> None:
    env = engine_env(paths)

    assert env["URVC_MODELS_DIR"] == str(paths.home / "models")
    # The engine looks for voices in <models>/rvc/voice_models/<name>.
    assert env["URVC_VOICE_MODELS_DIR"] == str(paths.home / "models" / "rvc" / "voice_models")
    assert env["URVC_AUDIO_DIR"] == str(paths.home / "audio")
    assert env["URVC_TEMP_DIR"] == str(paths.home / "temp")
    assert set(env) == {
        "URVC_MODELS_DIR",
        "URVC_VOICE_MODELS_DIR",
        "URVC_AUDIO_DIR",
        "URVC_TEMP_DIR",
        "URVC_CONFIG_DIR",
        "URVC_LOGS_DIR",
    }


def test_engine_prepare_keeps_the_first_choice_of_directories(
    paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from songcloner import engine
    from songcloner.paths import Paths

    monkeypatch.setattr(engine, "_prepared_for", None)
    monkeypatch.setattr(engine, "_require_engine", lambda: None)
    monkeypatch.setattr(engine, "_use_system_sox", lambda: None)
    # prepare() exports these; registering them first makes pytest restore them.
    for name in [*engine_env(paths), "URVC_NO_LOGGING"]:
        monkeypatch.setenv(name, "restored-after-test")

    assert engine.prepare(paths) == paths
    # A job changes directory after preparing. Internal callers pass no paths
    # and must get the prepared ones, not defaults for the new directory.
    monkeypatch.chdir(tmp_path)
    assert engine.prepare() == paths
    assert engine.prepare(paths) == paths
    with pytest.raises(engine.EngineError, match="different directories"):
        engine.prepare(Paths(home=tmp_path / "elsewhere", outputs=tmp_path / "out"))
