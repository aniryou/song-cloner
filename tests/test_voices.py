from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from songcloner import voices
from songcloner.voices import VoiceError


def test_add_then_list_and_get(paths, model_files) -> None:
    model, index = model_files

    added = voices.add_voice(paths, "singer", [model, index])

    assert added.name == "singer"
    assert added.model.read_bytes() == b"model-bytes"
    assert added.index is not None and added.index.read_bytes() == b"index-bytes"
    assert [v.name for v in voices.list_voices(paths)] == ["singer"]
    assert voices.get_voice(paths, "singer") == added


def test_index_is_optional(paths, model_files) -> None:
    model, _ = model_files

    added = voices.add_voice(paths, "singer", [model])

    assert added.index is None


def test_add_refuses_to_overwrite_without_force(paths, model_files) -> None:
    model, index = model_files
    voices.add_voice(paths, "singer", [model, index])
    model.write_bytes(b"new-model")

    with pytest.raises(VoiceError, match="already exists"):
        voices.add_voice(paths, "singer", [model])
    replaced = voices.add_voice(paths, "singer", [model], force=True)

    assert replaced.model.read_bytes() == b"new-model"
    assert replaced.index is None  # the old index must not linger next to a new model


@pytest.mark.parametrize(
    "names",
    [[], ["a.index"], ["a.pth", "b.pth"], ["a.pth", "a.index", "b.index"], ["a.pth", "notes.txt"]],
)
def test_add_rejects_wrong_file_sets(paths, tmp_path: Path, names: list[str]) -> None:
    files = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"x")
        files.append(path)

    with pytest.raises(VoiceError):
        voices.add_voice(paths, "singer", files)
    assert voices.list_voices(paths) == []


def test_add_reports_a_missing_file(paths, tmp_path: Path) -> None:
    with pytest.raises(VoiceError, match="File not found"):
        voices.add_voice(paths, "singer", [tmp_path / "absent.pth"])


@pytest.mark.parametrize("name", ["", "../escape", "has space", "-leading", "a/b"])
def test_invalid_names_are_rejected(paths, model_files, name: str) -> None:
    with pytest.raises(VoiceError, match="Invalid voice name"):
        voices.add_voice(paths, name, [model_files[0]])


def test_get_unknown_voice_lists_what_is_available(paths, model_files) -> None:
    voices.add_voice(paths, "singer", [model_files[0]])

    with pytest.raises(VoiceError, match=r"No voice named 'other' \(available: singer\)"):
        voices.get_voice(paths, "other")


def test_remove(paths, model_files) -> None:
    voices.add_voice(paths, "singer", list(model_files))

    voices.remove_voice(paths, "singer")

    assert voices.list_voices(paths) == []


def test_resolve_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(voices.HF_REPO_ENV, raising=False)
    assert voices.resolve_repo("me/voices") == "me/voices"
    with pytest.raises(VoiceError, match="No Hugging Face repository"):
        voices.resolve_repo(None)
    with pytest.raises(VoiceError, match="<user>/<name>"):
        voices.resolve_repo("just-a-name")

    monkeypatch.setenv(voices.HF_REPO_ENV, "me/from-env")
    assert voices.resolve_repo(None) == "me/from-env"
    assert voices.resolve_repo("me/explicit") == "me/explicit"


class FakeHfApi:
    """Stands in for huggingface_hub.HfApi; records what would be uploaded."""

    private = True
    instances: list[FakeHfApi] = []

    def __init__(self) -> None:
        self.created: list[tuple[str, bool]] = []
        self.uploaded: list[str] = []
        FakeHfApi.instances.append(self)

    def create_repo(self, repo_id: str, *, private: bool, exist_ok: bool, repo_type: str) -> None:
        self.created.append((repo_id, private))

    def repo_info(self, repo_id: str, *, repo_type: str) -> SimpleNamespace:
        return SimpleNamespace(private=self.private)

    def upload_file(self, *, path_or_fileobj: str, path_in_repo: str, repo_id: str, repo_type: str):
        self.uploaded.append(path_in_repo)


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch) -> type[FakeHfApi]:
    import huggingface_hub

    FakeHfApi.instances = []
    FakeHfApi.private = True
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    return FakeHfApi


def test_push_creates_a_private_repo_and_uploads_both_files(paths, model_files, fake_hf) -> None:
    voices.add_voice(paths, "singer", list(model_files))

    repo = voices.push_voice(paths, "singer", "me/voices")

    api = fake_hf.instances[0]
    assert repo == "me/voices"
    assert api.created == [("me/voices", True)]
    assert api.uploaded == ["singer/singer.pth", "singer/singer.index"]


def test_push_refuses_a_public_repo(paths, model_files, fake_hf) -> None:
    voices.add_voice(paths, "singer", list(model_files))
    fake_hf.private = False  # the repo already existed and is public

    with pytest.raises(VoiceError, match="is public"):
        voices.push_voice(paths, "singer", "me/voices")

    assert fake_hf.instances[0].uploaded == []


def test_pull_downloads_only_the_named_voice(paths, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    calls: list[dict] = []

    def fake_snapshot_download(*, repo_id: str, allow_patterns: list[str], local_dir: str) -> str:
        calls.append({"repo_id": repo_id, "allow_patterns": allow_patterns})
        folder = Path(local_dir) / "singer"
        folder.mkdir(parents=True)
        (folder / "singer.pth").write_bytes(b"remote-model")
        (folder / "singer.index").write_bytes(b"remote-index")
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    pulled = voices.pull_voice(paths, "singer", "me/voices")

    assert calls == [{"repo_id": "me/voices", "allow_patterns": ["singer/*.pth", "singer/*.index"]}]
    assert pulled.model.read_bytes() == b"remote-model"
    assert pulled.index is not None and pulled.index.read_bytes() == b"remote-index"


def test_pull_reports_a_voice_missing_from_the_repo(paths, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_kwargs: None)

    with pytest.raises(VoiceError, match="no voice files under singer/"):
        voices.pull_voice(paths, "singer", "me/voices")
