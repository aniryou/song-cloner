from __future__ import annotations

from pathlib import Path

import pytest

from songcloner.paths import Paths


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths(home=tmp_path / "data", outputs=tmp_path / "outputs")


@pytest.fixture
def model_files(tmp_path: Path) -> tuple[Path, Path]:
    """A fake model/index pair. Real ones are large binaries; content is irrelevant here."""
    source = tmp_path / "source"
    source.mkdir()
    model = source / "singer.pth"
    index = source / "singer.index"
    model.write_bytes(b"model-bytes")
    index.write_bytes(b"index-bytes")
    return model, index
