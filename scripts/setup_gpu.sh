#!/usr/bin/env bash
# Install song-cloner on a machine with an NVIDIA GPU (Google Colab, a GPU server),
# into its own Python environment.
#
#   bash scripts/setup_gpu.sh [ENV_DIR]     default ENV_DIR: .venv-gpu in this repository
#
# scripts/setup.sh installs whatever PyTorch build PyPI offers, which is fine on a
# CPU. This script installs the engine's "cuda" variant instead: the PyTorch build
# (CUDA 12.8) that the engine's authors use on GPUs, from PyTorch's own package index.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="${1:-$ROOT/.venv-gpu}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

if ! command -v sox >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq --no-install-recommends sox ffmpeg
  else
    echo "Please install sox and ffmpeg, then run this script again." >&2
    exit 1
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --quiet uv
fi

# The engine version is pinned in one place, pyproject.toml.
ENGINE_VERSION="$(sed -n 's/^engine = \["ultimate-rvc==\([^"]*\)"\]$/\1/p' "$ROOT/pyproject.toml")"
if [ -z "$ENGINE_VERSION" ]; then
  echo "Could not read the engine version from pyproject.toml." >&2
  exit 1
fi

uv venv --quiet --allow-existing --python 3.12 "$ENV_DIR"
# unsafe-best-match: take torch from PyTorch's index and everything else from
# PyPI. Without it uv would insist on PyTorch's index for every package that
# index happens to mirror, and the engine's exact pins are not all there.
uv pip install --quiet --python "$ENV_DIR/bin/python" \
  --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
  "ultimate-rvc[cuda]==$ENGINE_VERSION" -e "$ROOT[engine]"

"$ENV_DIR/bin/songcloner" doctor
