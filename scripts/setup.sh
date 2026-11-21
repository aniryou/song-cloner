#!/usr/bin/env bash
# Prepare a machine for song-cloner: audio tools, Python environment, base models.
# Safe to run again; each step skips work that is already done.
#
#   bash scripts/setup.sh              conversion only (about 1.5 GB of models)
#   bash scripts/setup.sh --training   also fetch the pretrained models for training
set -euo pipefail

cd "$(dirname "$0")/.."

# sox is needed for pitch shifting. Without a system copy the engine tries to
# download its own, which restricted networks (Claude cloud sessions) block.
if ! command -v sox >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq --no-install-recommends sox ffmpeg
  elif command -v brew >/dev/null 2>&1; then
    brew install sox ffmpeg
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

uv sync --quiet --extra engine --extra dev

if [ "${1:-}" = "--training" ]; then
  uv run songcloner setup --training-sample-rate 40000
else
  uv run songcloner setup
fi
uv run songcloner doctor
