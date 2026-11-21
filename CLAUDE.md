# song-cloner

Personal tool. The owner gives a song; the lead vocal is re-voiced with a voice model the
owner trained, and the remixed song comes back. Private listening only: nothing made here is
published or shared.

## Converting a song for the owner (the usual request)

1. Check the machine: `uv run songcloner doctor`. If the engine is not ready, run
   `bash scripts/setup.sh` (idempotent; installs sox/ffmpeg, the Python environment and about
   1.5 GB of base models).
2. Check the voice: `uv run songcloner voices list`. A fresh cloud machine has none. Fetch it with
   `uv run songcloner voices pull <name> --repo <user>/<repo>` (the owner's private Hugging Face
   repo; `SONGCLONER_HF_REPO` may hold the default). If huggingface.co answers 401/403, the
   environment has no credential for it: ask for one with the credential tool, never in chat.
3. Get the song. An uploaded file works as is. A YouTube link needs `www.youtube.com` and the
   media hosts `*.googlevideo.com` on the environment's allowed hosts; if the download is
   refused, ask the owner for the audio file instead of looking for another route.
4. Convert: `uv run songcloner convert <file-or-url> --voice <name>`. On a 4-core CPU this
   takes about 10 times the song's length (about 7 times with `--quality standard`), so run
   it in the background and tell the owner how long to expect.
5. Send the owner the file written to `outputs/`.

## Rented GPUs (`scripts/remote`)

When the environment has a huggingface.co credential, convert and train on Hugging Face's
GPUs instead of this machine's CPU: `scripts/remote convert <song> --voice <name>` takes
minutes and costs cents; `scripts/remote train <name> <songs...>` builds the dataset and
trains. Results live in the owner's private bucket `<user>/song-cloner`.

- The owner asked to keep the cost low. Use the default hardware (`t4-small`) and the default
  `--max-cost`; do not raise either, or pick pricier hardware, without asking. Say what a job
  can cost at most before starting it (the command prints it).
- For training, run `--dataset-only` first and read `reports/train-<name>.json`
  (`scripts/remote fetch reports/`) before paying for the training run.
- A job runs the commit that is checked out here, downloaded from GitHub: commit and push
  first. `scripts/remote` refuses otherwise.
- Never leave a job running unattended without a reason: check `scripts/remote status <id>`,
  and `scripts/remote cancel <id>` anything stuck.
- No token is passed into jobs. They read and write the bucket as a mounted directory.

Pitch: the converted vocal keeps the original singer's notes. When the original is sung by a
woman and the voice is a man's (or the reverse), add `--octaves -1` (or `1`), or try
`--auto-pitch`. `--semitones N` changes the key of the whole song instead.

## Rules

- This repository is public. Never commit songs, datasets, voice models or converted audio.
  `.gitignore` covers `data/`, `outputs/`, `songs/`, `datasets/` and audio/model extensions;
  keep new working folders inside those.
- Voices come only from the owner: trained with `notebooks/song_cloner_colab.ipynb`,
  `scripts/remote train` or `songcloner train`, or files the owner hands over. Do not download third-party voice models.
  A `.pth` file is a pickle and runs code when loaded, and the engine's bundled sample
  celebrity voices are switched off on purpose (`engine.prepare` writes the marker that
  prevents their download).
- Do not add features that publish or upload covers.
- Do not call `urvc` (the engine's own CLI) directly: it would use different directories and
  download the sample voices. Go through `songcloner`.

## Code map

- `src/songcloner/engine.py` is the only module that imports the engine (`ultimate-rvc`,
  pinned in `pyproject.toml`). All of its engine imports are lazy and run after `prepare()`,
  because the engine reads its directories from the environment at import time.
- `voices.py` (local registry and private Hugging Face sync), `convert.py`, `dataset.py`,
  `training.py` hold the logic; `cli.py` only parses arguments and prints.
- `engine.run_cover` composes the engine's own steps (separate, convert, mix) instead of
  calling its `run_pipeline`, so that the vocal separator can be chosen (`--quality`). The
  choice was measured on a track with ground-truth stems; re-measure before changing it.
- `paths.py`: engine data lives in `./data` (`SONGCLONER_HOME`), results in `./outputs`
  (`SONGCLONER_OUTPUTS`).
- The Colab notebook runs `job.py` too, with a Google Drive folder in place of the bucket,
  inside an environment built by `scripts/setup_gpu.sh`. Its `CODE_VERSION` setting names the
  branch it installs: keep it pointing at a branch that exists (`main` once merged).
- `remote.py` launches Hugging Face Jobs and `job.py` is what runs inside one. `remote.py`
  needs `huggingface_hub` 2.4+, which the engine's pins forbid in the project environment,
  so `scripts/remote` runs it in its own environment; `job.py` uses only the bucket
  directory, never the Hub API.

## Checks

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

Unit tests replace the engine with stubs, so they pass without it installed. After changing
`engine.py`, or the pinned engine version, also run a real conversion and a short training
run (README, "Smoke test"): the stubs cannot tell you the engine still accepts our calls.

Training needs a GPU. On a 4-core CPU it is only good for a 1-2 epoch smoke test.
