# song-cloner

Give it a song and a voice model, and it returns the song with the lead vocal sung in that
voice. The music and backing vocals are kept.

It is a thin command-line wrapper around [ultimate-rvc](https://github.com/JackismyShephard/ultimate-rvc),
which does the audio work:

1. **Separate** the song into lead vocal, backing vocals and instrumental (UVR MDX-Net models).
2. **Convert** the lead vocal with an RVC voice model (voice conversion: the melody, words and
   phrasing stay; the timbre changes).
3. **Remix** the converted vocal with the untouched instrumental and backing vocals.

## Personal use only

This is for private listening. Do not publish or share covers made with it, or the voice
models. Cloning a real singer's voice without their consent can infringe their personality
and publicity rights, and courts have already ruled against services offering it; the songs
themselves are copyrighted too. For the same reason this repository contains code only:
songs, datasets, voice models and results are git-ignored.

## What you need

- A **voice model** you supply: one `.pth` file and, ideally, its `.index` file. Train one
  from your own recordings with the [Colab notebook](notebooks/train_voice_colab.ipynb), or
  with `songcloner train` on a machine with an NVIDIA GPU.
- A **song**: an audio file, or a YouTube link if your network can reach YouTube.

A `.pth` file is a Python pickle: loading one runs code from it. Only use models you trained
yourself or fully trust.

## Setup

```bash
bash scripts/setup.sh
```

This installs `sox` and `ffmpeg` if missing, creates the Python environment with
[uv](https://docs.astral.sh/uv/) (Python 3.12, about 7 GB with PyTorch) and downloads about
1 GB of base models into `data/`. Run it again any time; finished steps are skipped.

## Converting a song

```bash
uv run songcloner voices add myvoice path/to/model.pth path/to/model.index
uv run songcloner convert path/to/song.mp3 --voice myvoice
```

The result lands in `outputs/`. Without a GPU, expect about 7 times the length of the song:
a 4-minute song took 27 minutes on a 4-core CPU, most of it in the three separation passes.
A GPU is far faster.

Useful options (`uv run songcloner convert --help` lists all):

| Option | Use it when |
| --- | --- |
| `--octaves -1` / `--octaves 1` | The original singer is a woman and the voice is a man's, or the reverse. Shifts only the vocal. |
| `--auto-pitch` | You want the engine to move the vocal into the voice's natural range for you. |
| `--semitones N` | The song sits too high or low for the voice: changes the key of the whole song. |
| `--index-rate 0.5` | The result does not sound enough like the voice (higher) or has artefacts (lower). |
| `--vocal-gain`, `--inst-gain`, `--backing-gain` | The mix balance is off (values in dB). |
| `--format wav` | You want lossless output instead of mp3. |

Set `SONGCLONER_VOICE=myvoice` to stop typing `--voice`.

## Training a voice

Training needs a GPU. On a 4-core CPU one epoch over 6 minutes of audio takes about 3
minutes, and a usable model needs a few hundred epochs.

**On Google Colab (free GPU):** open `notebooks/train_voice_colab.ipynb` in Colab, point it
at a Google Drive folder of recordings and run the cells. It saves the model to your Drive.

**On your own GPU machine:**

```bash
bash scripts/setup.sh --training
uv run songcloner train myvoice path/to/recordings --isolate-vocals
```

About the recordings:

- 10 to 30 minutes of one singer, in total. Quality beats quantity.
- Clean solo vocals are best. If you only have full songs, pass `--isolate-vocals` and the
  music, backing vocals and reverb are stripped first. Avoid duets and heavily processed
  vocals.
- `--epochs` defaults to 200. Training stops early when the model stops improving.

## Keeping a voice for cloud sessions

Machines that start empty each time (such as Claude cloud sessions) need to download the
voice. A private Hugging Face model repository works well:

```bash
export HF_TOKEN=...                       # a token with write access
uv run songcloner voices push myvoice --repo your-name/song-cloner-voices
uv run songcloner voices pull myvoice --repo your-name/song-cloner-voices   # on the other machine
```

`push` creates the repository as private and refuses to upload to a public one. Set
`SONGCLONER_HF_REPO` to avoid repeating `--repo`.

## Commands

| Command | What it does |
| --- | --- |
| `songcloner setup` | Download the base models. |
| `songcloner doctor` | Show what is installed, the compute device and the available voices. |
| `songcloner convert SONG --voice NAME` | Convert a song. |
| `songcloner train NAME FOLDER` | Train a voice from recordings. |
| `songcloner voices list / add / remove / pull / push` | Manage voice models. |

Files live in `data/` (engine data, voices) and `outputs/` (results), both relative to the
current directory. `SONGCLONER_HOME` and `SONGCLONER_OUTPUTS` override them.

## Development

```bash
uv sync --extra dev          # add --extra engine for the real thing
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

The unit tests replace the engine with stubs, so they run in seconds without PyTorch.

### Smoke test

After changing `src/songcloner/engine.py` or the pinned engine version, run the real pipeline
once. Any few minutes of speech and any short song will do:

```bash
uv run songcloner train smoke path/to/speech --epochs 2 --batch-size 4 --save-interval 1
uv run songcloner convert path/to/short-song.wav --voice smoke
uv run songcloner voices remove smoke
```

A two-epoch model sounds bad; the point is that every stage runs and a mixed song comes out.
