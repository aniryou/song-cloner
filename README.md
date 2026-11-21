# song-cloner

Give it a song and a voice model, and it returns the song with the lead vocal sung in that
voice. The music and backing vocals are kept.

It is a thin command-line wrapper around [ultimate-rvc](https://github.com/JackismyShephard/ultimate-rvc),
which does the audio work:

1. **Separate** the song into lead vocal, backing vocals and instrumental (BS-Roformer and
   UVR MDX-Net models).
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
  from songs you provide (see [Training a voice](#training-a-voice)); that needs a GPU.
- A **song**: an audio file, or a YouTube link if your network can reach YouTube.

A `.pth` file is a Python pickle: loading one runs code from it. Only use models you trained
yourself or fully trust.

## Setup

```bash
bash scripts/setup.sh
```

This installs `sox` and `ffmpeg` if missing, creates the Python environment with
[uv](https://docs.astral.sh/uv/) (Python 3.12, about 7 GB with PyTorch) and downloads about
1.5 GB of base models into `data/`. Run it again any time; finished steps are skipped.

## Converting a song

```bash
uv run songcloner voices add myvoice path/to/model.pth path/to/model.index
uv run songcloner convert path/to/song.mp3 --voice myvoice
```

The result lands in `outputs/`. Without a GPU this is slow, and most of the time goes into
the three separation passes. Measured on a 4-core CPU: about 10 times the length of the song
at the default quality (a one-minute excerpt took just under 10 minutes), about 7 times with
`--quality standard` (27 minutes for a 4-minute song). A GPU is far faster.

The default vocal separator (BS-Roformer) was chosen by measurement: on a MUSDB18 track with
ground-truth stems it was about 1 dB better than the standard one (UVR MDX-Net Voc FT) on
both the isolated vocal and the instrumental, which means less of the original singer left
behind in the music.

Useful options (`uv run songcloner convert --help` lists all):

| Option | Use it when |
| --- | --- |
| `--octaves -1` / `--octaves 1` | The original singer is a woman and the voice is a man's, or the reverse. Shifts only the vocal. |
| `--auto-pitch` | You want the engine to move the vocal into the voice's natural range for you. |
| `--semitones N` | The song sits too high or low for the voice: changes the key of the whole song. |
| `--index-rate 0.5` | The result does not sound enough like the voice (higher) or has artefacts (lower). |
| `--quality standard` | You want a faster conversion on a CPU and accept a little more of the original singer bleeding into the music. |
| `--vocal-gain`, `--inst-gain`, `--backing-gain` | The mix balance is off (values in dB). |
| `--format wav` | You want lossless output instead of mp3. |

Set `SONGCLONER_VOICE=myvoice` to stop typing `--voice`.

### Old recordings

The converted vocal is re-synthesised, so it comes out clean whatever the age of the song.
The backing track does not: on an old recording it stays hissy, narrow and dull. The most
effective fix is to convert a modern, well-recorded cover of the song instead, since the tool
keeps whatever backing track it is given. When there is none, `--restore` freshens the old
one. It cannot replace the instruments; it makes the same recording cleaner and fuller.

| Step | What it does |
| --- | --- |
| `denoise` | Lowers steady tape hiss. |
| `tone` | More bass and treble, a little less of the boxy midrange. |
| `air` | Adds the top end that old tape never captured. |
| `widen` | Spreads a mono or narrow recording across left and right. Played on one speaker, the result is unchanged. |
| `glue` | Gentle compression, for a denser sound. |

```bash
uv run songcloner convert old-song.mp3 --voice myvoice --restore denoise,tone
uv run songcloner convert old-song.mp3 --voice myvoice --restore tone:3,air:1
uv run songcloner convert old-song.mp3 --voice myvoice --restore all
```

Combine steps with commas; add `:1` for light or `:3` for strong; `all` is every step at
medium strength (`all:1`, `all:3`). Each combination gets its own file name, such as
`Old Song (myvoice Ver) [denoise+tone].mp3`, and separation and voice conversion are reused
between runs, so trying another combination on the same song takes seconds. Which one sounds
best depends on the recording: compare them by ear.

What was measured: a modern instrumental was aged artificially (mono, no bass or treble,
boxy mids, hiss) and restored. `all` cut the gap in octave-band balance to the original from
6.8 dB to 4.0 dB (`all:3`: 2.6 dB), and `denoise` took 10 dB off the hiss without lowering
the music. That says the steps move in the right direction, not that the result sounds good.

## Training a voice

Training needs a GPU. On a 4-core CPU one epoch over 6 minutes of audio takes about 3
minutes, and a usable model needs a few hundred epochs.

**On Google Colab (free GPU):** open
[`notebooks/song_cloner_colab.ipynb`](https://colab.research.google.com/github/aniryou/song-cloner/blob/main/notebooks/song_cloner_colab.ipynb)
in Colab, choose the **T4 GPU** runtime (a TPU does not work) and run the cells in order. It
keeps everything in a `song-cloner` folder on your Google Drive: you put training songs in
`songs/<voice>/`, it isolates the vocals, trains the voice and converts songs you drop into
`inbox/`. Progress is saved to Drive as it goes, so a disconnected session can be continued.

**On your own GPU machine:**

```bash
bash scripts/setup_gpu.sh                            # installs into .venv-gpu
.venv-gpu/bin/songcloner dataset myvoice path/to/songs      # files, folders or YouTube URLs
.venv-gpu/bin/songcloner train myvoice datasets/myvoice
```

`setup_gpu.sh` installs the PyTorch build for NVIDIA GPUs that the engine's authors use;
`setup.sh` installs the default build, which is the right one for a machine without a GPU.

The voice can only be as good as what it learns from, so the dataset is its own step:

- `songcloner dataset` strips the music, backing vocals and reverb from each song and writes
  the singer's dry lead vocal to `datasets/NAME/`. It reports how many minutes of singing it
  found: under 10 is too little, 20 or more is good. Run it again to add songs; ones already
  there are skipped.
- Choose songs where the singer is alone: no duets, little chorus, no heavy vocal effects.
  Listen to a few of the clips and delete any that contain other voices before training.
- If you already have clean solo vocals, skip `dataset` and point `train` at their folder.
- `--epochs` defaults to 200. Training stops early when the model stops improving.

## Renting a GPU

`scripts/remote` runs the slow parts on Hugging Face's GPUs instead of your machine. It
starts a [Hugging Face Job](https://huggingface.co/docs/hub/jobs) that installs this
project, does the work and writes the result to a private storage bucket in your account
(`<you>/song-cloner`), which holds your songs, isolated vocals, voices and covers.

You need a Hugging Face account with prepaid credit and an access token of type Write
(`hf auth login`, or the `HF_TOKEN` environment variable).

```bash
scripts/remote smoke                                    # cheap check that jobs work
scripts/remote train myvoice path/to/songs --dataset-only   # isolate vocals, report minutes
scripts/remote train myvoice                            # train on what is in the bucket
scripts/remote convert path/to/song.mp3 --voice myvoice # result lands in outputs/
scripts/remote logs JOB    scripts/remote status JOB    scripts/remote cancel JOB
```

`train` accepts files, folders and URLs, uploads the files to the bucket and keeps the
isolated vocals there, so later runs only process new songs.

Keeping the cost down:

- Jobs default to the cheapest GPU (`t4-small`, $0.40 an hour) and are billed by the minute.
- Every job has a hard time limit (`--hours`). A job whose time limit could cost more than
  `--max-cost` (default $2) is refused before anything is uploaded or started.
- While training, the best model so far is copied to the bucket every two minutes, so a job
  that hits its time limit still leaves a usable voice.
- `train` refuses to start training on less than five minutes of singing.

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
| `songcloner dataset NAME SONGS...` | Isolate the singer's vocal from songs, for training. |
| `songcloner train NAME FOLDER` | Train a voice from recordings. |
| `songcloner voices list / add / remove / pull / push` | Manage voice models. |
| `scripts/remote smoke / train / convert / logs / status / cancel / fetch` | The same work on a rented GPU. |

Files live in `data/` (engine data, voices), `datasets/` (training vocals) and `outputs/`
(results), all relative to the current directory. `SONGCLONER_HOME` and `SONGCLONER_OUTPUTS` override them.

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
