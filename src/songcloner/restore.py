"""Freshen the backing track of an old recording.

The converted vocal is re-synthesised, so it sounds clean whatever the age of
the source. The backing track is not: on an old recording it stays hissy,
narrow and dull, and the new vocal sits oddly on top of it. The steps here
work on the separated instrumental before it is mixed back in. They cannot
replace the instruments; they make the same recording cleaner and fuller.

Each step is independent and has three strengths, because what helps depends
on the recording and has to be judged by ear:

``denoise``  lower steady hiss
``tone``     more bass and treble, slightly less of the boxy midrange
``air``      add the top octave that old tape never captured
``widen``    spread a mono or narrow recording across the stereo field
``glue``     gentle compression, for a denser, more even sound

A spec is a comma-separated list such as ``"denoise,tone"``; a strength from 1
(light) to 3 (strong) follows a colon, as in ``"denoise:3,air:1"``. ``"all"``
means every step at medium strength, ``"all:1"`` and ``"all:3"`` every step at
that strength. Steps always run in the order above.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

STEPS = ("denoise", "tone", "air", "widen", "glue")
LEVELS = (1, 2, 3)
DEFAULT_LEVEL = 2


class RestoreError(Exception):
    """A problem with a restore spec that the user can act on."""


@dataclass(frozen=True)
class Step:
    name: str
    level: int = DEFAULT_LEVEL


def parse(spec: str | None) -> list[Step]:
    """Turn a spec such as ``"tone,denoise:3"`` into steps in their running order."""
    text = (spec or "").strip().lower()
    if text in {"", "none", "off"}:
        return []
    chosen: dict[str, int] = {}
    for part in text.split(","):
        name, _, level_text = part.strip().partition(":")
        if name not in STEPS and name != "all":
            raise RestoreError(
                f"Unknown restore step {name!r}; choose from: {', '.join(STEPS)}, or 'all'."
            )
        try:
            level = int(level_text) if level_text else DEFAULT_LEVEL
        except ValueError:
            level = 0
        if level not in LEVELS:
            raise RestoreError(
                f"Strength of {name!r} must be 1, 2 or 3 (light, medium, strong), "
                f"got {level_text!r}."
            )
        for step in STEPS if name == "all" else (name,):
            if step in chosen:
                raise RestoreError(f"Restore step {step!r} is listed twice.")
            chosen[step] = level
    return [Step(name, chosen[name]) for name in STEPS if name in chosen]


def tag(steps: list[Step]) -> str:
    """A short label for file names, e.g. ``"denoise+tone3"``; empty when there are no steps."""
    return "+".join(
        step.name if step.level == DEFAULT_LEVEL else f"{step.name}{step.level}" for step in steps
    )


# --- The steps. Audio is a float array of shape (samples, 2). -----------------


def _denoise(audio: np.ndarray, sample_rate: int, level: int) -> np.ndarray:
    """Lower steady hiss by spectral gating.

    The hiss level in each frequency band is taken to be about the quietest
    that band gets (music leaves every band silent now and then; hiss never
    does). Whatever sits near that floor is turned down; anything clearly
    above it is left alone. The reduction is capped, which keeps the watery
    artefacts of heavy noise reduction away.
    """
    import numpy as np
    from scipy import ndimage, signal

    reduction_db = {1: 6.0, 2: 10.0, 3: 14.0}[level]
    window = 2048
    cleaned = np.empty_like(audio)
    for channel in range(audio.shape[1]):
        _, _, spectrum = signal.stft(audio[:, channel], sample_rate, nperseg=window)
        power = np.abs(spectrum) ** 2
        # Hiss flickers from frame to frame. Averaging about a fifth of a second
        # steadies it, so that its floor can be read off and compared against.
        steady_db = 10 * np.log10(ndimage.uniform_filter1d(power, size=9, axis=1) + 1e-20)
        floor_db = np.percentile(steady_db, 10, axis=1, keepdims=True)
        # Where the music never pauses, a band's quietest moment is still music,
        # and gating against it would turn the music itself down. Hiss is about
        # equally strong in every band, so the bands with the lowest floors show
        # its true level; no band's floor may sit far above that.
        audible = slice(int(200 * window / sample_rate), int(16000 * window / sample_rate))
        hiss_db = np.percentile(floor_db[audible], 10)
        floor_db = np.minimum(floor_db, hiss_db + 6.0)
        keep = np.clip((steady_db - floor_db - 4.0) / 8.0, 0.0, 1.0)
        # A sudden loud sound is music even before the average has caught up.
        instant_db = 10 * np.log10(power + 1e-20)
        keep = np.maximum(keep, np.clip((instant_db - floor_db - 12.0) / 6.0, 0.0, 1.0))
        keep = ndimage.uniform_filter1d(keep, size=3, axis=0)
        gain = 10 ** (-(1.0 - keep) * reduction_db / 20)
        _, restored = signal.istft(spectrum * gain, sample_rate, nperseg=window)
        cleaned[:, channel] = _fit(restored, len(audio))
    return cleaned


def _shelf(kind: str, frequency: float, gain_db: float, sample_rate: int) -> np.ndarray:
    """Second-order shelving filter (Audio EQ Cookbook), as one second-order section."""
    import numpy as np

    amplitude = 10 ** (gain_db / 40)
    omega = 2 * np.pi * frequency / sample_rate
    cos, alpha = np.cos(omega), np.sin(omega) / np.sqrt(2)
    root = 2 * np.sqrt(amplitude) * alpha
    sign = 1 if kind == "low" else -1
    b0 = amplitude * ((amplitude + 1) - sign * (amplitude - 1) * cos + root)
    b1 = sign * 2 * amplitude * ((amplitude - 1) - sign * (amplitude + 1) * cos)
    b2 = amplitude * ((amplitude + 1) - sign * (amplitude - 1) * cos - root)
    a0 = (amplitude + 1) + sign * (amplitude - 1) * cos + root
    a1 = -sign * 2 * ((amplitude - 1) + sign * (amplitude + 1) * cos)
    a2 = (amplitude + 1) + sign * (amplitude - 1) * cos - root
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _peak(frequency: float, gain_db: float, q: float, sample_rate: int) -> np.ndarray:
    """Peaking filter (Audio EQ Cookbook), as one second-order section."""
    import numpy as np

    amplitude = 10 ** (gain_db / 40)
    omega = 2 * np.pi * frequency / sample_rate
    alpha = np.sin(omega) / (2 * q)
    a0 = 1 + alpha / amplitude
    return np.array(
        [
            [
                (1 + alpha * amplitude) / a0,
                -2 * np.cos(omega) / a0,
                (1 - alpha * amplitude) / a0,
                1.0,
                -2 * np.cos(omega) / a0,
                (1 - alpha / amplitude) / a0,
            ]
        ]
    )


def _tone(audio: np.ndarray, sample_rate: int, level: int) -> np.ndarray:
    """Lift the bass and treble that old recordings lack, and ease the boxy midrange."""
    import numpy as np
    from scipy import signal

    shelf_db = {1: 2.0, 2: 3.5, 3: 5.0}[level]
    mid_db = {1: -1.0, 2: -2.0, 3: -3.0}[level]
    sections = np.vstack(
        [
            signal.butter(2, 30, "highpass", fs=sample_rate, output="sos"),  # rumble
            _shelf("low", 110, shelf_db, sample_rate),
            _peak(2500, mid_db, 0.9, sample_rate),
            _shelf("high", 7000, shelf_db, sample_rate),
        ]
    )
    return signal.sosfilt(sections, audio, axis=0)


def _air(audio: np.ndarray, sample_rate: int, level: int) -> np.ndarray:
    """Add high frequencies that are not on the recording.

    Old tape stops somewhere between 8 and 12 kHz. An exciter makes new
    overtones from the upper midrange that is there, and only what lands above
    6 kHz is mixed in, quietly.
    """
    import numpy as np
    from scipy import signal

    amount = {1: 0.45, 2: 0.8, 3: 1.4}[level]
    source = signal.sosfilt(
        signal.butter(2, [2500, 7000], "bandpass", fs=sample_rate, output="sos"), audio, axis=0
    )
    scale = np.sqrt(np.mean(source**2)) + 1e-9
    # Soft clipping creates third and fifth harmonics, which land at 7.5 kHz
    # and above. The high-pass keeps those and drops the band they came from.
    # Nothing is subtracted, so highs already on the recording are never thinned.
    overtones = signal.sosfilt(
        signal.butter(4, 7500, "highpass", fs=sample_rate, output="sos"),
        np.tanh(3.0 * source / scale) * (scale / 3.0),
        axis=0,
    )
    return audio + amount * overtones


def _widen(audio: np.ndarray, sample_rate: int, level: int) -> np.ndarray:
    """Spread a mono or narrow recording across the stereo field.

    A short echo of the centre is added to the left and subtracted from the
    right, which makes the two sides differ without changing their sum: played
    on one speaker the result is exactly the original. The bass stays central.
    A recording that is already wide is left as it is.
    """
    import numpy as np
    from scipy import signal

    left, right = audio[:, 0], audio[:, 1]
    if np.corrcoef(left, right)[0, 1] < 0.6:
        return audio
    depth = {1: 0.25, 2: 0.40, 3: 0.55}[level]
    centre = signal.sosfilt(
        signal.butter(2, 200, "highpass", fs=sample_rate, output="sos"), (left + right) / 2
    )
    delay = int(sample_rate * 0.011)
    echo = np.concatenate([np.zeros(delay), centre[:-delay]]) * depth
    return np.stack([left + echo, right - echo], axis=1)


def _glue(audio: np.ndarray, sample_rate: int, level: int) -> np.ndarray:
    """Gentle compression: the loudest moments come down a little, for a denser sound."""
    import numpy as np
    from scipy import signal

    ratio = {1: 1.5, 2: 2.0, 3: 3.0}[level]
    smooth = 1 - np.exp(-1 / (0.05 * sample_rate))  # 50 ms
    power = signal.lfilter([smooth], [1, smooth - 1], np.mean(audio**2, axis=1))
    level_db = 10 * np.log10(power + 1e-12)
    # Work on whatever is louder than the track's typical level.
    threshold_db = 10 * np.log10(np.mean(audio**2) + 1e-12)
    over_db = np.maximum(level_db - threshold_db, 0.0)
    gain = 10 ** (-over_db * (1 - 1 / ratio) / 20)
    return audio * gain[:, None]


_FUNCTIONS = {"denoise": _denoise, "tone": _tone, "air": _air, "widen": _widen, "glue": _glue}


def _fit(samples: np.ndarray, length: int) -> np.ndarray:
    import numpy as np

    if len(samples) >= length:
        return samples[:length]
    return np.concatenate([samples, np.zeros(length - len(samples))])


def apply(audio: np.ndarray, sample_rate: int, steps: list[Step]) -> np.ndarray:
    """Run ``steps`` over ``audio`` (samples x channels); return stereo audio.

    The result is brought back to the loudness of the input, so that the
    balance between the backing track and the vocal is whatever it was before.
    """
    import numpy as np

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if not steps or not np.any(audio):
        return audio
    loudness = np.sqrt(np.mean(audio**2))
    for step in steps:
        audio = _FUNCTIONS[step.name](audio, sample_rate, step.level)
    audio = audio * (loudness / (np.sqrt(np.mean(audio**2)) + 1e-12))
    peak = np.max(np.abs(audio))
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    return audio


def restore_file(source: Path, destination: Path, steps: list[Step]) -> Path:
    """Apply ``steps`` to the audio file ``source`` and write the result to ``destination``."""
    import soundfile as sf

    audio, sample_rate = sf.read(str(source), always_2d=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), apply(audio, sample_rate, steps), sample_rate)
    return destination
