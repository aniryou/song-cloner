from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from songcloner import restore
from songcloner.restore import RestoreError, Step

SR = 44100
rng = np.random.default_rng(7)


def sine(frequency: float, seconds: float = 2.0, amplitude: float = 0.1) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * frequency * np.arange(int(SR * seconds)) / SR)


def noise(seconds: float = 3.0, amplitude: float = 0.05) -> np.ndarray:
    return amplitude * rng.standard_normal(int(SR * seconds))


def stereo(mono: np.ndarray) -> np.ndarray:
    return np.stack([mono, mono], axis=1)


def db(values: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.square(values)) + 1e-20))


def band_db(audio: np.ndarray, low: float, high: float) -> float:
    """Level of ``audio`` (first channel) between ``low`` and ``high`` Hz."""
    spectrum = np.abs(np.fft.rfft(audio[:, 0])) ** 2
    frequencies = np.fft.rfftfreq(len(audio), 1 / SR)
    return float(10 * np.log10(spectrum[(frequencies >= low) & (frequencies < high)].sum() + 1e-20))


# --- Specs ---------------------------------------------------------------------


@pytest.mark.parametrize("spec", [None, "", "  ", "none", "off", "NONE"])
def test_no_spec_means_no_steps(spec: str | None) -> None:
    assert restore.parse(spec) == []


def test_all_is_every_step_at_medium_strength() -> None:
    assert restore.parse("all") == [Step(name, 2) for name in restore.STEPS]


def test_all_takes_a_strength_too() -> None:
    assert restore.parse("all:3") == [Step(name, 3) for name in restore.STEPS]
    assert restore.tag(restore.parse("all:1")) == "denoise1+tone1+air1+widen1+glue1"


def test_steps_run_in_a_fixed_order_whatever_order_they_are_written_in() -> None:
    # Hiss must go before the treble is lifted, or the lift raises the hiss too.
    assert restore.parse(" Tone , AIR:1,denoise:3 ") == [
        Step("denoise", 3),
        Step("tone", 2),
        Step("air", 1),
    ]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("sparkle", "Unknown restore step 'sparkle'"),
        ("tone,tone:3", "listed twice"),
        ("all,tone:3", "listed twice"),
        ("tone:4", "must be 1, 2 or 3"),
        ("tone:0", "must be 1, 2 or 3"),
        ("tone:loud", "must be 1, 2 or 3"),
        ("tone,", "Unknown restore step ''"),
    ],
)
def test_bad_specs_are_explained(spec: str, message: str) -> None:
    with pytest.raises(RestoreError, match=message):
        restore.parse(spec)


def test_tag_names_the_combination_for_file_names() -> None:
    assert restore.tag([]) == ""
    assert restore.tag(restore.parse("denoise,tone:3,air:1")) == "denoise+tone3+air1"
    assert restore.tag(restore.parse("all")) == "denoise+tone+air+widen+glue"


# --- What each step does to sound ------------------------------------------------


def _bursts_with_hiss() -> tuple[np.ndarray, np.ndarray]:
    """A 1 kHz note switching on and off every half second, over steady hiss."""
    samples = SR * 4
    on = (np.arange(samples) // (SR // 2)) % 2 == 0
    audio = sine(1000, 4.0, 0.2) * on + 0.003 * rng.standard_normal(samples)
    return stereo(audio), on


def test_denoise_lowers_the_hiss_between_notes_and_leaves_the_notes_alone() -> None:
    audio, on = _bursts_with_hiss()
    edge = np.convolve(on.astype(float), np.ones(4410) / 4410, "same")
    gaps, notes = edge < 0.001, edge > 0.999

    light = restore.apply(audio, SR, [Step("denoise", 1)])
    strong = restore.apply(audio, SR, [Step("denoise", 3)])

    def hiss_drop(result: np.ndarray) -> float:
        # Loudness is matched afterwards, so compare the hiss to the notes.
        before = db(audio[gaps]) - db(audio[notes])
        return before - (db(result[gaps]) - db(result[notes]))

    assert hiss_drop(light) > 4.0
    assert hiss_drop(strong) > 10.0
    assert hiss_drop(strong) > hiss_drop(light) + 4.0


def test_denoise_does_not_turn_down_music_that_never_pauses() -> None:
    # With no silence to learn the hiss from, an earlier version took the
    # music's own quiet moments for hiss and turned the music down by 3-6 dB.
    dense = signal.sosfilt(signal.butter(8, 7000, "lowpass", fs=SR, output="sos"), noise(4.0))
    audio = stereo(dense + 0.0005 * rng.standard_normal(len(dense)))  # hiss 40 dB below

    result = restore.apply(audio, SR, [Step("denoise", 2)])

    assert abs(band_db(result, 100, 5000) - band_db(audio, 100, 5000)) < 0.3
    assert band_db(result, 12000, 16000) < band_db(audio, 12000, 16000) - 7.0


def _tone_gains(level: int) -> dict[int, float]:
    """How much each frequency changes relative to 600 Hz, in dB."""
    frequencies = (15, 60, 600, 2500, 12000)
    audio = stereo(sum(sine(f, 3.0, 0.05) for f in frequencies))
    result = restore.apply(audio, SR, [Step("tone", level)])

    def level_at(sound: np.ndarray, frequency: float) -> float:
        return band_db(sound[SR:], frequency * 0.9, frequency * 1.1)  # skip the filter's start

    change = {f: level_at(result, f) - level_at(audio, f) for f in frequencies}
    return {f: change[f] - change[600] for f in frequencies}


def test_tone_lifts_bass_and_treble_eases_the_midrange_and_removes_rumble() -> None:
    gains = _tone_gains(2)

    assert 2.0 < gains[60] < 4.5
    assert 2.5 < gains[12000] < 4.5
    assert -3.0 < gains[2500] < -1.0
    assert gains[15] < -6.0


def test_tone_strength_scales_the_lift() -> None:
    light, strong = _tone_gains(1), _tone_gains(3)

    assert strong[60] > light[60] + 2.0
    assert strong[12000] > light[12000] + 2.0


def test_air_adds_a_top_octave_that_was_not_there() -> None:
    dull = signal.sosfilt(signal.butter(8, 6000, "lowpass", fs=SR, output="sos"), noise())
    audio = stereo(dull)

    result = restore.apply(audio, SR, [Step("air", 2)])

    assert band_db(result, 9000, 16000) > band_db(audio, 9000, 16000) + 15.0
    # The part of the recording that was there is not rewritten.
    assert abs(band_db(result, 100, 5000) - band_db(audio, 100, 5000)) < 0.5


def test_air_never_thins_highs_that_are_already_on_the_recording() -> None:
    # An earlier version subtracted its source band and took about 1 dB off
    # the top of recordings that reached 9 kHz.
    for cutoff in (9000, 20000):
        bright = signal.sosfilt(signal.butter(8, cutoff, "lowpass", fs=SR, output="sos"), noise())
        audio = stereo(bright)

        result = restore.apply(audio, SR, [Step("air", 3)])

        change = band_db(result, 9000, 16000) - band_db(audio, 9000, 16000)
        assert 0.0 <= change < 3.0


def test_widen_makes_the_sides_differ_but_keeps_the_mono_sum() -> None:
    audio = stereo(noise())

    result = restore.apply(audio, SR, [Step("widen", 2)])

    assert np.corrcoef(result[:, 0], result[:, 1])[0, 1] < 0.9
    # On one speaker the result must be the original, only scaled by the
    # loudness matching; otherwise mono listeners would hear an echo.
    mono_in, mono_out = audio.sum(axis=1), result.sum(axis=1)
    scale = np.dot(mono_out, mono_in) / np.dot(mono_in, mono_in)
    assert np.allclose(mono_out, scale * mono_in, atol=1e-9)


def test_widen_leaves_the_bass_in_the_centre() -> None:
    result = restore.apply(stereo(sine(80)), SR, [Step("widen", 3)])

    assert np.corrcoef(result[SR // 2 :, 0], result[SR // 2 :, 1])[0, 1] > 0.99


def test_widen_does_not_touch_a_recording_that_is_already_stereo() -> None:
    audio = np.stack([noise(), noise()], axis=1)

    result = restore.apply(audio, SR, [Step("widen", 3)])

    assert np.allclose(result, audio)


def test_glue_brings_loud_passages_closer_to_quiet_ones() -> None:
    samples = SR * 4
    loud = (np.arange(samples) >= SR * 2).astype(float)
    audio = stereo(noise(4.0, 0.02) * (1 + 3 * loud))  # second half 12 dB louder
    quiet_part, loud_part = slice(SR // 2, SR * 2 - SR // 2), slice(SR * 2 + SR // 2, samples)

    result = restore.apply(audio, SR, [Step("glue", 2)])

    before = db(audio[loud_part]) - db(audio[quiet_part])
    after = db(result[loud_part]) - db(result[quiet_part])
    assert before == pytest.approx(12.0, abs=0.3)
    assert 1.0 < before - after < 6.0


# --- The whole chain ---------------------------------------------------------------


def test_apply_keeps_the_loudness_so_the_vocal_balance_is_unchanged() -> None:
    audio = stereo(noise(3.0, 0.03))

    result = restore.apply(audio, SR, restore.parse("all"))

    assert db(result) == pytest.approx(db(audio), abs=0.05)
    assert result.shape == audio.shape


def test_apply_without_steps_returns_the_audio_as_stereo() -> None:
    mono = noise(1.0)[:, None]

    result = restore.apply(mono, SR, [])

    assert result.shape == (len(mono), 2)
    assert np.array_equal(result[:, 0], mono[:, 0]) and np.array_equal(result[:, 1], mono[:, 0])


def test_apply_survives_silence() -> None:
    result = restore.apply(np.zeros((SR, 2)), SR, restore.parse("all"))

    assert not np.any(result)


def test_apply_never_clips() -> None:
    audio = stereo(np.clip(noise(2.0, 0.6), -1, 1))

    result = restore.apply(audio, SR, restore.parse("tone:3,air:3"))

    assert np.max(np.abs(result)) <= 0.98 + 1e-9


def test_restore_file_writes_a_stereo_file_of_the_same_length(tmp_path: Path) -> None:
    source, destination = tmp_path / "in.wav", tmp_path / "out" / "restored.wav"
    sf.write(source, noise(1.0), SR)  # a mono file

    restore.restore_file(source, destination, restore.parse("tone,widen"))

    audio, sample_rate = sf.read(destination, always_2d=True)
    assert sample_rate == SR and audio.shape == (SR, 2)
    assert np.corrcoef(audio[:, 0], audio[:, 1])[0, 1] < 0.95
