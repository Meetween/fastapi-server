"""Energy gate that keeps silent chunks away from the ASR model."""

import numpy as np
import pytest

from utils import pcm16_has_speech

SR = 16000
FRAME = 320  # 20 ms
THRESHOLD = 150.0


def _pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def _speech(seconds: float = 2.0, amplitude: float = 3000.0) -> bytes:
    t = np.arange(int(SR * seconds)) / SR
    wave = np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 360 * t) + 0.25 * np.sin(2 * np.pi * 900 * t)
    return _pcm(amplitude * wave / 1.75)


def _noise(seconds: float = 2.0, std: float = 20.0) -> np.ndarray:
    return np.random.default_rng(7).normal(0, std, int(SR * seconds))


def _gate(audio: bytes, **kwargs) -> bool:
    return pcm16_has_speech(audio, rms_threshold=THRESHOLD, **kwargs)


class TestPcm16HasSpeech:
    def test_zeros_are_silence(self):
        assert not _gate(_pcm(np.zeros(SR * 2)))

    def test_low_noise_is_silence(self):
        assert not _gate(_pcm(_noise(std=30.0)))

    def test_inaudible_tone_is_silence(self):
        t = np.arange(SR * 2) / SR
        assert not _gate(_pcm(40 * np.sin(2 * np.pi * 440 * t)))

    def test_speech_like_signal_passes(self):
        assert _gate(_speech())

    def test_speech_with_pauses_passes(self):
        # Mean RMS over the chunk is low, but a second of voiced frames is in it.
        samples = np.zeros(SR * 2)
        samples[SR // 2 : SR // 2 + SR] = np.frombuffer(_speech(1.0), dtype="<i2")
        assert _gate(_pcm(samples))

    def test_quiet_speech_above_threshold_passes(self):
        assert _gate(_speech(amplitude=600.0))

    def test_isolated_clicks_are_silence(self):
        samples = _noise()
        for start in (SR // 2, SR, SR + SR // 2):
            samples[start : start + 80] = 12000  # 5 ms click
        assert not _gate(_pcm(samples))

    def test_frame_rms_exactly_at_threshold_counts_as_active(self):
        assert _gate(_pcm(np.full(SR * 2, 150.0)))

    def test_frame_rms_just_below_threshold_is_silence(self):
        assert not _gate(_pcm(np.full(SR * 2, 149.0)))

    def test_active_time_exactly_at_minimum_passes(self):
        samples = np.zeros(SR * 2)
        samples[: 5 * FRAME] = 3000  # 100 ms
        assert _gate(_pcm(samples))

    def test_active_time_just_below_minimum_is_silence(self):
        samples = np.zeros(SR * 2)
        samples[: 4 * FRAME] = 3000  # 80 ms
        assert not _gate(_pcm(samples))

    def test_minimum_active_time_is_configurable(self):
        samples = np.zeros(SR * 2)
        samples[: 2 * FRAME] = 3000
        assert not _gate(_pcm(samples), min_active_ms=100)
        assert _gate(_pcm(samples), min_active_ms=40)

    def test_threshold_zero_disables_the_gate(self):
        assert pcm16_has_speech(_pcm(np.zeros(SR * 2)), rms_threshold=0)

    def test_empty_chunk_is_silence(self):
        assert not _gate(b"")

    def test_odd_byte_count_is_tolerated(self):
        assert _gate(_speech() + b"\x01")

    def test_chunk_shorter_than_a_frame_is_judged_whole(self):
        assert _gate(_pcm(np.full(100, 3000.0)))
        assert not _gate(_pcm(np.zeros(100)))

    @pytest.mark.parametrize("seconds", [0.5, 1.5, 2.0, 3.7])
    def test_any_chunk_length(self, seconds):
        assert _gate(_speech(seconds))
        assert not _gate(_pcm(np.zeros(int(SR * seconds))))
