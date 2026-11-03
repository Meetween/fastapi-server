"""Speech detector: Silero VAD v6 on CPU, the vendored model file and its windows."""

import hashlib

import numpy as np

from utils.vad import MODEL_PATH, WINDOW_MS, SileroVad, pcm16_speech_ms

SR = 16000


def _pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def test_model_file_is_the_pinned_release():
    # silero_vad_16k_op15.onnx of snakers4/silero-vad v6.2.3.
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    assert digest == "7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49"


def test_one_probability_per_32_ms_window():
    probabilities = SileroVad().probabilities(_pcm(np.zeros(SR * 2)))
    assert WINDOW_MS == 32
    assert probabilities.shape == (62,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_silence_has_no_speech():
    assert pcm16_speech_ms(_pcm(np.zeros(SR * 2)), threshold=0.5) == 0


def test_loud_noise_has_no_speech():
    # Loud enough for the RMS gate at any threshold the backend uses.
    noise = np.random.default_rng(7).normal(0, 2000, SR * 2)
    assert pcm16_speech_ms(_pcm(noise), threshold=0.5) == 0


def test_steady_tone_has_no_speech():
    t = np.arange(SR * 2) / SR
    assert pcm16_speech_ms(_pcm(3000 * np.sin(2 * np.pi * 200 * t)), threshold=0.5) == 0


def test_short_or_odd_input():
    assert pcm16_speech_ms(b"", threshold=0.5) == 0
    assert pcm16_speech_ms(b"\x01", threshold=0.5) == 0
    assert SileroVad().probabilities(_pcm(np.zeros(600))).shape == (1,)


def test_profile_gives_voice_and_the_silence_at_both_ends(monkeypatch):
    # 62 windows: 10 silent, 20 voiced, 5 silent, 7 voiced, 20 silent.
    probabilities = np.array([0.0] * 10 + [0.9] * 20 + [0.1] * 5 + [0.8] * 7 + [0.0] * 20, dtype=np.float32)
    monkeypatch.setattr(SileroVad, "probabilities", lambda self, audio: probabilities)

    assert SileroVad().profile(b"", threshold=0.5) == (27 * WINDOW_MS, 10 * WINDOW_MS, 20 * WINDOW_MS)


def test_profile_of_a_chunk_without_voice_is_silent_at_both_ends():
    assert SileroVad().profile(_pcm(np.zeros(SR * 2)), threshold=0.5) == (0, 62 * WINDOW_MS, 62 * WINDOW_MS)
