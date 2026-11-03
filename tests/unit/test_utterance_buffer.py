"""Utterance buffer: joins speech chunks and reports their length."""

import numpy as np

from utils import UtteranceBuffer, pcm16_active_ms


def test_empty_buffer():
    buffer = UtteranceBuffer(participant_id="p1", language="en-US", timestamp=1)
    assert buffer.duration_s == 0.0
    assert buffer.audio() == b""


def test_chunks_are_joined_in_order_and_timed():
    buffer = UtteranceBuffer(participant_id="p1", language="en-US", timestamp=1)
    buffer.append(b"\x01\x00" * 32000)  # 2 s at 16 kHz PCM16
    buffer.append(b"\x02\x00" * 16000)  # 1 s
    assert buffer.duration_s == 3.0
    assert buffer.audio() == b"\x01\x00" * 32000 + b"\x02\x00" * 16000


def test_speech_ms_adds_up():
    buffer = UtteranceBuffer(participant_id="p1", language="en-US", timestamp=1)
    buffer.append(b"\x00\x00" * 32000, speech_ms=300)
    buffer.append(b"\x00\x00" * 32000, speech_ms=1200)
    assert buffer.speech_ms == 1500


def test_active_ms_counts_frames_at_or_above_the_threshold():
    loud = np.full(16000 * 3 // 10, 1000, dtype="<i2")  # 300 ms
    quiet = np.zeros(16000 * 17 // 10, dtype="<i2")  # 1.7 s
    audio = np.concatenate([loud, quiet]).tobytes()
    assert pcm16_active_ms(audio, rms_threshold=150.0) == 300
    assert pcm16_active_ms(np.full(320, 150, dtype="<i2").tobytes(), rms_threshold=150.0) == 20


def test_active_ms_counts_everything_with_the_gate_off():
    assert pcm16_active_ms(np.zeros(32000, dtype="<i2").tobytes(), rms_threshold=0) == 2000
    assert pcm16_active_ms(b"", rms_threshold=150.0) == 0
