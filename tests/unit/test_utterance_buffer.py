"""Utterance buffer: joins speech chunks and reports their length."""

from utils import UtteranceBuffer


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
