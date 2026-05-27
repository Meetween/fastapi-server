"""Tests for shared audio decode helpers."""
import base64
from unittest.mock import Mock, patch

import pytest

from utils.audio import decode_uploaded_audio_to_pcm, uploaded_audio_to_tensor


def test_decode_uploaded_audio_to_pcm_rejects_empty():
    with pytest.raises(ValueError, match="Empty audio data"):
        decode_uploaded_audio_to_pcm(b"")


def test_uploaded_audio_to_tensor_uses_decode_and_converter():
    pcm = b"\x00\x01" * 80
    fake_tensor = Mock()
    fake_sr = 16000

    with patch("utils.audio.decode_uploaded_audio_to_pcm", return_value=pcm) as mock_decode:
        with patch(
            "butler.jitsi_api_functions.AudioConverter.bytes_to_tensor",
            return_value=(fake_tensor, fake_sr),
        ) as mock_tensor:
            tensor, sample_rate = uploaded_audio_to_tensor(base64.b64decode("YQ=="))

    mock_decode.assert_called_once_with(b"a", sample_rate=16000)
    mock_tensor.assert_called_once_with(pcm, 16000)
    assert tensor is fake_tensor
    assert sample_rate == fake_sr
