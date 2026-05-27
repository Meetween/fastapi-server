import os
import tempfile
from typing import Any, Tuple

import numpy as np
from pydub import AudioSegment

MODEL_SAMPLE_RATE = 16000


def pydub_to_pcm16(audio_segment: AudioSegment) -> bytes:
    """
    Convert a pydub AudioSegment (already mono/16k) to PCM16 bytes safely.

    Why: pydub often decodes to 32-bit samples (sample_width=4). Casting those
    directly to int16 truncates/mangles the waveform and can break ASR.
    """
    sample_width = int(audio_segment.sample_width)
    raw = np.array(audio_segment.get_array_of_samples())

    if sample_width == 2:
        pcm16 = raw.astype(np.int16, copy=False)
        return pcm16.tobytes()

    # Downconvert other widths via float normalization -> int16.
    if sample_width == 4:
        f = raw.astype(np.float32) / float(2**31)
    elif sample_width == 1:
        # 8-bit PCM is typically unsigned in WAV, but pydub gives signed values.
        f = raw.astype(np.float32) / 128.0
    else:
        # Fallback: scale by maximum representable integer of that width.
        f = raw.astype(np.float32) / float(2 ** (8 * sample_width - 1))

    f = np.clip(f, -1.0, 1.0)
    pcm16 = (f * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def decode_uploaded_audio_to_pcm(
    audio_bytes: bytes,
    *,
    temp_suffix: str = ".webm",
    sample_rate: int = MODEL_SAMPLE_RATE,
) -> bytes:
    """
    Decode uploaded container audio (e.g. WebM from Butler) to PCM LINEAR16 bytes.

    Writes to a temp file and uses pydub/ffmpeg, matching the historical /api/transcribe path.
    """
    if not audio_bytes:
        raise ValueError("Empty audio data")

    with tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    try:
        audio_segment = AudioSegment.from_file(temp_path)
        audio_segment = audio_segment.set_channels(1).set_frame_rate(sample_rate)
        return pydub_to_pcm16(audio_segment)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def uploaded_audio_to_tensor(
    audio_bytes: bytes,
    *,
    sample_rate: int = MODEL_SAMPLE_RATE,
) -> Tuple[Any, int]:
    """
    Decode uploaded container audio and convert to a model-ready tensor.

    Used by /api/chat (_generate_text). /api/transcribe stops at PCM + transcribe_audio.
    """
    from butler.jitsi_api_functions import AudioConverter

    audio_pcm = decode_uploaded_audio_to_pcm(audio_bytes, sample_rate=sample_rate)
    return AudioConverter.bytes_to_tensor(audio_pcm, sample_rate)

