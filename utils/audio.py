import math

import numpy as np
from pydub import AudioSegment


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



def pcm16_has_speech(
    audio_bytes: bytes,
    *,
    rms_threshold: float,
    min_active_ms: int = 100,
    sample_rate: int = 16000,
    frame_ms: int = 20,
) -> bool:
    """
    Energy gate for little endian 16 bit mono PCM: tell whether a chunk is worth sending to ASR.

    The chunk is cut into frames of ``frame_ms``. A frame is active when its RMS is at or above
    ``rms_threshold``, in raw int16 units (0 to 32767, so 328 is about -40 dBFS). The chunk has
    speech when the active frames add up to at least ``min_active_ms``. The duration condition
    keeps one or two keyboard clicks from passing as speech.

    A threshold of 0 or less disables the gate and every chunk passes.
    """
    if rms_threshold <= 0:
        return True

    samples = np.frombuffer(audio_bytes[: len(audio_bytes) // 2 * 2], dtype="<i2")
    if samples.size == 0:
        return False

    frame_len = max(1, sample_rate * frame_ms // 1000)
    if samples.size < frame_len:
        frames = samples[None, :]
    else:
        frames = samples[: samples.size // frame_len * frame_len].reshape(-1, frame_len)

    rms = np.sqrt(np.mean(np.square(frames.astype(np.float32)), axis=1))
    required = min(max(1, math.ceil(min_active_ms / frame_ms)), frames.shape[0])
    return int(np.count_nonzero(rms >= rms_threshold)) >= required
