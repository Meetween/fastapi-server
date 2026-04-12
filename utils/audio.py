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

