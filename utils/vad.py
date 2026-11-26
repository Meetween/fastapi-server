"""
Speech detector for the caption websocket: Silero VAD v6, the 16 kHz ONNX export, on CPU.

The RMS gate passes anything loud enough, so a fan, a breath or the faint echo of another speaker
through a laptop microphone reaches the model, which answers it as a chatbot ("I'm sorry, I didn't
catch that"). Silero tells voice from noise and rates each 32 ms window with a speech probability.

Model: src/silero_vad/data/silero_vad_16k_op15.onnx of snakers4/silero-vad at tag v6.2.3, MIT
licence in data/silero_vad.LICENSE, sha256 7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49.
1.3 MB, about 4 ms per 2 s chunk on one CPU thread.
"""

import threading
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).parent / "data" / "silero_vad_16k_op15.onnx"

SAMPLE_RATE = 16000
# The model takes 512 new samples per call, 32 ms at 16 kHz, after the last 64 samples of the
# previous window, and carries an RNN state from one window to the next.
WINDOW = 512
CONTEXT = 64
WINDOW_MS = WINDOW * 1000 // SAMPLE_RATE


class SileroVad:
    """
    Speech probability per 32 ms window of little endian 16 bit mono PCM at 16 kHz.

    Each call starts from a fresh state, so one instance serves every participant and thread.
    """

    def __init__(self, path: Path = MODEL_PATH):
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def probabilities(self, audio_bytes: bytes) -> np.ndarray:
        samples = np.frombuffer(audio_bytes[: len(audio_bytes) // 2 * 2], dtype="<i2")
        audio = samples.astype(np.float32) / 32768.0
        windows = audio.size // WINDOW
        probabilities = np.zeros(windows, dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(CONTEXT, dtype=np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        for i in range(windows):
            window = audio[i * WINDOW:(i + 1) * WINDOW]
            frame = np.concatenate([context, window])[None, :]
            output, state = self._session.run(None, {"input": frame, "state": state, "sr": sr})
            probabilities[i] = output[0][0]
            context = window[-CONTEXT:]
        return probabilities

    def speech_ms(self, audio_bytes: bytes, *, threshold: float) -> int:
        """Time in ms of the 32 ms windows whose speech probability is at or above ``threshold``."""
        return int(np.count_nonzero(self.probabilities(audio_bytes) >= threshold)) * WINDOW_MS

    def profile(self, audio_bytes: bytes, *, threshold: float) -> tuple[int, int, int]:
        """
        ``speech_ms`` with the silence at both ends of the chunk: ms of 32 ms windows before the first and
        after the last one at or above ``threshold``. A chunk with no voice is all silence at both ends.
        """
        voiced = self.probabilities(audio_bytes) >= threshold
        indices = np.flatnonzero(voiced)
        if indices.size == 0:
            return 0, voiced.size * WINDOW_MS, voiced.size * WINDOW_MS
        leading = int(indices[0])
        trailing = voiced.size - 1 - int(indices[-1])
        return int(indices.size) * WINDOW_MS, leading * WINDOW_MS, trailing * WINDOW_MS


_vad: Optional[SileroVad] = None
_vad_lock = threading.Lock()


def get_vad() -> SileroVad:
    """The process wide detector, loaded on first use."""
    global _vad
    with _vad_lock:
        if _vad is None:
            _vad = SileroVad()
        return _vad


def pcm16_speech_ms(audio_bytes: bytes, *, threshold: float) -> int:
    """Time in ms of voice in a chunk, by Silero VAD windows at or above ``threshold``."""
    return get_vad().speech_ms(audio_bytes, threshold=threshold)


def pcm16_speech_profile(audio_bytes: bytes, *, threshold: float) -> tuple[int, int, int]:
    """Voice, leading silence and trailing silence of a chunk in ms, one pass of the detector."""
    return get_vad().profile(audio_bytes, threshold=threshold)
