from dataclasses import dataclass, field
from typing import List, Optional

BYTES_PER_SECOND = 16000 * 2  # 16 kHz mono PCM16, the caption websocket format


@dataclass
class UtteranceBuffer:
    """
    Speech chunks of one participant, joined so the utterance is transcribed once.

    `timestamp` is when the first chunk arrived, in epoch milliseconds like the caption results.
    """

    participant_id: Optional[str]
    language: Optional[str]
    timestamp: int
    chunks: List[bytes] = field(default_factory=list)
    nbytes: int = 0

    def append(self, audio: bytes) -> None:
        self.chunks.append(audio)
        self.nbytes += len(audio)

    @property
    def duration_s(self) -> float:
        return self.nbytes / BYTES_PER_SECOND

    def audio(self) -> bytes:
        return b"".join(self.chunks)
