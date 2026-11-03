from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import uuid4

BYTES_PER_SECOND = 16000 * 2  # 16 kHz mono PCM16, the caption websocket format


@dataclass
class UtteranceBuffer:
    """
    Speech chunks of one participant, joined so the utterance is transcribed once.

    `timestamp` is when the first chunk arrived, in epoch milliseconds like the caption results.
    `message_id` names the utterance in the caption stream: its interim captions and its final one.
    """

    participant_id: Optional[str]
    language: Optional[str]
    timestamp: int
    chunks: List[bytes] = field(default_factory=list)
    nbytes: int = 0
    # Time of the 20 ms frames above the silence gate threshold, summed over the chunks.
    speech_ms: int = 0
    message_id: str = field(default_factory=lambda: str(uuid4()))
    # Caption text of each chunk, and its translations by target language, in chunk order.
    captions: List[str] = field(default_factory=list)
    translations: Dict[str, List[str]] = field(default_factory=dict)

    def append(self, audio: bytes, speech_ms: int = 0) -> None:
        self.chunks.append(audio)
        self.nbytes += len(audio)
        self.speech_ms += speech_ms

    def add_caption(self, text: str, translations: Optional[Dict[str, str]] = None) -> None:
        self.captions.append(text)
        for language, translated in (translations or {}).items():
            self.translations.setdefault(language, []).append(translated)

    def caption_text(self) -> str:
        """The chunk captions so far, joined: the interim caption of the utterance."""
        return " ".join(text for text in self.captions if text)

    def joined_translations(self) -> Dict[str, str]:
        return {
            language: " ".join(text for text in texts if text)
            for language, texts in self.translations.items()
        }

    @property
    def duration_s(self) -> float:
        return self.nbytes / BYTES_PER_SECOND

    def audio(self) -> bytes:
        return b"".join(self.chunks)
