"""Plausibility of an ASR text against the audio it came from.

On a short or garbled chunk the model can go on talking: in C2 a 0.5 s chunk came back as an
invented speech of about 1,500 characters. Nobody speaks more than about 20 characters a second,
so a text longer than ASR_CAPTION_MAX_CHARS_PER_S per second of audio plus
ASR_CAPTION_CHARS_SLACK is not a transcript. A caption loop ("very very very": one word three
times or more) is the other silence pattern; the report already keeps such a line out of its
evidence. Either is dropped where the caption arrives, so it is never sent to the room, never
stored, and so never read by the Agentar prompt or the report.

Jigasi sends no duration with a chunk. The chunk is 16 kHz mono PCM16, so its byte length gives
the duration exactly.

Dropped texts are counted by reason in this process (one uvicorn process serves the API), shown
on /health as `captions_dropped`.
"""
import logging
import re
import threading
from collections import Counter
from typing import Optional

from config import settings
from utils.utterance import BYTES_PER_SECOND

logger = logging.getLogger(__name__)

IMPLAUSIBLE_LENGTH = "implausible_length"
REPEAT_LOOP = "repeat_loop"
MODEL_REPLY = "model_reply"

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_counts: Counter = Counter()
_lock = threading.Lock()


def audio_seconds(nbytes_or_audio) -> float:
    nbytes = nbytes_or_audio if isinstance(nbytes_or_audio, int) else len(nbytes_or_audio)
    return nbytes / BYTES_PER_SECOND


def max_chars(seconds: float) -> Optional[float]:
    """Longest plausible text for `seconds` of audio, or None when the cap is off."""
    rate = settings.asr_caption_max_chars_per_s
    if rate <= 0:
        return None
    return rate * max(seconds, 0.0) + settings.asr_caption_chars_slack


def _is_loop(text: str) -> bool:
    words = _WORD.findall(text.casefold())
    return len(words) >= 3 and len(set(words)) == 1


def drop_reason(text: str, seconds: float) -> Optional[str]:
    """Why `text` cannot be the transcript of `seconds` of audio, or None when it can."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    cap = max_chars(seconds)
    if cap is not None and len(stripped) > cap:
        return IMPLAUSIBLE_LENGTH
    if _is_loop(stripped):
        return REPEAT_LOOP
    return None


def record_drop(reason: str, *, meeting_id: str, participant_id: Optional[str], chars: int, seconds: float) -> None:
    """Count one dropped text. The log line carries sizes only, not the text."""
    with _lock:
        _counts[reason] += 1
        total = _counts[reason]
    logger.warning(
        "Caption dropped reason=%s meeting=%s participant=%s chars=%d audio_s=%.2f total=%d",
        reason, meeting_id, participant_id, chars, seconds, total,
    )


def dropped_counts() -> dict[str, int]:
    with _lock:
        return dict(_counts)


def reset_counts() -> None:
    with _lock:
        _counts.clear()
