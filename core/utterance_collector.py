import asyncio
import logging
import time
from typing import Dict, Optional, Set

from butler import transcribe_audio
from core.engine_manager import EngineManager
from core.session_manager import SessionManager
from utils import UtteranceBuffer

logger = logging.getLogger(__name__)

# Same token budget as a caption call. A 10 s utterance is about 30 words.
UTTERANCE_MAX_NEW_TOKENS = 256


class UtteranceCollector:
    """
    Joins the speech chunks of each participant on one caption websocket into utterances,
    transcribes each utterance once and stores it as one transcript line.

    An utterance closes on the participant's first silent chunk, at `max_s` of audio, or when
    the participant sends nothing for `idle_s`. Jigasi drops non-speech audio with its own VAD,
    so the chunks of a speaker who stops talking usually just stop instead of turning silent.
    Transcription runs in the background so live captions never wait for it.
    """

    def __init__(self, meeting_id: str, *, max_s: float, idle_s: float):
        self.meeting_id = meeting_id
        self.max_s = max_s
        self.idle_s = idle_s
        self.room_name: Optional[str] = None
        self._open: Dict[Optional[str], UtteranceBuffer] = {}
        self._timers: Dict[Optional[str], asyncio.TimerHandle] = {}
        self._tasks: Set[asyncio.Task] = set()

    def feed(
        self,
        *,
        participant_id: Optional[str],
        language: Optional[str],
        room_name: Optional[str],
        audio: bytes,
        has_speech: bool,
    ) -> None:
        self.room_name = room_name
        if not has_speech:
            self._close(participant_id)
            return

        utterance = self._open.get(participant_id)
        if utterance is None:
            utterance = UtteranceBuffer(
                participant_id=participant_id,
                language=language,
                timestamp=time.time_ns() // 1_000_000,
            )
            self._open[participant_id] = utterance
        utterance.append(audio)

        if utterance.duration_s >= self.max_s:
            self._close(participant_id)
        elif self.idle_s > 0:
            self._cancel_timer(participant_id)
            self._timers[participant_id] = asyncio.get_running_loop().call_later(
                self.idle_s, self._close, participant_id
            )

    async def flush(self) -> None:
        """Close every open utterance and wait for every transcription in flight."""
        for participant_id in list(self._open):
            self._close(participant_id)
        if SessionManager.is_meeting_ended(self.meeting_id):
            # The end signal cleared the transcript these lines would join.
            for task in self._tasks:
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def _close(self, participant_id: Optional[str]) -> None:
        self._cancel_timer(participant_id)
        utterance = self._open.pop(participant_id, None)
        if utterance is None or SessionManager.is_meeting_ended(self.meeting_id):
            return
        task = asyncio.create_task(self._store(utterance, self.room_name))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _cancel_timer(self, participant_id: Optional[str]) -> None:
        timer = self._timers.pop(participant_id, None)
        if timer is not None:
            timer.cancel()

    async def _store(self, utterance: UtteranceBuffer, room_name: Optional[str]) -> None:
        try:
            result = await transcribe_audio(
                engine=EngineManager.get_lmm(),
                audio_bytes=utterance.audio(),
                sample_rate=16000,
                participant_id=utterance.participant_id,
                language=utterance.language,
                max_new_tokens=UTTERANCE_MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.2,
            )
        except Exception as e:
            logger.error(
                "Utterance transcription failed: meeting=%s participant=%s seconds=%.1f: %s",
                self.meeting_id, utterance.participant_id, utterance.duration_s, e,
            )
            return

        SessionManager.update_transcript(
            meeting_id=self.meeting_id,
            room_name=room_name,
            transcript=result.text,
            participant_id=utterance.participant_id,
            timestamp=utterance.timestamp,
        )
