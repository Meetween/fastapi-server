import asyncio
import logging
import time
from typing import Awaitable, Callable, Dict, Optional, Set

from butler import transcribe_audio
from core.engine_manager import EngineManager
from core.session_manager import SessionManager
from config import settings
from utils import UtteranceBuffer, is_model_reply

logger = logging.getLogger(__name__)

# Same token budget as a caption call. A 10 s utterance is about 30 words.
UTTERANCE_MAX_NEW_TOKENS = 256


class UtteranceCollector:
    """
    Joins the speech chunks of each participant on one caption websocket into utterances,
    transcribes each utterance once and stores it as one transcript line.

    An utterance closes on the participant's first silent chunk, at `max_s` of audio, or when
    the participant sends nothing for `idle_s`. One with less than `min_speech_ms` of speech is
    dropped: the model answers a lone fragment as a chatbot ("End.", "I'm sorry, I didn't
    understand the question") instead of transcribing it. With ASR_OUTPUT_GUARD a text that is such a
    reply is dropped as well, after the call. Jigasi drops non-speech audio with its own VAD,
    so the chunks of a speaker who stops talking usually just stop instead of turning silent.
    Transcription runs in the background so live captions never wait for it.

    `on_close`, when given, is awaited once for every utterance that closes, with its text, or None
    when it was dropped or its transcription failed. The caption websocket sends it as the final caption.
    """

    def __init__(
        self,
        meeting_id: str,
        *,
        max_s: float,
        idle_s: float,
        min_speech_ms: int,
        end_silence_ms: int = 0,
        on_close: Optional[Callable[[UtteranceBuffer, Optional[str]], Awaitable[None]]] = None,
    ):
        self.meeting_id = meeting_id
        self.max_s = max_s
        self.idle_s = idle_s
        self.min_speech_ms = min_speech_ms
        self.end_silence_ms = end_silence_ms
        self.on_close = on_close
        self.room_name: Optional[str] = None
        self._open: Dict[Optional[str], UtteranceBuffer] = {}
        self._timers: Dict[Optional[str], asyncio.TimerHandle] = {}
        self._tasks: Set[asyncio.Task] = set()
        # Non speech at the end of each participant's latest speech chunk, ms.
        self._trailing: Dict[Optional[str], int] = {}

    def feed(
        self,
        *,
        participant_id: Optional[str],
        language: Optional[str],
        room_name: Optional[str],
        audio: bytes,
        has_speech: bool,
        speech_ms: int = 0,
        leading_silence_ms: int = 0,
        trailing_silence_ms: int = 0,
    ) -> Optional[UtteranceBuffer]:
        """
        Add a chunk with its measured speech and the non speech it starts and ends with. Returns the
        utterance a speech chunk joined, which may already be closed by the cap or by the end silence.
        """
        self.room_name = room_name
        if not has_speech:
            self._close(participant_id)
            return None

        utterance = self._open.get(participant_id)
        if utterance is not None and self._pause_ended(utterance, participant_id, leading_silence_ms):
            # The pause straddled the chunk boundary: this chunk opens the next utterance.
            self._close(participant_id)
            utterance = None
        if utterance is None:
            utterance = UtteranceBuffer(
                participant_id=participant_id,
                language=language,
                timestamp=time.time_ns() // 1_000_000,
            )
            self._open[participant_id] = utterance
        utterance.append(audio, speech_ms)
        self._trailing[participant_id] = trailing_silence_ms

        if utterance.duration_s >= self.max_s:
            self._close(participant_id)
        elif self._speaker_finished(utterance, trailing_silence_ms):
            self._close(participant_id)
        elif self.idle_s > 0:
            self._cancel_timer(participant_id)
            self._timers[participant_id] = asyncio.get_running_loop().call_later(
                self.idle_s, self._close, participant_id
            )
        return utterance

    def _speaker_finished(self, utterance: UtteranceBuffer, trailing_silence_ms: int) -> bool:
        # Under the speech minimum a lone fragment would be dropped at close, and the words that
        # follow the pause would lose their first half. It waits for the next chunk or the idle timer.
        return (
            self.end_silence_ms > 0
            and trailing_silence_ms >= self.end_silence_ms
            and utterance.speech_ms >= self.min_speech_ms
        )

    def _pause_ended(self, utterance: UtteranceBuffer, participant_id: Optional[str], leading_silence_ms: int) -> bool:
        # Silence at the end of the previous chunk plus silence at the start of this one.
        return self._speaker_finished(utterance, self._trailing.get(participant_id, 0) + leading_silence_ms)

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
        if utterance.speech_ms < self.min_speech_ms:
            logger.info(
                "Dropping short utterance: meeting=%s participant=%s speech_ms=%d",
                self.meeting_id, utterance.participant_id, utterance.speech_ms,
            )
            if self.on_close is not None:
                self._spawn(self._finish(utterance, None))
            return
        self._spawn(self._store(utterance, self.room_name))

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
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
            await self._finish(utterance, None)
            return

        if settings.asr_output_guard_enabled and is_model_reply(result.text):
            # Not speech: the final caption falls back to the interim text, the transcript gets nothing.
            logger.debug(
                "Dropping model reply utterance: meeting=%s participant=%s text=%r",
                self.meeting_id, utterance.participant_id, result.text,
            )
            await self._finish(utterance, None)
            return

        SessionManager.update_transcript(
            meeting_id=self.meeting_id,
            room_name=room_name,
            transcript=result.text,
            participant_id=utterance.participant_id,
            timestamp=utterance.timestamp,
        )
        # Stored first, so a question asked on this caption finds its line in the transcript.
        await self._finish(utterance, result.text)

    async def _finish(self, utterance: UtteranceBuffer, text: Optional[str]) -> None:
        if self.on_close is None:
            return
        try:
            await self.on_close(utterance, text)
        except Exception as e:
            logger.warning(
                "Final caption not sent: meeting=%s participant=%s: %s",
                self.meeting_id, utterance.participant_id, e,
            )
