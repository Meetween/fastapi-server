import asyncio
import json
import logging
import re
import time
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security.utils import get_authorization_scheme_param
from starlette.status import WS_1008_POLICY_VIOLATION
from starlette.websockets import WebSocketState
from api.deps.service_token import token_matches
from src.schemas.virtual_participants import ROOM_PATTERN
from utils.storage import UnsafePathError
from utils import caption_plausibility
from core import EngineManager, SessionManager
from core.utterance_collector import UtteranceCollector
from butler import caption_audio, translate_audio, ResultType
from config import settings
from utils import (
    AudioRecorder,
    UUIDHelper,
    UtteranceBuffer,
    is_model_reply,
    pcm16_active_ms,
    pcm16_has_speech,
    pcm16_speech_ms,
    pcm16_speech_profile,
)
from pydantic import ValidationError

from models import (
    MeetweenWebsocketRequest,
    MeetweenWebsocketResponse,
    MeetweenTranslationItem,
    MeetweenErrorResponse,
)


ws_router = APIRouter()

logger = logging.getLogger(__name__)

# A close sent before accept rejects the handshake (uvicorn answers HTTP 403, the code stays in
# process). Jigasi retries a failed handshake a bounded number of times, but reconnects without end
# after a close that follows accept, so every handshake refusal happens before accept.
WS_CLOSE_UNAUTHORIZED = 4401

# The Jitsi meetingUniqueId is a UUID. ASCII letters, digits, `_` and `-` only: no dot, no separator,
# no Unicode look-alike. Checked with re.fullmatch, so a trailing newline does not pass as `$` would.
MEETING_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}"
_MEETING_ID = re.compile(MEETING_ID_PATTERN)
# The Jitsi room segment, the platform's room rule.
_ROOM_NAME = re.compile(ROOM_PATTERN)


def valid_meeting_id(value: object) -> bool:
    return isinstance(value, str) and _MEETING_ID.fullmatch(value) is not None


def valid_room_name(value: object) -> bool:
    return isinstance(value, str) and _ROOM_NAME.fullmatch(value) is not None


def _bearer(websocket: WebSocket) -> Optional[HTTPAuthorizationCredentials]:
    """The Authorization bearer of the upgrade request, parsed as HTTPBearer does."""
    scheme, token = get_authorization_scheme_param(websocket.headers.get("authorization"))
    if scheme.lower() != "bearer" or not token:
        return None
    return HTTPAuthorizationCredentials(scheme=scheme, credentials=token)


def jigasi_authorized(websocket: WebSocket) -> bool:
    """The Jigasi service key, under the same ENABLE_AUTH switch as Jigasi's HTTP routes."""
    if not settings.enable_auth:
        return True
    return token_matches(_bearer(websocket), settings.jigasi_meetween_api_key)


@ws_router.websocket("/ws/{meeting_id}")
async def websocket_caption(websocket: WebSocket, meeting_id: str):
    """
    WebSocket endpoint for real-time streaming transcription and translation.

    Clients send audio chunks and receive results in real-time.

    Three scenarios based on target_languages:
    - None / omitted:               transcription only
    - same language as source:      transcription only (no translation needed)
    - different language(s):        transcription + translation per target language

    In utterance mode with ASR_UTTERANCE_FINAL_CAPTION, each 2 s caption goes out as interim, named
    by its utterance and carrying the utterance's captions so far, and the utterance text goes out
    once as the final caption under the same message id. Jigasi relays both to the room.

    The upgrade request carries the Jigasi service key as `Authorization: Bearer`; without it the
    handshake is closed with 4401 before accept. A meeting id outside MEETING_ID_PATTERN is closed
    with 1008. An audio message whose room_name breaks ROOM_PATTERN is answered INVALID_REQUEST and
    dropped before any path is built; the socket stays open.
    """
    if not jigasi_authorized(websocket):
        client = websocket.client.host if websocket.client else None
        logger.warning("WebSocket refused, missing or invalid service key: client=%s", client)
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED, reason="Unauthorized")
        return

    if not valid_meeting_id(meeting_id):
        logger.warning("WebSocket refused, invalid meeting id: %r", meeting_id[:160])
        await websocket.close(code=WS_1008_POLICY_VIOLATION, reason="Invalid meeting id")
        return

    if SessionManager.is_meeting_ended(meeting_id):
        # Reject the handshake (not accept-then-close): Jigasi retries failed handshakes
        # with bounded backoff, but treats a server close after accept as a drop and reconnects forever.
        await websocket.send({"type": "websocket.http.response.start", "status": 409, "headers": []})
        await websocket.send({"type": "websocket.http.response.body", "body": b"Meeting has ended"})
        return

    await websocket.accept()

    SessionManager.add_connection(meeting_id, websocket)
    caption_handler = None
    audio_recorder_initialized = False
    room_name = None
    final_caption = settings.asr_utterance_mode and settings.asr_utterance_final_caption
    # Final captions are sent from background tasks. One lock orders every send on this socket.
    # A caption holds it from its model call to its reply, so an utterance closed while its last
    # chunk is still captioned sends its final only after that chunk's interim.
    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def send_final(utterance: UtteranceBuffer, text: Optional[str]) -> None:
        async with send_lock:
            # A dropped or failed utterance still ends its interim line, with the captions it had.
            transcript = text or utterance.caption_text()
            if not transcript:
                return
            if (
                websocket.client_state != WebSocketState.CONNECTED
                or websocket.application_state != WebSocketState.CONNECTED
            ):
                logger.info("Final caption not sent, socket closed: meeting=%s", meeting_id)
                return
            translations = [
                MeetweenTranslationItem(target_language=language, text=translated)
                for language, translated in utterance.joined_translations().items()
            ]
            response = MeetweenWebsocketResponse(
                meeting_id=meeting_id,
                participant_id=utterance.participant_id,
                transcript=transcript,
                message_id=utterance.message_id,
                timestamp=utterance.timestamp,
                is_interim=False,
                language=utterance.language,
                stability=1.0,
                translations=translations or None,
            )
            await websocket.send_json(response.model_dump())

    utterances = (
        UtteranceCollector(
            meeting_id,
            max_s=settings.asr_utterance_max_s,
            idle_s=settings.asr_utterance_idle_s,
            min_speech_ms=settings.asr_utterance_min_speech_ms,
            end_silence_ms=settings.asr_utterance_end_silence_ms if settings.asr_utterance_end_silence_enabled else 0,
            on_close=send_final if final_caption else None,
        )
        if settings.asr_utterance_mode
        else None
    )

    logger.info(f"WebSocket connected: {meeting_id}")

    try:
        if EngineManager.get_lmm():
            caption_handler = SessionManager.get_or_create_caption_handler(
                meeting_id=meeting_id,
                engine=EngineManager.get_lmm(),
            )

        while True:
            message = await websocket.receive_text()

            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "audio":
                    try:
                        request = MeetweenWebsocketRequest(**data)

                        if not valid_room_name(request.room_name):
                            logger.warning(
                                "Audio dropped, invalid room_name: meeting=%s room=%r",
                                meeting_id, request.room_name[:160],
                            )
                            await send(MeetweenErrorResponse(
                                message="Invalid room_name",
                                code="INVALID_REQUEST",
                            ).model_dump(mode="json"))
                            continue

                        audio_data = request.get_audio_bytes()
                        try:
                            if not audio_recorder_initialized:
                                AudioRecorder.init_recording(room_name=request.room_name, meeting_id=meeting_id)
                                room_name = request.room_name
                                audio_recorder_initialized = True

                            await AudioRecorder.save_chunk(
                                room_name=room_name,
                                meeting_id=meeting_id,
                                audio_data=audio_data,
                                chunk_duration=2.0,
                            )
                        except UnsafePathError:
                            # Names passed the patterns, so the tree itself was changed (a link).
                            logger.error(
                                "Audio dropped, recording path leaves the base: meeting=%s room=%s",
                                meeting_id, request.room_name,
                            )
                            await send(MeetweenErrorResponse(
                                message="Recording path refused",
                                code="INVALID_REQUEST",
                            ).model_dump(mode="json"))
                            continue

                        if SessionManager.is_virtual_participant(meeting_id, request.participant_id):
                            # The bot's own speech: no caption, no transcript line, no GPU time.
                            continue

                        if not EngineManager.get_lmm() or not caption_handler:
                            error_response = MeetweenErrorResponse(
                                message="SpeechLMM not available",
                                code="SERVICE_UNAVAILABLE",
                            )
                            logger.error("SpeechLMM not available")
                            await send(error_response.model_dump())
                            continue

                        # Silence goes to the model as a chatbot prompt and comes back as text,
                        # so a chunk without speech is dropped before any model call.
                        has_speech, speech_ms, leading_silence_ms, trailing_silence_ms = await _measure_speech(audio_data)
                        utterance = None
                        if utterances is not None:
                            utterance = utterances.feed(
                                participant_id=request.participant_id,
                                language=request.language,
                                room_name=room_name,
                                audio=audio_data,
                                has_speech=has_speech,
                                speech_ms=speech_ms,
                                leading_silence_ms=leading_silence_ms,
                                trailing_silence_ms=trailing_silence_ms,
                            )
                        if not has_speech:
                            logger.debug(
                                "Dropping silent chunk: meeting=%s bytes=%d speech_ms=%d",
                                meeting_id, len(audio_data), speech_ms,
                            )
                            continue

                        try:
                            async with send_lock:
                                # Run transcription and speech translation concurrently
                                result, translations = await asyncio.gather(
                                    caption_audio(
                                        engine=EngineManager.get_lmm(),
                                        audio_bytes=audio_data,
                                        sample_rate=16000,
                                        caption_handler=caption_handler,
                                        timestamp=time.time_ns() // 1_000_000,
                                        language=request.language,
                                    ),
                                    _get_translations(request, audio_data),
                                )

                                if getattr(result, "result_type", None) == ResultType.NO_RESULT:
                                    continue

                                if settings.asr_output_guard_enabled and is_model_reply(result.text):
                                    # Not speech: the model answered noise or echo. Nothing is sent,
                                    # and the utterance's interim caption does not carry it.
                                    logger.debug(
                                        "Dropping model reply caption: meeting=%s participant=%s text=%r",
                                        meeting_id, request.participant_id, result.text,
                                    )
                                    caption_plausibility.record_drop(
                                        caption_plausibility.MODEL_REPLY, meeting_id=meeting_id,
                                        participant_id=request.participant_id, chars=len(result.text or ""),
                                        seconds=caption_plausibility.audio_seconds(audio_data),
                                    )
                                    continue

                                # Too long for its audio, or a word loop: the model kept talking.
                                seconds = caption_plausibility.audio_seconds(audio_data)
                                reason = caption_plausibility.drop_reason(result.text, seconds)
                                if reason is not None:
                                    caption_plausibility.record_drop(
                                        reason, meeting_id=meeting_id, participant_id=request.participant_id,
                                        chars=len(result.text or ""), seconds=seconds,
                                    )
                                    continue

                                # In utterance mode the transcript line comes from the joined utterance.
                                if not result.is_interim and utterances is None:
                                    SessionManager.update_transcript(
                                        meeting_id=meeting_id,
                                        room_name=room_name,
                                        transcript=result.text,
                                        participant_id=request.participant_id,
                                        timestamp=result.timestamp,
                                    )

                                response = MeetweenWebsocketResponse(
                                    meeting_id=meeting_id,
                                    participant_id=request.participant_id,
                                    transcript=result.text,
                                    message_id=str(UUIDHelper.uuid_from_int(result.message_id)),
                                    timestamp=result.timestamp,
                                    is_interim=result.is_interim,
                                    language=request.language,
                                    stability=result.stability,
                                    translations=translations or None,
                                )
                                if final_caption and utterance is not None:
                                    utterance.add_caption(
                                        result.text, {t.target_language: t.text for t in translations}
                                    )
                                    response = response.model_copy(update={
                                        "transcript": utterance.caption_text(),
                                        "message_id": utterance.message_id,
                                        "timestamp": utterance.timestamp,
                                        "is_interim": True,
                                    })
                                await websocket.send_json(response.model_dump())

                        except Exception as e:
                            logger.error(f"Error processing audio: {e}", exc_info=True)
                            error_response = MeetweenErrorResponse(
                                message=f"Processing error: {str(e)}",
                                code="PROCESSING_ERROR",
                            )
                            await send(error_response.model_dump())

                    except ValidationError as e:
                        error = MeetweenErrorResponse(
                            message=f"Invalid audio request: {e}",
                            code="INVALID_REQUEST",
                        )
                        logger.warning(e)
                        await send(error.model_dump(mode="json"))

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON: {e}")
                error = MeetweenErrorResponse(
                    message=f"Invalid JSON: {e}",
                    code="INVALID_JSON",
                )
                await send(error.model_dump(mode="json"))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {meeting_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        try:
            SessionManager.remove_connection(meeting_id)
            if room_name:
                AudioRecorder.stop_recording(room_name=room_name, meeting_id=meeting_id)
            # Session state is NOT cleared here. A dropped socket must not wipe a running
            # meeting's transcript; only the jigasi end-of-meeting signal clears it.
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        if utterances is not None:
            try:
                await utterances.flush()
            except Exception as e:
                logger.warning(f"Error flushing utterances: {e}")
        logger.info(f"WebSocket session closed: {meeting_id}")


async def _measure_speech(audio_data: bytes) -> tuple[bool, int, int, int]:
    """
    Whether a chunk holds speech, how many ms of it, and how many ms of non speech it starts and ends with
    (0 unless ASR_UTTERANCE_END_SILENCE_MS is set). The RMS gate is the loudness floor; with
    ASR_VAD the speech detector then decides whether a loud enough chunk holds voice, so a quiet echo
    stays out by the floor and a loud fan or breath by the detector.
    """
    loud = pcm16_has_speech(
        audio_data,
        rms_threshold=settings.asr_silence_rms,
        min_active_ms=settings.asr_silence_min_active_ms,
    )
    if not settings.asr_vad_enabled:
        return loud, pcm16_active_ms(audio_data, rms_threshold=settings.asr_silence_rms), 0, 0
    if not loud:
        return False, 0, 0, 0
    if settings.asr_utterance_end_silence_enabled:
        speech_ms, leading_ms, trailing_ms = await asyncio.to_thread(
            pcm16_speech_profile, audio_data, threshold=settings.asr_vad_threshold
        )
    else:
        speech_ms = await asyncio.to_thread(pcm16_speech_ms, audio_data, threshold=settings.asr_vad_threshold)
        leading_ms = trailing_ms = 0
    return speech_ms >= settings.asr_vad_min_speech_ms, speech_ms, leading_ms, trailing_ms


async def _get_translations(
    request: MeetweenWebsocketRequest,
    audio_data: bytes,
) -> list[MeetweenTranslationItem]:
    """
    Translate audio directly into each requested target language that differs
    from the source. Returns an empty list when no target languages are requested.
    """
    if not request.target_languages:
        return []

    source_base = (request.language or "en").split("-")[0].lower()
    targets = [t for t in request.target_languages if t.split("-")[0].lower() != source_base]

    if not targets:
        return []

    results = await asyncio.gather(
        *[
            translate_audio(
                engine=EngineManager.get_lmm(),
                audio_bytes=audio_data,
                language=request.language,
                target_lang=target,
                participant_id=request.participant_id,
            )
            for target in targets
        ],
        return_exceptions=True,
    )

    translations = []
    for target, result in zip(targets, results):
        if isinstance(result, Exception):
            logger.error(f"Translation failed for {source_base}->{target}: {result}", exc_info=result)
        else:
            translations.append(MeetweenTranslationItem(target_language=target, text=result))

    return translations
