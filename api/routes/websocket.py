import asyncio
import json
import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core import EngineManager, SessionManager
from butler import caption_audio, translate_audio, ResultType
from utils import AudioRecorder, UUIDHelper
from pydantic import ValidationError

from models import (
    MeetweenWebsocketRequest,
    MeetweenWebsocketResponse,
    MeetweenTranslationItem,
    MeetweenErrorResponse,
)


ws_router = APIRouter()

logger = logging.getLogger(__name__)


@ws_router.websocket("/ws/{meeting_id}")
async def websocket_caption(websocket: WebSocket, meeting_id: str):
    """
    WebSocket endpoint for real-time streaming transcription and translation.

    Clients send audio chunks and receive results in real-time.

    Three scenarios based on target_languages:
    - None / omitted:               transcription only
    - same language as source:      transcription only (no translation needed)
    - different language(s):        transcription + translation per target language
    """
    await websocket.accept()

    if SessionManager.is_meeting_ended(meeting_id):
        await websocket.close(code=1008, reason="meeting ended")
        return

    SessionManager.add_connection(meeting_id, websocket)
    caption_handler = None
    audio_recorder_initialized = False
    room_name = None

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

                        if not audio_recorder_initialized:
                            room_name = request.room_name
                            AudioRecorder.init_recording(room_name=room_name, meeting_id=meeting_id)
                            audio_recorder_initialized = True

                        audio_data = request.get_audio_bytes()
                        await AudioRecorder.save_chunk(
                            room_name=room_name,
                            meeting_id=meeting_id,
                            audio_data=audio_data,
                            chunk_duration=2.0,
                        )

                        if not EngineManager.get_lmm() or not caption_handler:
                            error_response = MeetweenErrorResponse(
                                message="SpeechLMM not available",
                                code="SERVICE_UNAVAILABLE",
                            )
                            logger.error("SpeechLMM not available")
                            await websocket.send_json(error_response.model_dump())
                            continue

                        try:
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

                            if not result.is_interim:
                                SessionManager.update_transcript(
                                    meeting_id=meeting_id,
                                    room_name=room_name,
                                    transcript=result.text,
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
                            await websocket.send_json(response.model_dump())

                        except Exception as e:
                            logger.error(f"Error processing audio: {e}", exc_info=True)
                            error_response = MeetweenErrorResponse(
                                message=f"Processing error: {str(e)}",
                                code="PROCESSING_ERROR",
                            )
                            await websocket.send_json(error_response.model_dump())

                    except ValidationError as e:
                        error = MeetweenErrorResponse(
                            message=f"Invalid audio request: {e}",
                            code="INVALID_REQUEST",
                        )
                        logger.warning(e)
                        await websocket.send_text(error.model_dump_json())

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON: {e}")
                error = MeetweenErrorResponse(
                    message=f"Invalid JSON: {e}",
                    code="INVALID_JSON",
                )
                await websocket.send_text(error.model_dump_json())

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
        logger.info(f"WebSocket session closed: {meeting_id}")


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
