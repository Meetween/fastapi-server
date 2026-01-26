import uuid
import json
import logging
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from typing import Optional
from core import EngineManager, SessionManager
from butler import CaptionHandler, caption_audio
from utils import AudioRecorder, UUIDHelper
from pydantic import ValidationError

from models import MeetweenTranscriptionRequest, MeetweenTranscriptionResponse, MeetweenErrorResponse


router = APIRouter(prefix="/api", tags=["caption"])
ws_router = APIRouter()

logger = logging.getLogger(__name__)


@ws_router.websocket("/ws/{meeting_id}")
async def websocket_caption(websocket: WebSocket, meeting_id: str):
    """
    WebSocket endpoint for real-time streaming transcription

    Clients send audio chunks and receive transcription updates in real-time.
    Supports both interim and final results.
    """
    await websocket.accept()

    SessionManager.add_connection(meeting_id, websocket)
    caption_handler = None
    audio_recorder_initialized = False
    room_name = None

    logger.info(f"WebSocket connected: {meeting_id}")

    try:
        # Initialize caption handler for this session
        if EngineManager.get_lmm():
            caption_handler = SessionManager.get_or_create_caption_handler(
                meeting_id=meeting_id,
                engine=EngineManager.get_lmm(),
                vad_model=EngineManager.get_vad()
            )
        while True:
            # Receive message from client
            message = await websocket.receive_text()

            try:
                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type == "audio":
                    # Binary audio data
                    try:
                        request = MeetweenTranscriptionRequest(**data)

                        if not audio_recorder_initialized:
                            room_name = request.room_name
                            AudioRecorder.init_recording(room_name=room_name, meeting_id=meeting_id)
                            audio_recorder_initialized = True


                        audio_data = request.get_audio_bytes()
                        await AudioRecorder.save_chunk(room_name=room_name, meeting_id=meeting_id, audio_data=audio_data, chunk_duration=2.0)

                        if not EngineManager.get_lmm() or not caption_handler:
                            error_response = MeetweenErrorResponse(
                                message="SpeechLMM not available",
                                code="SERVICE_UNAVAILABLE"
                            )
                            logger.error("SpeechLMM not available")
                            await websocket.send_json(error_response.model_dump())
                            continue

                        try:
                            # Process audio through caption handler
                            result = await caption_audio(
                                engine=EngineManager.get_lmm(),
                                audio_bytes=audio_data,
                                sample_rate=16000,
                                caption_handler=caption_handler,
                                timestamp=time.time_ns() // 1_000_000,
                                language=request.language,
                                vad=EngineManager.get_vad()
                            )

                            # Send transcription result
                            response = MeetweenTranscriptionResponse(
                                meeting_id=meeting_id,
                                participant_id=request.participant_id,
                                transcript=result.text,
                                message_id=str(UUIDHelper.uuid_from_int(result.message_id)),
                                timestamp=result.timestamp,
                                is_interim=result.is_interim,
                                language=request.language,
                                stability=result.stability
                            )
                            await websocket.send_json(response.model_dump())

                        except Exception as e:
                            logger.error(f"Error processing audio: {e}", exc_info=True)
                            error_response = MeetweenErrorResponse(
                                message=f"Processing error: {str(e)}",
                                code="PROCESSING_ERROR"
                            )
                            await websocket.send_json(error_response.model_dump())

                    except ValidationError as e:
                        error = MeetweenErrorResponse(
                            message=f"Invalid audio request: {e}",
                            code="INVALID_REQUEST"
                        )
                        logger.warning(e)
                        await websocket.send_text(error.model_dump_json())

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON: {e}")
                error = MeetweenErrorResponse(
                    message=f"Invalid JSON: {e}",
                    code="INVALID_JSON"
                )
                await websocket.send_text(error.model_dump_json())
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {meeting_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Cleanup
        try:
            if room_name:
                AudioRecorder.stop_recording(room_name=room_name, meeting_id=meeting_id)
            SessionManager.remove_connection(meeting_id)
        except Exception as e:
            logger.warning(f"Error during cleanup, {e}")
        logger.info(f"WebSocket session closed: {meeting_id}")
