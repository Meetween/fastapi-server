import uuid
import json
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from typing import Optional
from core import EngineManager, SessionManager
from butler import CaptionHandler, caption_audio
from utils import AudioRecorder
from pydantic import ValidationError

from models import MeetweenTranscriptionRequest, MeetweenTranscriptionResponse, MeetweenErrorResponse


router = APIRouter(prefix="/api", tags=["caption"])
ws_router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/caption", response_model=MeetweenTranscriptionResponse)
async def api_caption(
    audio: UploadFile = File(..., description="Audio chunk for real-time captioning"),
    language: str = Form("en-US", description="BCP-47 language code (e.g., 'en-US', 'de-DE')"),
    participant_id: Optional[str] = Form(None, description="Participant identifier"),
    session_id: Optional[str] = Form(None, description="Caption session ID")
):
    """
    Process audio chunk for real-time captioning
    
    Returns interim captions that can be updated with subsequent audio chunks.
    Use /api/caption/reset to reset the caption session.
    """
    if not EngineManager.get_lmm():
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded"
        )
    
    try:
        audio_data = await audio.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Use session_id or participant_id as caption session key
        caption_key = session_id or participant_id or str(uuid.uuid4())
        
        # Get or create caption handler
        handler = SessionManager.get_or_create_caption_handler(
                room_name = caption_key,
                engine=EngineManager.get_lmm(),
                vad_model=EngineManager.get_vad()
            )
                
        # Process audio chunk
        result = caption_audio(
            engine=EngineManager.get_lmm(),
            audio_bytes=audio_data,
            sample_rate=16000,
            caption_handler=handler,
            language=language,
            vad=EngineManager.get_vad()
        )
        
        return MeetweenTranscriptionResponse(
            transcript=result.text,
            is_final=False,
            confidence=result.confidence * 0.6,  # Lower confidence for interim results
            transcript_id=str(uuid.uuid4()),
            participant_id=participant_id,
            language=language,
            message_id=0,
            timestamp=0.0,
            is_interim=True,
            stability=0.6
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Captioning error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Captioning failed: {str(e)}")


@router.post("/caption/reset")
async def api_caption_reset(
    session_id: Optional[str] = Form(None),
    participant_id: Optional[str] = Form(None)
):
    """
    Reset caption session, clearing accumulated context
    """
    caption_key = session_id or participant_id
        
    if SessionManager.reset_caption_handler(caption_key):
        return {"status": "success", "message": f"Caption session {caption_key} reset"}
    
    return {"status": "success", "message": "No active caption session found"}


@ws_router.websocket("/ws/{room_name}")
async def websocket_caption(websocket: WebSocket, room_name: str):
    """
    WebSocket endpoint for real-time streaming transcription
    
    Clients send audio chunks and receive transcription updates in real-time.
    Supports both interim and final results.
    """
    await websocket.accept()
    
    SessionManager.add_connection(room_name, websocket)
    caption_handler = None
    
    logger.info(f"WebSocket connected: {room_name}")
    
    AudioRecorder.init_recording(meeting_id=room_name)
    try:
        # Initialize caption handler for this session
        if EngineManager.get_lmm():
            caption_handler = SessionManager.get_or_create_caption_handler(
                room_name=room_name,
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
                        audio_data = request.get_audio_bytes()
                        await AudioRecorder.save_chunk(meeting_id=room_name, audio_data=audio_data, chunk_duration=2.0)
                    
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
                            result = caption_audio(
                                engine=EngineManager.get_lmm(),
                                audio_bytes=audio_data,
                                sample_rate=16000,
                                caption_handler=caption_handler,
                                language=request.language,
                                vad=EngineManager.get_vad()
                            )
                            
                            # Send transcription result
                            response = MeetweenTranscriptionResponse(
                                meeting_id=room_name,
                                transcript=result.text,
                                is_final=False,
                                confidence=result.confidence,
                                transcript_id=str(uuid.uuid4()),
                                participant_id=request.participant_id,
                                language=request.language,
                                is_interim=True,
                                stability=0.6
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
        logger.info(f"WebSocket disconnected: {room_name}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Cleanup
        try:
            AudioRecorder.stop_recording(meeting_id=room_name)
            SessionManager.remove_connection(room_name)
        except Exception as e:
            logger.warning(f"Error during cleanup, {e}")
        logger.info(f"WebSocket session closed: {room_name}")
