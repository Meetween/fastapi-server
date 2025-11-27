import uuid
import json
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from typing import Optional
from core import EngineManager, SessionManager
from butler import CaptionHandler, caption_audio

from models import MeetweenTranscriptionResponse, MeetweenErrorResponse


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
            audio_data=audio_data,
            handler=handler,
            language=language
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
    session_id = str(uuid.uuid4())
    await websocket.accept()
    
    SessionManager.add_connection(room_name, websocket)
    caption_handler = None
    
    logger.info(f"WebSocket connected: : {room_name}")
    
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
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary audio data
                audio_data = message["bytes"]
                
                if not EngineManager.get_lmm() or not caption_handler:
                    error_response = MeetweenErrorResponse(
                        message="SpeechLMM not available",
                        code="SERVICE_UNAVAILABLE"
                    )
                    await websocket.send_json(error_response.model_dump())
                    continue
                
                try:
                    # Process audio through caption handler
                    result = caption_audio(
                        audio_data=audio_data,
                        handler=caption_handler,
                        language="en-US"
                    )
                    
                    # Send transcription result
                    response = MeetweenTranscriptionResponse(
                        transcript=result.text,
                        is_final=False,
                        confidence=result.confidence,
                        transcript_id=str(uuid.uuid4()),
                        language="en-US",
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
            
            elif "text" in message:
                # Text message (could be control messages)
                try:
                    data = json.loads(message["text"])
                    
                    if data.get("type") == "reset":
                        # Reset caption session
                        if caption_handler:
                            caption_handler = SessionManager.get_or_create_caption_handler(
                                room_name=room_name,
                                engine=EngineManager.get_lmm(),
                                vad_model=EngineManager.get_vad()
                            )
                        await websocket.send_json({
                            "type": "control",
                            "message": "Session reset"
                        })
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message['text']}")
                except Exception as e:
                    logger.error(f"Error handling text message: {e}")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {room_name}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Cleanup
        SessionManager.remove_connection(room_name)
        logger.info(f"WebSocket session closed: {room_name}")
