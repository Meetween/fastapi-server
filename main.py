"""
Meetween FastAPI Server
Provides SpeechLMM-based API endpoints for transcription, captions, and summarization
Supports WebSocket streaming for real-time transcription
"""
import json
import logging
import uuid
import os
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from models import (
    MeetweenTranscriptionResponse,
    MeetweenSummarizationResponse,
    MeetweenErrorResponse,
    TranscriptionResult
)

from butler import (
    SpeechLMMEngine,
    transcribe_audio,
    caption_audio,
    summarize_audio,
    CaptionHandler,
    SileroVAD
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Meetween FastAPI Server",
    version="1.0.0",
    description="SpeechLMM-powered transcription, captioning, and summarization API"
)

# Global SpeechLMM engine and VAD model
speechlmm_engine: Optional[SpeechLMMEngine] = None
vad_model: Optional[SileroVAD] = None

# Active WebSocket connections and caption handlers
active_connections: Dict[str, WebSocket] = {}
caption_handlers: Dict[str, CaptionHandler] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize server and load SpeechLMM model"""
    global speechlmm_engine, vad_model
    
    logger.info("Server starting up...")
    
    # Load SpeechLMM model
    model_path = os.getenv("SPEECHLMM_MODEL_PATH")
    if not model_path:
        logger.error("SPEECHLMM_MODEL_PATH environment variable not set")
        raise RuntimeError("SPEECHLMM_MODEL_PATH is required")
    
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        raise RuntimeError(f"Model path not found: {model_path}")
    
    try:
        logger.info(f"Loading SpeechLMM model from {model_path}")
        speechlmm_engine = SpeechLMMEngine(
            model_path=model_path,
            device=None,  # Auto-detect device (CUDA if available)
            verbose=True
        )
        logger.info("SpeechLMM model loaded successfully")
        
        # Initialize VAD model
        try:
            vad_model = SileroVAD(sample_rate=16000, threshold=0.5)
            logger.info("VAD model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load VAD model: {e}")
            vad_model = None
            
    except Exception as e:
        logger.error(f"Failed to load SpeechLMM model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize SpeechLMM: {e}")
    
    logger.info("Server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("Server shutting down...")
    
    # Close all active WebSocket connections
    for session_id, ws in list(active_connections.items()):
        try:
            await ws.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket {session_id}: {e}")
    
    active_connections.clear()
    caption_handlers.clear()
    
    logger.info("Server shutdown complete")


# ===== REST API ENDPOINTS =====

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Meetween FastAPI Server",
        "version": "1.0.0",
        "status": "running",
        "speechlmm_loaded": speechlmm_engine is not None,
        "vad_loaded": vad_model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "speechlmm": "loaded" if speechlmm_engine else "not loaded",
        "vad": "loaded" if vad_model else "not loaded"
    }


@app.post("/api/transcribe", response_model=MeetweenTranscriptionResponse)
async def api_transcribe(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form("en-US", description="BCP-47 language code (e.g., 'en-US', 'es-ES', 'fr-FR')"),
    participant_id: Optional[str] = Form(None, description="Optional participant identifier")
):
    """
    Transcribe audio file and return the full transcription
    
    - **audio**: Audio file (WAV, MP3, etc.)
    - **language**: BCP-47 language code (default: 'en-US')
    - **participant_id**: Optional participant identifier
    """
    if not speechlmm_engine:
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded. Check server configuration."
        )
    
    try:
        # Read audio file
        audio_data = await audio.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Transcribing {len(audio_data)} bytes, language: {language}")
        
        # Perform transcription
        result = transcribe_audio(
            audio_data=audio_data,
            engine=speechlmm_engine,
            language=language,
            vad_model=vad_model
        )
        
        # Build response
        return MeetweenTranscriptionResponse(
            transcript=result.text,
            is_final=True,
            confidence=result.confidence,
            transcript_id=str(uuid.uuid4()),
            participant_id=participant_id,
            language=language,
            message_id=0,
            timestamp=0.0,
            is_interim=False,
            stability=1.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/caption", response_model=MeetweenTranscriptionResponse)
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
    if not speechlmm_engine:
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
        if caption_key not in caption_handlers:
            caption_handlers[caption_key] = CaptionHandler(
                engine=speechlmm_engine,
                vad_model=vad_model
            )
        
        handler = caption_handlers[caption_key]
        
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


@app.post("/api/caption/reset")
async def api_caption_reset(
    session_id: Optional[str] = Form(None),
    participant_id: Optional[str] = Form(None)
):
    """
    Reset caption session, clearing accumulated context
    """
    caption_key = session_id or participant_id
    
    if caption_key and caption_key in caption_handlers:
        del caption_handlers[caption_key]
        return {"status": "success", "message": f"Caption session {caption_key} reset"}
    
    return {"status": "success", "message": "No active caption session found"}


@app.post("/api/summarize", response_model=MeetweenSummarizationResponse)
async def api_summarize(
    audio: UploadFile = File(..., description="Audio file to summarize"),
    language: str = Form("en-US", description="BCP-47 language code (e.g., 'en-US', 'de-DE')"),
    participant_id: Optional[str] = Form(None, description="Participant identifier")
):
    """
    Generate a summary of the audio content
    
    - **audio**: Audio file containing meeting or conversation
    - **language**: BCP-47 language code (default: 'en-US')
    - **participant_id**: Optional participant identifier
    """
    if not speechlmm_engine:
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded"
        )
    
    try:
        audio_data = await audio.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Summarizing {len(audio_data)} bytes, language: {language}")
        
        # Generate summary
        summary = summarize_audio(
            audio_data=audio_data,
            engine=speechlmm_engine,
            language=language,
            vad_model=vad_model
        )
        
        return MeetweenSummarizationResponse(
            summary=summary,
            participant_id=participant_id,
            language=language
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


# ===== WEBSOCKET ENDPOINT =====

@app.websocket("/ws")
async def websocket_transcription(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming transcription
    
    Clients send audio chunks and receive transcription updates in real-time.
    Supports both interim and final results.
    """
    session_id = str(uuid.uuid4())
    await websocket.accept()
    
    active_connections[session_id] = websocket
    caption_handler = None
    
    logger.info(f"WebSocket connected: {session_id}")
    
    try:
        # Initialize caption handler for this session
        if speechlmm_engine:
            caption_handler = CaptionHandler(
                engine=speechlmm_engine,
                vad_model=vad_model
            )
        
        while True:
            # Receive message from client
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary audio data
                audio_data = message["bytes"]
                
                if not speechlmm_engine or not caption_handler:
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
                            caption_handler = CaptionHandler(
                                engine=speechlmm_engine,
                                vad_model=vad_model
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
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Cleanup
        if session_id in active_connections:
            del active_connections[session_id]
        logger.info(f"WebSocket session closed: {session_id}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
