"""
Meetween FastAPI Server
Provides SpeechLMM-based API endpoints for transcription, captions, and summarization
Supports WebSocket streaming for real-time transcription
"""
import json
import logging
import uuid
import os
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime
from config import settings
from core import EngineManager

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
    title=settings.app_name,
    version=settings.app_version,
    description="SpeechLMM-powered transcription, captioning, and summarization API"
)

# Active WebSocket connections and caption handlers
active_connections: Dict[str, WebSocket] = {}
caption_handlers: Dict[str, CaptionHandler] = {}


# Active meeting recordings
active_meetings: Dict[str, dict] = {}  # meeting_id -> metadata


@app.on_event("startup")
def startup_event():
    """Initialize server and load SpeechLMM model"""
    
    logger.info("Server starting up...")
    
    EngineManager.initialize()
    settings.recordings_base_path.mkdir(parents=True, exist_ok=True) 
    
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
        "speechlmm_loaded": EngineManager.get_lmm() is not None,
        "vad_loaded": EngineManager.get_vad() is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "speechlmm": "loaded" if EngineManager.get_lmm() else "not loaded",
        "vad": "loaded" if EngineManager.get_vad() else "not loaded"
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
    if not EngineManager.get_lmm():
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
            engine=EngineManager.get_lmm(),
            language=language,
            vad_model=EngineManager.get_vad()
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
        if caption_key not in caption_handlers:
            caption_handlers[caption_key] = CaptionHandler(
                engine=EngineManager.get_lmm(),
                vad_model=EngineManager.get_vad()
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
    
    - **audio**: Audio file containing meeting or conversation (supports WebM, WAV, MP3)
    - **language**: BCP-47 language code (default: 'en-US')
    - **participant_id**: Optional participant identifier
    """
    if not EngineManager.get_lmm():
        raise HTTPException(
            status_code=503,
            detail="SpeechLMM model not loaded"
        )
    
    try:
        import tempfile
        import os
        import numpy as np
        from pydub import AudioSegment
        
        audio_data = await audio.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Summarizing {len(audio_data)} bytes from {audio.content_type or 'unknown'}, language: {language}")
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            # Load audio with pydub (better ffmpeg support)
            audio_segment = AudioSegment.from_file(temp_path)
            
            # Convert to mono, 16kHz (model's native sample rate - avoids resampling)
            audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
            
            # Get raw PCM data
            audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            audio_pcm = audio_array.tobytes()
            
            logger.info(f"Decoded audio: {len(audio_pcm)} bytes at 16000Hz")
            
            # Generate summary
            summary = summarize_audio(
                audio_bytes=audio_pcm,
                sample_rate=16000,
                engine=EngineManager.get_lmm(),
                language=language
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
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
        if EngineManager.get_lmm():
            caption_handler = CaptionHandler(
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
                            caption_handler = CaptionHandler(
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
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Cleanup
        if session_id in active_connections:
            del active_connections[session_id]
        logger.info(f"WebSocket session closed: {session_id}")


# ============================================================================
# Meeting Recording Endpoints (for "Catch Me Up" feature)
# ============================================================================

@app.post("/api/recording/start")
async def start_recording(
    meeting_id: str = Form(..., description="Unique meeting identifier"),
    participant_id: Optional[str] = Form(None, description="Participant who started recording")
):
    """
    Initialize recording for a meeting.
    Creates storage directory and metadata file.
    """
    meeting_path = settings.recordings_base_path / meeting_id
    meeting_path.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "meeting_id": meeting_id,
        "start_time": datetime.utcnow().isoformat(),
        "started_by": participant_id,
        "chunk_count": 0,
        "total_duration": 0.0
    }
    
    # Save metadata
    metadata_path = meeting_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    active_meetings[meeting_id] = metadata
    
    logger.info(f"Started recording for meeting: {meeting_id}")
    return JSONResponse({
        "status": "success",
        "meeting_id": meeting_id,
        "message": "Recording started"
    })


@app.post("/api/recording/chunk")
async def save_audio_chunk(
    meeting_id: str = Form(..., description="Meeting identifier"),
    audio: UploadFile = File(..., description="Audio chunk to store"),
    chunk_duration: float = Form(..., description="Duration of chunk in seconds"),
    timestamp: Optional[float] = Form(None, description="Timestamp in meeting")
):
    """
    Save an audio chunk for a meeting.
    Called continuously as meeting progresses.
    """
    meeting_path = settings.recordings_base_path / meeting_id
    
    if not meeting_path.exists():
        # Auto-initialize if not started
        await start_recording(meeting_id=meeting_id)
    
    # Load metadata
    metadata_path = meeting_path / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Increment chunk count
    chunk_num = metadata["chunk_count"] + 1
    chunk_filename = f"chunk_{chunk_num:04d}.webm"
    chunk_path = meeting_path / chunk_filename
    
    # Save audio chunk
    audio_data = await audio.read()
    with open(chunk_path, "wb") as f:
        f.write(audio_data)
    
    # Update metadata
    metadata["chunk_count"] = chunk_num
    metadata["total_duration"] += chunk_duration
    metadata["last_chunk_time"] = datetime.utcnow().isoformat()
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved chunk {chunk_num} for meeting {meeting_id} ({len(audio_data)} bytes, {chunk_duration}s)")
    
    return JSONResponse({
        "status": "success",
        "chunk_number": chunk_num,
        "total_chunks": chunk_num,
        "total_duration": metadata["total_duration"]
    })


@app.post("/api/recording/catchup", response_model=MeetweenSummarizationResponse)
async def catchup_on_meeting(
    meeting_id: str = Form(..., description="Meeting identifier"),
    participant_id: Optional[str] = Form(None, description="Participant requesting catch-up"),
    language: str = Form("en-US", description="BCP-47 language code"),
    max_duration: Optional[float] = Form(None, description="Max audio duration to process (seconds)")
):
    """
    Generate a summary of all recorded audio for a meeting.
    This is the "Catch Me Up" feature - summarizes everything that happened so far.
    """
    if not EngineManager.get_lmm():
        raise HTTPException(status_code=503, detail="SpeechLMM model not loaded")
    
    meeting_path = settings.recordings_base_path / meeting_id
    
    if not meeting_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No recording found for meeting: {meeting_id}"
        )
    
    # Load metadata
    metadata_path = meeting_path / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    chunk_count = metadata["chunk_count"]
    
    if chunk_count == 0:
        return MeetweenSummarizationResponse(
            summary="No audio recorded yet. The meeting just started.",
            participant_id=participant_id,
            language=language
        )
    
    logger.info(f"Processing catch-up for meeting {meeting_id}: {chunk_count} chunks, {metadata['total_duration']:.1f}s")
    
    try:
        import tempfile
        import numpy as np
        from pydub import AudioSegment
        
        # Combine all chunks into one audio file
        combined_audio = AudioSegment.empty()
        successful_chunks = 0
        
        for chunk_num in range(1, chunk_count + 1):
            chunk_path = meeting_path / f"chunk_{chunk_num:04d}.webm"
            if chunk_path.exists():
                try:
                    # Verify chunk has content and is not empty
                    if chunk_path.stat().st_size < 100:  # Skip tiny/empty files
                        logger.warning(f"Skipping empty chunk: {chunk_path.name}")
                        continue
                    
                    chunk_audio = AudioSegment.from_file(str(chunk_path))
                    
                    # Skip very short chunks (< 0.5 seconds)
                    if len(chunk_audio) < 500:
                        logger.warning(f"Skipping short chunk: {chunk_path.name} ({len(chunk_audio)}ms)")
                        continue
                    
                    combined_audio += chunk_audio
                    successful_chunks += 1
                    
                    # Limit duration if specified
                    if max_duration and len(combined_audio) / 1000.0 > max_duration:
                        combined_audio = combined_audio[:int(max_duration * 1000)]
                        break
                        
                except Exception as e:
                    # Skip corrupted chunks and continue
                    logger.warning(f"Skipping corrupted chunk {chunk_path.name}: {e}")
                    continue
        
        if len(combined_audio) == 0:
            return MeetweenSummarizationResponse(
                summary=f"No valid audio found. Recording contains {chunk_count} chunks but none could be decoded. The meeting may have just started.",
                participant_id=participant_id,
                language=language
            )
        
        logger.info(f"Successfully combined {successful_chunks}/{chunk_count} chunks")
        
        # Convert to mono, 16kHz
        combined_audio = combined_audio.set_channels(1).set_frame_rate(16000)
        
        # Get raw PCM data
        audio_array = np.array(combined_audio.get_array_of_samples(), dtype=np.int16)
        audio_pcm = audio_array.tobytes()
        
        duration_seconds = len(combined_audio) / 1000.0
        logger.info(f"Combined audio: {len(audio_pcm)} bytes, {duration_seconds:.1f}s at 16kHz")
        
        # Generate summary
        summary = summarize_audio(
            audio_bytes=audio_pcm,
            sample_rate=16000,
            engine=EngineManager.get_lmm(),
            language=language
        )
        
        return MeetweenSummarizationResponse(
            summary=summary,
            participant_id=participant_id,
            language=language
        )
        
    except Exception as e:
        logger.error(f"Catch-up error for meeting {meeting_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate catch-up summary: {str(e)}"
        )


@app.post("/api/recording/stop")
async def stop_recording(
    meeting_id: str = Form(..., description="Meeting identifier"),
    delete_recording: bool = Form(False, description="Delete recording after stopping")
):
    """
    Stop recording for a meeting.
    Optionally delete the recording files.
    """
    meeting_path = settings.recordings_base_path / meeting_id
    
    if not meeting_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No recording found for meeting: {meeting_id}"
        )
    
    # Update metadata
    metadata_path = meeting_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        metadata["end_time"] = datetime.utcnow().isoformat()
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    # Remove from active meetings
    if meeting_id in active_meetings:
        del active_meetings[meeting_id]
    
    # Optionally delete recording
    if delete_recording:
        import shutil
        shutil.rmtree(meeting_path)
        logger.info(f"Deleted recording for meeting: {meeting_id}")
        return JSONResponse({
            "status": "success",
            "message": "Recording stopped and deleted",
            "meeting_id": meeting_id
        })
    else:
        logger.info(f"Stopped recording for meeting: {meeting_id}")
        return JSONResponse({
            "status": "success",
            "message": "Recording stopped",
            "meeting_id": meeting_id,
            "recording_path": str(meeting_path)
        })


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False
    )
