"""
Meetween FastAPI Server
Provides SpeechLMM-based API endpoints for transcription, captions, and summarization
Supports WebSocket streaming for real-time transcription
"""
import json
import logging
import uvicorn

from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

from config import settings
from core import EngineManager, SessionManager

from api.routes import (
    health,
    caption,
    transcription,
    summary
)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse


from models import (
    MeetweenSummarizationResponse
)

from butler import (
    summarize_audio,
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


    await SessionManager.cleanup()
    await EngineManager.cleanup()

    logger.info("Server shutdown complete")

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

app.include_router(health.router)
app.include_router(caption.router)
app.include_router(caption.ws_router)
app.include_router(transcription.router)
app.include_router(summary.router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False
    )
