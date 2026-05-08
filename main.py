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

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import settings
from core import EngineManager, SessionManager
from utils.storage import StorageManager

from api.routes import (
    health,
    caption,
    transcription,
    summary,
    config,
    chat,
    agenda,
    auth,
)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse



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

# Scheduler for periodic tasks
scheduler = AsyncIOScheduler()


def cleanup_old_recordings_job():
    """Scheduled job to delete old recordings"""
    try:
        result = StorageManager.cleanup_old_recordings()
        logger.info(f"Recording cleanup job completed: {result['deleted_count']} recordings deleted")
        if result['errors']:
            logger.warning(f"Cleanup job encountered {len(result['errors'])} errors")
    except Exception as e:
        logger.error(f"Error in recording cleanup job: {str(e)}", exc_info=True)


@app.on_event("startup")
def startup_event():
    """Initialize server and load SpeechLMM model"""

    logger.info("Server starting up...")

    EngineManager.initialize()
    settings.recordings_base_path.mkdir(parents=True, exist_ok=True)

    # Schedule daily cleanup of old recordings
    scheduler.add_job(
        cleanup_old_recordings_job,
        trigger=CronTrigger(hour=2, minute=0),  # Run daily at 2:00 AM UTC
        id="cleanup_old_recordings",
        name="Cleanup old recordings",
        replace_existing=True
    )
    scheduler.start()
    logger.info("Scheduled recording cleanup job (daily at 2:00 AM UTC)")

    logger.info("Server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("Server shutting down...")

    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler shut down")

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
app.include_router(config.router)
app.include_router(chat.router)
app.include_router(agenda.router)
app.include_router(auth.router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False
    )
