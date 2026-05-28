"""Meeting recording endpoints (Catch Me Up storage)."""

import json
import logging
import shutil
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.deps.auth import REQUIRE_AUTH
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/recording",
    tags=["recording"],
    dependencies=REQUIRE_AUTH,
)

_active_meetings: Dict[str, dict] = {}


def _init_recording(meeting_id: str, participant_id: Optional[str] = None) -> dict:
    meeting_path = settings.recordings_base_path / meeting_id
    meeting_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "meeting_id": meeting_id,
        "start_time": datetime.utcnow().isoformat(),
        "started_by": participant_id,
        "chunk_count": 0,
        "total_duration": 0.0,
    }

    metadata_path = meeting_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    _active_meetings[meeting_id] = metadata
    logger.info("Started recording for meeting: %s", meeting_id)
    return metadata


@router.post("/start")
async def start_recording(
    meeting_id: str = Form(..., description="Unique meeting identifier"),
    participant_id: Optional[str] = Form(None, description="Participant who started recording"),
):
    """Initialize recording for a meeting."""
    _init_recording(meeting_id, participant_id)
    return JSONResponse({
        "status": "success",
        "meeting_id": meeting_id,
        "message": "Recording started",
    })


@router.post("/chunk")
async def save_audio_chunk(
    meeting_id: str = Form(..., description="Meeting identifier"),
    audio: UploadFile = File(..., description="Audio chunk to store"),
    chunk_duration: float = Form(..., description="Duration of chunk in seconds"),
    timestamp: Optional[float] = Form(None, description="Timestamp in meeting"),
):
    """Save an audio chunk for a meeting."""
    meeting_path = settings.recordings_base_path / meeting_id

    if not meeting_path.exists():
        _init_recording(meeting_id)

    metadata_path = meeting_path / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    chunk_num = metadata["chunk_count"] + 1
    chunk_filename = f"chunk_{chunk_num:04d}.webm"
    chunk_path = meeting_path / chunk_filename

    audio_data = await audio.read()
    with open(chunk_path, "wb") as f:
        f.write(audio_data)

    metadata["chunk_count"] = chunk_num
    metadata["total_duration"] += chunk_duration
    metadata["last_chunk_time"] = datetime.utcnow().isoformat()

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Saved chunk %s for meeting %s (%s bytes, %ss)",
        chunk_num,
        meeting_id,
        len(audio_data),
        chunk_duration,
    )

    return JSONResponse({
        "status": "success",
        "chunk_number": chunk_num,
        "total_chunks": chunk_num,
        "total_duration": metadata["total_duration"],
    })


@router.post("/stop")
async def stop_recording(
    meeting_id: str = Form(..., description="Meeting identifier"),
    delete_recording: bool = Form(False, description="Delete recording after stopping"),
):
    """Stop recording for a meeting."""
    meeting_path = settings.recordings_base_path / meeting_id

    if not meeting_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No recording found for meeting: {meeting_id}",
        )

    metadata_path = meeting_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        metadata["end_time"] = datetime.utcnow().isoformat()

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    if meeting_id in _active_meetings:
        del _active_meetings[meeting_id]

    if delete_recording:
        shutil.rmtree(meeting_path)
        logger.info("Deleted recording for meeting: %s", meeting_id)
        return JSONResponse({
            "status": "success",
            "message": "Recording stopped and deleted",
            "meeting_id": meeting_id,
        })

    logger.info("Stopped recording for meeting: %s", meeting_id)
    return JSONResponse({
        "status": "success",
        "message": "Recording stopped",
        "meeting_id": meeting_id,
        "recording_path": str(meeting_path),
    })
