"""
Meeting recording business logic
"""
import logging
import shutil
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime

from utils import StorageManager
from config import settings

logger = logging.getLogger(__name__)

class AudioRecorder:
    """Service for handling meeting recordings"""

    # Track active meetings
    _active_meetings: Dict[str, Dict] = {}
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def init_recording(cls, meeting_id: str, participant_id: Optional[str] = None):
        """Initialize recording for a meeting"""
        meeting_path = StorageManager.create_meeting_directory(
            settings.recordings_base_path,
            meeting_id
        )

        metadata = {
            "meeting_id": meeting_id,
            "start_time": datetime.utcnow().isoformat(),
            "started_by": participant_id,
            "chunk_count": 0,
            "total_duration": 0.0
        }

        StorageManager.save_metadata(meeting_path, metadata)
        cls._active_meetings[meeting_id] = metadata

    @classmethod
    async def save_chunk(
        cls,
        meeting_id: str,
        audio_data: bytes,
        chunk_duration: float
    ) -> dict:
        """Save an audio chunk for a meeting"""
        meeting_path = settings.recordings_base_path / meeting_id

        if not meeting_path.exists():
            cls.init_recording(meeting_id)

        metadata = StorageManager.load_metadata(meeting_path)
 
        chunk_num = metadata["chunk_count"] + 1
        chunk_filename = StorageManager.get_chunk_filename(chunk_num)
        chunk_path = meeting_path / chunk_filename

        with open(chunk_path, "wb") as f:
            f.write(audio_data)

        # Save metadata
        metadata["chunk_count"] = chunk_num
        metadata["total_duration"] += chunk_duration
        metadata["last_chunk_time"] = datetime.utcnow().isoformat()

        StorageManager.save_metadata(meeting_path, metadata)
        return {
            "chunk_number": chunk_num,
            "total_chunks": chunk_num,
            "total_duration": metadata["total_duration"]
        }

    @classmethod
    def stop_recording(cls, meeting_id: str, delete_recording: bool = False) -> dict:
        """Stop recording for a meeting"""
        meeting_path = settings.recordings_base_path / meeting_id

        if not meeting_path.exists():
            logging.error(f"Could not stop recording. {meeting_path} does not exist")
            return {
                "status": "error",
                "message": "Could not stop recording",
            } 
            
        # Update metadata
        metadata = StorageManager.load_metadata(meeting_path)
        metadata["end_time"] = datetime.utcnow().isoformat()
        StorageManager.save_metadata(meeting_path, metadata)

        # Remove from active meetings
        if meeting_id in cls._active_meetings:
            del cls._active_meetings[meeting_id]

        # Optionally delete recording
        if delete_recording:
            shutil.rmtree(meeting_path)
            logger.info(f"Deleted recording for meeting: {meeting_id}")
            return {
                "status": "success",
                "message": "Recording stopped and deleted",
                "meeting_id": meeting_id
            }
        else:
            return {
                "status": "success",
                "message": "Recording stopped",
                "meeting_id": meeting_id,
                "recording_path": str(meeting_path)
            }
