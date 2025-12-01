import json
from pathlib import Path
from datetime import datetime
from models import MeetweenTranscriptionResponse
import logging

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages file storage for recordings"""

    @staticmethod
    def create_meeting_directory(base_path: Path, meeting_id: str) -> Path:
        meeting_path = base_path / meeting_id
        meeting_path.mkdir(parents=True, exist_ok=True)
        return meeting_path

    @staticmethod
    def save_metadata(meeting_path: Path, metadata: dict):
        metadata_path = meeting_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Saved metadata to {metadata_path}")

    @staticmethod
    def load_metadata(meeting_path: Path) -> dict:
        metadata_path = meeting_path / "metadata.json"
        with open(metadata_path, "r") as f:
            return json.load(f)

    @staticmethod
    def get_chunk_filename(chunk_number: int) -> str:
        return f"chunk_{chunk_number:04d}.webm"

    
