import json
import shutil
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional
from config import settings
import logging

logger = logging.getLogger(__name__)

_SEPARATORS = frozenset("/\\")
_MAX_SEGMENT = 255


class UnsafePathError(ValueError):
    """A recording path segment, or the place it resolves to, is not `<base>/<room>/<meeting>`."""


def _check_segment(value: object, what: str) -> str:
    """One plain path component: not empty, not `.` or `..`, no separator, no control character."""
    if not isinstance(value, str) or not value or len(value) > _MAX_SEGMENT:
        raise UnsafePathError(f"unsafe {what}")
    if value in {".", ".."} or any(ch in _SEPARATORS or unicodedata.category(ch) == "Cc" for ch in value):
        raise UnsafePathError(f"unsafe {what}")
    return value


class StorageManager:
    """Manages file storage for recordings"""

    @staticmethod
    def meeting_path(base_path: Path, room_name: str, meeting_id: str) -> Path:
        """
        The recording directory of one meeting, `<base>/<room_name>/<meeting_id>`, resolved.

        Each name must be a single plain component, and the resolved path must equal the lexical one
        under the resolved base: `..`, an absolute name, a separator, or a symlink at the room or
        meeting level (even one pointing inside the base) raises UnsafePathError.
        """
        _check_segment(room_name, "room_name")
        _check_segment(meeting_id, "meeting_id")
        base = Path(base_path).resolve()
        expected = base / room_name / meeting_id
        if expected.resolve() != expected:
            raise UnsafePathError("recording path leaves the recordings base")
        return expected

    @staticmethod
    def find_meeting_directory(base_path: Path, room_name: str, meeting_id: str) -> Optional[Path]:
        """The meeting's recording directory when it is safe and exists, else None."""
        try:
            path = StorageManager.meeting_path(base_path, room_name, meeting_id)
        except UnsafePathError:
            logger.warning("Refused recording path: room=%r meeting=%r", room_name, meeting_id)
            return None
        return path if path.is_dir() else None

    @staticmethod
    def create_meeting_directory(base_path: Path, room_name: str, meeting_id: str) -> Path:
        meeting_path = StorageManager.meeting_path(base_path, room_name, meeting_id)
        meeting_path.mkdir(parents=True, exist_ok=True)
        # A link swapped in between the check and mkdir is caught here, before any file is written.
        if meeting_path.resolve() != meeting_path:
            raise UnsafePathError("recording path leaves the recordings base")
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
    def get_chunk_filename(chunk_number: int, suffix="wav") -> str:
        suffix = suffix if settings.recording_format is None else settings.recording_format
        return f"chunk_{chunk_number:04d}.{suffix}"

    @staticmethod
    def cleanup_old_recordings() -> dict:
        """
        Delete recordings older than the retention period.
        Checks metadata.json for end_time (or start_time if no end_time).
        Returns statistics about deleted recordings.
        """
        base_path = settings.recordings_base_path
        retention_days = settings.recording_retention_days
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        deleted_count = 0
        errors = []
        
        if not base_path.exists():
            logger.warning(f"Recordings base path does not exist: {base_path}")
            return {
                "deleted_count": 0,
                "errors": []
            }
        
        logger.info(f"Starting cleanup of recordings older than {retention_days} days (cutoff: {cutoff_date.isoformat()})")
        
        for room_path in base_path.iterdir():
            # A symlink is never followed: rmtree through a linked room would delete outside the base.
            if room_path.is_symlink() or not room_path.is_dir():
                continue
            
            for meeting_path in room_path.iterdir():
                if meeting_path.is_symlink() or not meeting_path.is_dir():
                    continue
                
                try:
                    metadata_path = meeting_path / "metadata.json"
                    
                    if not metadata_path.exists():
                        dir_mtime = datetime.fromtimestamp(meeting_path.stat().st_mtime)
                        if dir_mtime < cutoff_date:
                            shutil.rmtree(meeting_path)
                            deleted_count += 1
                            logger.info(f"Deleted recording without metadata: {meeting_path} (last modified: {dir_mtime.isoformat()})")
                        continue
                    
                    metadata = StorageManager.load_metadata(meeting_path)
                    
                    time_key = "end_time" if "end_time" in metadata else "start_time"
                    if time_key not in metadata:
                        logger.warning(f"No timestamp found in metadata for {meeting_path}, skipping")
                        continue
                    
                    time_str = metadata[time_key]
                    if time_str.endswith('Z'):
                        time_str = time_str[:-1] + '+00:00'
                    
                    recording_time = datetime.fromisoformat(time_str)
                    if recording_time.tzinfo is not None:
                        recording_time = recording_time.astimezone(timezone.utc).replace(tzinfo=None)
                    
                    if recording_time < cutoff_date:
                        shutil.rmtree(meeting_path)
                        deleted_count += 1
                        logger.info(f"Deleted old recording: {meeting_path} ({time_key}: {recording_time.isoformat()})")
                    
                except Exception as e:
                    error_msg = f"Error processing {meeting_path}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        logger.info(f"Cleanup completed: deleted {deleted_count} recordings")
        
        return {
            "deleted_count": deleted_count,
            "errors": errors
        }

    
