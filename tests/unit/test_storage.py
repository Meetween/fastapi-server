"""
Tests for StorageManager.cleanup_old_recordings() functionality
"""
import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from utils.storage import StorageManager
from config import settings


@pytest.mark.unit
class TestStorageCleanup:
    """Test suite for StorageManager.cleanup_old_recordings()"""

    def test_cleanup_deletes_old_recordings_with_end_time(self, temp_recordings_dir, create_test_recording):
        """Test that recordings older than retention period are deleted"""
        retention_days = 14
        
        old_time = datetime.utcnow() - timedelta(days=15)
        old_recording = create_test_recording(
            room_name="room1",
            meeting_id="meeting1",
            start_time=old_time,
            end_time=old_time + timedelta(hours=1)
        )
        
        recent_time = datetime.utcnow() - timedelta(days=5)
        recent_recording = create_test_recording(
            room_name="room1",
            meeting_id="meeting2",
            start_time=recent_time,
            end_time=recent_time + timedelta(hours=1)
        )
        
        with patch.object(settings, 'recording_retention_days', retention_days):
            result = StorageManager.cleanup_old_recordings()
        
        assert result["deleted_count"] == 1
        assert not old_recording.exists(), "Old recording should be deleted"
        assert recent_recording.exists(), "Recent recording should be kept"
        assert "meeting1" not in [d.name for d in (temp_recordings_dir / "room1").iterdir() if d.is_dir()]
        assert "meeting2" in [d.name for d in (temp_recordings_dir / "room1").iterdir() if d.is_dir()]

    def test_cleanup_uses_start_time_when_no_end_time(self, temp_recordings_dir, create_test_recording):
        """Test that start_time is used when end_time is not available"""
        retention_days = 14
        old_time = datetime.utcnow() - timedelta(days=20)
        
        recording = create_test_recording(
            room_name="room2",
            meeting_id="meeting3",
            start_time=old_time,
            end_time=None
        )
        
        with patch.object(settings, 'recording_retention_days', retention_days):
            result = StorageManager.cleanup_old_recordings()
        
        assert result["deleted_count"] == 1
        assert not recording.exists(), "Recording should be deleted based on start_time"

    def test_cleanup_keeps_recent_recordings(self, temp_recordings_dir, create_test_recording):
        """Test that recent recordings are not deleted"""
        retention_days = 14
        recent_time = datetime.utcnow() - timedelta(days=1)
        
        recent_recording = create_test_recording(
            room_name="room3",
            meeting_id="meeting4",
            start_time=recent_time,
            end_time=recent_time + timedelta(hours=1)
        )
        
        with patch.object(settings, 'recording_retention_days', retention_days):
            result = StorageManager.cleanup_old_recordings()
        
        assert result["deleted_count"] == 0
        assert recent_recording.exists(), "Recent recording should be kept"

    def test_cleanup_handles_timezone_aware_datetimes(self, temp_recordings_dir):
        """Test that timezone-aware datetimes are handled correctly"""
        retention_days = 14
        old_time_utc = datetime.now(timezone.utc) - timedelta(days=20)
        
        meeting_path = temp_recordings_dir / "room4" / "meeting5"
        meeting_path.mkdir(parents=True)
        
        metadata = {
            "meeting_id": "meeting5",
            "room_name": "room4",
            "start_time": old_time_utc.isoformat(),
            "end_time": (old_time_utc + timedelta(hours=1)).isoformat(),
            "chunk_count": 1,
            "total_duration": 5.0
        }
        
        metadata_path = meeting_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        chunk_path = meeting_path / "chunk_0001.wav"
        chunk_path.write_bytes(b"dummy audio data" * 100)
        
        with patch.object(settings, 'recording_retention_days', retention_days):
            result = StorageManager.cleanup_old_recordings()
        
        assert result["deleted_count"] == 1
        assert not meeting_path.exists(), "Recording with timezone-aware datetime should be deleted"

    def test_cleanup_handles_z_suffix_in_iso_string(self, temp_recordings_dir):
        """Test that ISO strings ending with 'Z' are parsed correctly"""
        retention_days = 14
        old_time = datetime.utcnow() - timedelta(days=20)
        
        meeting_path = temp_recordings_dir / "room5" / "meeting6"
        meeting_path.mkdir(parents=True)
        
        metadata = {
            "meeting_id": "meeting6",
            "room_name": "room5",
            "start_time": old_time.isoformat() + "Z",
            "end_time": (old_time + timedelta(hours=1)).isoformat() + "Z",
            "chunk_count": 1,
            "total_duration": 5.0
        }
        
        metadata_path = meeting_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        chunk_path = meeting_path / "chunk_0001.wav"
        chunk_path.write_bytes(b"dummy audio data" * 100)
        
        with patch.object(settings, 'recording_retention_days', retention_days):
            result = StorageManager.cleanup_old_recordings()
        
        assert result["deleted_count"] == 1
        assert not meeting_path.exists(), "Recording with 'Z' suffix should be deleted"

    def test_cleanup_handles_recordings_without_metadata(self, temp_recordings_dir):
        """Test that recordings without metadata use directory modification time"""
        retention_days = 14
        
        meeting_path = temp_recordings_dir / "room6" / "meeting7"
        meeting_path.mkdir(parents=True)
        
        chunk_path = meeting_path / "chunk_0001.wav"
        chunk_path.write_bytes(b"dummy audio data" * 100)
        
        old_timestamp = (datetime.utcnow() - timedelta(days=20)).timestamp()
        import os
        os.utime(meeting_path, (old_timestamp, old_timestamp))
        
        with patch.object(settings, 'recording_retention_days', retention_days):
            result = StorageManager.cleanup_old_recordings()
        
        assert result["deleted_count"] == 1
        assert not meeting_path.exists(), "Recording without metadata should be deleted based on directory mtime"

    def test_cleanup_skips_recordings_without_timestamps(self, temp_recordings_dir):
        """Test that recordings without any timestamps are skipped"""
        meeting_path = temp_recordings_dir / "room7" / "meeting8"
        meeting_path.mkdir(parents=True)
        
        metadata = {
            "meeting_id": "meeting8",
            "room_name": "room7",
            "chunk_count": 1,
            "total_duration": 5.0
        }
        
        metadata_path = meeting_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        chunk_path = meeting_path / "chunk_0001.wav"
        chunk_path.write_bytes(b"dummy audio data" * 100)
        
        result = StorageManager.cleanup_old_recordings()
        
        assert result["deleted_count"] == 0
        assert meeting_path.exists(), "Recording without timestamps should be skipped"
        assert len(result["errors"]) == 0, "Should not have errors for skipped recordings"

    def test_cleanup_handles_multiple_rooms(self, temp_recordings_dir, create_test_recording):
        """Test cleanup across multiple rooms"""
        retention_days = 14
        old_time = datetime.utcnow() - timedelta(days=20)
        recent_time = datetime.utcnow() - timedelta(days=5)
        
        create_test_recording("room_a", "meeting_a1", old_time, old_time + timedelta(hours=1))
        create_test_recording("room_b", "meeting_b1", old_time, old_time + timedelta(hours=1))
        create_test_recording("room_a", "meeting_a2", recent_time, recent_time + timedelta(hours=1))
        
        with patch.object(settings, 'recording_retention_days', retention_days):
            result = StorageManager.cleanup_old_recordings()
        
        assert result["deleted_count"] == 2
        assert not (temp_recordings_dir / "room_a" / "meeting_a1").exists()
        assert not (temp_recordings_dir / "room_b" / "meeting_b1").exists()
        assert (temp_recordings_dir / "room_a" / "meeting_a2").exists()

    def test_cleanup_handles_errors_gracefully(self, temp_recordings_dir, create_test_recording):
        """Test that errors during cleanup are caught and reported"""
        retention_days = 14
        old_time = datetime.utcnow() - timedelta(days=20)
        
        create_test_recording("room9", "meeting10", old_time, old_time + timedelta(hours=1))
        
        bad_meeting_path = temp_recordings_dir / "room9" / "meeting11"
        bad_meeting_path.mkdir(parents=True)
        bad_metadata_path = bad_meeting_path / "metadata.json"
        bad_metadata_path.write_text("invalid json{")
        
        with patch.object(settings, 'recording_retention_days', retention_days):
            result = StorageManager.cleanup_old_recordings()
        
        assert result["deleted_count"] == 1, "Valid recording should be deleted"
        assert len(result["errors"]) > 0, "Should have errors for corrupted metadata"
        assert "meeting11" in str(result["errors"][0]), "Error should mention the problematic meeting"

    def test_cleanup_handles_nonexistent_base_path(self):
        """Test cleanup when base path doesn't exist"""
        original_path = settings.recordings_base_path
        settings.recordings_base_path = Path("/nonexistent/path/that/does/not/exist")
        
        try:
            result = StorageManager.cleanup_old_recordings()
            
            assert result["deleted_count"] == 0
            assert len(result["errors"]) == 0
        finally:
            settings.recordings_base_path = original_path

    def test_cleanup_at_cutoff_boundary(self, temp_recordings_dir, create_test_recording):
        """Test cleanup behavior at the exact cutoff boundary"""
        retention_days = 14
        now = datetime.utcnow()
        cutoff_date = now - timedelta(days=retention_days)
        
        recording_at_cutoff = create_test_recording(
            room_name="room10",
            meeting_id="meeting12",
            start_time=cutoff_date - timedelta(hours=1),
            end_time=cutoff_date + timedelta(minutes=1)
        )
        
        recording_before_cutoff = create_test_recording(
            room_name="room10",
            meeting_id="meeting13",
            start_time=cutoff_date - timedelta(days=1),
            end_time=cutoff_date - timedelta(days=1)
        )
        
        with patch.object(settings, 'recording_retention_days', retention_days):
            result = StorageManager.cleanup_old_recordings()
        
        assert result["deleted_count"] == 1, f"Expected 1 deletion, got {result['deleted_count']}"
        assert recording_at_cutoff.exists(), "Recording at cutoff should be kept"
        assert not recording_before_cutoff.exists(), "Recording before cutoff should be deleted"
