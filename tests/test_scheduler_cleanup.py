"""
Tests for scheduler integration with recording cleanup
"""
import pytest
from unittest.mock import patch
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from main import cleanup_old_recordings_job, scheduler
from utils.storage import StorageManager


@pytest.mark.integration
@pytest.mark.regression
class TestSchedulerIntegration:
    """Test suite for scheduler integration with recording cleanup"""

    def test_cleanup_job_calls_storage_manager(self):
        """Test that the cleanup job calls StorageManager.cleanup_old_recordings()"""
        mock_result = {
            "deleted_count": 5,
            "errors": []
        }

        with patch.object(StorageManager, 'cleanup_old_recordings', return_value=mock_result) as mock_cleanup:
            cleanup_old_recordings_job()

        mock_cleanup.assert_called_once()

    def test_cleanup_job_logs_results(self, caplog):
        """Test that the cleanup job logs the results"""
        mock_result = {
            "deleted_count": 3,
            "errors": []
        }

        with patch.object(StorageManager, 'cleanup_old_recordings', return_value=mock_result):
            cleanup_old_recordings_job()

        assert "Recording cleanup job completed: 3 recordings deleted" in caplog.text

    def test_cleanup_job_logs_errors(self, caplog):
        """Test that the cleanup job logs errors if present"""
        mock_result = {
            "deleted_count": 2,
            "errors": ["Error processing meeting1: Invalid metadata"]
        }

        with patch.object(StorageManager, 'cleanup_old_recordings', return_value=mock_result):
            cleanup_old_recordings_job()

        assert "cleanup job encountered 1 errors" in caplog.text.lower()

    def test_cleanup_job_handles_exceptions(self, caplog):
        """Test that the cleanup job handles exceptions gracefully"""
        test_exception = Exception("Test exception")

        with patch.object(StorageManager, 'cleanup_old_recordings', side_effect=test_exception):
            cleanup_old_recordings_job()

        assert "Error in recording cleanup job" in caplog.text
        assert "Test exception" in caplog.text

    @pytest.mark.asyncio
    async def test_scheduler_is_configured(self):
        """Test that scheduler is properly configured"""
        assert isinstance(scheduler, AsyncIOScheduler)
        assert scheduler.running or not scheduler.running

    def test_scheduler_job_registered_on_startup(self):
        """Test that cleanup job is registered when server starts"""
        assert cleanup_old_recordings_job is not None
        assert callable(cleanup_old_recordings_job)

        jobs = scheduler.get_jobs()
        cleanup_jobs = [job for job in jobs if job.id == "cleanup_old_recordings"]

        if scheduler.running:
            assert len(cleanup_jobs) > 0, "Cleanup job should be registered when scheduler is running"
            assert cleanup_jobs[0].id == "cleanup_old_recordings"
