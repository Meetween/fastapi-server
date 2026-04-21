"""
Tests for summary endpoints (/api/summarize, /api/summarize-download)
"""
import pytest
from fastapi.testclient import TestClient
from tests.utils.assertions import (
    assert_successful_response,
    assert_task_response,
    assert_file_download_response
)
from tests.utils.helpers import (
    create_audio_payload,
    create_text_payload,
    create_generic_payload
)
from models import TASK_TYPE_SPEECH_SUMMARY, TASK_TYPE_TEXT_SUMMARY
from core import SessionManager


@pytest.mark.integration
@pytest.mark.regression
class TestSummarizeEndpoint:
    """Test suite for /api/summarize endpoint"""

    def test_summarize_with_audio(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        dummy_audio_bytes: bytes
    ):
        """Test summary endpoint with base64-encoded audio input"""
        payload = create_audio_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            audio_data=dummy_audio_bytes,
            participant_id="test-participant-1"
        )
        
        response = client.post("/api/summarize", json=payload)
        
        data = assert_successful_response(response)
        assert_task_response(data, expected_type=TASK_TYPE_SPEECH_SUMMARY)
        assert "result" in data or "summary" in data

    def test_summarize_with_text(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str
    ):
        """Test summary endpoint with plain text input"""
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=test_room_name,
            transcript=sample_text,
        )

        payload = create_text_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            text=sample_text,
            participant_id="test-participant-2"
        )
        
        response = client.post("/api/summarize", json=payload)
        
        data = assert_successful_response(response)
        assert_task_response(data, expected_type=TASK_TYPE_TEXT_SUMMARY)

    def test_summarize_from_recordings(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str,
    ):
        """Test summary endpoint that loads from recordings"""
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=test_room_name,
            transcript=sample_text,
        )

        payload = create_generic_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            participant_id="test-participant-3"
        )
        
        response = client.post("/api/summarize", json=payload)
        
        data = assert_successful_response(response)
        assert_task_response(data, expected_type=TASK_TYPE_TEXT_SUMMARY)

    @pytest.mark.parametrize("language", ["en-US", "en-GB", "de-DE", "fr-FR"])
    def test_summarize_with_different_languages(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str,
        language: str
    ):
        """Test summary endpoint with different language codes"""
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=test_room_name,
            transcript=sample_text,
        )

        payload = create_text_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            text=sample_text,
            language=language,
            participant_id="test-participant-lang"
        )
        
        response = client.post("/api/summarize", json=payload)
        
        data = assert_successful_response(response)
        assert_task_response(data, expected_type=TASK_TYPE_TEXT_SUMMARY)
        if "language" in data:
            assert data["language"] == language

    def test_summarize_missing_required_fields(
        self,
        client: TestClient
    ):
        """Test summary endpoint with missing required fields"""
        payload = {"type": TASK_TYPE_TEXT_SUMMARY}
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code == 422, "Should return validation error"


@pytest.mark.integration
@pytest.mark.regression
class TestSummarizeDownloadEndpoint:
    """Test suite for /api/summarize-download endpoint"""

    def test_summarize_download(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str,
    ):
        """Test summary download endpoint returns downloadable file"""
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=test_room_name,
            transcript=sample_text,
        )

        payload = create_generic_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            participant_id="test-participant-4"
        )
        
        response = client.post("/api/summarize-download", json=payload)
        
        file_content = assert_file_download_response(response)
        assert len(file_content) > 0

    def test_summarize_download_with_audio(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        dummy_audio_bytes: bytes
    ):
        """Test summary download with audio input"""
        payload = create_audio_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            audio_data=dummy_audio_bytes,
            participant_id="test-participant-5"
        )
        
        response = client.post("/api/summarize-download", json=payload)
        
        assert_file_download_response(response)

