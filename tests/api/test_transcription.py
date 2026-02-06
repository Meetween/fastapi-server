"""
Tests for transcription endpoints (/api/transcribe)
"""
import pytest
from fastapi.testclient import TestClient
from tests.utils.assertions import (
    assert_successful_response,
    assert_error_response,
    assert_transcription_response
)
from tests.utils.helpers import (
    create_audio_payload,
    create_text_payload,
    create_generic_payload
)


@pytest.mark.integration
@pytest.mark.regression
class TestTranscribeEndpoint:
    """Test suite for /api/transcribe endpoint"""

    def test_transcribe_from_recordings(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str
    ):
        """Test transcription endpoint that loads from recordings"""
        # Arrange
        payload = create_generic_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            task_type="transcription",
            participant_id="test-participant-1"
        )
        
        # Act
        response = client.post("/api/transcribe", json=payload)
        
        # Assert
        data = assert_successful_response(response)
        assert_transcription_response(data)

    def test_transcribe_with_audio(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        dummy_audio_bytes: bytes
    ):
        """Test transcription endpoint with audio input"""
        # Arrange
        payload = create_audio_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            audio_data=dummy_audio_bytes,
            task_type="transcription",
            participant_id="test-participant-2"
        )
        
        # Act
        response = client.post("/api/transcribe", json=payload)
        
        # Assert
        data = assert_successful_response(response)
        assert_transcription_response(data)

    def test_transcribe_with_text(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str
    ):
        """Test transcription endpoint with text input (not yet implemented)"""
        payload = create_text_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            text=sample_text,
            task_type="transcription",
            participant_id="test-participant-3"
        )
        
        response = client.post("/api/transcribe", json=payload)
        
        # Assert - text transcription is not yet implemented, should return 400
        assert_error_response(response, expected_status=400, expected_detail="Text transcription not yet implemented")

    def test_transcribe_missing_required_fields(self, client: TestClient):
        """Test transcription endpoint with missing required fields"""
        # Arrange
        payload = {"type": "transcription"}  # Missing required fields
        
        # Act
        response = client.post("/api/transcribe", json=payload)
        
        # Assert
        assert response.status_code == 422, "Should return validation error"
