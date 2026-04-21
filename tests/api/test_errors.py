"""
Tests for error handling and edge cases
"""
import base64
import pytest
from fastapi.testclient import TestClient
from tests.utils.assertions import assert_error_response
from tests.utils.helpers import create_audio_payload, create_text_payload
from models import TASK_TYPE_TEXT_SUMMARY


@pytest.mark.integration
class TestErrorHandling:
    """Test suite for error handling and validation"""

    def test_invalid_endpoint(self, client: TestClient):
        """Test that invalid endpoints return 404"""
        response = client.get("/api/nonexistent")
        
        assert response.status_code == 404

    def test_invalid_json(self, client: TestClient):
        """Test that invalid JSON returns 422"""
        response = client.post(
            "/api/summarize",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_missing_required_field_room_name(
        self,
        client: TestClient,
        test_meeting_id: str
    ):
        """Test that missing room_name returns validation error"""
        payload = {
            "type": TASK_TYPE_TEXT_SUMMARY,
            "meeting_id": test_meeting_id,
            "language": "en-US"
        }
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_missing_required_field_meeting_id(
        self,
        client: TestClient,
        test_room_name: str
    ):
        """Test that missing meeting_id returns validation error"""
        payload = {
            "type": TASK_TYPE_TEXT_SUMMARY,
            "room_name": test_room_name,
            "language": "en-US"
        }
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_invalid_audio_base64(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str
    ):
        """Test that invalid base64 audio returns validation error"""
        payload = {
            "type": TASK_TYPE_TEXT_SUMMARY,
            "room_name": test_room_name,
            "meeting_id": test_meeting_id,
            "language": "en-US",
            "audio": "not-valid-base64!!!"
        }
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code in [400, 422]

    def test_empty_audio_data(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str
    ):
        """Test that empty audio data returns validation error"""
        payload = create_audio_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            audio_data=b''
        )
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code in [400, 422]

    def test_invalid_task_type(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str
    ):
        """Test that invalid task type returns validation error"""
        payload = {
            "type": "invalid_task_type",
            "room_name": test_room_name,
            "meeting_id": test_meeting_id,
            "language": "en-US"
        }
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code == 422

    def test_empty_text(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str
    ):
        """Test that empty text input may be handled"""
        payload = create_text_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            text=""
        )
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code in [200, 400, 422]

    @pytest.mark.parametrize("invalid_language", ["", "invalid", "123", "en"])
    def test_invalid_language_code(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str,
        invalid_language: str
    ):
        """Test that invalid language codes are handled"""
        payload = create_text_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            text=sample_text,
            language=invalid_language
        )
        
        response = client.post("/api/summarize", json=payload)
        
        assert response.status_code in [200, 400, 422]
