"""
Tests for catchup endpoint (/api/catchup)
"""
import pytest
from fastapi.testclient import TestClient
from core import SessionManager
from tests.utils.assertions import (
    assert_successful_response,
    assert_task_response
)
from tests.utils.helpers import create_generic_payload


@pytest.mark.integration
@pytest.mark.regression
class TestCatchupEndpoint:
    """Test suite for /api/catchup endpoint"""

    def test_catchup_default(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str,
    ):
        """Test catchup endpoint with default parameters"""
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=test_room_name,
            transcript=sample_text,
        )
        payload = create_generic_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            task_type="catch_me_up",
            participant_id="test-participant-1"
        )
        
        response = client.post("/api/catchup", json=payload)
        
        data = assert_successful_response(response)
        assert_task_response(data, expected_type="catch_me_up")

    @pytest.mark.parametrize("max_duration", [30.0, 60.0, 120.0, 300.0])
    def test_catchup_with_max_duration(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str,
        max_duration: float
    ):
        """Test catchup endpoint with different max_duration values"""
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=test_room_name,
            transcript=sample_text,
        )
        payload = create_generic_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            task_type="catch_me_up",
            participant_id="test-participant-2",
            max_duration=max_duration
        )

        response = client.post("/api/catchup", json=payload)
        
        data = assert_successful_response(response)
        assert_task_response(data, expected_type="catch_me_up")

    def test_catchup_missing_required_fields(self, client: TestClient):
        """Test catchup endpoint with missing required fields"""
        payload = {"type": "catch_me_up"}
        
        response = client.post("/api/catchup", json=payload)
        
        assert response.status_code == 422, "Should return validation error"
