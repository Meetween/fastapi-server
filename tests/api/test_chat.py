"""
Tests for /api/chat endpoint (Q&A over text with optional history/context)
"""
import pytest
from fastapi.testclient import TestClient

from tests.utils.assertions import assert_successful_response


@pytest.mark.integration
class TestChatEndpoint:
    """Test suite for /api/chat endpoint"""

    def test_chat_basic_request(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
    ):
        """Chat with minimal required fields"""
        payload = {
            "user_id": "user-123",
            "meeting_id": test_meeting_id,
            "room_name": test_room_name,
            "input_type": "text",
            "prompt": "What were the main topics?",
        }

        response = client.post("/api/chat", json=payload)

        data = assert_successful_response(response)
        assert "result" in data
        assert isinstance(data["result"], str)
        assert data["meeting_id"] == test_meeting_id
        assert data["room_name"] == test_room_name

    def test_chat_with_history_and_context(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
    ):
        """Chat with full history and context payload"""
        payload = {
            "user_id": "user-456",
            "meeting_id": test_meeting_id,
            "room_name": test_room_name,
            "input_type": "text",
            "prompt": "Any open action items?",
            "context": "Transcript: We discussed roadmap, budget, and timelines.",
            "history": [
                {"role": "user", "content": "Summarize the meeting."},
                {"role": "assistant", "content": "The meeting covered roadmap and budget."},
            ],
        }

        response = client.post("/api/chat", json=payload)

        data = assert_successful_response(response)
        assert "result" in data
        assert isinstance(data["result"], str)

    def test_chat_missing_required_fields(self, client: TestClient):
        """Missing prompt should trigger validation error"""
        payload = {
            "user_id": "user-789",
            "meeting_id": "meeting-1",
            "room_name": "room-1",
            "input_type": "text",
            # prompt missing
        }

        response = client.post("/api/chat", json=payload)

        assert response.status_code == 422

