"""
Tests for /api/chat endpoint (Q&A over text with optional history/context)
"""
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from core import SessionManager
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
            "type": "chat",
            "meeting_id": test_meeting_id,
            "room_name": test_room_name,
            "participant_id": "user-123",
            "language": "en-US",
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
            "type": "chat",
            "meeting_id": test_meeting_id,
            "room_name": test_room_name,
            "participant_id": "user-456",
            "language": "en-US",
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
        """Missing prompt should trigger bad request"""
        payload = {
            "type": "chat",
            "meeting_id": "meeting-1",
            "room_name": "room-1",
            "participant_id": "user-789",
            "language": "en-US",
            # prompt missing
        }

        response = client.post("/api/chat", json=payload)

        assert response.status_code == 400


def test_chat_uses_session_manager_context_and_history_only(
    client: TestClient,
    test_room_name: str,
    test_meeting_id: str,
):
    SessionManager.update_transcript(
        meeting_id=test_meeting_id,
        room_name=test_room_name,
        transcript="Server transcript says budget was approved.",
    )

    payload = {
        "type": "chat",
        "meeting_id": test_meeting_id,
        "room_name": test_room_name,
        "participant_id": "user-1",
        "language": "en-US",
        "prompt": "What happened with budget?",
        "context": "Request context says timeline remains open.",
        "history": [{"role": "user", "content": "Any decision yet?"}],
    }

    mock_engine = Mock()
    captured = {"prompt": ""}
    mock_engine.max_context_chars = 300_000
    mock_engine.max_history_chars = 600_000

    async def mock_generate_text_from_text(*args, **kwargs):
        captured["prompt"] = kwargs["prompt"]
        return "Budget approved, timeline open."

    mock_engine.generate_text_from_text = mock_generate_text_from_text

    with patch("api.routes.chat.EngineManager.get_lmm", return_value=mock_engine):
        response = client.post("/api/chat", json=payload)

    data = assert_successful_response(response)
    assert data["result"] == "Budget approved, timeline open."
    assert "<context>" in captured["prompt"]
    assert "Server transcript says budget was approved." in captured["prompt"]
    assert "Request context says timeline remains open." not in captured["prompt"]
    assert "<history>" in captured["prompt"]
    assert "User: Any decision yet?" not in captured["prompt"]
    assert "<question>" in captured["prompt"]
    assert "What happened with budget?" in captured["prompt"]
    SessionManager.clear_session(
        meeting_id=test_meeting_id,
        room_name=test_room_name,
        force=True,
    )

