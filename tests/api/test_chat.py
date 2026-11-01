"""
Tests for /api/chat endpoint (Q&A over text with optional history/context)
"""
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from api.routes.chat import _build_chat_prompt
from core import SessionManager
from models import HistoryEntry, TaskType
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
            "type": TaskType.SPOKEN_QUESTION_ANSWERING,
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
            "type": TaskType.SPOKEN_QUESTION_ANSWERING,
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
            "type": TaskType.SPOKEN_QUESTION_ANSWERING,
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
        "type": TaskType.SPOKEN_QUESTION_ANSWERING,
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

    async def mock_generate_text(*args, **kwargs):
        captured["prompt"] = kwargs["prompt"]
        return "Budget approved, timeline open."

    mock_engine._generate_text = mock_generate_text

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



def _chat_payload(meeting_id: str, room_name: str, participant_id: str = "user-empty") -> dict:
    return {
        "type": TaskType.SPOKEN_QUESTION_ANSWERING,
        "meeting_id": meeting_id,
        "room_name": room_name,
        "participant_id": participant_id,
        "language": "en-US",
        "prompt": "What was decided on the budget?",
    }


def test_chat_without_transcript_skips_model_and_returns_reason(client: TestClient):
    meeting_id, room_name = "chat-empty-meeting", "chat-empty-room"
    mock_engine = Mock()
    mock_engine.max_context_chars = 300_000
    mock_engine.max_history_chars = 600_000
    calls = {"n": 0}

    async def mock_generate_text(*args, **kwargs):
        calls["n"] += 1
        return "Based on the meeting transcript, the deadline is June 15."

    mock_engine._generate_text = mock_generate_text

    with patch("api.routes.chat.EngineManager.get_lmm", return_value=mock_engine):
        response = client.post("/api/chat", json=_chat_payload(meeting_id, room_name))

    data = assert_successful_response(response)
    assert data["result"] == ""
    assert data["error"] == "no_transcript"
    assert data["meeting_id"] == meeting_id
    assert data["room_name"] == room_name
    assert data["participant_id"] == "user-empty"
    assert calls["n"] == 0
    assert SessionManager.get_participant_history(meeting_id, room_name, "user-empty") == []


def test_chat_with_transcript_has_no_error_field_set(
    client: TestClient,
    test_room_name: str,
    test_meeting_id: str,
):
    SessionManager.update_transcript(
        meeting_id=test_meeting_id,
        room_name=test_room_name,
        transcript="Server transcript says budget was approved.",
    )
    try:
        response = client.post("/api/chat", json=_chat_payload(test_meeting_id, test_room_name))
    finally:
        SessionManager.clear_session(
            meeting_id=test_meeting_id,
            room_name=test_room_name,
            force=True,
        )

    data = assert_successful_response(response)
    assert data["result"] == "Mock chat response text for testing"
    assert data["error"] is None


def _prompt(**overrides) -> str:
    args = dict(
        query="How are you?",
        context="Alice: the deadline is June 15.",
        history=None,
        max_context_chars=300_000,
        max_history_chars=600_000,
    )
    args.update(overrides)
    return _build_chat_prompt(**args)


def test_chat_prompt_answers_directly_without_a_transcript_preamble():
    prompt = _prompt()
    assert "Never open with a preamble" in prompt
    assert "small talk" in prompt
    assert "one plain" in prompt
    assert 'you may say things like "Based on the meeting' not in prompt


def test_chat_prompt_keeps_markdown_and_length_for_web_participants():
    for participant_id in (None, "user-123", "jitsi-abc"):
        prompt = _prompt(participant_id=participant_id)
        assert "Use bullet points" in prompt
        assert "read aloud" not in prompt
        assert "two short sentences" not in prompt


def test_chat_prompt_adds_spoken_style_for_the_virtual_participant():
    prompt = _prompt(participant_id="vp:bot1-3")
    assert "will be read aloud" in prompt
    assert "at most two short sentences" in prompt
    assert "no bullet points, headings, markdown or emoji" in prompt


def test_chat_prompt_grounds_the_virtual_participant_in_the_transcript():
    prompt = _prompt(participant_id="vp:bot1-3")
    assert "Answer only from lines of the meeting transcript" in prompt
    assert "reply exactly: That was not mentioned in the meeting." in prompt
    assert "Never guess" in prompt
    assert "Question: Who drives the van? Answer: That was not mentioned in the meeting." in prompt


def test_chat_prompt_leaves_the_web_prompt_free_of_the_grounding_example():
    prompt = _prompt(participant_id="user-123")
    assert "Answer only from lines" not in prompt
    assert "Who drives the van?" not in prompt


def test_chat_endpoint_passes_participant_id_to_the_prompt(
    client: TestClient,
    test_room_name: str,
    test_meeting_id: str,
):
    SessionManager.update_transcript(
        meeting_id=test_meeting_id,
        room_name=test_room_name,
        transcript="Alice: the deadline is June 15.",
    )
    seen: list[str] = []

    async def mock_generate_text(*args, **kwargs):
        seen.append(kwargs["prompt"])
        return "June 15."

    mock_engine = Mock()
    mock_engine.max_context_chars = 300_000
    mock_engine.max_history_chars = 600_000
    mock_engine._generate_text = mock_generate_text
    try:
        with patch("api.routes.chat.EngineManager.get_lmm", return_value=mock_engine):
            for pid in ("vp:bot1-1", "user-web"):
                payload = _chat_payload(test_meeting_id, test_room_name)
                payload["participant_id"] = pid
                assert_successful_response(client.post("/api/chat", json=payload))
    finally:
        SessionManager.clear_session(
            meeting_id=test_meeting_id,
            room_name=test_room_name,
            force=True,
        )
    assert "will be read aloud" in seen[0]
    assert "will be read aloud" not in seen[1]


def test_chat_prompt_carries_transcript_history_and_question():
    history = [
        HistoryEntry(role="user", content="Hello"),
        HistoryEntry(role="assistant", content="Hi there"),
    ]
    prompt = _prompt(history=history)
    assert "<context>\nAlice: the deadline is June 15.\n</context>" in prompt
    assert "User: Hello\nAssistant: Hi there" in prompt
    assert "<question>\nHow are you?\n</question>" in prompt
