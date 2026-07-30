"""
Tests for /api/agenda-tracking endpoint.
"""
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from core import SessionManager
from models import TaskType


def _payload(meeting_id: str, room_id: str):
    return {
        "type": TaskType.AGENDA_TRACKING,
        "meeting_id": meeting_id,
        "room_id": room_id,
        "requested_by": "moderator",
        "items": [
            {"index": 0, "text": "Budget discussion", "status": "pending"},
            {"index": 1, "text": "Hiring plan", "status": "pending"},
            {"index": 2, "text": "Marketing strategy", "status": "pending"},
        ],
    }


def test_agenda_tracking_without_transcript_returns_given_items(
    client: TestClient,
    test_meeting_id: str,
):
    room_id = "agenda-room-1"
    payload = _payload(meeting_id=test_meeting_id, room_id=room_id)

    response = client.post("/api/agenda-tracking", json=payload)

    assert response.status_code == 204
    assert response.content == b""


def test_agenda_tracking_with_transcript_infers_item_statuses(
    client: TestClient,
    test_meeting_id: str,
):
    room_id = "agenda-room-2"
    payload = _payload(meeting_id=test_meeting_id, room_id=room_id)
    try:
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=room_id,
            transcript=(
                "We finished discussing budget. "
                "For hiring plan, we decided to skip for now. "
                "Marketing strategy is not discussed yet."
            ),
        )

        mock_engine = Mock()

        async def mock_generate_text(*args, **kwargs):
            return (
                '{"items":['
                '{"index":0,"text":"Budget discussion","status":"completed"},'
                '{"index":1,"text":"Hiring plan","status":"skipped"},'
                '{"index":2,"text":"Marketing strategy","status":"pending"}'
                "]}"
            )

        mock_engine._generate_text = mock_generate_text

        with patch("api.routes.agenda.EngineManager.get_lmm", return_value=mock_engine):
            response = client.post("/api/agenda-tracking", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert [item["status"] for item in data["items"]] == [
            "completed",
            "skipped",
            "pending",
        ]
    finally:
        SessionManager.clear_session(
            meeting_id=test_meeting_id, room_name=room_id, force=True
        )


def test_agenda_tracking_unparseable_output_preserves_existing_statuses(
    client: TestClient,
    test_meeting_id: str,
):
    room_id = "agenda-room-3"
    payload = _payload(meeting_id=test_meeting_id, room_id=room_id)
    try:
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=room_id,
            transcript="Budget done. Hiring still under discussion.",
        )

        mock_engine = Mock()

        async def mock_generate_text(*args, **kwargs):
            return "<answer><item>invalid-output</item></answer>"

        mock_engine._generate_text = mock_generate_text

        with patch("api.routes.agenda.EngineManager.get_lmm", return_value=mock_engine):
            response = client.post("/api/agenda-tracking", json=payload)

        assert response.status_code == 204
        assert response.content == b""
    finally:
        SessionManager.clear_session(
            meeting_id=test_meeting_id, room_name=room_id, force=True
        )


def test_agenda_tracking_uses_text_generation_method(
    client: TestClient,
    test_meeting_id: str,
):
    room_id = "agenda-room-4"
    payload = _payload(meeting_id=test_meeting_id, room_id=room_id)
    try:
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=room_id,
            transcript="Budget complete. Hiring in progress.",
        )

        mock_engine = Mock()
        called = {"text": False, "task": None}

        async def mock_generate_text(*args, **kwargs):
            called["text"] = True
            called["task"] = kwargs.get("task")
            return (
                '{"items":['
                '{"index":0,"text":"Budget discussion","status":"completed"},'
                '{"index":1,"text":"Hiring plan","status":"skipped"},'
                '{"index":2,"text":"Marketing strategy","status":"pending"}'
                "]}"
            )

        mock_engine._generate_text = mock_generate_text

        with patch("api.routes.agenda.EngineManager.get_lmm", return_value=mock_engine):
            response = client.post("/api/agenda-tracking", json=payload)

        assert response.status_code == 200
        assert called["text"] is True
        assert called["task"] == TaskType.AGENDA_TRACKING

        data = response.json()
        assert [item["status"] for item in data["items"]] == [
            "completed",
            "skipped",
            "pending",
        ]
    finally:
        SessionManager.clear_session(
            meeting_id=test_meeting_id, room_name=room_id, force=True
        )


def test_agenda_tracking_parses_json_wrapped_in_markdown_fences(
    client: TestClient,
    test_meeting_id: str,
):
    room_id = "agenda-room-markdown"
    payload = _payload(meeting_id=test_meeting_id, room_id=room_id)
    try:
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=room_id,
            transcript="Budget done. Hiring skipped. Marketing not discussed.",
        )

        mock_engine = Mock()

        async def mock_generate_text(*args, **kwargs):
            return """```json
{
  "items": [
    {"index": 0, "text": "Budget discussion", "status": "completed"},
    {"index": 1, "text": "Hiring plan", "status": "skipped"},
    {"index": 2, "text": "Marketing strategy", "status": "pending"}
  ]
}
```"""

        mock_engine._generate_text = mock_generate_text

        with patch("api.routes.agenda.EngineManager.get_lmm", return_value=mock_engine):
            response = client.post("/api/agenda-tracking", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert [item["status"] for item in data["items"]] == [
            "completed",
            "skipped",
            "pending",
        ]
    finally:
        SessionManager.clear_session(
            meeting_id=test_meeting_id, room_name=room_id, force=True
        )


def test_agenda_tracking_accepts_punctuation_only_item_text_differences(
    client: TestClient,
    test_meeting_id: str,
):
    room_id = "agenda-room-5"
    payload = _payload(meeting_id=test_meeting_id, room_id=room_id)
    try:
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=room_id,
            transcript="We wrapped the budget discussion and postponed the hiring plan.",
        )

        mock_engine = Mock()

        async def mock_generate_text(*args, **kwargs):
            return (
                '{"items":['
                '{"index":0,"text":"Budget discussion!","status":"completed"},'
                '{"index":1,"text":" hiring PLAN ","status":"skipped"},'
                '{"index":2,"text":"Marketing-strategy","status":"pending"}'
                "]}"
            )

        mock_engine._generate_text = mock_generate_text

        with patch("api.routes.agenda.EngineManager.get_lmm", return_value=mock_engine):
            response = client.post("/api/agenda-tracking", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert [item["status"] for item in data["items"]] == [
            "completed",
            "skipped",
            "pending",
        ]
    finally:
        SessionManager.clear_session(
            meeting_id=test_meeting_id, room_name=room_id, force=True
        )


def test_agenda_tracking_invalid_task_type_returns_422(
    client: TestClient,
    test_meeting_id: str,
):
    payload = _payload(meeting_id=test_meeting_id, room_id="agenda-room-invalid")
    payload["type"] = TaskType.SPOKEN_QUESTION_ANSWERING

    response = client.post("/api/agenda-tracking", json=payload)

    assert response.status_code == 422


def test_agenda_tracking_preserves_completed_status_against_llm_uncomplete(
    client: TestClient,
    test_meeting_id: str,
):
    """Completed items must not be reverted by Butler/LLM sync."""
    room_id = "agenda-room-completed-lock"
    payload = _payload(meeting_id=test_meeting_id, room_id=room_id)
    payload["items"][0]["status"] = "completed"
    try:
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=room_id,
            transcript="Budget was somehow not finished. Hiring is done.",
        )

        mock_engine = Mock()
        captured_prompt = {"value": None}

        async def mock_generate_text(*args, **kwargs):
            captured_prompt["value"] = kwargs.get("prompt") or (
                args[0] if args else None
            )
            # LLM incorrectly tries to un-complete index 0 and only returns
            # processable items (1 and 2) — but even if it returns index 0,
            # completed must stay locked.
            return (
                '{"items":['
                '{"index":1,"text":"Hiring plan","status":"completed"},'
                '{"index":2,"text":"Marketing strategy","status":"pending"}'
                "]}"
            )

        mock_engine._generate_text = mock_generate_text

        with patch("api.routes.agenda.EngineManager.get_lmm", return_value=mock_engine):
            response = client.post("/api/agenda-tracking", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert [item["status"] for item in data["items"]] == [
            "completed",
            "completed",
            "pending",
        ]
        # Completed item must not be included in the LLM prompt.
        assert captured_prompt["value"] is not None
        assert "Budget discussion" not in captured_prompt["value"]
        assert "Hiring plan" in captured_prompt["value"]
    finally:
        SessionManager.clear_session(
            meeting_id=test_meeting_id, room_name=room_id, force=True
        )


def test_agenda_tracking_all_completed_skips_inference(
    client: TestClient,
    test_meeting_id: str,
):
    room_id = "agenda-room-all-completed"
    payload = _payload(meeting_id=test_meeting_id, room_id=room_id)
    for item in payload["items"]:
        item["status"] = "completed"
    try:
        SessionManager.update_transcript(
            meeting_id=test_meeting_id,
            room_name=room_id,
            transcript="Everything is done.",
        )

        mock_engine = Mock()
        called = {"text": False}

        async def mock_generate_text(*args, **kwargs):
            called["text"] = True
            return '{"items":[]}'

        mock_engine._generate_text = mock_generate_text

        with patch("api.routes.agenda.EngineManager.get_lmm", return_value=mock_engine):
            response = client.post("/api/agenda-tracking", json=payload)

        assert response.status_code == 204
        assert called["text"] is False
    finally:
        SessionManager.clear_session(
            meeting_id=test_meeting_id, room_name=room_id, force=True
        )
