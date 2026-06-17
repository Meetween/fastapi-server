"""
Tests for /api/agenda-tracking endpoint.
"""
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from core import SessionManager


def _payload(meeting_id: str, room_id: str):
    return {
        "type": "agendaTracking",
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
        called = {"text": False}

        async def mock_generate_text(*args, **kwargs):
            called["text"] = True
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
