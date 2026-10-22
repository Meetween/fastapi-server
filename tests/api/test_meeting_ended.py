"""Jigasi end-of-meeting signal: clears in-memory state, rejects later writes, is idempotent."""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from config import settings
from core import SessionManager
from models import TaskType

JIGASI_KEY = "jigasi-test-key"
MEETING_ID = "5506be1e-b870-4922-845d-d053b1d0b4db"
AUTH = {"Authorization": f"Bearer {JIGASI_KEY}"}


def _participant_update():
    return {"participants": [{"participant_id": "p1", "username": "Ada"}]}


def _translation_body():
    return {
        "type": "MT",
        "room_name": "room-1",
        "meeting_id": MEETING_ID,
        "text": "hello",
        "language": "en",
        "target_language": "de",
    }


@pytest.fixture(autouse=True)
def _isolate_ended_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "enable_auth", True)
    monkeypatch.setattr(settings, "jigasi_meetween_api_key", JIGASI_KEY)
    monkeypatch.setattr(SessionManager, "_ended_meetings", {})


@pytest.mark.integration
class TestMeetingEnded:
    def test_end_signal_clears_session_and_returns_success(self, test_client: TestClient):
        SessionManager.add_participants(
            meeting_id=MEETING_ID,
            room_name=None,
            participants=[{"participant_id": "p1", "username": "Ada"}],
        )
        assert MEETING_ID in SessionManager._sessions

        response = test_client.post(f"/meetings/{MEETING_ID}/ended", headers=AUTH)

        assert response.status_code == 200
        assert response.json() == {
            "status": "success",
            "meeting_id": MEETING_ID,
            "already_ended": False,
        }
        assert MEETING_ID not in SessionManager._sessions
        assert SessionManager.is_meeting_ended(MEETING_ID)

    def test_repeated_end_signal_is_idempotent(self, test_client: TestClient):
        test_client.post(f"/meetings/{MEETING_ID}/ended", headers=AUTH)

        response = test_client.post(f"/meetings/{MEETING_ID}/ended", headers=AUTH)

        assert response.status_code == 200
        assert response.json()["already_ended"] is True

    def test_end_signal_without_key_is_rejected(self, test_client: TestClient):
        response = test_client.post(f"/meetings/{MEETING_ID}/ended")

        assert response.status_code == 401
        assert not SessionManager.is_meeting_ended(MEETING_ID)

    def test_participant_update_after_end_is_rejected(self, test_client: TestClient):
        test_client.post(f"/meetings/{MEETING_ID}/ended", headers=AUTH)

        response = test_client.post(
            f"/meetings/{MEETING_ID}/participants",
            json=_participant_update(),
            headers=AUTH,
        )

        assert response.status_code == 409
        assert MEETING_ID not in SessionManager._sessions

    def test_translation_after_end_is_rejected(self, test_client: TestClient):
        test_client.post(f"/meetings/{MEETING_ID}/ended", headers=AUTH)

        response = test_client.post("/api/translate", json=_translation_body(), headers=AUTH)

        assert response.status_code == 409

    def test_websocket_after_end_is_closed_with_meeting_ended(self, test_client: TestClient):
        test_client.post(f"/meetings/{MEETING_ID}/ended", headers=AUTH)

        with pytest.raises(WebSocketDisconnect) as exc:
            with test_client.websocket_connect(f"/ws/{MEETING_ID}") as ws:
                ws.receive_text()

        assert exc.value.code == 1008
        assert MEETING_ID not in SessionManager._connections

    def test_open_websocket_is_closed_by_end_signal(self, test_client: TestClient):
        with test_client.websocket_connect(f"/ws/{MEETING_ID}") as ws:
            assert MEETING_ID in SessionManager._connections

            test_client.post(f"/meetings/{MEETING_ID}/ended", headers=AUTH)

            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()

        assert exc.value.code == 1008
        assert MEETING_ID not in SessionManager._connections

    def test_in_flight_transcript_after_end_does_not_recreate_session(
        self, test_client: TestClient
    ):
        test_client.post(f"/meetings/{MEETING_ID}/ended", headers=AUTH)

        SessionManager.update_transcript(
            meeting_id=MEETING_ID,
            room_name=None,
            transcript="late words",
            participant_id="p1",
            timestamp=1.0,
        )

        assert MEETING_ID not in SessionManager._sessions
        assert SessionManager.get_transcript(meeting_id=MEETING_ID, room_name=None) is None

    def test_chat_after_end_is_rejected_and_does_not_recreate_session(self, client: TestClient):
        end_response = client.post(f"/meetings/{MEETING_ID}/ended", headers=AUTH)
        assert end_response.status_code == 200

        response = client.post(
            "/api/chat",
            json={
                "type": TaskType.SPOKEN_QUESTION_ANSWERING,
                "meeting_id": MEETING_ID,
                "room_name": "room-1",
                "participant_id": "user-123",
                "language": "en-US",
                "prompt": "What were the main topics?",
            },
        )

        assert response.status_code == 409
        assert MEETING_ID not in SessionManager._sessions

    def test_agenda_after_end_is_rejected(self, client: TestClient):
        client.post(f"/meetings/{MEETING_ID}/ended", headers=AUTH)

        response = client.post(
            "/api/agenda-tracking",
            json={
                "type": TaskType.AGENDA_TRACKING,
                "meeting_id": MEETING_ID,
                "room_id": "room-1",
                "items": [{"index": 0, "text": "Budget", "status": "pending"}],
            },
        )

        assert response.status_code == 409
        assert MEETING_ID not in SessionManager._sessions

    def test_other_meetings_are_unaffected(self, test_client: TestClient):
        other = "00000000-0000-4000-8000-000000000001"
        test_client.post(f"/meetings/{MEETING_ID}/ended", headers=AUTH)

        response = test_client.post(
            f"/meetings/{other}/participants",
            json=_participant_update(),
            headers=AUTH,
        )

        assert response.status_code == 200
