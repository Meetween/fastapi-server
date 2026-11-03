"""The virtual participant worker authenticates /api/chat with its own service key."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from config import settings
from core import SessionManager
from models import TaskType

VP_KEY = "vp-test-key"
JIGASI_KEY = "jigasi-test-key"
MEETING_ID = "5506be1e-b870-4922-845d-d053b1d0b4db"
ROOM = "room-1"
BOT_PARTICIPANT = "vp:0123456789abcdef0123456789abcdef"
VP_AUTH = {"Authorization": f"Bearer {VP_KEY}"}


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "enable_auth", True)
    monkeypatch.setattr(settings, "vp_meetween_api_key", VP_KEY)
    monkeypatch.setattr(settings, "jigasi_meetween_api_key", JIGASI_KEY)
    yield
    SessionManager._sessions.clear()
    SessionManager._last_activity.clear()


def _chat_body(participant_id: str | None = BOT_PARTICIPANT) -> dict:
    body = {
        "type": TaskType.SPOKEN_QUESTION_ANSWERING,
        "meeting_id": MEETING_ID,
        "room_name": ROOM,
        "language": "en-US",
        "prompt": "What did we decide?",
    }
    if participant_id is not None:
        body["participant_id"] = participant_id
    return body


@pytest.mark.integration
class TestChatServiceToken:
    def test_service_key_is_accepted(self, test_client: TestClient):
        response = test_client.post("/api/chat", json=_chat_body(), headers=VP_AUTH)

        assert response.status_code == 200
        body = response.json()
        assert body["result"] == "Mock chat response text for testing"
        assert body["participant_id"] == BOT_PARTICIPANT

    def test_service_turn_is_stored_under_the_bot_participant(self, test_client: TestClient):
        test_client.post("/api/chat", json=_chat_body(), headers=VP_AUTH)

        history = SessionManager.get_participant_history(MEETING_ID, ROOM, BOT_PARTICIPANT)
        assert [entry.role for entry in history] == ["user", "assistant"]

    def test_wrong_key_is_rejected(self, test_client: TestClient):
        response = test_client.post(
            "/api/chat",
            json=_chat_body(),
            headers={"Authorization": "Bearer not-the-key"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid access token"
        assert response.headers["WWW-Authenticate"] == 'Bearer error="invalid_token"'

    def test_no_token_is_rejected(self, test_client: TestClient):
        response = test_client.post("/api/chat", json=_chat_body())

        assert response.status_code == 401
        assert response.json()["detail"] == "Missing bearer token"
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_non_ascii_bearer_is_rejected(self, test_client: TestClient):
        response = test_client.post(
            "/api/chat",
            json=_chat_body(),
            # Raw latin-1 bytes: httpx refuses non-ASCII str headers, and this is what hits the server.
            headers={"Authorization": b"Bearer caf\xe9"},
        )

        assert response.status_code == 401

    def test_unset_key_rejects_the_service_caller(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "vp_meetween_api_key", "")

        response = test_client.post("/api/chat", json=_chat_body(), headers=VP_AUTH)

        assert response.status_code == 401

    def test_auth_off_accepts_a_call_without_a_token(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "enable_auth", False)

        response = test_client.post("/api/chat", json=_chat_body("user-1"))

        assert response.status_code == 200


@pytest.mark.integration
class TestChatUserToken:
    def test_user_token_is_still_accepted(
        self, test_client: TestClient, auth_headers: dict[str, str]
    ):
        response = test_client.post(
            "/api/chat", json=_chat_body("user-123"), headers=auth_headers
        )

        assert response.status_code == 200
        assert response.json()["participant_id"] == "user-123"

    def test_expired_user_token_is_reported_as_expired(self, test_client: TestClient):
        aj = settings.app_jwt
        expired = jwt.encode(
            {
                "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
                "sub": "expired@example.com",
                "aud": aj.audience_access,
            },
            aj.secret_key,
            algorithm=aj.algorithm,
        )

        response = test_client.post(
            "/api/chat",
            json=_chat_body("user-123"),
            headers={"Authorization": f"Bearer {expired}"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Access token expired"


@pytest.mark.integration
class TestKeyScopes:
    def test_jigasi_key_is_rejected_on_chat(self, test_client: TestClient):
        response = test_client.post(
            "/api/chat",
            json=_chat_body(),
            headers={"Authorization": f"Bearer {JIGASI_KEY}"},
        )

        assert response.status_code == 401

    def test_service_key_is_rejected_on_the_jigasi_routes(self, test_client: TestClient):
        participants = test_client.post(
            f"/meetings/{MEETING_ID}/participants",
            json={"participants": [{"participant_id": "p1", "username": "Ada"}]},
            headers=VP_AUTH,
        )
        ended = test_client.post(f"/meetings/{MEETING_ID}/ended", headers=VP_AUTH)

        assert participants.status_code == 401
        assert ended.status_code == 401


@pytest.mark.integration
class TestParticipantNamespace:
    @pytest.mark.parametrize("participant_id", ["user-123", "default", "VP:abc", "vp:", "vp:a/b"])
    def test_service_caller_cannot_use_another_participant_id(
        self, test_client: TestClient, participant_id: str
    ):
        response = test_client.post(
            "/api/chat", json=_chat_body(participant_id), headers=VP_AUTH
        )

        assert response.status_code == 403

    def test_service_caller_must_send_a_participant_id(self, test_client: TestClient):
        response = test_client.post("/api/chat", json=_chat_body(None), headers=VP_AUTH)

        assert response.status_code == 403
        assert SessionManager.get_participant_history(MEETING_ID, ROOM, "default") == []

    def test_user_cannot_take_the_bot_namespace(
        self, test_client: TestClient, auth_headers: dict[str, str]
    ):
        response = test_client.post(
            "/api/chat", json=_chat_body(BOT_PARTICIPANT), headers=auth_headers
        )

        assert response.status_code == 403
