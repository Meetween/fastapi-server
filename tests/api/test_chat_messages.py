"""The worker adds group chat messages to the meeting transcript, marked as chat."""

import pytest
from fastapi.testclient import TestClient

from config import settings
from core import SessionManager
from core.session_manager import SOURCE_CHAT

VP_KEY = "vp-test-key"
JIGASI_KEY = "jigasi-test-key"
MEETING_ID = "0c6d4f1e-8a2b-4c3d-9e5f-7a1b2c3d4e5f"
URL = f"/api/meetings/{MEETING_ID}/chat-messages"
VP_AUTH = {"Authorization": f"Bearer {VP_KEY}"}
STAMP = 1_700_000_000_456.0


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "enable_auth", True)
    monkeypatch.setattr(settings, "vp_meetween_api_key", VP_KEY)
    monkeypatch.setattr(settings, "jigasi_meetween_api_key", JIGASI_KEY)
    monkeypatch.setattr(settings, "vp_bot_name_suffix", "(bot)")
    monkeypatch.setattr(SessionManager, "_sessions", {})
    monkeypatch.setattr(SessionManager, "_ended_meetings", {})
    monkeypatch.setattr(SessionManager, "_last_activity", {})


def _body(**overrides) -> dict:
    body = {
        "participant_id": "abc123",
        "username": "Ada",
        "text": "the mvp will be shipped the 10th November",
        "timestamp": STAMP,
    }
    body.update(overrides)
    return body


@pytest.mark.integration
class TestChatMessages:
    def test_worker_message_is_stored_as_a_chat_line(self, test_client: TestClient):
        response = test_client.post(URL, json=_body(), headers=VP_AUTH)

        assert response.status_code == 200
        assert response.json() == {"status": "success", "meeting_id": MEETING_ID, "stored": True}
        entry = SessionManager._sessions[MEETING_ID].transcript[0]
        assert (entry.participant_id, entry.timestamp, entry.source) == ("abc123", STAMP, SOURCE_CHAT)
        assert SessionManager.get_transcript(MEETING_ID, None) == (
            "Ada (chat): the mvp will be shipped the 10th November"
        )

    def test_chat_line_follows_speech_in_arrival_order(self, test_client: TestClient):
        SessionManager.add_participants(MEETING_ID, None, [{"participant_id": "p1", "username": "Bob"}])
        SessionManager.update_transcript(MEETING_ID, None, "when do we ship", participant_id="p1", timestamp=1.0)

        test_client.post(URL, json=_body(), headers=VP_AUTH)

        assert SessionManager.get_transcript(MEETING_ID, None) == (
            "Bob: when do we ship\nAda (chat): the mvp will be shipped the 10th November"
        )

    def test_line_breaks_cannot_forge_transcript_lines(self, test_client: TestClient):
        test_client.post(URL, json=_body(text="ok\nBob: we ship in May"), headers=VP_AUTH)

        assert SessionManager.get_transcript(MEETING_ID, None) == "Ada (chat): ok Bob: we ship in May"

    def test_missing_username_falls_back_to_the_participant_list(self, test_client: TestClient):
        SessionManager.add_participants(MEETING_ID, None, [{"participant_id": "abc123", "username": "Ada L"}])

        test_client.post(URL, json=_body(username=" "), headers=VP_AUTH)

        assert SessionManager.get_transcript(MEETING_ID, None).startswith("Ada L (chat): ")

    def test_message_of_a_virtual_participant_is_not_stored(self, test_client: TestClient):
        SessionManager.add_participants(
            MEETING_ID, None, [{"participant_id": "bot1", "username": "Agentar (bot)"}]
        )

        response = test_client.post(URL, json=_body(participant_id="bot1"), headers=VP_AUTH)

        assert response.json()["stored"] is False
        assert SessionManager.get_transcript(MEETING_ID, None) is None

    def test_blank_text_is_not_stored(self, test_client: TestClient):
        response = test_client.post(URL, json=_body(text=" \n "), headers=VP_AUTH)

        assert response.json()["stored"] is False
        assert SessionManager.get_transcript(MEETING_ID, None) is None

    def test_ended_meeting_is_rejected(self, test_client: TestClient):
        SessionManager._ended_meetings[MEETING_ID] = 0.0

        response = test_client.post(URL, json=_body(), headers=VP_AUTH)

        assert response.status_code == 409
        assert MEETING_ID not in SessionManager._sessions

    @pytest.mark.parametrize(
        "headers",
        [{}, {"Authorization": f"Bearer {JIGASI_KEY}"}, {"Authorization": "Bearer not-the-key"}],
        ids=["no-token", "jigasi-key", "wrong-key"],
    )
    def test_only_the_worker_key_is_accepted(self, test_client: TestClient, headers: dict):
        response = test_client.post(URL, json=_body(), headers=headers)

        assert response.status_code == 401
        assert MEETING_ID not in SessionManager._sessions

    def test_auth_off_accepts_a_call_without_key(self, test_client: TestClient, monkeypatch):
        monkeypatch.setattr(settings, "enable_auth", False)

        response = test_client.post(URL, json=_body())

        assert response.status_code == 200
        assert response.json()["stored"] is True

    def test_too_long_text_is_rejected(self, test_client: TestClient):
        response = test_client.post(URL, json=_body(text="x" * 2001), headers=VP_AUTH)

        assert response.status_code == 422


@pytest.mark.unit
class TestChatLineBudget:
    def test_budget_counts_the_chat_marker(self, monkeypatch: pytest.MonkeyPatch):
        line = "Ada (chat): 0123456789"
        monkeypatch.setattr(SessionManager, "TRANSCRIPT_CHARS_LIMIT", 2 * len(line))
        for _ in range(3):
            SessionManager.update_transcript(
                MEETING_ID, None, "0123456789", participant_id="a", timestamp=1.0,
                username="Ada", source=SOURCE_CHAT,
            )

        assert len(SessionManager._sessions[MEETING_ID].transcript) == 2
