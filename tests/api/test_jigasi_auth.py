"""Jigasi authenticates participant updates and translation with one shared key."""

import pytest
from fastapi.testclient import TestClient

from config import settings

JIGASI_KEY = "jigasi-test-key"
MEETING_ID = "5506be1e-b870-4922-845d-d053b1d0b4db"


def _participant_update():
    return {
        "participants": [
            {"participant_id": "p1", "username": "Ada"},
        ]
    }


def _use_jigasi_key(monkeypatch: pytest.MonkeyPatch, key: str = JIGASI_KEY) -> None:
    monkeypatch.setattr(settings, "enable_auth", True)
    monkeypatch.setattr(settings, "jigasi_meetween_api_key", key)


def _translation_body():
    return {
        "type": "MT",
        "room_name": "room-1",
        "meeting_id": MEETING_ID,
        "text": "hello",
        "language": "en",
        "target_language": "de",
    }


@pytest.mark.integration
class TestJigasiParticipantAuth:
    def test_matching_key_updates_participants(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        _use_jigasi_key(monkeypatch)

        response = test_client.post(
            f"/meetings/{MEETING_ID}/participants",
            json=_participant_update(),
            headers={"Authorization": f"Bearer {JIGASI_KEY}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["meeting_id"] == MEETING_ID
        assert body["participants_updated"] == 1

    def test_missing_bearer_is_rejected(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        _use_jigasi_key(monkeypatch)

        response = test_client.post(
            f"/meetings/{MEETING_ID}/participants",
            json=_participant_update(),
        )

        assert response.status_code == 401

    def test_wrong_key_is_rejected(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        _use_jigasi_key(monkeypatch)

        response = test_client.post(
            f"/meetings/{MEETING_ID}/participants",
            json=_participant_update(),
            headers={"Authorization": "Bearer not-the-key"},
        )

        assert response.status_code == 401

    def test_user_access_token_is_rejected(
        self,
        test_client: TestClient,
        auth_headers: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ):
        _use_jigasi_key(monkeypatch)

        response = test_client.post(
            f"/meetings/{MEETING_ID}/participants",
            json=_participant_update(),
            headers=auth_headers,
        )

        assert response.status_code == 401

    def test_unset_key_is_rejected(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        _use_jigasi_key(monkeypatch, key="")

        response = test_client.post(
            f"/meetings/{MEETING_ID}/participants",
            json=_participant_update(),
            headers={"Authorization": f"Bearer {JIGASI_KEY}"},
        )

        assert response.status_code == 401

    def test_auth_off_updates_participants_without_a_key(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "enable_auth", False)
        monkeypatch.setattr(settings, "jigasi_meetween_api_key", "")

        response = test_client.post(
            f"/meetings/{MEETING_ID}/participants",
            json=_participant_update(),
        )

        assert response.status_code == 200
        assert response.json()["participants_updated"] == 1


@pytest.mark.integration
class TestJigasiTranslationAuth:
    def test_matching_key_translates(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        _use_jigasi_key(monkeypatch)

        response = test_client.post(
            "/api/translate",
            json=_translation_body(),
            headers={"Authorization": f"Bearer {JIGASI_KEY}"},
        )

        assert response.status_code == 200
        assert response.json()["result"] == "Mock translation text for testing"

    def test_missing_bearer_is_rejected(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        _use_jigasi_key(monkeypatch)

        response = test_client.post("/api/translate", json=_translation_body())

        assert response.status_code == 401

    def test_user_access_token_is_rejected(
        self,
        test_client: TestClient,
        auth_headers: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ):
        _use_jigasi_key(monkeypatch)

        response = test_client.post(
            "/api/translate",
            json=_translation_body(),
            headers=auth_headers,
        )

        assert response.status_code == 401
