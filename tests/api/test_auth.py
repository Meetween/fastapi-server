from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from config import MeetingJwtSettings, settings
from tests.utils.helpers import (
    create_user_via_api_and_login,
    generate_test_meeting_id,
    obtain_bearer_headers,
    register_user_via_api,
)

_REGISTER_PAYLOAD = {
    "email": "blocked@example.com",
    "password": "secret123",
}


def _encode_expired_token(*, sub: str, audience: str) -> str:
    """A correctly signed token whose exp passed one minute ago."""
    aj = settings.app_jwt
    expired_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    return jwt.encode(
        {"exp": expired_at, "sub": sub, "aud": audience},
        aj.secret_key,
        algorithm=aj.algorithm,
    )


@pytest.mark.integration
class TestAuthEndpoints:
    def test_register_requires_admin(self, test_client, admin_headers):
        assert (
            register_user_via_api(test_client, _REGISTER_PAYLOAD).status_code
            == 401
        )

        created = register_user_via_api(
            test_client, _REGISTER_PAYLOAD, headers=admin_headers
        )
        assert created.status_code == 200

        user_headers = obtain_bearer_headers(
            test_client,
            email="regular@example.com",
            password="secret123",
        )
        forbidden = register_user_via_api(
            test_client,
            {"email": "another@example.com", "password": "secret123"},
            headers=user_headers,
        )
        assert forbidden.status_code == 403

    def test_register_and_login(self, test_client, admin_headers):
        register_payload = {
            "email": "auth-user@example.com",
            "password": "secret123",
            "first_name": "Auth",
            "last_name": "User",
        }
        register_response = register_user_via_api(
            test_client, register_payload, headers=admin_headers
        )
        assert register_response.status_code == 200
        register_data = register_response.json()
        assert register_data["email"] == "auth-user@example.com"
        assert register_data["is_active"] is True

        dup = register_user_via_api(
            test_client, register_payload, headers=admin_headers
        )
        assert dup.status_code == 409

        login_response = test_client.post(
            "/api/auth/login",
            json={"email": "auth-user@example.com", "password": "secret123"},
        )
        assert login_response.status_code == 200
        login_data = login_response.json()
        assert "access_token" in login_data
        assert "refresh_token" in login_data
        assert login_data["token_type"] == "bearer"

    def test_refresh_issues_new_access_token(self, test_client, admin_headers):
        register_user_via_api(
            test_client,
            {
                "email": "refresh-user@example.com",
                "password": "secret123",
            },
            headers=admin_headers,
        )
        login_response = test_client.post(
            "/api/auth/login",
            json={"email": "refresh-user@example.com", "password": "secret123"},
        )
        assert login_response.status_code == 200
        login_data = login_response.json()

        refresh_response = test_client.post(
            "/api/auth/refresh",
            json={"refresh_token": login_data["refresh_token"]},
        )
        assert refresh_response.status_code == 200
        refresh_data = refresh_response.json()
        assert refresh_data["token_type"] == "bearer"
        assert "access_token" in refresh_data
        assert refresh_data["refresh_token"] != login_data["refresh_token"]

        protected = test_client.get(
            "/api/auth/context/some-room",
            headers={"Authorization": f"Bearer {refresh_data['access_token']}"},
        )
        assert protected.status_code == 404

    def test_expired_access_token_is_reported_as_expired(self, test_client):
        expired = _encode_expired_token(
            sub="expired@example.com", audience=settings.app_jwt.audience_access
        )
        response = test_client.get(
            "/api/auth/context/some-room",
            headers={"Authorization": f"Bearer {expired}"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Access token expired"
        assert response.headers["WWW-Authenticate"] == (
            'Bearer error="invalid_token", '
            'error_description="The access token expired"'
        )

    def test_malformed_access_token_is_reported_as_invalid(self, test_client):
        response = test_client.get(
            "/api/auth/context/some-room",
            headers={"Authorization": "Bearer not-a-valid-token"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid access token"
        assert response.headers["WWW-Authenticate"] == 'Bearer error="invalid_token"'

    def test_missing_access_token_carries_challenge(self, test_client):
        response = test_client.get("/api/auth/context/some-room")
        assert response.status_code == 401
        assert response.json()["detail"] == "Missing bearer token"
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_refresh_works_after_access_token_expired(
        self, test_client, admin_headers
    ):
        register_user_via_api(
            test_client,
            {"email": "expired-refresh@example.com", "password": "secret123"},
            headers=admin_headers,
        )
        login = test_client.post(
            "/api/auth/login",
            json={"email": "expired-refresh@example.com", "password": "secret123"},
        ).json()
        expired = _encode_expired_token(
            sub="expired-refresh@example.com",
            audience=settings.app_jwt.audience_access,
        )
        rejected = test_client.get(
            "/api/auth/context/some-room",
            headers={"Authorization": f"Bearer {expired}"},
        )
        assert rejected.status_code == 401

        refreshed = test_client.post(
            "/api/auth/refresh", json={"refresh_token": login["refresh_token"]}
        )
        assert refreshed.status_code == 200
        accepted = test_client.get(
            "/api/auth/context/some-room",
            headers={"Authorization": f"Bearer {refreshed.json()['access_token']}"},
        )
        assert accepted.status_code == 404

    def test_refresh_rejects_expired_refresh_token(self, test_client):
        expired = _encode_expired_token(
            sub="expired@example.com", audience=settings.app_jwt.audience_refresh
        )
        response = test_client.post(
            "/api/auth/refresh", json={"refresh_token": expired}
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Refresh token expired"

    def test_refresh_rejects_invalid_token(self, test_client):
        response = test_client.post(
            "/api/auth/refresh",
            json={"refresh_token": "not-a-valid-token"},
        )
        assert response.status_code == 401

    def test_refresh_rejects_access_token(self, test_client, admin_headers):
        headers = create_user_via_api_and_login(
            test_client,
            email="refresh-access@example.com",
            password="secret123",
            admin_headers=admin_headers,
        )
        access_token = headers["Authorization"].removeprefix("Bearer ")
        response = test_client.post(
            "/api/auth/refresh",
            json={"refresh_token": access_token},
        )
        assert response.status_code == 401

    def test_jitsi_issue_and_context_with_access_token(
        self, test_client, admin_headers, monkeypatch
    ):
        monkeypatch.setattr(
            settings,
            "_meeting_jwt",
            MeetingJwtSettings(
                issuer=settings.meeting_jwt.issuer,
                expire_minutes=settings.meeting_jwt.expire_minutes,
                meet_base_url=None,
            ),
        )
        auth_headers = create_user_via_api_and_login(
            test_client,
            email="jwt-user@example.com",
            password="secret123",
            admin_headers=admin_headers,
            first_name="JWT",
            last_name="User",
        )

        meeting_uuid = generate_test_meeting_id()
        issue_response = test_client.post(
            "/api/auth/jitsi/issue",
            json={
                "room_name": "room-123",
                "meeting_id": meeting_uuid,
                "moderator": True,
                "metadata": {"source": "jwt-auth"},
            },
            headers=auth_headers,
        )
        assert issue_response.status_code == 200
        issue_data = issue_response.json()
        assert "jitsi_token" in issue_data
        assert issue_data["room"] == "room-123"
        assert issue_data["meeting_context"]["meeting_id"] == meeting_uuid
        assert issue_data["meeting_context"]["room_name"] == "room-123"
        assert issue_data["meeting_context"]["user_id"] == "jwt-user@example.com"
        assert issue_data["meeting_context"]["metadata"]["source"] == "jwt-auth"
        assert issue_data.get("meet_join_url") is None

        validate_response = test_client.post(
            "/api/auth/jitsi/validate",
            json={"token": issue_data["jitsi_token"]},
        )
        assert validate_response.status_code == 200
        validate_data = validate_response.json()
        assert validate_data["valid"] is True
        assert isinstance(validate_data["claims"], dict)
        assert validate_data["claims"]["room"] == "room-123"
        assert validate_data["claims"]["iss"] == settings.meeting_jwt.issuer
        assert "iat" in validate_data["claims"]
        assert "meeting_id" not in validate_data["claims"]

        context_response = test_client.get(
            "/api/auth/context/room-123",
            headers=auth_headers,
        )
        assert context_response.status_code == 200
        context_data = context_response.json()
        assert context_data["user_id"] == "jwt-user@example.com"
        assert context_data["meeting_id"] == meeting_uuid

    def test_jitsi_issue_omitted_meeting_id_keys_context_by_room(
        self, test_client, admin_headers
    ):
        headers = create_user_via_api_and_login(
            test_client,
            email="legacy-room-key@example.com",
            password="secret123",
            admin_headers=admin_headers,
        )
        issue_response = test_client.post(
            "/api/auth/jitsi/issue",
            json={"room_name": "legacy-room", "moderator": False},
            headers=headers,
        )
        assert issue_response.status_code == 200
        data = issue_response.json()
        assert data["meeting_context"]["meeting_id"] is None
        assert data["meeting_context"]["room_name"] == "legacy-room"
        ctx = test_client.get("/api/auth/context/legacy-room", headers=headers)
        assert ctx.status_code == 200
        assert ctx.json()["meeting_id"] is None

    def test_context_requires_bearer(self, test_client):
        response = test_client.get("/api/auth/context/some-room")
        assert response.status_code == 401

    def test_jitsi_issue_requires_room_name(self, test_client, admin_headers):
        headers = create_user_via_api_and_login(
            test_client,
            email="noroom@example.com",
            password="secret123",
            admin_headers=admin_headers,
        )
        bad = test_client.post(
            "/api/auth/jitsi/issue",
            json={},
            headers=headers,
        )
        assert bad.status_code == 422

    def test_jitsi_meet_join_url_when_base_configured(
        self, test_client, admin_headers, monkeypatch
    ):
        monkeypatch.setattr(
            settings,
            "_meeting_jwt",
            MeetingJwtSettings(
                issuer=settings.meeting_jwt.issuer,
                expire_minutes=settings.meeting_jwt.expire_minutes,
                meet_base_url="https://meet.example.com",
            ),
        )
        headers = create_user_via_api_and_login(
            test_client,
            email="joinurl@example.com",
            password="secret123",
            admin_headers=admin_headers,
        )
        issue_response = test_client.post(
            "/api/auth/jitsi/issue",
            json={
                "room_name": "My Room",
                "moderator": False,
            },
            headers=headers,
        )
        assert issue_response.status_code == 200
        url = issue_response.json()["meet_join_url"]
        assert url is not None
        assert url.startswith("https://meet.example.com/")
        assert "My%20Room" in url or "My+Room" in url
        assert "jwt=" in url
