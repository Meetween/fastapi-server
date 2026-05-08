import pytest


@pytest.mark.integration
class TestAuthEndpoints:
    def test_register_and_login(self, client):
        register_payload = {
            "email": "auth-user@example.com",
            "password": "secret123",
            "first_name": "Auth",
            "last_name": "User",
        }
        register_response = client.post("/api/auth/users", json=register_payload)
        assert register_response.status_code == 200
        register_data = register_response.json()
        assert register_data["email"] == "auth-user@example.com"
        assert register_data["is_active"] is True

        login_response = client.post(
            "/api/auth/login",
            json={"email": "auth-user@example.com", "password": "secret123"},
        )
        assert login_response.status_code == 200
        login_data = login_response.json()
        assert "access_token" in login_data
        assert "refresh_token" in login_data
        assert login_data["token_type"] == "bearer"

    def test_jitsi_issue_and_context_with_access_token(self, client):
        register_payload = {
            "email": "jwt-user@example.com",
            "password": "secret123",
            "first_name": "JWT",
            "last_name": "User",
        }
        register_response = client.post("/api/auth/users", json=register_payload)
        assert register_response.status_code == 200

        login_response = client.post(
            "/api/auth/login",
            json={"email": "jwt-user@example.com", "password": "secret123"},
        )
        assert login_response.status_code == 200
        access_token = login_response.json()["access_token"]
        auth_headers = {"Authorization": f"Bearer {access_token}"}

        upsert_payload = {
            "meeting_id": "meeting-123",
            "room_name": "room-123",
            "metadata": {"source": "jwt-auth"},
        }
        upsert_response = client.post(
            "/api/auth/context",
            json=upsert_payload,
            headers=auth_headers,
        )
        assert upsert_response.status_code == 200
        upsert_data = upsert_response.json()
        assert upsert_data["meeting_context"]["meeting_id"] == "meeting-123"

        issue_response = client.post(
            "/api/auth/jitsi/issue",
            json={"room_name": "room-123", "moderator": True},
            headers=auth_headers,
        )
        assert issue_response.status_code == 200
        issue_data = issue_response.json()
        assert "jitsi_token" in issue_data
        assert issue_data["room"] == "room-123"

        validate_response = client.post(
            "/api/auth/jitsi/validate",
            json={"token": issue_data["jitsi_token"]},
        )
        assert validate_response.status_code == 200
        validate_data = validate_response.json()
        assert validate_data["valid"] is True
        assert validate_data["claims"]["room"] == "room-123"

        context_response = client.get(
            "/api/auth/context/meeting-123",
            headers=auth_headers,
        )
        assert context_response.status_code == 200
        context_data = context_response.json()
        assert context_data["user_id"] == "jwt-user@example.com"
        assert context_data["meeting_id"] == "meeting-123"
