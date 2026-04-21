"""
Test helper functions
"""
import asyncio
import base64
import uuid
from typing import Any, Dict

from fastapi.testclient import TestClient

from models import TASK_TYPE_TEXT_SUMMARY, TASK_TYPE_SPEECH_SUMMARY
from src.database.models.user import User
from utils.auth import create_user
from utils.uuid_helper import new_meeting_id

TEST_ADMIN_EMAIL = "pytest-admin@meetween.test"
TEST_ADMIN_PASSWORD = "pytest-admin-secret-123"


def create_audio_payload(
    room_name: str,
    meeting_id: str,
    audio_data: bytes,
    participant_id: str = None,
    language: str = "en-US",
    task_type: str = TASK_TYPE_SPEECH_SUMMARY
) -> Dict[str, Any]:
    """
    Create a payload with base64-encoded audio.

    Args:
        room_name: Room name
        meeting_id: Meeting ID
        audio_data: Raw audio bytes
        participant_id: Optional participant ID
        language: Language code (default: "en-US")
        task_type: Task type (default: TASK_TYPE_SPEECH_SUMMARY)
        
    Returns:
        Payload dictionary
    """
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    payload = {
        "type": task_type,
        "room_name": room_name,
        "meeting_id": meeting_id,
        "language": language,
        "audio": audio_base64
    }
    if participant_id:
        payload["participant_id"] = participant_id
    return payload


def create_text_payload(
    room_name: str,
    meeting_id: str,
    text: str,
    participant_id: str = None,
    language: str = "en-US",
    task_type: str = TASK_TYPE_TEXT_SUMMARY
) -> Dict[str, Any]:
    """
    Create a payload with text input.

    Args:
        room_name: Room name
        meeting_id: Meeting ID
        text: Text content
        participant_id: Optional participant ID
        language: Language code (default: "en-US")
        task_type: Task type (default: TASK_TYPE_TEXT_SUMMARY)
        
    Returns:
        Payload dictionary
    """
    payload = {
        "type": task_type,
        "room_name": room_name,
        "meeting_id": meeting_id,
        "language": language,
        "text": text
    }
    if participant_id:
        payload["participant_id"] = participant_id
    return payload


def create_generic_payload(
    room_name: str,
    meeting_id: str,
    participant_id: str = None,
    language: str = "en-US",
    task_type: str = TASK_TYPE_TEXT_SUMMARY,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a generic payload (no audio or text, loads from recordings).

    Args:
        room_name: Room name
        meeting_id: Meeting ID
        participant_id: Optional participant ID
        language: Language code (default: "en-US")
        task_type: Task type (default: TASK_TYPE_TEXT_SUMMARY)
        **kwargs: Additional fields to include

    Returns:
        Payload dictionary
    """
    payload = {
        "type": task_type,
        "room_name": room_name,
        "meeting_id": meeting_id,
        "language": language
    }
    if participant_id:
        payload["participant_id"] = participant_id
    payload.update(kwargs)
    return payload


def ensure_user_in_db(
    *,
    email: str,
    password: str,
    first_name: str | None = "Test",
    last_name: str | None = "User",
    is_admin: bool = False,
) -> None:
    """Insert a user when missing (tests bypass admin-only registration API)."""

    async def _ensure() -> None:
        if await User.find_by_email(email) is not None:
            return
        await create_user(
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            is_admin=is_admin,
        )

    asyncio.run(_ensure())


def ensure_admin_user() -> None:
    """Seed bootstrap admin via the same path as `python cli.py create-admin`."""
    ensure_user_in_db(
        email=TEST_ADMIN_EMAIL,
        password=TEST_ADMIN_PASSWORD,
        first_name="Pytest",
        last_name="Admin",
        is_admin=True,
    )


def login_headers(
    client: TestClient,
    *,
    email: str,
    password: str,
) -> dict[str, str]:
    response = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def obtain_admin_headers(client: TestClient) -> dict[str, str]:
    """Bearer token for the bootstrap admin user."""
    ensure_admin_user()
    return login_headers(
        client, email=TEST_ADMIN_EMAIL, password=TEST_ADMIN_PASSWORD
    )


def register_user_via_api(
    client: TestClient,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> Any:
    return client.post("/api/auth/users", json=payload, headers=headers)


def create_user_via_api_and_login(
    client: TestClient,
    *,
    email: str,
    password: str,
    admin_headers: dict[str, str],
    **extra_fields: Any,
) -> dict[str, str]:
    """Admin creates a user via API, then returns that user's bearer headers."""
    response = register_user_via_api(
        client,
        {"email": email, "password": password, **extra_fields},
        headers=admin_headers,
    )
    assert response.status_code == 200
    return login_headers(client, email=email, password=password)


def obtain_bearer_headers(
    client: TestClient,
    *,
    email: str,
    password: str,
    first_name: str = "Test",
    last_name: str = "User",
) -> dict[str, str]:
    """Ensure user exists, log in, and return Authorization headers."""
    ensure_user_in_db(
        email=email,
        password=password,
        first_name=first_name,
        last_name=last_name,
    )
    return login_headers(client, email=email, password=password)


def create_mt_payload(
    room_name: str,
    meeting_id: str,
    text: str,
    target_language: str = "de",
    language: str = "en",
    participant_id: str = None,
) -> Dict[str, Any]:
    """Create a MeetweenMTRequest payload for text translation."""
    payload = {
        "type": "MT",
        "room_name": room_name,
        "meeting_id": meeting_id,
        "text": text,
        "language": language,
        "target_language": target_language,
    }
    if participant_id:
        payload["participant_id"] = participant_id
    return payload


def create_st_payload(
    room_name: str,
    meeting_id: str,
    audio_data: bytes,
    target_language: str = "de",
    language: str = "en",
    participant_id: str = None,
) -> Dict[str, Any]:
    """Create a MeetweenSTRequest payload for speech translation."""
    payload = {
        "type": "ST",
        "room_name": room_name,
        "meeting_id": meeting_id,
        "audio": base64.b64encode(audio_data).decode("utf-8"),
        "language": language,
        "target_language": target_language,
    }
    if participant_id:
        payload["participant_id"] = participant_id
    return payload


def generate_test_meeting_id() -> str:
    """Generate a unique test meeting ID."""
    return new_meeting_id()


def generate_test_room_name() -> str:
    """Generate a unique test room name."""
    return f"test-room-{new_meeting_id()}"

