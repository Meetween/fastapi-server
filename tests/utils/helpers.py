"""
Test helper functions
"""
import base64
import uuid
from typing import Dict, Any

from models import TASK_TYPE_TEXT_SUMMARY, TASK_TYPE_SPEECH_SUMMARY
from utils.uuid_helper import new_meeting_id


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


def generate_test_meeting_id() -> str:
    """Generate a unique test meeting ID."""
    return new_meeting_id()


def generate_test_room_name() -> str:
    """Generate a unique test room name."""
    return f"test-room-{new_meeting_id()}"

