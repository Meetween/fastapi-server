from fastapi import HTTPException

from core import SessionManager


def reject_if_meeting_ended(meeting_id: str) -> None:
    """Refuse requests for a meeting Jigasi reported as ended."""
    if SessionManager.is_meeting_ended(meeting_id):
        raise HTTPException(status_code=409, detail="Meeting has ended")
