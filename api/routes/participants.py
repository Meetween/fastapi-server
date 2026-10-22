import logging
from fastapi import APIRouter, HTTPException

from api.deps.service_token import REQUIRE_SERVICE_TOKEN

from core import SessionManager
from models import ParticipantUpdateRequest

router = APIRouter(tags=["participants"], dependencies=REQUIRE_SERVICE_TOKEN)
logger = logging.getLogger(__name__)


@router.post("/meetings/{meeting_id}/participants")
async def update_participants(
    meeting_id: str,
    request: ParticipantUpdateRequest,
):
    """
    Receive participant updates from jigasi.

    Adds or updates participants in the session usermap.
    Participants are never removed even if they leave the meeting.
    """
    if SessionManager.is_meeting_ended(meeting_id):
        raise HTTPException(status_code=409, detail="Meeting has ended")

    try:
        participants = [
            {"participant_id": p.participant_id, "username": p.username}
            for p in request.participants
        ]

        SessionManager.add_participants(
            meeting_id=meeting_id,
            room_name=None,
            participants=participants,
        )

        logger.info(
            "Received participant update for meeting_id=%s: %d participant(s)",
            meeting_id,
            len(participants),
        )

        return {
            "status": "success",
            "meeting_id": meeting_id,
            "participants_updated": len(participants),
        }

    except Exception:
        logger.error("Error updating participants", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update participants")


@router.post("/meetings/{meeting_id}/ended")
async def meeting_ended(meeting_id: str):
    """
    Receive the end-of-meeting signal from jigasi.

    Force-clears in-memory session state for the meeting and marks it ended.
    Later participant updates and new websocket connections for the meeting are rejected.
    Idempotent: a repeated call returns success with already_ended=true.
    """
    newly_ended = await SessionManager.mark_meeting_ended(meeting_id=meeting_id)
    return {
        "status": "success",
        "meeting_id": meeting_id,
        "already_ended": not newly_ended,
    }
