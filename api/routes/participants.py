import logging
from fastapi import APIRouter, HTTPException
from core import SessionManager
from models import ParticipantUpdateRequest

router = APIRouter()
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

    except Exception as e:
        logger.error(f"Error updating participants: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
