"""
Group chat messages as meeting context.

The virtual participant worker sees every message typed in the meeting's group chat and forwards
the ones people wrote here. Each becomes a transcript line marked as chat, so the model answering
questions reads what was typed next to what was said.
"""
import logging

from fastapi import APIRouter, Depends

from api.deps.meeting_ended import reject_if_meeting_ended
from api.deps.service_token import require_vp_worker
from core import SessionManager
from core.session_manager import SOURCE_CHAT
from models import MeetingChatMessage, MeetingChatMessageResponse

router = APIRouter(
    prefix="/api",
    tags=["chat"],
    dependencies=[Depends(require_vp_worker)],
)
logger = logging.getLogger(__name__)


@router.post(
    "/meetings/{meeting_id}/chat-messages",
    response_model=MeetingChatMessageResponse,
    dependencies=[Depends(reject_if_meeting_ended)],
)
async def add_chat_message(meeting_id: str, message: MeetingChatMessage):
    """
    Append a group chat message to the meeting transcript as "Name (chat): text".

    - Accepts the virtual participant worker key only.
    - Line breaks are folded so a message cannot pose as further transcript lines.
    - A message from a virtual participant is not stored.
    """
    text = " ".join(message.text.split())
    if not text or SessionManager.is_virtual_participant(meeting_id, message.participant_id):
        return MeetingChatMessageResponse(status="success", meeting_id=meeting_id, stored=False)

    SessionManager.update_transcript(
        meeting_id=meeting_id,
        room_name=None,
        transcript=text,
        participant_id=message.participant_id,
        timestamp=message.timestamp,
        username=" ".join(message.username.split()) or None,
        source=SOURCE_CHAT,
    )
    logger.info("Stored chat message for meeting_id=%s participant_id=%s", meeting_id, message.participant_id)
    return MeetingChatMessageResponse(status="success", meeting_id=meeting_id, stored=True)
