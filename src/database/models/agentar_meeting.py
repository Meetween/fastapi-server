from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, Field

from src.database.models.agentar_types import UtcDatetime, now_field
from src.database.mongo_model import MongoModel

MeetingStatus = Literal[
    "draft", "scheduled", "waiting_for_human", "joining", "live", "leaving", "ended", "missed", "failed"
]
# Statuses in which a bot may be in the room. `live_lock` is set while a meeting is in
# waiting_for_human, joining, live or leaving.
BOT_PRESENT_STATUSES: tuple[str, ...] = ("waiting_for_human", "joining", "live", "leaving")


class MeetingQuestion(BaseModel):
    id: str = Field(..., pattern=r"^q[0-9]{1,2}$", description="Assigned by position on every write")
    text: str


class AgentarMeeting(MongoModel):
    """One Agentar meeting. The lifecycle tick moves `status` with atomic `find_one_and_update`
    claims and `$set`; `save()` replaces the whole document and is for creation only."""

    owner: str = Field(..., description="Token subject, the lower-cased email of the owner")
    title: Optional[str] = None
    room_name: str
    mode: Literal["now", "scheduled"] = "now"
    start_at: Optional[UtcDatetime] = None
    grace_minutes: int = 15
    purpose: str = ""
    questions: list[MeetingQuestion] = Field(default_factory=list)
    relevant_info: str = ""
    file_ids: list[str] = Field(default_factory=list)
    use_general_kb: bool = False
    picture_id: Optional[str] = None
    status: MeetingStatus = "draft"
    status_reason: Optional[str] = None
    display_name: Optional[str] = None
    voice: Optional[str] = None
    bot_id: Optional[str] = None
    jitsi_meeting_id: Optional[str] = None
    # True while the meeting holds the owner's single live slot, otherwise absent or None.
    live_lock: Optional[bool] = None
    # Transcript lines copied by the tick and by the end hook, moved to the report afterwards.
    transcript_snapshot: Optional[list[dict[str, Any]]] = None
    dispatched_at: Optional[UtcDatetime] = None
    joined_at: Optional[UtcDatetime] = None
    ended_at: Optional[UtcDatetime] = None
    report_status: Literal["none", "pending", "ready", "failed"] = "none"
    created_at: UtcDatetime = Field(default_factory=now_field)
    updated_at: UtcDatetime = Field(default_factory=now_field)

    collection_name: ClassVar[str] = "agentar_meetings"

    @classmethod
    async def setup_indexes(cls) -> None:
        collection = cls.get_collection()
        await collection.create_index([("owner", 1), ("start_at", -1), ("created_at", -1)])
        # The tick scans by status and start_at.
        await collection.create_index([("status", 1), ("start_at", 1)])
        # A bot belongs to one meeting. Meetings without a bot hold no value, so they do not collide.
        await collection.create_index(
            "bot_id", unique=True, partialFilterExpression={"bot_id": {"$type": "string"}}
        )
        # One live Agentar per user, held by the database as well as by the pre check.
        await collection.create_index(
            "owner", unique=True, partialFilterExpression={"live_lock": True}, name="owner_live_lock"
        )
