from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, Optional

from pydantic import Field

from config import settings
from src.database.mongo_model import MongoModel


class MeetingContext(MongoModel):
    """Per-user, per-meeting context stored in the `meeting_context` collection."""

    user_id: str = Field(..., description="Owner of this meeting context")
    meeting_id: str = Field(..., description="Meeting identifier")
    room_name: Optional[str] = Field(default=None, description="Room name if known")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last upsert timestamp",
    )

    collection_name: ClassVar[str] = settings.mongo_meeting_context_collection

    @classmethod
    async def setup_indexes(cls) -> None:
        await cls.get_collection().create_index(
            [("user_id", 1), ("meeting_id", 1)], unique=True
        )
