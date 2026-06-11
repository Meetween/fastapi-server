from datetime import datetime, timezone
from typing import Any, ClassVar, Dict
from uuid import UUID

from pydantic import Field, field_validator

from config import settings
from src.database.mongo_model import MongoModel


class MeetingContext(MongoModel):
    """Per-user join context. Upsert key is (user_id, room_name); `meeting_id` is an optional UUID string."""

    user_id: str = Field(..., description="Owner of this meeting context")
    room_name: str = Field(..., description="Room name if known")
    meeting_id: str | None = Field(default=None, description="Meeting identifier (RFC 4122 UUID string)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last upsert timestamp",
    )

    collection_name: ClassVar[str] = settings.mongo.meeting_context_collection

    @field_validator("meeting_id", mode="before")
    @classmethod
    def _coerce_meeting_id(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, str):
            return value
        raise TypeError(f"meeting_id must be str, UUID, or None, got {type(value)!r}")

    @classmethod
    async def setup_indexes(cls) -> None:
        await cls.get_collection().create_index(
            [("user_id", 1), ("room_name", 1)], unique=True
        )
