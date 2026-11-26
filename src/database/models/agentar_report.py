from typing import Any, ClassVar, Literal, Optional

from pydantic import Field

from src.database.models.agentar_types import UtcDatetime
from src.database.mongo_model import MongoModel


class AgentarReport(MongoModel):
    """Post meeting report, the source of truth. Item shapes are the `ActionItem`, `Answer`,
    `OpenPoint` and `TranscriptLine` schemas of the contract."""

    meeting_id: str
    owner: str
    status: Literal["pending", "ready", "failed"] = "pending"
    error: Optional[str] = None
    room_name: Optional[str] = None
    title: Optional[str] = None
    started_at: Optional[UtcDatetime] = None
    ended_at: Optional[UtcDatetime] = None
    generated_at: Optional[UtcDatetime] = None
    generation_ms: Optional[int] = None
    summary: Optional[str] = None
    action_items: list[dict[str, Any]] = Field(default_factory=list)
    answers: list[dict[str, Any]] = Field(default_factory=list)
    open_points: list[dict[str, Any]] = Field(default_factory=list)
    transcript: list[dict[str, Any]] = Field(default_factory=list)
    model: Optional[str] = None
    md_key: Optional[str] = Field(default=None, description="Object store key of the .md export")

    collection_name: ClassVar[str] = "agentar_reports"

    @classmethod
    async def setup_indexes(cls) -> None:
        collection = cls.get_collection()
        await collection.create_index("meeting_id", unique=True)
        await collection.create_index("owner")
