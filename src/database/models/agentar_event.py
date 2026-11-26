from typing import Any, ClassVar, Literal, Optional

from pydantic import Field

from src.database.models.agentar_types import UtcDatetime, now_field
from src.database.mongo_model import MongoModel

EVENT_TTL_SECONDS = 30 * 24 * 3600


class AgentarEvent(MongoModel):
    """Worker or server event, also the dashboard notification feed. The notification cursor is
    the `_id`, so the index on (owner, _id) serves the polling query."""

    event_id: str = Field(..., description="Idempotency key, a uuid from the worker or the server")
    owner: str
    meeting_id: str
    bot_id: Optional[str] = None
    type: str
    source: Literal["worker", "server"] = "worker"
    data: dict[str, Any] = Field(default_factory=dict)
    at: Optional[UtcDatetime] = Field(default=None, description="Worker clock")
    received_at: UtcDatetime = Field(default_factory=now_field)

    collection_name: ClassVar[str] = "agentar_events"

    @classmethod
    async def setup_indexes(cls) -> None:
        collection = cls.get_collection()
        await collection.create_index("event_id", unique=True)
        await collection.create_index([("owner", 1), ("_id", 1)])
        await collection.create_index("received_at", expireAfterSeconds=EVENT_TTL_SECONDS)
