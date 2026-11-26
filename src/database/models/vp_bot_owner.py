from typing import ClassVar

from pydantic import Field

from src.database.models.agentar_types import UtcDatetime, now_field
from src.database.mongo_model import MongoModel

BOT_OWNER_TTL_SECONDS = 24 * 3600


class VpBotOwner(MongoModel):
    """Owner of a Butler panel bot, so the control routes can answer 404 across users."""

    bot_id: str
    owner: str
    room_name: str
    created_at: UtcDatetime = Field(default_factory=now_field)

    collection_name: ClassVar[str] = "vp_bot_owners"

    @classmethod
    async def setup_indexes(cls) -> None:
        collection = cls.get_collection()
        await collection.create_index("bot_id", unique=True)
        await collection.create_index("created_at", expireAfterSeconds=BOT_OWNER_TTL_SECONDS)
