from typing import ClassVar, Optional

from pydantic import Field

from src.database.models.agentar_types import UtcDatetime, now_field
from src.database.mongo_model import MongoModel


def room_key(room_name: str) -> str:
    """Ownership key for a room. Jitsi treats room names case insensitively."""
    return room_name.lower()


class RoomOwner(MongoModel):
    """The user who first claimed a room, and so moderates it.

    A room is claimed by the first Jitsi token issue, Agentar meeting create or dispatch for it,
    or by an operator with `cli.py room-owner set`. The owner key is the token subject, the
    lower-cased email, as everywhere in the Agentar collections.
    """

    room_name: str = Field(..., description="Room name, lower-cased")
    owner_user_id: str = Field(..., description="Token subject of the owner (lower-cased email)")
    created_at: UtcDatetime = Field(default_factory=now_field, description="When the room was claimed")

    collection_name: ClassVar[str] = "room_owner"

    @classmethod
    async def find_by_room(cls, room_name: str) -> Optional["RoomOwner"]:
        return await cls.find_one({"room_name": room_key(room_name)})

    @classmethod
    async def setup_indexes(cls) -> None:
        collection = cls.get_collection()
        await collection.create_index("room_name", unique=True)
        # Claims per user and UTC day are counted on this pair.
        await collection.create_index([("owner_user_id", 1), ("created_at", -1)])
