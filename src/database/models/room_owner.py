from typing import ClassVar

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError
from pydantic import Field

from config import settings
from src.database.mongo_model import MongoModel


class RoomOwner(MongoModel):
    """Owner of a Jitsi room, the gate for spawning virtual participants."""

    room_name: str = Field(..., description="Jitsi room name (unique)")
    owner_email: str = Field(..., description="Email of the owning user (lower-cased)")

    collection_name: ClassVar[str] = settings.mongo.room_owner_collection

    @classmethod
    async def claim_if_unowned(cls, *, room_name: str, owner_email: str) -> "RoomOwner":
        """Return the room's owner, recording `owner_email` as owner if none exists yet.

        An existing owner is never overwritten, so the first moderator to get a
        token for the room stays its owner.
        """
        query = {"room_name": room_name}
        try:
            doc = await cls.get_collection().find_one_and_update(
                query,
                {"$setOnInsert": {"room_name": room_name, "owner_email": owner_email.lower()}},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError:
            # Lost an upsert race to a concurrent moderator; the winner's record is authoritative.
            doc = await cls.get_collection().find_one(query)
        return cls.from_dict(doc)

    @classmethod
    async def find_by_room(cls, room_name: str) -> "RoomOwner | None":
        return await cls.find_one({"room_name": room_name})

    @classmethod
    async def setup_indexes(cls) -> None:
        await cls.get_collection().create_index("room_name", unique=True)
