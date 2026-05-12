"""
Generic base class for MongoDB-backed Pydantic models.

Each concrete model declares `collection_name` and may override
`setup_indexes()` to register collection indexes at app startup.
"""
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pydantic import BaseModel, ConfigDict, Field

from src.database.mongo import get_collection, get_database

T = TypeVar("T", bound="MongoModel")


class MongoModel(BaseModel):
    """Base model for MongoDB documents with collection and instance operations."""

    id: Optional[str] = Field(default=None, description="MongoDB document ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Time of creation",
    )

    model_config = ConfigDict(populate_by_name=True)

    collection_name: ClassVar[str] = "documents"

    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

    @classmethod
    def get_collection(cls) -> AsyncIOMotorCollection:
        return get_collection(cls.collection_name)

    @classmethod
    async def get_database(cls) -> AsyncIOMotorDatabase:
        return await get_database()

    @classmethod
    async def setup_indexes(cls) -> None:
        """Override in subclasses to declare collection indexes."""
        return None

    @classmethod
    async def find_one(cls: Type[T], filter_dict: Dict[str, Any]) -> Optional[T]:
        doc = await cls.get_collection().find_one(filter_dict)
        return cls.from_dict(doc) if doc else None

    @classmethod
    async def find_by_id(cls: Type[T], document_id: str) -> Optional[T]:
        try:
            return await cls.find_one({"_id": ObjectId(document_id)})
        except Exception as exc:
            cls.logger.error("Error finding document by ID %s: %s", document_id, exc)
            return None

    @classmethod
    async def find_all(
        cls: Type[T],
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort: Optional[List[tuple]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[T]:
        cursor = cls.get_collection().find(filter_dict or {})
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        documents = await cursor.to_list(length=limit if limit is not None else None)
        return [cls.from_dict(doc) for doc in documents]

    @classmethod
    async def create(cls: Type[T], **kwargs: Any) -> T:
        doc = cls(**kwargs)
        return await doc.save()

    @classmethod
    async def count_documents(
        cls, filter_dict: Optional[Dict[str, Any]] = None
    ) -> int:
        return await cls.get_collection().count_documents(filter_dict or {})

    @classmethod
    async def delete_many(cls, filter_dict: Dict[str, Any]) -> int:
        result = await cls.get_collection().delete_many(filter_dict)
        return result.deleted_count

    async def save(self: T) -> T:
        """Insert if no id, otherwise replace by id.

        `pymongo.errors.DuplicateKeyError` is propagated so callers can map it
        to a domain-specific error with a meaningful message.
        """
        collection = self.__class__.get_collection()
        doc_dict = self.to_dict()

        if self.id:
            result = await collection.replace_one(
                {"_id": ObjectId(self.id)}, doc_dict
            )
            if result.matched_count == 0:
                raise ValueError(f"Document with ID {self.id} not found")
        else:
            result = await collection.insert_one(doc_dict)
            self.id = str(result.inserted_id)
        return self

    async def delete(self) -> bool:
        if not self.id:
            raise ValueError("Cannot delete document without ID")
        result = await self.__class__.get_collection().delete_one(
            {"_id": ObjectId(self.id)}
        )
        return result.deleted_count > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to a Mongo-storable dict (drops the application-side id)."""
        doc_dict = self.model_dump(exclude={"id"})
        for key, value in doc_dict.items():
            if isinstance(value, Enum):
                doc_dict[key] = value.value
        return doc_dict

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        payload = dict(data) if data else {}
        if "_id" in payload:
            payload["id"] = str(payload.pop("_id"))
        return cls(**payload)
