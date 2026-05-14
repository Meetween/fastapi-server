"""
Async MongoDB client lifecycle.

The `AsyncMongoDBManager` is constructed once at app startup via `connect()`
and disposed in `close()` during shutdown. Models and services read the
database through `get_database()` / `get_collection()` so they never reference
the underlying Motor client directly.
"""
import logging
from typing import Optional

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

from config import settings

logger = logging.getLogger(__name__)


def _create_client(uri: str) -> AsyncIOMotorClient:
    """Factory for the underlying Motor client. Patched in tests."""
    return AsyncIOMotorClient(uri)


class AsyncMongoDBManager:
    def __init__(self) -> None:
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None

    async def connect(
        self,
        connection_string: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> AsyncIOMotorDatabase:
        """Connect to MongoDB and return the database instance."""
        if self.database is not None:
            return self.database

        uri = connection_string or settings.mongo.uri
        name = db_name or settings.mongo.db_name

        try:
            self.client = _create_client(uri)
            await self.client.admin.command("ping")
            self.database = self.client[name]
        except Exception:
            logger.exception("Failed to connect to MongoDB")
            raise

        await _setup_indexes()
        logger.info("Connected to MongoDB database '%s'", name)
        return self.database

    def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        if self.database is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.database[collection_name]

    async def close(self) -> None:
        if self.client is None:
            return
        self.client.close()
        self.client = None
        self.database = None
        logger.info("MongoDB connection closed")


async_mongo_manager = AsyncMongoDBManager()


async def get_database() -> AsyncIOMotorDatabase:
    if async_mongo_manager.database is None:
        await async_mongo_manager.connect()
    return async_mongo_manager.database


def get_collection(collection_name: str) -> AsyncIOMotorCollection:
    return async_mongo_manager.get_collection(collection_name)


async def _setup_indexes() -> None:
    """Create indexes for every registered MongoModel that defines them.

    Imported lazily to avoid circular imports between mongo.py and models.
    """
    from src.database.models.meeting_context import MeetingContext
    from src.database.models.user import User

    for model in (User, MeetingContext):
        await model.setup_indexes()
