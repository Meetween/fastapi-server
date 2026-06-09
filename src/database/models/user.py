from typing import ClassVar, Optional

from pydantic import Field

from config import settings
from src.database.mongo_model import MongoModel


class User(MongoModel):
    """Application user persisted in the `users` collection."""

    email: str = Field(..., description="User's email address (lower-cased)")
    password_hash: str = Field(..., description="Hashed password")
    first_name: Optional[str] = Field(default=None, description="User's first name")
    last_name: Optional[str] = Field(default=None, description="User's last name")
    is_active: bool = Field(default=True, description="Whether the user is active")
    is_admin: bool = Field(default=False, description="Whether the user has admin privileges")

    collection_name: ClassVar[str] = settings.mongo.users_collection

    @classmethod
    async def find_by_email(cls, email: str) -> Optional["User"]:
        return await cls.find_one({"email": email.lower()})

    @classmethod
    async def setup_indexes(cls) -> None:
        await cls.get_collection().create_index("email", unique=True)
