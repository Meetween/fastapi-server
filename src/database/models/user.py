from typing import ClassVar, Literal, Optional

from pydantic import Field

from config import settings
from src.database.mongo_model import MongoModel

UserRole = Literal["user", "admin"]


class User(MongoModel):
    """Application user persisted in the `users` collection."""

    email: str = Field(..., description="User's email address (lower-cased)")
    password_hash: str = Field(..., description="Hashed password")
    first_name: Optional[str] = Field(default=None, description="User's first name")
    last_name: Optional[str] = Field(default=None, description="User's last name")
    is_active: bool = Field(default=True, description="Whether the user is active")
    role: UserRole = Field(default="user", description="Application role")

    collection_name: ClassVar[str] = settings.mongo.users_collection

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @classmethod
    async def find_by_email(cls, email: str) -> Optional["User"]:
        return await cls.find_one({"email": email.lower()})

    @classmethod
    async def setup_indexes(cls) -> None:
        await cls.get_collection().create_index("email", unique=True)
