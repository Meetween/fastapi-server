from typing import Any, ClassVar, Dict, Optional

from pydantic import BaseModel, Field

from config import settings
from src.database.models.agentar_types import UtcDatetime
from src.database.mongo_model import MongoModel


class UserProfile(BaseModel):
    """The user's Agentar profile, embedded in the user document.

    `MongoModel.save()` replaces the whole document and drops undeclared fields, so every profile
    field is declared here. Routes change the profile with `$set` on `profile.<field>`, never
    with `save()` of a stale copy.
    """

    agent_name: Optional[str] = None
    voice: Optional[str] = None
    language: str = "en"
    picture_id: Optional[str] = Field(default=None, description="Id of the picture record in agentar_files")
    picture_sha256: Optional[str] = None
    onboarded_at: Optional[UtcDatetime] = None
    onboarding_skipped_at: Optional[UtcDatetime] = None


class User(MongoModel):
    """Application user persisted in the `users` collection."""

    email: str = Field(..., description="User's email address (lower-cased)")
    password_hash: str = Field(..., description="Hashed password")
    first_name: Optional[str] = Field(default=None, description="User's first name")
    last_name: Optional[str] = Field(default=None, description="User's last name")
    is_active: bool = Field(default=True, description="Whether the user is active")
    is_admin: bool = Field(default=False, description="Whether the user has admin privileges")
    profile: Optional[UserProfile] = Field(
        default=None, description="Agentar profile, absent until the user first saves it"
    )

    collection_name: ClassVar[str] = settings.mongo.users_collection

    def to_dict(self) -> Dict[str, Any]:
        """Leave `profile` out while it is unset.

        A stored `profile: null` makes MongoDB refuse `$set` on `profile.<field>` ("Cannot create
        field in element {profile: null}"). An absent key lets the first `$set` create the object.
        """
        doc = super().to_dict()
        if doc.get("profile") is None:
            doc.pop("profile", None)
        return doc

    @classmethod
    async def find_by_email(cls, email: str) -> Optional["User"]:
        return await cls.find_one({"email": email.lower()})

    @classmethod
    async def setup_indexes(cls) -> None:
        await cls.get_collection().create_index("email", unique=True)
