from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, Field

from src.database.models.agentar_types import UtcDatetime, now_field
from src.database.mongo_model import MongoModel


class FileChunk(BaseModel):
    """About 150 tokens of extracted text. Chunks never leave the API."""

    n: int
    text: str
    page: Optional[int] = None


class AgentarFile(MongoModel):
    """Knowledge base file or picture record. Bytes live in the object store at `storage_key`."""

    owner: str = Field(..., description="Token subject, the lower-cased email of the owner")
    scope: Literal["general", "meeting"] = "general"
    meeting_id: Optional[str] = None
    kind: Literal["document", "picture"] = "document"
    name: str = Field(default="", description="Client filename, display only, never part of a key")
    size: int = 0
    content_type: str = ""
    sha256: str = ""
    storage_key: str = Field(default="", description="Key relative to the bucket, built from users._id")
    text_status: Optional[Literal["ok", "empty", "truncated", "failed"]] = None
    text_chars: int = 0
    chunks: list[FileChunk] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: UtcDatetime = Field(default_factory=now_field)

    collection_name: ClassVar[str] = "agentar_files"

    @classmethod
    async def setup_indexes(cls) -> None:
        await cls.get_collection().create_index(
            [("owner", 1), ("scope", 1), ("meeting_id", 1), ("created_at", -1)]
        )
