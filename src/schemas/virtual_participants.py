"""Pydantic request/response schemas for the virtual participants router."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Shared with vp-worker. Jitsi room segments are lowercase.
ROOM_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,62}$"
NAME_PATTERN = r"^[A-Za-z0-9 _.()-]{1,48}$"
BOT_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
# vp-worker appends this suffix to every bot display name.
BOT_NAME_SUFFIX = " (bot)"

BotState = Literal["joining", "joined", "speaking", "leaving", "left", "error", "lost"]


class VirtualParticipantCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_name: str = Field(
        ...,
        pattern=ROOM_PATTERN,
        description="Room segment of the Jitsi join URL. Lowercased before validation.",
    )
    display_name: str = Field(
        default="Agentar",
        pattern=NAME_PATTERN,
        description=f"Name shown in the meeting. The worker appends '{BOT_NAME_SUFFIX}'.",
    )
    script: str | None = Field(default=None, max_length=2000)
    meeting_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "Jitsi meeting id, the one Jigasi reports as ended. Optional. When given and the "
            "meeting has ended the request is refused with 409. A room name cannot stand in "
            "for it: rooms are reused, the meeting id is not."
        ),
    )

    @field_validator("room_name", mode="before")
    @classmethod
    def lowercase_room_name(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value


class VirtualParticipantSayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=2000)


class VirtualParticipantSpeech(BaseModel):
    model_config = ConfigDict(extra="allow")

    state: Literal["idle", "queued", "rendering", "playing"] = "idle"
    queue_position: int | None = None
    last: dict[str, Any] | None = None


class VirtualParticipantView(BaseModel):
    """Bot view as reported by vp-worker, relayed unchanged."""

    model_config = ConfigDict(extra="allow")

    id: str
    room: str
    display_name: str
    state: BotState
    replica_id: str
    boot_id: str
    created_at: float
    joined_at: float | None = None
    left_at: float | None = None
    error: str | None = None
    humans: int | None = None
    speech: VirtualParticipantSpeech
    media: dict[str, Any] | None = None


class VirtualParticipantError(BaseModel):
    detail: Any
