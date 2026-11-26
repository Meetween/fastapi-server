"""Pydantic request/response schemas for the virtual participants router."""
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

from config import settings
from src.schemas.names import bot_display_name

# Shared with vp-worker. Jitsi room segments are lowercase.
ROOM_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,62}$"
BOT_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
# vp-worker appends its VP_BOT_NAME_SUFFIX to every bot display name. It is the "(AI)" disclosure
# on the Agentar stacks, and it is how a bot is told apart from a human in the room. The same
# environment value is read here; unset, the Agentar default applies.
DEFAULT_BOT_NAME_SUFFIX = " (AI)"
BOT_NAME_SUFFIX = settings.vp_bot_name_suffix if settings.vp_bot_name_suffix.strip() else DEFAULT_BOT_NAME_SUFFIX

BotState = Literal["joining", "joined", "speaking", "leaving", "left", "error", "lost"]


class VirtualParticipantCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_name: str = Field(
        ...,
        pattern=ROOM_PATTERN,
        description="Room segment of the Jitsi join URL. Lowercased before validation.",
    )
    display_name: Annotated[str, AfterValidator(bot_display_name)] = Field(
        default="Agentar",
        description=(
            "Name shown in the meeting: Unicode letters and marks, digits, space and `_ . ' -`, "
            "1 to 48 characters, no parentheses, so only the worker suffix can add the "
            f"'{BOT_NAME_SUFFIX.strip()}' disclosure."
        ),
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
