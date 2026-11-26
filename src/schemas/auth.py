"""Pydantic request/response schemas for the auth router."""
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, field_validator

from src.schemas.names import person_name
from src.schemas.virtual_participants import ROOM_PATTERN


class UserCreateRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=6)
    first_name: str | None = None
    last_name: str | None = None


class UserCreateResponse(BaseModel):
    email: str
    is_active: bool


class LoginRequest(BaseModel):
    # Caps keep a hostile body out of the hasher and the lookup.
    email: str = Field(..., min_length=1, max_length=254)
    password: str = Field(..., min_length=1, max_length=128)


class TokenPairResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class JitsiIssueResponse(BaseModel):
    """Join data for a room. Only the room owner and admins get a token; anyone else joins as a guest."""

    room: str
    jitsi_token: str | None = Field(
        ..., description="Room token for the owner and admins; null means join as a guest."
    )
    moderator: bool
    meeting_context: dict[str, Any]
    meet_join_url: str | None = Field(
        ...,
        description=(
            "`<JITSI_MEET_BASE_URL>/<room>#jwt=%22<token>%22` for the owner, `<base>/<room>` "
            "otherwise, null when the base URL is unset."
        ),
    )


class JitsiIssueRequest(BaseModel):
    """`room_name` is the `room` claim. Context is upserted per (user, room); `meeting_id` is optional app metadata."""

    room_name: str = Field(
        ...,
        pattern=ROOM_PATTERN,
        description="Room segment of the Jitsi join URL. Lower-cased before validation.",
    )
    meeting_id: UUID | None = Field(
        default=None,
        description="Optional app meeting id (UUID), stored on the context document only (not sent in the Jitsi JWT).",
    )
    moderator: bool = Field(
        default=False,
        deprecated=True,
        description=(
            "Deprecated and ignored. The server sets the moderator claim: true for the room owner "
            "(the first user who claimed the room) and for admins."
        ),
    )
    metadata: dict[str, Any] | None = None

    @field_validator("room_name", mode="before")
    @classmethod
    def _lowercase_room_name(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value


class AuthConfig(BaseModel):
    """Public feature flags the login panel reads before it renders (GET /api/auth/config)."""

    model_config = ConfigDict(extra="forbid")

    signup_enabled: bool
    email_enabled: bool
    calendar_enabled: bool


class SignupRequest(BaseModel):
    """Open sign up. The email is stored lower-cased."""

    model_config = ConfigDict(extra="forbid")

    email: Annotated[str, StringConstraints(max_length=254, strip_whitespace=True)]
    password: str = Field(..., min_length=10, max_length=128)
    first_name: Annotated[
        str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True), AfterValidator(person_name)
    ]
    last_name: Annotated[
        str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True), AfterValidator(person_name)
    ]

    @field_validator("email")
    @classmethod
    def _plausible_email(cls, value: str) -> str:
        local, sep, domain = value.partition("@")
        if not sep or not local or "." not in domain or " " in value or "@" in domain:
            raise ValueError("not a valid email address")
        return value.lower()
