"""Pydantic request/response schemas for the auth router."""
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, field_validator

from src.schemas.names import person_name


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


class JitsiTokenResponse(BaseModel):
    jitsi_token: str
    room: str
    meeting_context: dict[str, Any]
    meet_join_url: str | None = None


class JitsiValidationResponse(BaseModel):
    valid: Literal[True] = True
    claims: dict[str, Any]


class JitsiIssueRequest(BaseModel):
    """Jitsi JWT uses `room_name` for the `room` claim. Context is upserted per (user, room); `meeting_id` is optional app metadata."""

    room_name: str = Field(..., min_length=1, description="Must match the room segment in the Jitsi join URL.")
    meeting_id: UUID | None = Field(
        default=None,
        description="Optional app meeting id (UUID), stored on the context document only (not sent in the Jitsi JWT).",
    )
    moderator: bool = False
    metadata: dict[str, Any] | None = None


class TokenValidationRequest(BaseModel):
    token: str


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
