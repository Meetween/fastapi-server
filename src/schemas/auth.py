"""Pydantic request/response schemas for the auth router."""
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class UserCreateRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=6)
    first_name: str | None = None
    last_name: str | None = None


class UserCreateResponse(BaseModel):
    email: str
    is_active: bool


class LoginRequest(BaseModel):
    email: str
    password: str


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
