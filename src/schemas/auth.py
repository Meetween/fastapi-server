"""Pydantic request/response schemas for the auth router."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class UserCreateRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=6)
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserCreateResponse(BaseModel):
    email: str
    is_active: bool


class LoginRequest(BaseModel):
    email: str
    password: str


class MeetingContextRequest(BaseModel):
    meeting_id: str
    room_name: Optional[str] = None
    metadata: Optional[dict] = None


class TokenPairResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"


class JitsiTokenResponse(BaseModel):
    jitsi_token: str
    room: str


class JitsiIssueRequest(BaseModel):
    room_name: Optional[str] = None
    moderator: bool = False


class TokenValidationRequest(BaseModel):
    token: str
