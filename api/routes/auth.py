import logging

from fastapi import APIRouter, Depends, HTTPException
from jose import JWTError

from api.deps.auth import REQUIRE_ADMIN, get_current_subject
from config import settings
from src.schemas.auth import (
    UserCreateRequest,
    UserCreateResponse,
    LoginRequest,
    TokenPairResponse,
    JitsiTokenResponse,
    JitsiIssueRequest,
    JitsiValidationResponse,
    TokenValidationRequest,
)
from utils.auth import (
    create_user,
    verify_user,
    create_access_token,
    create_refresh_token,
    create_jitsi_token,
    verify_jitsi_token,
    store_user_meeting_context,
    get_user_meeting_context,
    build_meet_join_url,
)

logger = logging.getLogger(__name__)

public_router = APIRouter(prefix="/api/auth", tags=["auth"])
protected_router = APIRouter(prefix="/api/auth", tags=["auth"])


@protected_router.post(
    "/users",
    response_model=UserCreateResponse,
    dependencies=REQUIRE_ADMIN,
)
async def register_user(request: UserCreateRequest):
    try:
        user = await create_user(
            email=request.email,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name,
        )
        return UserCreateResponse(email=user.email, is_active=user.is_active)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@public_router.post("/login", response_model=TokenPairResponse)
async def login(request: LoginRequest):
    is_valid = await verify_user(request.email, request.password)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenPairResponse(
        access_token=create_access_token(sub=request.email.lower()),
        refresh_token=create_refresh_token(sub=request.email.lower()),
    )


@public_router.post("/jitsi/validate", response_model=JitsiValidationResponse)
async def validate_jitsi(request: TokenValidationRequest):
    try:
        claims = verify_jitsi_token(request.token)
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid Jitsi token") from exc
    return JitsiValidationResponse(claims=claims)


@protected_router.post("/jitsi/issue", response_model=JitsiTokenResponse)
async def issue_jitsi_token(
    request: JitsiIssueRequest,
    current_subject: str = Depends(get_current_subject),
):
    room = request.room_name
    meeting_id = request.meeting_id
    context_record = await store_user_meeting_context(
        user_id=current_subject,
        meeting_id=meeting_id,
        room_name=room,
        metadata=request.metadata,
    )
    jitsi_token = create_jitsi_token(
        sub=current_subject,
        room=room,
        moderator=request.moderator,
    )
    mj = settings.meeting_jwt
    meet_join_url = (
        build_meet_join_url(
            base_url=mj.meet_base_url,
            room=room,
            jwt_token=jitsi_token,
        )
        if mj.meet_base_url
        else None
    )
    return JitsiTokenResponse(
        jitsi_token=jitsi_token,
        room=room,
        meeting_context=context_record.model_dump(mode="json"),
        meet_join_url=meet_join_url,
    )


@protected_router.get("/context/{room_name}")
async def get_context(
    room_name: str,
    current_subject: str = Depends(get_current_subject),
):
    """Return stored join context for this user and Jitsi room (same key as /jitsi/issue)."""
    record = await get_user_meeting_context(current_subject, room_name)
    if record is None:
        raise HTTPException(status_code=404, detail="Meeting context not found")
    return record
