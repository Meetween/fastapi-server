import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from api.deps.auth import REQUIRE_ADMIN, get_current_subject, get_current_user
from api.errors import error_responses
from config import settings
from src.database.models.user import User
from src.schemas.agentar import PlainError
from src.schemas.auth import (
    UserCreateRequest,
    UserCreateResponse,
    LoginRequest,
    RefreshTokenRequest,
    TokenPairResponse,
    JitsiIssueResponse,
    JitsiIssueRequest,
)
from utils.auth import (
    create_user,
    verify_user,
    create_access_token,
    create_refresh_token,
    RefreshTokenError,
    resolve_refresh_subject,
    create_jitsi_token,
    jitsi_signing_key,
    jitsi_user_name,
    store_user_meeting_context,
    get_user_meeting_context,
    build_meet_join_url,
)
from utils.rate_limit import RateLimiter, client_ip, get_rate_limiter, longest_wait, too_many_requests
from utils.room_owner import claim_room_moderator

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


@public_router.post(
    "/login",
    response_model=TokenPairResponse,
    responses={401: {"model": PlainError, "description": "Invalid credentials"}, **error_responses(429)},
)
async def login(
    credentials: LoginRequest,
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
):
    """Failed logins are throttled before any hashing: `LOGIN_FAIL_LIMIT` per window for one
    client address and email, `LOGIN_FAIL_IP_LIMIT` per address over all emails. A success
    clears the address and email counter. No permanent lockout: the window slides."""
    email = credentials.email.strip().lower()
    cfg = settings.agentar
    window = cfg.login_fail_window_minutes * 60
    ip = client_ip(request)
    pair_key = f"login:fail:{ip}:{email}"
    ip_key = f"login:fail:{ip}"

    wait = longest_wait(
        limiter,
        [(pair_key, cfg.login_fail_limit, window), (ip_key, cfg.login_fail_ip_limit, window)],
    )
    if wait is not None:
        logger.warning("Login throttled for client %s", ip)
        raise too_many_requests(wait)

    if not await verify_user(email, credentials.password):
        limiter.hit(pair_key, window)
        limiter.hit(ip_key, window)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    limiter.reset(pair_key)
    return TokenPairResponse(
        access_token=create_access_token(sub=email),
        refresh_token=create_refresh_token(sub=email),
    )


@public_router.post("/refresh", response_model=TokenPairResponse)
async def refresh_tokens(request: RefreshTokenRequest):
    try:
        subject = resolve_refresh_subject(request.refresh_token)
    except RefreshTokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user = await User.find_by_email(subject)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    return TokenPairResponse(
        access_token=create_access_token(sub=subject),
        refresh_token=create_refresh_token(sub=subject),
    )


@protected_router.post(
    "/jitsi/issue",
    response_model=JitsiIssueResponse,
    responses=error_responses(401, 429, 503),
)
async def issue_jitsi_token(
    request: JitsiIssueRequest,
    user: User = Depends(get_current_user),
):
    """Join data for a room. The first user who claims a room owns it; the owner and admins get a
    moderator token, everyone else a plain URL to join as a guest."""
    # Checked before any write, so a server without the Jitsi key claims no room.
    jitsi_signing_key()
    if not user.is_active:
        raise HTTPException(status_code=401, detail="User not found")

    room = request.room_name
    # The request's moderator field is ignored: the room owner and admins moderate.
    moderator = await claim_room_moderator(room, user)
    context_record = await store_user_meeting_context(
        user_id=user.email,
        meeting_id=request.meeting_id,
        room_name=room,
        metadata=request.metadata,
    )
    jitsi_token = (
        create_jitsi_token(
            room=room,
            user_id=user.id,
            name=jitsi_user_name(user.first_name, user.last_name),
            moderator=True,
        )
        if moderator
        else None
    )
    mj = settings.meeting_jwt
    meet_join_url = (
        build_meet_join_url(base_url=mj.meet_base_url, room=room, jwt_token=jitsi_token)
        if mj.meet_base_url
        else None
    )
    logger.info("jitsi issue room=%s moderator=%s", room, moderator)
    return JitsiIssueResponse(
        room=room,
        jitsi_token=jitsi_token,
        moderator=moderator,
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
