import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from config import settings
from src.schemas.auth import (
    UserCreateRequest,
    UserCreateResponse,
    LoginRequest,
    MeetingContextRequest,
    TokenPairResponse,
    JitsiTokenResponse,
    JitsiIssueRequest,
    TokenValidationRequest,
)
from utils.auth import (
    create_user,
    verify_user,
    create_access_token,
    create_refresh_token,
    create_jitsi_token,
    verify_jwt_token,
    verify_jitsi_token,
    store_user_meeting_context,
    get_user_meeting_context,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger(__name__)
bearer_security = HTTPBearer(auto_error=False)


def _get_current_subject(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_security),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        claims = verify_jwt_token(
            token=credentials.credentials,
            audience=settings.jwt_audience_access,
        )
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid access token") from exc

    subject = str(claims.get("sub") or "").strip()
    if not subject:
        raise HTTPException(status_code=401, detail="Token subject missing")
    return subject


@router.post("/users", response_model=UserCreateResponse)
async def register_user(request: UserCreateRequest):
    try:
        return await create_user(
            email=request.email,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/login", response_model=TokenPairResponse)
async def login(request: LoginRequest):
    is_valid = await verify_user(request.email, request.password)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenPairResponse(
        access_token=create_access_token(sub=request.email.lower()),
        refresh_token=create_refresh_token(sub=request.email.lower()),
    )


@router.post("/context")
async def upsert_context(
    request: MeetingContextRequest,
    current_subject: str = Depends(_get_current_subject),
):
    context_record = await store_user_meeting_context(
        user_id=current_subject,
        meeting_id=request.meeting_id,
        room_name=request.room_name,
        metadata=request.metadata,
    )
    return {
        "meeting_context": context_record,
    }


@router.post("/jitsi/validate")
async def validate_jitsi(request: TokenValidationRequest):
    try:
        claims = verify_jitsi_token(request.token)
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid Jitsi token") from exc
    return {"valid": True, "claims": claims}


@router.post("/jitsi/issue", response_model=JitsiTokenResponse)
async def issue_jitsi_token(
    request: JitsiIssueRequest,
    current_subject: str = Depends(_get_current_subject),
):
    room = request.room_name or "*"
    return JitsiTokenResponse(
        jitsi_token=create_jitsi_token(
            sub=current_subject,
            room=room,
            moderator=request.moderator,
        ),
        room=room,
    )


@router.get("/context/{meeting_id}")
async def get_context(
    meeting_id: str,
    current_subject: str = Depends(_get_current_subject),
):
    record = await get_user_meeting_context(user_id=current_subject, meeting_id=meeting_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Meeting context not found")
    return record
