import hashlib
import hmac
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from uuid import UUID, uuid4

from jose import JWTError, jwt
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from config import settings
from src.database.models.meeting_context import MeetingContext
from src.database.models.user import User

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    payload = f"{settings.app_jwt.secret_key}:{password}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def verify_user(email: str, password: str) -> bool:
    user = await User.find_one({"email": email.lower()})
    if not user:
        return False
    return hmac.compare_digest(user.password_hash, hash_password(password))


async def create_user(
    email: str,
    password: str,
    first_name: str | None = None,
    last_name: str | None = None,
) -> User:
    try:
        return await User.create(
            email=email.lower(),
            password_hash=hash_password(password),
            first_name=first_name,
            last_name=last_name,
            is_active=True,
        )
    except DuplicateKeyError as exc:
        raise ValueError("User with this email already exists") from exc


def create_access_token(*, sub: str) -> str:
    aj = settings.app_jwt
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=aj.access_expire_minutes
    )
    to_encode = {"exp": expire, "sub": sub, "aud": aj.audience_access}
    return jwt.encode(
        to_encode, aj.secret_key, algorithm=aj.algorithm
    )


def create_refresh_token(*, sub: str) -> str:
    aj = settings.app_jwt
    expire = datetime.now(timezone.utc) + timedelta(
        days=aj.refresh_expire_days
    )
    jti = str(uuid4())
    to_encode = {
        "exp": expire,
        "sub": sub,
        "jti": jti,
        "aud": aj.audience_refresh,
    }
    return jwt.encode(
        to_encode, aj.secret_key, algorithm=aj.algorithm
    )


def create_jitsi_token(*, sub: str, room: str, moderator: bool = False) -> str:
    aj = settings.app_jwt
    mj = settings.meeting_jwt
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=mj.expire_minutes)
    claims = {
        "iss": mj.issuer,
        "aud": "jitsi",
        "sub": sub,
        "room": room,
        "moderator": moderator,
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(claims, aj.secret_key, algorithm=aj.algorithm)


def verify_jwt_token(token: str, audience: str) -> dict:
    aj = settings.app_jwt
    return jwt.decode(
        token,
        aj.secret_key,
        algorithms=[aj.algorithm],
        audience=audience,
    )


class AccessTokenError(ValueError):
    """Access JWT is missing, invalid, or has no subject."""


def resolve_access_subject(token: str) -> str:
    """Return the email subject from a valid access token."""
    try:
        claims = verify_jwt_token(
            token.strip(),
            audience=settings.app_jwt.audience_access,
        )
    except JWTError as exc:
        raise AccessTokenError("Invalid access token") from exc

    subject = str(claims.get("sub") or "").strip()
    if not subject:
        raise AccessTokenError("Token subject missing")
    return subject


def verify_jitsi_token(token: str) -> dict:
    aj = settings.app_jwt
    mj = settings.meeting_jwt
    return jwt.decode(
        token,
        aj.secret_key,
        algorithms=[aj.algorithm],
        audience="jitsi",
        issuer=mj.issuer,
    )


async def store_user_meeting_context(
    *,
    user_id: str,
    room_name: str,
    meeting_id: UUID | None = None,
    metadata: dict | None = None,
) -> MeetingContext:
    """Atomically upsert one document per (user_id, room_name), matching the unique index on those fields."""
    now = datetime.now(timezone.utc)
    meeting_id_str = str(meeting_id) if meeting_id is not None else None
    doc = await MeetingContext.get_collection().find_one_and_update(
        {"user_id": user_id, "room_name": room_name},
        {
            "$set": {
                "user_id": user_id,
                "room_name": room_name,
                "meeting_id": meeting_id_str,
                "metadata": metadata or {},
                "updated_at": now,
            },
            "$setOnInsert": {"timestamp": now},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return MeetingContext.from_dict(doc)


def build_meet_join_url(*, base_url: str, room: str, jwt_token: str) -> str:
    """Build https://meet.example.com/<room>?jwt=<token> (JWT URL-encoded)."""
    base = base_url.rstrip("/")
    return f"{base}/{quote(room, safe='')}?jwt={quote(jwt_token, safe='')}"


async def get_user_meeting_context(
    user_id: str, room_name: str
) -> MeetingContext | None:
    return await MeetingContext.find_one(
        {"user_id": user_id, "room_name": room_name}
    )
