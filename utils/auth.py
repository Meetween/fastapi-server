import hashlib
import hmac
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from jose import jwt

from config import settings

logger = logging.getLogger(__name__)

_users: dict[str, dict] = {}
_meeting_context: dict[str, dict] = {}


def hash_password(password: str) -> str:
    payload = f"{settings.jwt_secret_key}:{password}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def verify_user(email: str, password: str) -> bool:
    user = _users.get(email.lower())
    if not user:
        return False
    return hmac.compare_digest(user["password_hash"], hash_password(password))


async def create_user(
    email: str,
    password: str,
    first_name: str | None = None,
    last_name: str | None = None,
) -> dict:
    key = email.lower()
    existing = _users.get(key)
    if existing:
        raise ValueError("User with this email already exists")

    user = {
        "email": key,
        "password_hash": hash_password(password),
        "first_name": first_name,
        "last_name": last_name,
        "is_active": True,
    }
    _users[key] = user
    return user


def create_access_token(*, sub: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.jwt_access_token_expire_minutes
    )
    to_encode = {"exp": expire, "sub": sub, "aud": settings.jwt_audience_access}
    return jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def create_refresh_token(*, sub: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.jwt_refresh_token_expire_days
    )
    jti = str(uuid4())
    to_encode = {
        "exp": expire,
        "sub": sub,
        "jti": jti,
        "aud": settings.jwt_audience_refresh,
    }
    return jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def create_jitsi_token(*, sub: str, room: str, moderator: bool = False) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.jwt_access_token_expire_minutes
    )
    claims = {
        "aud": "jitsi",
        "sub": sub,
        "room": room,
        "moderator": moderator,
        "exp": expire,
    }
    return jwt.encode(claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def verify_jwt_token(token: str, audience: str) -> dict:
    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
        audience=audience,
    )


def verify_jitsi_token(token: str) -> dict:
    return verify_jwt_token(token=token, audience="jitsi")


def store_user_meeting_context(
    *,
    user_id: str,
    meeting_id: str,
    room_name: str | None,
    metadata: dict | None = None,
) -> dict:
    key = f"{user_id}:{meeting_id}"
    record = {
        "user_id": user_id,
        "meeting_id": meeting_id,
        "room_name": room_name,
        "metadata": metadata or {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _meeting_context[key] = record
    return record


def get_user_meeting_context(user_id: str, meeting_id: str) -> dict | None:
    return _meeting_context.get(f"{user_id}:{meeting_id}")
