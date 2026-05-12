import hashlib
import hmac
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from jose import jwt
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from config import settings
from src.database.models.meeting_context import MeetingContext
from src.database.models.user import User

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    payload = f"{settings.jwt_secret_key}:{password}".encode("utf-8")
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


async def store_user_meeting_context(
    *,
    user_id: str,
    meeting_id: str,
    room_name: str | None,
    metadata: dict | None = None,
) -> MeetingContext:
    """Atomically upsert the (user_id, meeting_id) context document."""
    now = datetime.now(timezone.utc)
    doc = await MeetingContext.get_collection().find_one_and_update(
        {"user_id": user_id, "meeting_id": meeting_id},
        {
            "$set": {
                "user_id": user_id,
                "meeting_id": meeting_id,
                "room_name": room_name,
                "metadata": metadata or {},
                "updated_at": now,
            },
            "$setOnInsert": {"timestamp": now},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return MeetingContext.from_dict(doc)


async def get_user_meeting_context(
    user_id: str, meeting_id: str
) -> MeetingContext | None:
    return await MeetingContext.find_one(
        {"user_id": user_id, "meeting_id": meeting_id}
    )
