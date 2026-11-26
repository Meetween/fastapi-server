import json
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from uuid import UUID, uuid4

from bson import ObjectId
from jose import ExpiredSignatureError, JWTError, jwt
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from api.errors import AgentarError
from config import settings
from src.database.models.meeting_context import MeetingContext
from src.database.models.user import User
from src.schemas.names import person_name
from utils import clock, passwords

logger = logging.getLogger(__name__)

# RFC 6750 section 3: a 401 carries WWW-Authenticate so clients can tell an
# expired token, which a refresh repairs, from a missing or forged one.
CHALLENGE_MISSING = {"WWW-Authenticate": "Bearer"}
CHALLENGE_INVALID = {"WWW-Authenticate": 'Bearer error="invalid_token"'}
CHALLENGE_EXPIRED = {
    "WWW-Authenticate": (
        'Bearer error="invalid_token", '
        'error_description="The access token expired"'
    )
}


def hash_password(password: str) -> str:
    """argon2id PHC string. Blocking; request paths use `passwords.hash_password_async`."""
    return passwords.hash_password(password)


async def verify_user(email: str, password: str) -> bool:
    """True when the account exists, is active, and the password matches.

    Every outcome costs one argon2id verification: an unknown email is checked against a dummy
    hash, so the answer time does not tell whether the account exists. A legacy or outdated hash
    is replaced after a successful check by a conditional update on the old value, so two
    concurrent logins cannot overwrite each other.
    """
    user = await User.find_by_email(email)
    if user is None:
        await passwords.burn_dummy_async(password)
        return False

    stored = user.password_hash
    result = await passwords.verify_password_async(stored, password)
    if not result.ok or not user.is_active:
        return False

    if result.new_hash is not None:
        await User.get_collection().update_one(
            {"_id": ObjectId(user.id), "password_hash": stored},
            {"$set": {"password_hash": result.new_hash}},
        )
        logger.info("Password hash of user %s upgraded to argon2id", user.id)
    return True


async def create_user(
    email: str,
    password: str,
    first_name: str | None = None,
    last_name: str | None = None,
    *,
    is_admin: bool = False,
) -> User:
    password_hash = await passwords.hash_password_async(password)
    try:
        return await User.create(
            email=email.lower(),
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            is_admin=is_admin,
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


# Jitsi room tokens. Prosody verifies them with JWT_APP_SECRET, which holds the same value as
# JWT_JITSI_SECRET_KEY. HS256 is fixed here, apart from JWT_ALGORITHM, because it is what Prosody
# is configured to accept; changing the API token algorithm must not break room joins.
JITSI_AUDIENCE = "jitsi"
JITSI_ALGORITHM = "HS256"
# Subject of a virtual participant's room token. A person's token carries the XMPP domain.
JITSI_BOT_SUBJECT = "vp-bot"


def jitsi_signing_key() -> str:
    """The Jitsi key, or 503 `jitsi_key_missing` while it is unset.

    Never falls back to JWT_SECRET_KEY: that key signs API tokens and must not leave fastapi.
    """
    key = settings.jwt_jitsi_secret_key
    if not key:
        logger.error("JWT_JITSI_SECRET_KEY is not set; Jitsi room tokens are disabled")
        raise AgentarError(503, "jitsi_key_missing", "Jitsi room tokens are not configured")
    return key


def _jitsi_claims(*, sub: str, room: str, lifetime: timedelta) -> dict:
    now = clock.utcnow()
    return {
        "iss": settings.meeting_jwt.issuer,
        "aud": JITSI_AUDIENCE,
        "sub": sub,
        "room": room,
        "jti": str(uuid4()),
        "iat": now,
        "exp": now + lifetime,
    }


def jitsi_user_name(first_name: str | None, last_name: str | None) -> str | None:
    """Name for `context.user.name`, or None to leave the name the person types in Jitsi.

    A name that breaks the person name rule (an account created before the rule existed) is
    left out rather than shown, so it cannot fake the "(AI)" disclosure of a bot.
    """
    name = " ".join(part.strip() for part in (first_name, last_name) if part and part.strip())
    if not name:
        return None
    try:
        return person_name(name)
    except ValueError:
        return None


def create_jitsi_token(*, room: str, user_id: str, name: str | None, moderator: bool) -> str:
    """Room token for a person, valid JWT_JITSI_EXPIRE_MINUTES (15) and only for ``room``.

    ``user_id`` is the account id, never the email: Prosody logs a token it refuses, so no claim
    carries personal data beyond the display name. 503 while the Jitsi key is unset.
    """
    key = jitsi_signing_key()
    claims = _jitsi_claims(
        sub=settings.virtual_participants.xmpp_domain,
        room=room,
        lifetime=timedelta(minutes=settings.meeting_jwt.expire_minutes),
    )
    user: dict = {"id": user_id, "moderator": moderator}
    if name:
        user["name"] = name
    claims["context"] = {"user": user}
    claims["moderator"] = moderator
    return jwt.encode(claims, key, algorithm=JITSI_ALGORITHM)


def create_jitsi_bot_token(room: str) -> str:
    """Room token for a virtual participant, valid as long as the worker keeps a bot alive.

    It has no `context.user`: a name claim would replace the worker's display name, which ends
    with the "(AI)" disclosure and keeps the bot's own speech out of the transcript.
    503 while the Jitsi key is unset.
    """
    key = jitsi_signing_key()
    claims = _jitsi_claims(
        sub=JITSI_BOT_SUBJECT,
        room=room,
        lifetime=timedelta(seconds=settings.vp_max_lifetime_s),
    )
    return jwt.encode(claims, key, algorithm=JITSI_ALGORITHM)


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


class AccessTokenExpiredError(AccessTokenError):
    """Access JWT was valid but its exp claim has passed. Clients should refresh."""


class RefreshTokenError(ValueError):
    """Refresh JWT is missing, invalid, or has no subject."""


def resolve_access_subject(token: str) -> str:
    """Return the email subject from a valid access token."""
    try:
        claims = verify_jwt_token(
            token.strip(),
            audience=settings.app_jwt.audience_access,
        )
    except ExpiredSignatureError as exc:
        raise AccessTokenExpiredError("Access token expired") from exc
    except JWTError as exc:
        raise AccessTokenError("Invalid access token") from exc

    subject = str(claims.get("sub") or "").strip()
    if not subject:
        raise AccessTokenError("Token subject missing")
    return subject


def resolve_refresh_subject(token: str) -> str:
    """Return the email subject from a valid refresh token."""
    try:
        claims = verify_jwt_token(
            token.strip(),
            audience=settings.app_jwt.audience_refresh,
        )
    except ExpiredSignatureError as exc:
        raise RefreshTokenError("Refresh token expired") from exc
    except JWTError as exc:
        raise RefreshTokenError("Invalid refresh token") from exc

    subject = str(claims.get("sub") or "").strip()
    if not subject:
        raise RefreshTokenError("Token subject missing")
    return subject


def verify_jitsi_token(token: str) -> dict:
    """Decode a room token as Prosody does: Jitsi key, audience, issuer, expiry."""
    return jwt.decode(
        token,
        jitsi_signing_key(),
        algorithms=[JITSI_ALGORITHM],
        audience=JITSI_AUDIENCE,
        issuer=settings.meeting_jwt.issuer,
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


def build_meet_join_url(*, base_url: str, room: str, jwt_token: str | None) -> str:
    """``<base>/<room>#jwt=%22<token>%22`` for a token, ``<base>/<room>`` without one.

    The token travels in the fragment, which the browser never sends to a server or in a
    Referer. Jitsi parses fragment values as JSON, so the token is a quoted JSON string.
    """
    url = f"{base_url.rstrip('/')}/{quote(room, safe='')}"
    if jwt_token is None:
        return url
    return f"{url}#jwt={quote(json.dumps(jwt_token), safe='')}"


async def get_user_meeting_context(
    user_id: str, room_name: str
) -> MeetingContext | None:
    return await MeetingContext.find_one(
        {"user_id": user_id, "room_name": room_name}
    )
