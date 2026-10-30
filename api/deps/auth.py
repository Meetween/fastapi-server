"""FastAPI dependencies for JWT bearer authentication."""

import re
from typing import Literal, NamedTuple

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.deps.service_token import token_matches
from config import settings
from src.database.models.user import User
from utils.auth import (
    AccessTokenError,
    AccessTokenExpiredError,
    resolve_access_subject,
    CHALLENGE_MISSING,
    CHALLENGE_INVALID,
    CHALLENGE_EXPIRED,
)

bearer_security = HTTPBearer(auto_error=False)


def get_current_subject(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_security),
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token",
            headers=CHALLENGE_MISSING,
        )
    try:
        return resolve_access_subject(credentials.credentials)
    except AccessTokenExpiredError as exc:
        raise HTTPException(
            status_code=401, detail=str(exc), headers=CHALLENGE_EXPIRED
        ) from exc
    except AccessTokenError as exc:
        raise HTTPException(
            status_code=401, detail=str(exc), headers=CHALLENGE_INVALID
        ) from exc


class Caller(NamedTuple):
    """Who a request to a user-or-worker route came from."""

    kind: Literal["user", "worker", "anonymous"]
    subject: str | None = None


def require_user_or_vp_worker(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_security),
) -> Caller:
    """Accept a user access token or the virtual participant worker key.

    Anything that is not the worker key goes through get_current_subject, so a missing,
    malformed or expired user token keeps the same 401 body and challenge as every other route.
    """
    if not settings.enable_auth:
        return Caller("anonymous")
    if token_matches(credentials, settings.vp_meetween_api_key):
        return Caller("worker")
    return Caller("user", get_current_subject(credentials))


# Participant ids of virtual participants are vp:<bot id>. The prefix is reserved: the worker may
# only use it and users may not, so neither side can read or write the other's chat history.
VP_PARTICIPANT_PREFIX = "vp:"
_VP_PARTICIPANT_ID = re.compile(r"vp:[A-Za-z0-9_-]{1,64}")


def enforce_participant_namespace(caller: Caller, participant_id: str | None) -> None:
    value = participant_id or ""
    if caller.kind == "worker" and not _VP_PARTICIPANT_ID.fullmatch(value):
        raise HTTPException(
            status_code=403,
            detail="Worker calls need a participant_id of the form vp:<bot id>",
        )
    if caller.kind == "user" and value.startswith(VP_PARTICIPANT_PREFIX):
        raise HTTPException(
            status_code=403,
            detail="The participant_id prefix vp: is reserved for virtual participants",
        )


async def get_current_user(
    current_subject: str = Depends(get_current_subject),
) -> User:
    user = await User.find_by_email(current_subject)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


REQUIRE_AUTH = [Depends(get_current_subject)] if settings.enable_auth else []
REQUIRE_ADMIN = [Depends(require_admin)] if settings.enable_auth else []
