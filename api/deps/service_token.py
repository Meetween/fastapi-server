"""Shared-secret authentication for service callers."""

import hmac

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from utils.auth import CHALLENGE_MISSING, CHALLENGE_INVALID
from config import settings

_bearer = HTTPBearer(auto_error=False)


def token_matches(
    credentials: HTTPAuthorizationCredentials | None, secret: str
) -> bool:
    """Constant time check of the presented bearer against one configured secret.

    An empty secret never matches, so a service whose key is unset is rejected, not open.
    """
    if not secret or credentials is None:
        return False
    # str compare_digest raises TypeError on non-ASCII; header text can carry it (latin-1 decoded).
    return hmac.compare_digest(credentials.credentials.encode(), secret.encode())


def require_service_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Accept the configured service token when auth is on. Otherwise allow the call."""
    if not settings.enable_auth:
        return

    if token_matches(credentials, settings.jigasi_meetween_api_key):
        return

    missing = credentials is None
    raise HTTPException(
        status_code=401,
        detail="Missing bearer token" if missing else "Invalid bearer token",
        headers=CHALLENGE_MISSING if missing else CHALLENGE_INVALID,
    )


REQUIRE_SERVICE_TOKEN = [Depends(require_service_token)] if settings.enable_auth else []
