"""Shared-secret authentication for service callers."""

import hmac

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from utils.auth import CHALLENGE_MISSING, CHALLENGE_INVALID
from config import settings

_bearer = HTTPBearer(auto_error=False)


def require_service_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Accept the configured service token when auth is on. Otherwise allow the call."""
    if not settings.enable_auth:
        return

    secret = settings.jigasi_meetween_api_key
    presented = credentials.credentials if credentials is not None else ""
    if secret and hmac.compare_digest(presented, secret):
        return

    missing = credentials is None
    raise HTTPException(
        status_code=401,
        detail="Missing bearer token" if missing else "Invalid bearer token",
        headers=CHALLENGE_MISSING if missing else CHALLENGE_INVALID,
    )


REQUIRE_SERVICE_TOKEN = [Depends(require_service_token)] if settings.enable_auth else []
