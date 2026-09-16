"""FastAPI dependencies for JWT bearer authentication."""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings
from src.database.models.user import User
from utils.auth import (
    AccessTokenError,
    AccessTokenExpiredError,
    resolve_access_subject,
)

bearer_security = HTTPBearer(auto_error=False)

# RFC 6750 section 3: a 401 carries WWW-Authenticate so clients can tell an
# expired token, which a refresh repairs, from a missing or forged one.
_CHALLENGE_MISSING = {"WWW-Authenticate": "Bearer"}
_CHALLENGE_INVALID = {"WWW-Authenticate": 'Bearer error="invalid_token"'}
_CHALLENGE_EXPIRED = {
    "WWW-Authenticate": (
        'Bearer error="invalid_token", '
        'error_description="The access token expired"'
    )
}


def get_current_subject(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_security),
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token",
            headers=_CHALLENGE_MISSING,
        )
    try:
        return resolve_access_subject(credentials.credentials)
    except AccessTokenExpiredError as exc:
        raise HTTPException(
            status_code=401, detail=str(exc), headers=_CHALLENGE_EXPIRED
        ) from exc
    except AccessTokenError as exc:
        raise HTTPException(
            status_code=401, detail=str(exc), headers=_CHALLENGE_INVALID
        ) from exc


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
