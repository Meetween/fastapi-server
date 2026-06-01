"""FastAPI dependencies for JWT bearer authentication."""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.database.models.user import User
from utils.auth import AccessTokenError, resolve_access_subject

bearer_security = HTTPBearer(auto_error=False)


def get_current_subject(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_security),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        return resolve_access_subject(credentials.credentials)
    except AccessTokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


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


REQUIRE_AUTH = [Depends(get_current_subject)]
REQUIRE_ADMIN = [Depends(require_admin)]
