"""FastAPI dependencies for JWT bearer authentication."""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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


REQUIRE_AUTH = [Depends(get_current_subject)]
