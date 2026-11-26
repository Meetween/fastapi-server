"""Public feature flags and open sign up.

Kept out of api/routes/auth.py so sign up and the Jitsi token route do not edit the same file.
Stubs until built: every body answers 501.
"""
from fastapi import APIRouter

from api.errors import error_responses, not_implemented
from src.schemas.auth import AuthConfig, SignupRequest, TokenPairResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/config", response_model=AuthConfig)
async def get_auth_config():
    """No auth. The login panel reads it before it renders the log in or sign up form."""
    not_implemented()


@router.post(
    "/signup",
    status_code=201,
    response_model=TokenPairResponse,
    responses=error_responses(403, 409, 422, 429),
)
async def signup(request: SignupRequest):
    not_implemented()
