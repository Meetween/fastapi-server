"""Public feature flags and open sign up.

Kept out of api/routes/auth.py so sign up and the Jitsi token route do not edit the same file.
With `EMAIL_ENABLED=false` (the demo) an account is active at once and sign up answers the same
token pair as login. Email verification and password reset by mail are backlog; meanwhile an
operator resets a password with `python cli.py reset-password <email>`.
"""
import logging

from fastapi import APIRouter, Depends, Request

from api.errors import AgentarError, error_responses
from config import settings
from src.database.models.user import User
from src.schemas.auth import AuthConfig, SignupRequest, TokenPairResponse
from utils.auth import create_access_token, create_refresh_token, create_user
from utils.rate_limit import RateLimiter, client_ip, get_rate_limiter, longest_wait, too_many_requests

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

SIGNUP_WINDOW_S = 3600


def _email_taken() -> AgentarError:
    return AgentarError(409, "email_taken", "An account with this email exists. Log in")


def require_signup_enabled() -> None:
    """A dependency, so a disabled sign up answers 403 before the body is validated."""
    if not settings.signup_enabled:
        raise AgentarError(403, "signup_disabled", "Sign up is disabled")


@router.get("/config", response_model=AuthConfig)
async def get_auth_config() -> AuthConfig:
    """No auth. The login panel reads it before it renders the log in or sign up form."""
    return AuthConfig(
        signup_enabled=settings.signup_enabled,
        email_enabled=settings.email_enabled,
        calendar_enabled=settings.calendar_enabled,
    )


@router.post(
    "/signup",
    status_code=201,
    response_model=TokenPairResponse,
    responses=error_responses(403, 409, 422, 429),
    dependencies=[Depends(require_signup_enabled)],
)
async def signup(
    body: SignupRequest,
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> TokenPairResponse:
    """Throttled before the lookup and the hash: `SIGNUP_RATE_PER_IP` and `SIGNUP_RATE_PER_EMAIL`
    attempts per hour. Every attempt past validation counts, a 409 included, so the throttle
    also caps probing for registered addresses (A3 ruling 1)."""
    cfg = settings.agentar
    ip = client_ip(request)
    ip_key = f"signup:ip:{ip}"
    email_key = f"signup:email:{body.email}"

    wait = longest_wait(
        limiter,
        [(ip_key, cfg.signup_rate_per_ip, SIGNUP_WINDOW_S), (email_key, cfg.signup_rate_per_email, SIGNUP_WINDOW_S)],
    )
    if wait is not None:
        logger.warning("Sign up throttled for client %s", ip)
        raise too_many_requests(wait)
    limiter.hit(ip_key, SIGNUP_WINDOW_S)
    limiter.hit(email_key, SIGNUP_WINDOW_S)

    if await User.find_by_email(body.email) is not None:
        raise _email_taken()
    try:
        user = await create_user(
            email=body.email,
            password=body.password,
            first_name=body.first_name,
            last_name=body.last_name,
        )
    except ValueError as exc:  # the unique index caught a concurrent sign up
        raise _email_taken() from exc

    logger.info("Account %s created by sign up", user.id)
    return TokenPairResponse(
        access_token=create_access_token(sub=user.email),
        refresh_token=create_refresh_token(sub=user.email),
    )
