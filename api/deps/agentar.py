"""Shared dependencies of the Agentar routes."""
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials

from api.deps.auth import get_current_user
from api.deps.service_token import _bearer, _unauthorized, token_matches
from api.errors import AgentarError
from config import settings
from src.database.models.user import User
from utils.object_store import ObjectStore, get_configured_object_store

# The owner of every Agentar document is the token subject (lower-cased email); object keys use
# `user.id`. These routes need an account, so they authenticate even when ENABLE_AUTH is off.
CurrentUser = Annotated[User, Depends(get_current_user)]


def get_object_store() -> ObjectStore:
    """The store selected by settings. 503 `storage_unavailable` while it is not configured.

    Tests replace it with ``app.dependency_overrides[get_object_store]``.
    """
    store = get_configured_object_store(settings.object_store)
    if store is None:
        raise AgentarError(503, "storage_unavailable", "Object storage is not configured")
    return store


ObjectStoreDep = Annotated[ObjectStore, Depends(get_object_store)]


def require_agentar_worker(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """The worker key, checked even when ENABLE_AUTH is false.

    `require_vp_worker` lets every call through while auth is off; the events route feeds the
    lifecycle of a meeting, so a wrong switch must not open it. An unset key rejects everything.
    """
    if token_matches(credentials, settings.vp_meetween_api_key):
        return
    raise _unauthorized(credentials)


async def require_content_length(request: Request) -> None:
    """Uploads must declare their size: 411 `length_required` for a chunked or unsized body.

    The size itself is checked by the route (413). Without this a chunked body skips that check
    while the form parser spools the parts.
    """
    raw = request.headers.get("content-length")
    if raw is None or not raw.isascii() or not raw.isdigit():
        raise AgentarError(411, "length_required", "A Content-Length header is required")
