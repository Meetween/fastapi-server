"""Error model of the Agentar routes.

New routes answer `{"detail": "<English message>", "code": "<ErrorCode>"}` through one
exception class and one handler. `detail` stays a string so clients that read it today keep
working. Request validation keeps the FastAPI default 422 body, and the 401 bodies and
challenges of the existing auth dependencies are untouched.
"""
import re
from typing import Mapping

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from src.schemas.agentar import Error, ErrorCode, PlainError


class AgentarError(HTTPException):
    """HTTPException that carries a machine readable `code`."""

    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        detail: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=dict(headers) if headers else None)
        self.code = code


async def agentar_error_handler(request: Request, exc: AgentarError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": exc.code},
        headers=exc.headers,
    )


def not_found() -> AgentarError:
    """An unknown id and an id owned by someone else look the same: 404, never 403."""
    return AgentarError(404, "not_found", "Not found")


_OBJECT_ID = re.compile(r"[a-f0-9]{24}")


def object_id_or_404(value: str) -> ObjectId:
    """Parse a path id. A malformed id is as unknown as a missing one. The contract allows
    lower case hex only."""
    if not _OBJECT_ID.fullmatch(value):
        raise not_found()
    try:
        return ObjectId(value)
    except (InvalidId, TypeError) as exc:
        raise not_found() from exc


_ERROR_DESCRIPTIONS = {
    401: "Missing, invalid or expired bearer token",
    403: "Sign up is disabled (signup_disabled)",
    404: "Unknown id, or an id owned by another user (not_found)",
    409: (
        "Not allowed in the current state (invalid_state, live_agentar_exists, agentar_capacity, "
        "room_not_owned, file_in_use, email_taken)"
    ),
    411: "No Content-Length header (length_required)",
    413: "Body over the cap (file_too_large)",
    415: "Content type not allowed (unsupported_type)",
    422: "Request invalid, or a quota is exceeded",
    429: "Throttled (rate_limited)",
    503: "A dependency is not configured or not reachable",
}


def error_responses(*codes: int) -> dict[int | str, dict]:
    """`responses=` entry for the given error statuses, so OpenAPI shows the contract bodies.
    422 keeps the FastAPI validation body and is documented by FastAPI itself."""
    out: dict[int | str, dict] = {}
    for code in codes:
        if code == 422:
            continue
        out[code] = {
            "model": PlainError if code == 401 else Error,
            "description": _ERROR_DESCRIPTIONS[code],
        }
    return out
