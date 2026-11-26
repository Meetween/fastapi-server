"""Worker events in, dashboard notifications out.

Stubs until built: every body answers 501. The events route takes the worker key
(`VP_MEETWEEN_API_KEY`), never a user token, and it is checked even when ENABLE_AUTH is false.
"""
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query

from api.deps.agentar import CurrentUser, require_agentar_worker
from api.errors import error_responses, not_implemented
from src.schemas.agentar import EventAck, NotificationList, WorkerEvent

router = APIRouter(prefix="/api/agentar", tags=["events"])


@router.post(
    "/events",
    status_code=202,
    response_model=EventAck,
    dependencies=[Depends(require_agentar_worker)],
    responses={
        200: {"model": EventAck, "description": "Duplicate, already stored"},
        **error_responses(401, 404, 409),
    },
)
async def post_worker_event(event: WorkerEvent):
    not_implemented()


@router.get("/notifications", response_model=NotificationList, responses=error_responses(401))
async def list_notifications(
    user: CurrentUser,
    since: Annotated[Optional[str], Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
):
    not_implemented()
