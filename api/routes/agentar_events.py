"""Worker events in, dashboard notifications out.

The events route takes the worker key (`VP_MEETWEEN_API_KEY`), never a user token, and checks it
even when ENABLE_AUTH is false. Notifications are scoped to the token subject.
"""
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query, Response

from api.deps.agentar import CurrentUser, require_agentar_worker
from api.errors import error_responses
from src.schemas.agentar import EventAck, NotificationList, WorkerEvent
from utils import agentar_events

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
async def post_worker_event(event: WorkerEvent, response: Response) -> EventAck:
    duplicate = await agentar_events.ingest_event(event)
    if duplicate:
        response.status_code = 200
    return EventAck(duplicate=duplicate)


@router.get("/notifications", response_model=NotificationList, responses=error_responses(401))
async def list_notifications(
    user: CurrentUser,
    since: Annotated[Optional[str], Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> NotificationList:
    return await agentar_events.list_notifications(owner=user.email, since=since, limit=limit)
