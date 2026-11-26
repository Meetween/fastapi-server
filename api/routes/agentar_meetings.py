"""Agentar meetings: create, list, read, edit, delete, picture override, dispatch, recall.

Thin routes over `utils.agentar_meetings`. The lifecycle tick that moves scheduled meetings is
separate. Path ids are plain strings: a malformed id is a 404 through
`api.errors.object_id_or_404`, not a 422.
"""
from typing import Annotated, Optional

import httpx
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response

from api.deps.agentar import CurrentUser, ObjectStoreDep, require_content_length
from api.errors import error_responses
from api.routes.virtual_participants import get_vp_http_client
from config import settings
from src.schemas.agentar import (
    Meeting,
    MeetingCreate,
    MeetingList,
    MeetingPatch,
    MeetingStatus,
    Picture,
)
from utils import agentar_meetings as meetings
from utils.uploads import FILE_PART, multipart_openapi, read_single_upload

router = APIRouter(prefix="/api/agentar/meetings", tags=["meetings"])

_JPEG = {200: {"content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}}}
_PICTURE_HEADERS = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"}

VpClient = Annotated[httpx.AsyncClient, Depends(get_vp_http_client)]


@router.post("", status_code=201, response_model=Meeting, responses=error_responses(401, 409, 422, 429))
async def create_meeting(body: MeetingCreate, user: CurrentUser):
    return meetings.to_view(await meetings.create_meeting(user, body))


@router.get("", response_model=MeetingList, responses=error_responses(401))
async def list_meetings(
    user: CurrentUser,
    status: Annotated[Optional[list[MeetingStatus]], Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[Optional[str], Query(max_length=64)] = None,
):
    items, next_cursor = await meetings.list_meetings(user.email, status, limit, cursor)
    return MeetingList(items=[meetings.to_view(m) for m in items], next_cursor=next_cursor)


@router.get("/{meeting_id}", response_model=Meeting, responses=error_responses(401, 404))
async def get_meeting(meeting_id: str, user: CurrentUser):
    return meetings.to_view(await meetings.load_owned(meeting_id, user.email))


@router.patch("/{meeting_id}", response_model=Meeting, responses=error_responses(401, 404, 409, 422))
async def update_meeting(meeting_id: str, body: MeetingPatch, user: CurrentUser):
    return meetings.to_view(await meetings.update_meeting(user, meeting_id, body))


@router.delete("/{meeting_id}", status_code=204, responses=error_responses(401, 404, 409))
async def delete_meeting(meeting_id: str, user: CurrentUser, store: ObjectStoreDep):
    await meetings.delete_meeting(user, meeting_id, store)
    return Response(status_code=204)


@router.put(
    "/{meeting_id}/picture",
    response_model=Picture,
    dependencies=[Depends(require_content_length)],
    responses=error_responses(401, 404, 409, 411, 413, 415, 422, 503),
    openapi_extra=multipart_openapi({"file": FILE_PART}, ["file"]),
)
async def put_meeting_picture(meeting_id: str, request: Request, user: CurrentUser, store: ObjectStoreDep):
    """Reads the form with `utils.uploads.read_single_upload`, not with an UploadFile parameter."""
    meeting = await meetings.load_owned(meeting_id, user.email)
    meetings.require_picture_editable(meeting)
    upload = await read_single_upload(request, max_bytes=settings.agentar.picture_max_bytes)
    return await meetings.put_picture(user, meeting, upload.data, store)


@router.get(
    "/{meeting_id}/picture",
    response_class=Response,
    responses={**_JPEG, **error_responses(401, 404)},
)
async def get_meeting_picture(meeting_id: str, user: CurrentUser, store: ObjectStoreDep):
    data = await meetings.read_picture(user, meeting_id, store)
    return Response(content=data, media_type="image/jpeg", headers=_PICTURE_HEADERS)


@router.delete("/{meeting_id}/picture", status_code=204, responses=error_responses(401, 404, 409))
async def delete_meeting_picture(meeting_id: str, user: CurrentUser, store: ObjectStoreDep):
    await meetings.delete_picture(user, meeting_id, store)
    return Response(status_code=204)


@router.post(
    "/{meeting_id}/dispatch",
    status_code=202,
    response_model=Meeting,
    responses=error_responses(401, 404, 409, 422, 503),
)
async def dispatch_meeting(meeting_id: str, user: CurrentUser):
    return meetings.to_view(await meetings.dispatch_meeting(user, meeting_id))


@router.post(
    "/{meeting_id}/recall",
    status_code=202,
    response_model=Meeting,
    responses=error_responses(401, 404, 409, 503),
)
async def recall_meeting(meeting_id: str, user: CurrentUser, client: VpClient):
    return meetings.to_view(await meetings.recall_meeting(user, meeting_id, client))
