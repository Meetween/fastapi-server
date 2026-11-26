"""Post meeting reports: read, regenerate, Markdown download.

Stubs until built: every body answers 501.
"""
from fastapi import APIRouter
from fastapi.responses import Response

from api.deps.agentar import CurrentUser, ObjectStoreDep
from api.errors import error_responses, not_implemented
from src.schemas.agentar import Report

router = APIRouter(prefix="/api/agentar/meetings", tags=["reports"])

_MARKDOWN = {200: {"content": {"text/markdown": {"schema": {"type": "string"}}}}}


@router.get("/{meeting_id}/report", response_model=Report, responses=error_responses(401, 404))
async def get_report(meeting_id: str, user: CurrentUser):
    not_implemented()


@router.post(
    "/{meeting_id}/report/regenerate",
    status_code=202,
    response_model=Report,
    responses=error_responses(401, 404, 409),
)
async def regenerate_report(meeting_id: str, user: CurrentUser):
    not_implemented()


@router.get(
    "/{meeting_id}/report.md",
    response_class=Response,
    responses={**_MARKDOWN, **error_responses(401, 404, 409)},
)
async def download_report_markdown(meeting_id: str, user: CurrentUser, store: ObjectStoreDep):
    not_implemented()
