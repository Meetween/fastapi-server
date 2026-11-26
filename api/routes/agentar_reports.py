"""Post meeting reports: read, regenerate, Markdown download.

Owner scoped on the meeting: an id that is unknown or belongs to someone else is 404. The
`agentar_reports` record is the source of truth; the `.md` download is rendered from it.
"""
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import Response

from api.deps.agentar import CurrentUser
from api.errors import AgentarError, error_responses, not_found, object_id_or_404
from config import settings
from src.database.models.agentar_meeting import AgentarMeeting
from src.database.models.agentar_report import AgentarReport
from src.schemas.agentar import Report
from utils import agentar_report
from utils.object_store import ObjectStore, get_configured_object_store

router = APIRouter(prefix="/api/agentar/meetings", tags=["reports"])

_MARKDOWN = {200: {"content": {"text/markdown": {"schema": {"type": "string"}}}}}


def _optional_store() -> Optional[ObjectStore]:
    """The export store, or None. The report does not need it: a missing store only skips the copy."""
    return get_configured_object_store(settings.object_store)


OptionalStore = Annotated[Optional[ObjectStore], Depends(_optional_store)]


async def _owned_meeting(meeting_id: str, user: CurrentUser) -> AgentarMeeting:
    oid = object_id_or_404(meeting_id)
    meeting = await AgentarMeeting.find_one({"_id": oid, "owner": user.email})
    if meeting is None:
        raise not_found()
    return meeting


async def _report_of(meeting: AgentarMeeting) -> Optional[AgentarReport]:
    doc = await AgentarReport.get_collection().find_one({"meeting_id": meeting.id, "owner": meeting.owner})
    return AgentarReport.from_dict(doc) if doc else None


def _pending_view(meeting: AgentarMeeting) -> AgentarReport:
    """An ended meeting whose report the lifecycle has not claimed yet reads as pending."""
    return AgentarReport(
        meeting_id=meeting.id,
        owner=meeting.owner,
        status="pending",
        room_name=meeting.room_name,
        title=meeting.title,
        started_at=meeting.joined_at or meeting.dispatched_at,
        ended_at=meeting.ended_at,
    )


def _view(report: AgentarReport) -> Report:
    return Report.model_validate(report.to_dict())


@router.get("/{meeting_id}/report", response_model=Report, responses=error_responses(401, 404))
async def get_report(meeting_id: str, user: CurrentUser) -> Report:
    meeting = await _owned_meeting(meeting_id, user)
    report = await _report_of(meeting)
    if report is None:
        if meeting.status != "ended":
            raise not_found()
        report = _pending_view(meeting)
    return _view(report)


@router.post(
    "/{meeting_id}/report/regenerate",
    status_code=202,
    response_model=Report,
    responses=error_responses(401, 404, 409),
)
async def regenerate_report(
    meeting_id: str, user: CurrentUser, background: BackgroundTasks, store: OptionalStore
) -> Report:
    meeting = await _owned_meeting(meeting_id, user)
    if meeting.status != "ended":
        raise AgentarError(409, "invalid_state", "The meeting has not ended")
    report = await agentar_report.start_report(meeting)
    if report is None:
        raise AgentarError(409, "invalid_state", "The report is being generated")
    background.add_task(agentar_report.generate_report, meeting.id, store=store, claimed=True)
    return _view(report)


@router.get(
    "/{meeting_id}/report.md",
    response_class=Response,
    responses={**_MARKDOWN, **error_responses(401, 404, 409)},
)
async def download_report_markdown(meeting_id: str, user: CurrentUser) -> Response:
    meeting = await _owned_meeting(meeting_id, user)
    report = await _report_of(meeting)
    if report is None:
        if meeting.status != "ended":
            raise not_found()
        raise AgentarError(409, "report_not_ready", "The report is not ready")
    if report.status != "ready":
        raise AgentarError(409, "report_not_ready", "The report is not ready")
    data = report.to_dict()
    filename = agentar_report.report_filename(data)
    return Response(
        content=agentar_report.render_markdown(data),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )
