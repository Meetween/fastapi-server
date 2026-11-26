"""Worker events in, dashboard notifications out.

Intake order for one event: the (meeting, bot) pair must match a meeting record (404), a repeat
of a stored `event_id` is a duplicate (200), the meeting must be in a state that accepts events
(409 `invalid_state`), `at` must be near the server clock (422 `event_clock_skew`), then the event
is stored and its effect on the meeting is applied. The duplicate check comes before the state and
clock checks because a worker retry carries the original `at` and may arrive late.

Every stored event is a dashboard notification. The notification cursor is the event `_id`.
Strings in `data` come from guests and the model, so only the known fields of each type are kept,
control and bidi characters are removed, and the UI renders them as text nodes.
"""
import importlib
import logging
import re
import uuid
from datetime import timedelta
from typing import Any, Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi.exceptions import RequestValidationError
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from api.errors import AgentarError, not_found, object_id_or_404
from config import settings
from src.database.models.agentar_event import AgentarEvent
from src.database.models.agentar_meeting import AgentarMeeting
from src.schemas.agentar import Notification, NotificationList, NotificationType, WorkerEvent
from utils import clock

logger = logging.getLogger(__name__)

# Statuses that accept every event type. `left` is also accepted shortly after `ended`.
ACCEPTING_STATUSES = ("joining", "live", "leaving")

NOTIFICATION_TYPES: tuple[str, ...] = (
    "joined",
    "question_asked",
    "answer_given",
    "escalation",
    "left",
    "error",
    "missed",
    "failed",
    "report_ready",
)
WARNING_TYPES = frozenset({"escalation", "missed"})
ERROR_TYPES = frozenset({"error", "failed"})

DEFAULT_WINDOW = timedelta(hours=24)
DEFAULT_LIMIT_WITHOUT_CURSOR = 20

_OBJECT_ID = re.compile(r"[a-f0-9]{24}")
# C0 and C1 controls except tab and newline, and the bidirectional overrides and isolates.
_UNSAFE_TEXT = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f‪-‮⁦-⁩]")


def severity_for(event_type: str) -> str:
    if event_type in WARNING_TYPES:
        return "warning"
    if event_type in ERROR_TYPES:
        return "error"
    return "info"


def plain_text(value: str) -> str:
    return _UNSAFE_TEXT.sub("", value)


def _clean(data: dict[str, Any]) -> dict[str, Any]:
    return {key: plain_text(value) if isinstance(value, str) else value for key, value in data.items()}


def event_payload(event: WorkerEvent) -> dict[str, Any]:
    """The known fields of the event data, as plain values. `kind` repeats `type`, so it is dropped."""
    return _clean(event.data.model_dump(exclude={"kind"}, exclude_unset=True))


async def _mark_ended(meeting_id: str, reason: str) -> None:
    """Hand a live or leaving meeting to the lifecycle module (B7).

    Assumed signature: ``async def mark_agentar_ended(meeting_id: str, reason: str)``. B7 owns the
    state change to ended, the live slot, the transcript flush and the report queue. The module
    is imported on use because it lives on another branch; once merged this becomes a plain import.
    """
    lifecycle = importlib.import_module("utils.agentar_lifecycle")
    await lifecycle.mark_agentar_ended(meeting_id=meeting_id, reason=reason)


async def _fail_joining(meeting_oid: ObjectId, reason: str) -> bool:
    """joining to failed, atomically, releasing the live slot. False when the meeting moved on."""
    now = clock.utcnow()
    updated = await AgentarMeeting.get_collection().find_one_and_update(
        {"_id": meeting_oid, "status": "joining"},
        {
            "$set": {"status": "failed", "status_reason": reason, "updated_at": now},
            "$unset": {"live_lock": ""},
        },
        return_document=ReturnDocument.AFTER,
    )
    return updated is not None


async def _load_meeting(event: WorkerEvent) -> dict[str, Any]:
    meeting = await AgentarMeeting.get_collection().find_one({"_id": object_id_or_404(event.agentar_meeting_id)})
    # The pair must match. A meeting without a bot, or with another bot, looks like no meeting.
    if meeting is None or meeting.get("bot_id") is None or meeting["bot_id"] != event.bot_id:
        raise not_found()
    return meeting


def _check_state(meeting: dict[str, Any], event: WorkerEvent) -> None:
    status = meeting["status"]
    if status in ACCEPTING_STATUSES:
        return
    if status == "ended" and event.type == "left" and meeting.get("ended_at") is not None:
        grace = timedelta(seconds=settings.agentar.event_left_grace_seconds)
        if clock.utcnow() - clock.as_utc(meeting["ended_at"]) <= grace:
            return
    raise AgentarError(409, "invalid_state", f"The meeting does not accept {event.type} events in status {status}")


def _check_clock(event: WorkerEvent) -> None:
    skew = abs(clock.utcnow() - clock.as_utc(event.at))
    if skew > timedelta(seconds=settings.agentar.event_max_skew_seconds):
        raise AgentarError(422, "event_clock_skew", "The event time is too far from the server time")


async def _apply_effect(meeting: dict[str, Any], event: WorkerEvent) -> None:
    """Move the meeting for `joined`, `left` and a fatal `error`. The rest are only stored.

    Every move is a filtered atomic update or an idempotent hand over, so a worker retry after a
    failed attempt applies it again without harm.
    """
    meeting_oid = meeting["_id"]
    status = meeting["status"]
    now = clock.utcnow()
    collection = AgentarMeeting.get_collection()

    if event.jitsi_meeting_id:
        # Bound once. `None` in the filter matches a missing field as well.
        await collection.update_one(
            {"_id": meeting_oid, "jitsi_meeting_id": None},
            {"$set": {"jitsi_meeting_id": plain_text(event.jitsi_meeting_id), "updated_at": now}},
        )

    if event.type == "joined":
        await collection.update_one(
            {"_id": meeting_oid, "status": "joining"},
            {"$set": {"status": "live", "joined_at": now, "updated_at": now}},
        )
    elif event.type == "left":
        if status == "ended":
            return
        if status == "joining" and event.data.reason == "error":
            await _fail_joining(meeting_oid, "worker_error")
            return
        await _mark_ended(str(meeting_oid), event.data.reason)
    elif event.type == "error" and event.data.fatal:
        if status == "joining":
            await _fail_joining(meeting_oid, "worker_error")
        else:
            # The bot was in the room: end it so a report is still made.
            await _mark_ended(str(meeting_oid), "error")


async def ingest_event(event: WorkerEvent) -> bool:
    """Validate, store and apply one worker event. Returns True for a duplicate."""
    meeting = await _load_meeting(event)
    event_id = str(event.event_id)
    events = AgentarEvent.get_collection()

    if await _is_duplicate(event_id, str(meeting["_id"])):
        return True

    _check_state(meeting, event)
    _check_clock(event)

    stored = AgentarEvent(
        event_id=event_id,
        owner=meeting["owner"],
        meeting_id=str(meeting["_id"]),
        bot_id=event.bot_id,
        type=event.type,
        source="worker",
        data=event_payload(event),
        at=clock.as_utc(event.at),
        received_at=clock.utcnow(),
    )
    try:
        await stored.save()
    except DuplicateKeyError:
        # Two attempts of the same event raced past the lookup.
        if await _is_duplicate(event_id, str(meeting["_id"])):
            return True
        raise

    try:
        await _apply_effect(meeting, event)
    except Exception:
        # Keep retries useful: without the row the worker's next attempt applies the effect again.
        await events.delete_one({"event_id": event_id})
        raise
    return False


async def _is_duplicate(event_id: str, meeting_id: str) -> bool:
    """True when the event_id is stored for this meeting. An id stored for another meeting is
    answered as unknown, so one meeting cannot probe another's events."""
    row = await AgentarEvent.get_collection().find_one({"event_id": event_id}, {"meeting_id": 1})
    if row is None:
        return False
    if row["meeting_id"] != meeting_id:
        raise not_found()
    return True


async def record_server_event(
    *, owner: str, meeting_id: str, type: NotificationType, data: Optional[dict[str, Any]] = None, event_id: Optional[str] = None
) -> None:
    """Add a server side notification: `missed`, `failed`, `report_ready`, with `{reason}` as data.

    For the lifecycle tick (B7) and the report job (B9). Pass a stable `event_id` when the caller
    can run twice for the same outcome; a repeat is ignored.
    """
    if type not in NOTIFICATION_TYPES:
        raise ValueError(f"unknown notification type {type!r}")
    try:
        await AgentarEvent(
            event_id=event_id or f"server-{uuid.uuid4()}",
            owner=owner,
            meeting_id=meeting_id,
            type=type,
            source="server",
            data=_clean(data or {}),
            received_at=clock.utcnow(),
        ).save()
    except DuplicateKeyError:
        logger.debug("server event %s already stored", event_id)


def _bad_cursor(value: str) -> RequestValidationError:
    return RequestValidationError(
        [{"type": "value_error", "loc": ("query", "since"), "msg": "since is not a cursor", "input": value}]
    )


async def list_notifications(*, owner: str, since: Optional[str], limit: int) -> NotificationList:
    """Notifications of one owner. Without `since`: the newest 20 of the last 24 h. With `since`:
    everything after it, oldest first, at most `limit`. The cursor is the id of the last item, the
    given `since` when nothing is new, null when there is neither."""
    collection = AgentarEvent.get_collection()
    query: dict[str, Any] = {"owner": owner, "type": {"$in": list(NOTIFICATION_TYPES)}}

    if since is not None:
        if not _OBJECT_ID.fullmatch(since):
            raise _bad_cursor(since)
        try:
            query["_id"] = {"$gt": ObjectId(since)}
        except InvalidId as exc:
            raise _bad_cursor(since) from exc
        rows = await collection.find(query).sort("_id", 1).limit(limit).to_list(length=limit)
    else:
        query["received_at"] = {"$gte": clock.utcnow() - DEFAULT_WINDOW}
        size = min(limit, DEFAULT_LIMIT_WITHOUT_CURSOR)
        newest = await collection.find(query).sort("_id", -1).limit(size).to_list(length=size)
        rows = list(reversed(newest))

    events = [AgentarEvent.from_dict(row) for row in rows]
    titles = await _titles({event.meeting_id for event in events})
    items = [
        Notification(
            id=event.id,
            at=event.received_at,
            meeting_id=event.meeting_id,
            meeting_title=titles.get(event.meeting_id),
            type=event.type,
            severity=severity_for(event.type),
            data=event.data,
        )
        for event in events
    ]
    return NotificationList(items=items, cursor=items[-1].id if items else since)


async def _titles(meeting_ids: set[str]) -> dict[str, Optional[str]]:
    if not meeting_ids:
        return {}
    oids = [ObjectId(meeting_id) for meeting_id in meeting_ids if _OBJECT_ID.fullmatch(meeting_id)]
    rows = await AgentarMeeting.get_collection().find({"_id": {"$in": oids}}, {"title": 1}).to_list(length=len(oids))
    return {str(row["_id"]): row.get("title") for row in rows}
