"""Agentar meetings: create, edit, list, delete, picture override, dispatch and recall.

Rules kept here, not in the routes:

- Owner scope. Every query carries `owner=<token subject>`. An id that is malformed, unknown or
  somebody else's is a 404.
- Status moves only through `_transition`, an atomic `find_one_and_update` filtered on the
  statuses the move may start from, so two requests cannot both win. The table is CONTRACT-NOTES
  "Status transitions".
- One live Agentar per user is held by the partial unique index on `live_lock` as well as by a
  pre-check. The global cap `AGENTAR_MAX_LIVE_TOTAL` is claim then verify: the lock is taken,
  holders are counted, and a request that finds the cap exceeded gives its lock back. Two racing
  requests may both give it back (a retryable 409), never both keep it.
- A room is sent into only by the user who holds it in Jitsi (`claim_room_moderator`, admins
  exempt), because the bot is owner in Jitsi.
"""
import hashlib
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import unquote, urlsplit

import httpx
from bson import ObjectId
from fastapi.exceptions import RequestValidationError
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from api.errors import AgentarError, not_found, object_id_or_404
from config import settings
from src.database.models.agentar_event import AgentarEvent
from src.database.models.agentar_file import AgentarFile
from src.database.models.agentar_meeting import BOT_PRESENT_STATUSES, AgentarMeeting, MeetingQuestion
from src.database.models.agentar_report import AgentarReport
from src.database.models.user import User
from src.schemas.agentar import Meeting, MeetingCreate, MeetingPatch, Picture, Question
from src.schemas.virtual_participants import BOT_NAME_SUFFIX, ROOM_PATTERN
from utils import clock
from utils.agentar_worker import (
    WorkerDispatchError,
    events_url,
    read_picture_bytes,
    recall_bot,
    resolve_display_name,
    resolve_voice,
)
from utils.object_store import ObjectNotFound, ObjectStore, ObjectStoreError

logger = logging.getLogger(__name__)

DISPATCHABLE = ("draft", "missed", "failed")
EDITABLE = ("draft", "scheduled")
DELETABLE = ("draft", "scheduled", "missed", "failed", "ended")
MIN_LEAD = timedelta(minutes=1)
MAX_OFFSET = 100_000
PICTURE_NAME = "picture.jpg"
# A send starts clean: a failed attempt may have left the id of a bot that is gone.
_CLEAN_SLATE = ("bot_id", "jitsi_meeting_id")

_ROOM_RE = re.compile(ROOM_PATTERN)
_CURSOR_RE = re.compile(r"o[0-9]{1,6}")


# Validation ----------------------------------------------------------------------------------


def invalid(field: str, message: str, *, where: str = "body", value: Any = None) -> RequestValidationError:
    """A 422 in the FastAPI default body shape, for rules that need the stored meeting or the
    clock and so cannot live in the request model."""
    return RequestValidationError(
        [{"type": "value_error", "loc": (where, field), "msg": message, "input": value}]
    )


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _origin(url: str) -> Optional[tuple[str, int]]:
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return None
    return parts.hostname.lower(), port or _default_port(parts.scheme)


def parse_room(value: str) -> str:
    """Room name from a bare name or a link on the Jitsi host: the last path segment, lower case,
    checked against the room pattern. A link on another host, or any link while JITSI_MEET_BASE_URL
    is unset, is 422 `room_invalid`."""
    text = value.strip()
    bad = AgentarError(422, "room_invalid", "Use a room name or a link on this Meetween Jitsi host")
    if "://" in text or text.startswith("//"):
        base = settings.meeting_jwt.meet_base_url
        base_origin = _origin(base) if base else None
        parts = urlsplit(text if "://" in text else f"https:{text}")
        if base_origin is None or _origin(text if "://" in text else f"https:{text}") != base_origin:
            raise bad
        segments = [segment for segment in parts.path.split("/") if segment]
        if not segments:
            raise bad
        text = unquote(segments[-1])
    room = text.lower()
    if not _ROOM_RE.fullmatch(room):
        raise bad
    return room


def join_url(room: str) -> Optional[str]:
    base = settings.meeting_jwt.meet_base_url
    return f"{base.rstrip('/')}/{room}" if base else None


def _clean_text(value: str) -> str:
    return value.strip()


def _title(value: Optional[str]) -> Optional[str]:
    return (_clean_text(value) or None) if value is not None else None


def _questions(texts: Sequence[str]) -> list[MeetingQuestion]:
    """Ids follow the position on every write: q1, q2, ..."""
    cleaned = [_clean_text(text) for text in texts]
    if any(not text for text in cleaned):
        raise invalid("questions", "a question cannot be empty")
    return [MeetingQuestion(id=f"q{i}", text=text) for i, text in enumerate(cleaned, start=1)]


def check_schedule(mode: str, start_at: Optional[datetime]) -> Optional[datetime]:
    """The stored start for a mode. `now` keeps none until dispatch. `scheduled` needs a start from
    one minute to AGENTAR_SCHEDULE_HORIZON_DAYS ahead: past is 422 `start_in_past`."""
    if mode == "now":
        return None
    if start_at is None:
        raise invalid("start_at", "start_at is required for a scheduled meeting")
    start = clock.as_utc(start_at)
    now = clock.utcnow()
    if start < now + MIN_LEAD:
        raise AgentarError(422, "start_in_past", "The start must be at least one minute from now")
    if start > now + timedelta(days=settings.agentar.schedule_horizon_days):
        raise invalid(
            "start_at", f"start_at is more than {settings.agentar.schedule_horizon_days} days ahead"
        )
    return start


async def check_file_ids(owner: str, file_ids: Sequence[str], meeting_id: Optional[str]) -> None:
    """Every id is a document of this owner: from the general knowledge base, or a file of this
    same meeting. Anything else is a 422 that does not say whether the id exists."""
    for file_id in file_ids:
        record = await AgentarFile.find_by_id(file_id)
        usable = (
            record is not None
            and record.owner == owner
            and record.kind == "document"
            and (record.scope == "general" or (meeting_id is not None and record.meeting_id == meeting_id))
        )
        if not usable:
            raise invalid("file_ids", "unknown file id", value=file_id)


async def claim_room(room_name: str, user: User) -> None:
    """Claim a free room for the user or confirm the user holds it. Anything else is 409
    `room_not_owned`. Admins are exempt, and a user who claims too many new rooms gets the 429
    `rate_limited` that `claim_room_moderator` raises (both live there)."""
    # B2's module. Hoist to the top of the file once utils/room_owner.py is merged.
    from utils.room_owner import claim_room_moderator

    if not await claim_room_moderator(room_name, user):
        raise AgentarError(409, "room_not_owned", "This room belongs to another user")


async def _check_daily_quota(owner: str) -> None:
    now = clock.utcnow()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    created = await AgentarMeeting.count_documents({"owner": owner, "created_at": {"$gte": day_start}})
    if created >= settings.agentar.meetings_per_day:
        raise AgentarError(
            422,
            "quota_meetings_per_day",
            f"At most {settings.agentar.meetings_per_day} meetings can be created per day",
        )


# Reads ---------------------------------------------------------------------------------------


async def load_owned(meeting_id: str, owner: str) -> AgentarMeeting:
    oid = object_id_or_404(meeting_id)
    meeting = await AgentarMeeting.find_one({"_id": oid, "owner": owner})
    if meeting is None:
        raise not_found()
    return meeting


def to_view(meeting: AgentarMeeting) -> Meeting:
    return Meeting(
        id=meeting.id,
        title=meeting.title,
        room_name=meeting.room_name,
        join_url=join_url(meeting.room_name),
        mode=meeting.mode,
        start_at=meeting.start_at,
        grace_minutes=meeting.grace_minutes,
        purpose=meeting.purpose,
        questions=[Question(id=q.id, text=q.text) for q in meeting.questions],
        relevant_info=meeting.relevant_info,
        file_ids=meeting.file_ids,
        use_general_kb=meeting.use_general_kb,
        has_picture=meeting.picture_id is not None,
        status=meeting.status,
        status_reason=meeting.status_reason,
        display_name=meeting.display_name,
        voice=meeting.voice,
        bot_id=meeting.bot_id,
        jitsi_meeting_id=meeting.jitsi_meeting_id,
        dispatched_at=meeting.dispatched_at,
        joined_at=meeting.joined_at,
        ended_at=meeting.ended_at,
        report_status=meeting.report_status,
        created_at=meeting.created_at,
        updated_at=meeting.updated_at,
    )


async def list_meetings(
    owner: str, statuses: Optional[Iterable[str]], limit: int, cursor: Optional[str]
) -> tuple[list[AgentarMeeting], Optional[str]]:
    """The owner's meetings, start_at then created_at descending. The cursor is an offset
    (`o<number>`): a dashboard holds a few dozen meetings per user, so a keyset over a nullable
    start_at is not worth its cost here."""
    offset = 0
    if cursor is not None:
        if not _CURSOR_RE.fullmatch(cursor) or int(cursor[1:]) > MAX_OFFSET:
            raise invalid("cursor", "invalid cursor", where="query")
        offset = int(cursor[1:])
    query: dict[str, Any] = {"owner": owner}
    wanted = sorted(set(statuses)) if statuses else []
    if wanted:
        query["status"] = {"$in": wanted}
    cursor_ = (
        AgentarMeeting.get_collection()
        .find(query)
        .sort([("start_at", -1), ("created_at", -1), ("_id", -1)])
        .skip(offset)
        .limit(limit + 1)
    )
    docs = await cursor_.to_list(length=limit + 1)
    meetings = [AgentarMeeting.from_dict(doc) for doc in docs]
    next_cursor = f"o{offset + limit}" if len(meetings) > limit else None
    return meetings[:limit], next_cursor


# Writes --------------------------------------------------------------------------------------


async def _transition(
    meeting_id: str,
    owner: str,
    from_statuses: Sequence[str],
    set_fields: dict[str, Any],
    unset_fields: Sequence[str] = (),
) -> Optional[AgentarMeeting]:
    """Atomic move: apply the update only while the meeting is still in one of `from_statuses`.
    None when it is not (someone else moved it) or when it is not the owner's."""
    update: dict[str, Any] = {"$set": {**set_fields, "updated_at": clock.utcnow()}}
    if unset_fields:
        update["$unset"] = {name: "" for name in unset_fields}
    doc = await AgentarMeeting.get_collection().find_one_and_update(
        {"_id": ObjectId(meeting_id), "owner": owner, "status": {"$in": list(from_statuses)}},
        update,
        return_document=ReturnDocument.AFTER,
    )
    return AgentarMeeting.from_dict(doc) if doc else None


def invalid_state(message: str) -> AgentarError:
    return AgentarError(409, "invalid_state", message)


async def create_meeting(user: User, body: MeetingCreate) -> AgentarMeeting:
    room = parse_room(body.room)
    start_at = check_schedule(body.mode, body.start_at)
    questions = _questions(body.questions)
    await check_file_ids(user.email, body.file_ids, None)
    await _check_daily_quota(user.email)
    await claim_room(room, user)
    meeting = AgentarMeeting(
        owner=user.email,
        title=_title(body.title),
        room_name=room,
        mode=body.mode,
        start_at=start_at,
        grace_minutes=body.grace_minutes,
        purpose=_clean_text(body.purpose),
        questions=questions,
        relevant_info=_clean_text(body.relevant_info),
        file_ids=list(body.file_ids),
        use_general_kb=body.use_general_kb,
    )
    return await meeting.save()


async def update_meeting(user: User, meeting_id: str, body: MeetingPatch) -> AgentarMeeting:
    meeting = await load_owned(meeting_id, user.email)
    if meeting.status not in EDITABLE:
        raise invalid_state("Only a draft or scheduled meeting can be edited")
    sent = body.model_fields_set
    changes: dict[str, Any] = {}

    for required in ("room", "mode"):
        if required in sent and getattr(body, required) is None:
            raise invalid(required, f"{required} cannot be null")

    if "title" in sent:
        changes["title"] = _title(body.title)
    if "grace_minutes" in sent:
        changes["grace_minutes"] = body.grace_minutes
    if "purpose" in sent:
        changes["purpose"] = _clean_text(body.purpose)
    if "relevant_info" in sent:
        changes["relevant_info"] = _clean_text(body.relevant_info)
    if "use_general_kb" in sent:
        changes["use_general_kb"] = body.use_general_kb
    if "questions" in sent:
        changes["questions"] = [q.model_dump() for q in _questions(body.questions)]
    if "file_ids" in sent:
        await check_file_ids(user.email, body.file_ids, meeting.id)
        changes["file_ids"] = list(body.file_ids)

    mode = body.mode if "mode" in sent else meeting.mode
    if meeting.status == "scheduled" and mode != meeting.mode:
        raise invalid("mode", "The mode of an armed meeting cannot change; recall it first")
    if "mode" in sent or "start_at" in sent:
        requested_start = body.start_at if "start_at" in sent else meeting.start_at
        start_at = check_schedule(mode, requested_start)
        if mode != meeting.mode:
            changes["mode"] = mode
        if start_at != meeting.start_at:
            changes["start_at"] = start_at

    if "room" in sent:
        room = parse_room(body.room)
        if room != meeting.room_name:
            await claim_room(room, user)
            changes["room_name"] = room

    if not changes:
        return meeting
    updated = await _transition(meeting.id, user.email, EDITABLE, changes)
    if updated is None:
        raise invalid_state("The meeting changed state; reload it")
    return updated


def _picture_prefix(user_id: str, meeting_id: str) -> str:
    return f"u/{user_id}/m/{meeting_id}/"


async def delete_meeting(user: User, meeting_id: str, store: ObjectStore) -> None:
    """Delete the meeting with its files, picture, report and events.

    The meeting document goes first, atomically and only in a deletable status, so a bot can never
    be sent for a meeting that is half deleted. The objects under the meeting prefix and the
    dependent records follow. An object that cannot be removed is logged with its prefix and left
    for the purge script: the records that pointed at it are gone, so it is invisible and counts
    against no quota.
    """
    meeting = await load_owned(meeting_id, user.email)
    if meeting.status in BOT_PRESENT_STATUSES:
        raise invalid_state("Recall Agentar before deleting the meeting")
    result = await AgentarMeeting.get_collection().delete_one(
        {"_id": ObjectId(meeting.id), "owner": user.email, "status": {"$in": list(DELETABLE)}}
    )
    if result.deleted_count == 0:
        raise invalid_state("The meeting changed state; reload it")
    prefix = _picture_prefix(user.id, meeting.id)
    try:
        for info in await store.list(prefix):
            await store.delete(info.key)
    except ObjectStoreError as exc:
        logger.error(f"agentar meeting delete left objects under {prefix}: {type(exc).__name__}")
    await AgentarFile.delete_many({"owner": user.email, "meeting_id": meeting.id})
    await AgentarReport.delete_many({"owner": user.email, "meeting_id": meeting.id})
    await AgentarEvent.delete_many({"owner": user.email, "meeting_id": meeting.id})


# Picture override ----------------------------------------------------------------------------


def _storage_unavailable(exc: Exception) -> AgentarError:
    logger.error(f"agentar object store failed: {type(exc).__name__}")
    return AgentarError(503, "storage_unavailable", "Object storage is not reachable")


def _normalise(data: bytes) -> bytes:
    """512x512 JPEG of at most 400 KB from the upload. B4's `utils/picture.py`; it raises
    AgentarError (415 `unsupported_type` or 422 `picture_invalid`) for a file it cannot use.
    Hoist the import once that module is merged."""
    from utils.picture import normalise_picture

    return normalise_picture(data)


def require_picture_editable(meeting: AgentarMeeting) -> None:
    if meeting.status not in EDITABLE:
        raise invalid_state("The picture can change only in a draft or scheduled meeting")


async def put_picture(user: User, meeting: AgentarMeeting, upload: bytes, store: ObjectStore) -> Picture:
    """Store the normalised picture as the meeting override and drop the previous one.

    Order: object, record, then the conditional meeting update. A meeting that left draft or
    scheduled in between loses nothing: the new object and record are removed and the call is 409.
    """
    require_picture_editable(meeting)
    jpeg = _normalise(upload)
    file_id = str(ObjectId())
    key = f"u/{user.id}/m/{meeting.id}/avatar/{file_id}.jpg"
    digest = hashlib.sha256(jpeg).hexdigest()
    now = clock.utcnow()
    try:
        await store.put(key, jpeg, content_type="image/jpeg")
    except ObjectStoreError as exc:
        raise _storage_unavailable(exc) from exc
    record = AgentarFile(
        owner=user.email,
        scope="meeting",
        meeting_id=meeting.id,
        kind="picture",
        name=PICTURE_NAME,
        size=len(jpeg),
        content_type="image/jpeg",
        sha256=digest,
        storage_key=key,
    )
    document = record.to_dict()
    document["_id"] = ObjectId(file_id)
    try:
        await AgentarFile.get_collection().insert_one(document)
        updated = await _transition(
            meeting.id, user.email, EDITABLE, {"picture_id": file_id}
        )
    except Exception:
        await _discard_picture(file_id, key, store)
        raise
    if updated is None:
        await _discard_picture(file_id, key, store)
        raise invalid_state("The meeting changed state; reload it")
    if meeting.picture_id:
        await _discard_picture(meeting.picture_id, None, store)
    return Picture(bytes=len(jpeg), sha256=digest, updated_at=now)


async def _discard_picture(file_id: str, key: Optional[str], store: ObjectStore) -> None:
    """Remove a picture record and its object, best effort: a failure is logged, not raised."""
    try:
        record = await AgentarFile.find_by_id(file_id) if key is None else None
        storage_key = key or (record.storage_key if record else None)
        if storage_key:
            await store.delete(storage_key)
    except ObjectStoreError as exc:
        logger.error(f"agentar picture object left behind file={file_id}: {type(exc).__name__}")
    await AgentarFile.delete_many({"_id": ObjectId(file_id)})


async def read_picture(user: User, meeting_id: str, store: ObjectStore) -> bytes:
    meeting = await load_owned(meeting_id, user.email)
    if meeting.picture_id is None:
        raise not_found()
    record = await AgentarFile.find_by_id(meeting.picture_id)
    if record is None or record.owner != user.email:
        raise not_found()
    try:
        data = await read_picture_bytes(store, record.storage_key)
    except ObjectNotFound as exc:
        raise not_found() from exc
    except ObjectStoreError as exc:
        raise _storage_unavailable(exc) from exc
    if data is None:
        raise not_found()
    return data


async def delete_picture(user: User, meeting_id: str, store: ObjectStore) -> None:
    """Drop the override so the profile picture applies again. Idempotent."""
    meeting = await load_owned(meeting_id, user.email)
    require_picture_editable(meeting)
    if meeting.picture_id is None:
        return
    updated = await _transition(meeting.id, user.email, EDITABLE, {"picture_id": None})
    if updated is None:
        raise invalid_state("The meeting changed state; reload it")
    await _discard_picture(meeting.picture_id, None, store)


# Dispatch and recall -------------------------------------------------------------------------


def _require_dispatch_configuration() -> None:
    """503 before any change when the worker or the Jitsi key is not configured."""
    if not settings.virtual_participants.worker_secret or not settings.agentar.worker_events_base_url:
        raise AgentarError(503, "worker_unavailable", "The virtual participant worker is not configured")
    if not settings.jwt_jitsi_secret_key:
        raise AgentarError(503, "jitsi_key_missing", "The Jitsi token key is not configured")
    try:
        events_url()
    except WorkerDispatchError as exc:
        logger.error(f"agentar dispatch refused: {exc.detail}")
        raise AgentarError(503, "worker_unavailable", "The virtual participant worker is not configured") from exc


async def _live_holders() -> int:
    return await AgentarMeeting.count_documents({"live_lock": True})


async def _ensure_slot_free(owner: str) -> None:
    """Pre-check of one live Agentar per user and of the global cap. The unique index and the
    verify step after the claim are what hold under a race."""
    if await AgentarMeeting.find_one({"owner": owner, "live_lock": True}) is not None:
        raise AgentarError(409, "live_agentar_exists", "You already have an Agentar live in a meeting")
    if await _live_holders() >= settings.agentar.max_live_total:
        raise AgentarError(409, "agentar_capacity", "The Agentar capacity is in use; try again later")


def _restore_fields(previous: AgentarMeeting) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Set and unset lists that put a meeting back to what it was before a claim."""
    set_fields: dict[str, Any] = {"status": previous.status, "status_reason": previous.status_reason}
    unset: list[str] = ["live_lock"]
    for name in ("start_at", "dispatched_at", "bot_id", "jitsi_meeting_id"):
        value = getattr(previous, name)
        if value is None:
            unset.append(name)
        else:
            set_fields[name] = value
    return set_fields, tuple(unset)


async def dispatch_meeting(user: User, meeting_id: str) -> AgentarMeeting:
    """Send Agentar now, or arm the scheduled start.

    Checks, none of which changes anything: status, configuration (503), the start of a scheduled
    meeting (422), room ownership (409), and for `now` the live slot of the user and the global
    cap (409). A scheduled meeting is only armed: the tick applies the live and cap checks when
    its start comes, so arming for tomorrow is not refused because a meeting is live today.

    For `now` the meeting takes the user's live slot as `waiting_for_human` and the lifecycle
    takes over with `advance_waiting`: it looks for a human in the room, claims `joining` and
    calls the worker, or leaves the meeting waiting for the tick. A worker fault is recorded by
    the lifecycle as `failed`; the call still returns the meeting as it is afterwards.
    """
    meeting = await load_owned(meeting_id, user.email)
    if meeting.status not in DISPATCHABLE:
        raise invalid_state("Only a draft, missed or failed meeting can be sent")
    _require_dispatch_configuration()
    now = clock.utcnow()
    if meeting.mode == "scheduled" and (meeting.start_at is None or meeting.start_at <= now):
        raise AgentarError(422, "start_in_past", "The scheduled start has passed; edit the meeting first")
    await claim_room(meeting.room_name, user)

    resolved = {
        "display_name": f"{resolve_display_name(user)}{BOT_NAME_SUFFIX}",
        "voice": resolve_voice(user),
        "status_reason": None,
    }
    if meeting.mode == "scheduled":
        armed = await _transition(
            meeting.id,
            user.email,
            DISPATCHABLE,
            {"status": "scheduled", **resolved},
            unset_fields=_CLEAN_SLATE,
        )
        if armed is None:
            raise invalid_state("The meeting changed state; reload it")
        return armed

    await _ensure_slot_free(user.email)
    claim = {"status": "waiting_for_human", "live_lock": True, "start_at": now, **resolved}
    try:
        claimed = await _transition(meeting.id, user.email, DISPATCHABLE, claim, _CLEAN_SLATE)
    except DuplicateKeyError as exc:
        raise AgentarError(409, "live_agentar_exists", "You already have an Agentar live in a meeting") from exc
    if claimed is None:
        raise invalid_state("The meeting changed state; reload it")
    if await _live_holders() > settings.agentar.max_live_total:
        restore, unset = _restore_fields(meeting)
        await _transition(meeting.id, user.email, ("waiting_for_human",), restore, unset)
        raise AgentarError(409, "agentar_capacity", "The Agentar capacity is in use; try again later")

    # B7's module. Hoist to the top of the file once utils/agentar_lifecycle.py is merged.
    from utils.agentar_lifecycle import advance_waiting

    await advance_waiting(meeting.id)
    return await load_owned(meeting.id, user.email)


async def recall_meeting(
    user: User, meeting_id: str, client: Optional[httpx.AsyncClient] = None
) -> AgentarMeeting:
    """Stop Agentar. Armed or waiting: back to draft, no bot exists. Joining or live: the worker
    speaks the farewell and leaves, the meeting becomes `leaving` and the `left` event or the tick
    ends it. Leaving: idempotent. Anything else: 409.

    The worker is asked first. If it cannot be reached the call is 503 and the status is
    untouched, so the recall can be repeated; a status that says leaving must mean the bot was told.
    """
    meeting = await load_owned(meeting_id, user.email)
    if meeting.status == "leaving":
        return meeting
    if meeting.status in ("scheduled", "waiting_for_human"):
        draft = await _transition(
            meeting.id,
            user.email,
            ("scheduled", "waiting_for_human"),
            {"status": "draft", "status_reason": None, **({"start_at": None} if meeting.mode == "now" else {})},
            unset_fields=("live_lock",),
        )
        if draft is None:
            raise invalid_state("The meeting changed state; reload it")
        return draft
    if meeting.status not in ("joining", "live"):
        raise invalid_state("Only a scheduled or running meeting can be recalled")
    if not meeting.bot_id:
        raise invalid_state("The bot is still starting; try again in a moment")
    try:
        known = await recall_bot(meeting.bot_id, farewell=True, client=client)
    except WorkerDispatchError as exc:
        logger.warning(f"agentar recall failed meeting={meeting.id} {exc.detail}")
        raise AgentarError(503, "worker_unavailable", "The virtual participant worker is unavailable") from exc
    if known is None:
        logger.warning(f"agentar recall: worker does not know bot {meeting.bot_id} meeting={meeting.id}")
    leaving = await _transition(meeting.id, user.email, ("joining", "live"), {"status": "leaving"})
    if leaving is not None:
        return leaving
    current = await load_owned(meeting.id, user.email)
    if current.status in ("leaving", "ended"):
        return current
    raise invalid_state("The meeting changed state; reload it")
