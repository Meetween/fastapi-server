"""Agentar meeting lifecycle: the periodic tick and the end of a meeting.

The tick runs every ``AGENTAR_TICK_SECONDS`` (default 30) and does three things:

1. Claims due ``scheduled`` meetings. The claim is one ``find_one_and_update`` filtered on
   ``status: scheduled``, so of two processes sharing the collection exactly one wins. It sets
   ``live_lock``, whose partial unique index holds one live Agentar per owner. Room ownership is
   checked again after the claim (A3 H1): a room now held by another user fails the meeting.
2. Advances ``waiting_for_human`` meetings. A human in the room (the Prosody count the virtual
   participant route uses, skipped when ``VP_REQUIRE_HUMAN`` is off) and a free slot under
   ``AGENTAR_MAX_LIVE_TOTAL`` move the meeting to ``joining`` and call the worker. Past
   ``start_at + grace_minutes`` the meeting is ``missed``.
3. Reconciles ``joining``, ``live`` and ``leaving`` meetings with the worker's bot state: binds
   the Jitsi meeting id, moves ``joining`` to ``live``, fails a bot that errors or never joins,
   copies the live transcript from ``SessionManager`` into ``transcript_snapshot``, and ends the
   meeting when the bot is gone.

Every status change is a ``find_one_and_update`` filtered on the expected status, so a recall, a
worker event or a second process that moved the meeting first always wins and the tick backs off.

The end of a meeting has three entry points, all idempotent and all funnelled into one update:
the tick seeing a terminal bot, the worker ``left`` event (B10 calls ``mark_agentar_ended``), and
Jigasi's ``/meetings/{id}/ended`` through ``on_meeting_ended``, the one hook line in
``SessionManager.mark_meeting_ended``. That hook copies the transcript before the session is
cleared, so the report reads it from Mongo and never from memory.

Collaborators of other modules are reached through ``LifecycleDeps`` and imported lazily, so this
module imports neither ``core`` nor the API routes at load time and tests swap them for fakes.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable, Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from config import settings
from src.database.models.agentar_meeting import AgentarMeeting
from utils import clock

logger = logging.getLogger(__name__)

# A bot still joining this long after dispatch is failed with join_timeout. The worker gives up
# on its own after VP_JOIN_TIMEOUT_S (45 s) plus the Ditto setup (60 s); this is the backstop for
# a worker that never reports.
JOIN_TIMEOUT_SECONDS = 180
# Due scheduled meetings claimed per tick. The rest wait for the next tick.
CLAIM_BATCH = 50

# Worker bot states (vp-worker BotState).
BOT_JOINED_STATES = frozenset({"joined", "speaking"})
BOT_TERMINAL_STATES = frozenset({"left", "error", "lost"})

# status_reason values of the contract (Meeting.status_reason).
ROOM_STAYED_EMPTY = "room_stayed_empty"
ANOTHER_AGENTAR_LIVE = "another_agentar_live"
AGENTAR_CAPACITY = "agentar_capacity"
ROOM_NOT_OWNED = "room_not_owned"
WORKER_ERROR = "worker_error"
JOIN_TIMEOUT = "join_timeout"
BOT_LOST = "bot_lost"

# Reasons passed to mark_agentar_ended by this module.
END_BOT_LEFT = "bot_left"
END_BOT_LOST = "bot_lost"
END_MEETING_ENDED = "meeting_ended"

ACTIVE_STATUSES = ("joining", "live", "leaving")
ENDABLE_STATUSES = ("live", "leaving")


@dataclass(frozen=True)
class LifecycleDeps:
    """What the lifecycle needs from other modules. ``default_deps()`` wires the real ones.

    - ``human_in_room(room)``: True with a human present, False without, None when presence is
      unknown (Prosody unreachable). Unknown is treated as empty.
    - ``owns_room(meeting)``: the owner still holds the room (``claim_room_moderator``).
    - ``dispatch_bot(meeting)``: sends the bot, returns the worker bot view (``id``, ``state``).
    - ``get_bot(bot_id)``: the worker bot view, or None when the worker no longer knows the bot.
    - ``remove_bot(bot_id)``: makes a bot leave without the farewell.
    - ``enqueue_report(meeting_id)``: queues the report job of an ended meeting.
    - ``record_event(owner=, meeting_id=, type=, data=, event_id=)``: a server side notification.
    """

    human_in_room: Callable[[str], Awaitable[Optional[bool]]]
    owns_room: Callable[[AgentarMeeting], Awaitable[bool]]
    dispatch_bot: Callable[[AgentarMeeting], Awaitable[dict[str, Any]]]
    get_bot: Callable[[str], Awaitable[Optional[dict[str, Any]]]]
    remove_bot: Callable[[str], Awaitable[None]]
    enqueue_report: Callable[[str], Awaitable[None]]
    record_event: Callable[..., Awaitable[None]]


# Real collaborators -----------------------------------------------------------------------------


async def _human_in_room(room: str) -> Optional[bool]:
    from fastapi import HTTPException

    from api.routes import virtual_participants as vp_route

    vp = settings.virtual_participants
    if not vp.require_human:
        return True
    client = await vp_route.get_vp_http_client()
    try:
        return await vp_route._count_humans(client, vp, room) > 0
    except HTTPException:
        # The route already logged the cause. The meeting keeps waiting.
        return None


async def _owns_room(meeting: AgentarMeeting) -> bool:
    from src.database.models.user import User
    from utils.room_owner import claim_room_moderator  # B2

    user = await User.find_by_email(meeting.owner)
    if user is None:
        return False
    return bool(await claim_room_moderator(meeting.room_name, user))


async def _dispatch_bot(meeting: AgentarMeeting) -> dict[str, Any]:
    from utils.agentar_worker import dispatch_bot  # B6

    return await dispatch_bot(meeting)


async def _get_bot(bot_id: str) -> Optional[dict[str, Any]]:
    from utils.agentar_worker import get_bot  # B6

    return await get_bot(bot_id)


async def _remove_bot(bot_id: str) -> None:
    from utils.agentar_worker import recall_bot  # B6

    await recall_bot(bot_id, farewell=False)


async def _enqueue_report(meeting_id: str) -> None:
    from utils.agentar_report import enqueue_report  # B9

    result = enqueue_report(meeting_id)
    if inspect.isawaitable(result):
        await result


async def _record_event(**kwargs: Any) -> None:
    from utils.agentar_events import record_server_event  # B10

    await record_server_event(**kwargs)


def default_deps() -> LifecycleDeps:
    return LifecycleDeps(
        human_in_room=_human_in_room,
        owns_room=_owns_room,
        dispatch_bot=_dispatch_bot,
        get_bot=_get_bot,
        remove_bot=_remove_bot,
        enqueue_report=_enqueue_report,
        record_event=_record_event,
    )


# Mongo helpers ----------------------------------------------------------------------------------


def _collection():
    return AgentarMeeting.get_collection()


def _oid(meeting_id: str | ObjectId) -> Optional[ObjectId]:
    if isinstance(meeting_id, ObjectId):
        return meeting_id
    try:
        return ObjectId(meeting_id)
    except (InvalidId, TypeError):
        return None


async def _transition(
    meeting_id: str | ObjectId,
    from_statuses: Iterable[str],
    set_fields: dict[str, Any],
    *,
    unset: Iterable[str] = (),
    extra_filter: Optional[dict[str, Any]] = None,
) -> Optional[AgentarMeeting]:
    """Apply one change to a meeting still in one of ``from_statuses``. Returns the meeting after
    the change, or None when it had already moved on. ``DuplicateKeyError`` from the live lock or
    the bot id index is propagated."""
    oid = _oid(meeting_id)
    if oid is None:
        return None
    query: dict[str, Any] = {"_id": oid, "status": {"$in": list(from_statuses)}}
    if extra_filter:
        query.update(extra_filter)
    update: dict[str, Any] = {"$set": {**set_fields, "updated_at": clock.utcnow()}}
    unset_fields = list(unset)
    if unset_fields:
        update["$unset"] = {name: "" for name in unset_fields}
    doc = await _collection().find_one_and_update(query, update, return_document=ReturnDocument.AFTER)
    if doc is None:
        return None
    meeting = AgentarMeeting.from_dict(doc)
    if "status" in set_fields:
        logger.info(
            "agentar meeting %s -> %s reason=%s", meeting.id, meeting.status, meeting.status_reason
        )
    return meeting


async def _find(query: dict[str, Any], limit: Optional[int] = None) -> list[AgentarMeeting]:
    cursor = _collection().find(query).sort("start_at", 1)
    if limit:
        cursor = cursor.limit(limit)
    return [AgentarMeeting.from_dict(doc) for doc in await cursor.to_list(length=limit)]


def _deadline(meeting: AgentarMeeting) -> datetime:
    start = meeting.start_at or meeting.created_at
    return start + timedelta(minutes=meeting.grace_minutes)


def _attempt_key(meeting: AgentarMeeting) -> int:
    """Identifies one dispatch of a meeting, so a re-dispatched meeting gets fresh event ids."""
    start = meeting.start_at or meeting.dispatched_at or meeting.created_at
    return int(start.timestamp())


async def _notify(deps: LifecycleDeps, meeting: AgentarMeeting, event_type: str) -> None:
    try:
        await deps.record_event(
            owner=meeting.owner,
            meeting_id=meeting.id,
            type=event_type,
            data={"reason": meeting.status_reason},
            event_id=f"{meeting.id}:{event_type}:{_attempt_key(meeting)}",
        )
    except Exception:
        logger.exception("agentar meeting %s: %s notification not recorded", meeting.id, event_type)


async def _close_unstarted(
    deps: LifecycleDeps,
    meeting_id: str,
    from_statuses: Iterable[str],
    status: str,
    reason: str,
) -> Optional[AgentarMeeting]:
    """Move a meeting that never went live to missed or failed and release the live slot."""
    closed = await _transition(
        meeting_id, from_statuses, {"status": status, "status_reason": reason}, unset=("live_lock",)
    )
    if closed is not None:
        await _notify(deps, closed, status)
    return closed


# Tick -------------------------------------------------------------------------------------------


async def tick(deps: Optional[LifecycleDeps] = None) -> None:
    """One lifecycle pass. Registered as an interval job in ``main.startup_event``.

    A failure on one meeting is logged and never stops the pass for the others.
    """
    deps = deps or default_deps()

    due = await _find(
        {"status": "scheduled", "start_at": {"$lte": clock.utcnow()}}, limit=CLAIM_BATCH
    )
    for meeting in due:
        await _guarded("claim", meeting.id, claim_scheduled(meeting.id, deps))

    for meeting in await _find({"status": "waiting_for_human"}):
        await _guarded("advance", meeting.id, advance_waiting(meeting.id, deps))

    for meeting in await _find({"status": {"$in": list(ACTIVE_STATUSES)}}):
        await _guarded("reconcile", meeting.id, reconcile(meeting, deps))


async def _guarded(step: str, meeting_id: Optional[str], coro: Awaitable[Any]) -> None:
    try:
        await coro
    except Exception:
        logger.exception("agentar tick %s failed for meeting %s", step, meeting_id)


async def claim_scheduled(meeting_id: str, deps: LifecycleDeps) -> Optional[AgentarMeeting]:
    """Claim one due scheduled meeting: ``scheduled`` to ``waiting_for_human`` with the live lock.

    Returns the claimed meeting, or None when another process claimed it, it was recalled, the
    owner already has a live Agentar, or the room is no longer the owner's.
    """
    now = clock.utcnow()
    try:
        claimed = await _transition(
            meeting_id,
            ("scheduled",),
            {"status": "waiting_for_human", "live_lock": True, "status_reason": None},
            extra_filter={"start_at": {"$lte": now}},
        )
    except DuplicateKeyError:
        # The owner's live slot is held by another meeting. It stays scheduled and is retried by
        # every tick until its grace window closes.
        meeting = await AgentarMeeting.find_by_id(str(meeting_id))
        if meeting is not None and meeting.status == "scheduled" and now > _deadline(meeting):
            await _close_unstarted(deps, meeting.id, ("scheduled",), "missed", ANOTHER_AGENTAR_LIVE)
        return None
    if claimed is None:
        return None

    try:
        owned = await deps.owns_room(claimed)
    except Exception:
        # Unknown ownership never sends a bot: hand the meeting back for the next tick.
        logger.exception("agentar meeting %s: room ownership check failed", claimed.id)
        await _transition(claimed.id, ("waiting_for_human",), {"status": "scheduled"}, unset=("live_lock",))
        return None
    if not owned:
        await _close_unstarted(deps, claimed.id, ("waiting_for_human",), "failed", ROOM_NOT_OWNED)
        return None
    return claimed


async def _live_total(exclude_id: Optional[str] = None) -> int:
    query: dict[str, Any] = {"status": {"$in": list(ACTIVE_STATUSES)}}
    oid = _oid(exclude_id) if exclude_id else None
    if oid is not None:
        query["_id"] = {"$ne": oid}
    return await _collection().count_documents(query)


async def _join_blocker(meeting: AgentarMeeting, deps: LifecycleDeps) -> Optional[str]:
    """The reason the bot cannot be sent now, or None when it can."""
    if await deps.human_in_room(meeting.room_name) is not True:
        return ROOM_STAYED_EMPTY
    if await _live_total(exclude_id=meeting.id) >= settings.agentar.max_live_total:
        return AGENTAR_CAPACITY
    return None


async def advance_waiting(meeting_id: str, deps: Optional[LifecycleDeps] = None) -> Optional[AgentarMeeting]:
    """One step for a ``waiting_for_human`` meeting: send the bot, keep waiting, or miss.

    Also the inline first step of a mode ``now`` dispatch: the dispatch route sets
    ``waiting_for_human`` with the live lock and calls this. Returns the meeting as it stands.
    """
    deps = deps or default_deps()
    meeting = await AgentarMeeting.find_by_id(str(meeting_id))
    if meeting is None or meeting.status != "waiting_for_human":
        return meeting

    blocker = await _join_blocker(meeting, deps)
    if clock.utcnow() > _deadline(meeting):
        return await _close_unstarted(
            deps, meeting.id, ("waiting_for_human",), "missed", blocker or ROOM_STAYED_EMPTY
        ) or await AgentarMeeting.find_by_id(meeting.id)
    if blocker is not None:
        return meeting
    return await _start_join(meeting, deps)


async def _within_capacity(meeting: AgentarMeeting) -> bool:
    """Whether this joining meeting holds one of the ``AGENTAR_MAX_LIVE_TOTAL`` slots.

    Two processes may both pass the pre check and both move a meeting to joining. Ranking the
    active meetings (established ones first, then joining ones by dispatch time and id) gives
    every process the same answer, so exactly the extra ones back off.
    """
    docs = await _collection().find(
        {"status": {"$in": list(ACTIVE_STATUSES)}}, {"_id": 1, "status": 1, "dispatched_at": 1}
    ).to_list(length=None)

    def rank(doc: dict[str, Any]) -> tuple[int, datetime, str]:
        dispatched = doc.get("dispatched_at")
        dispatched = clock.as_utc(dispatched) if dispatched else datetime.min.replace(tzinfo=timezone.utc)
        return (0 if doc.get("status") != "joining" else 1, dispatched, str(doc["_id"]))

    ordered = [str(doc["_id"]) for doc in sorted(docs, key=rank)]
    return meeting.id in ordered[: settings.agentar.max_live_total]


async def _start_join(meeting: AgentarMeeting, deps: LifecycleDeps) -> Optional[AgentarMeeting]:
    """``waiting_for_human`` to ``joining``, then call the worker and bind the bot id.

    The status moves before the worker call: the claim is what stops two processes from both
    sending a bot.
    """
    joining = await _transition(
        meeting.id,
        ("waiting_for_human",),
        {"status": "joining", "dispatched_at": clock.utcnow(), "status_reason": None},
    )
    if joining is None:
        return await AgentarMeeting.find_by_id(meeting.id)
    if not await _within_capacity(joining):
        logger.info("agentar meeting %s: capacity taken by a concurrent dispatch, waiting", joining.id)
        return await _transition(
            joining.id,
            ("joining",),
            {"status": "waiting_for_human", "dispatched_at": None},
            extra_filter={"bot_id": None},
        )

    try:
        view = await deps.dispatch_bot(joining)
    except Exception as exc:
        logger.warning("agentar meeting %s: worker dispatch failed: %r", joining.id, exc)
        return await _close_unstarted(deps, joining.id, ("joining",), "failed", WORKER_ERROR)

    bot_id = view.get("id") if isinstance(view, dict) else None
    if not isinstance(bot_id, str) or not bot_id:
        logger.error("agentar meeting %s: worker answered without a bot id", joining.id)
        return await _close_unstarted(deps, joining.id, ("joining",), "failed", WORKER_ERROR)

    try:
        bound = await _transition(joining.id, ("joining", "leaving"), {"bot_id": bot_id})
    except DuplicateKeyError:
        # The worker handed back a bot that another meeting owns (same name in the same room).
        # It is not ours to remove.
        logger.error("agentar meeting %s: worker returned bot %s of another meeting", joining.id, bot_id)
        return await _close_unstarted(deps, joining.id, ("joining",), "failed", WORKER_ERROR)

    if bound is None or bound.status == "leaving":
        # Recalled while the worker call was in flight: the recall had no bot id to remove.
        logger.info("agentar meeting %s recalled during dispatch, removing bot %s", joining.id, bot_id)
        await deps.remove_bot(bot_id)
        return bound or await AgentarMeeting.find_by_id(joining.id)

    return await _apply_bot_view(bound, view, deps)


async def reconcile(meeting: AgentarMeeting, deps: LifecycleDeps) -> Optional[AgentarMeeting]:
    """Follow the worker state of a joining, live or leaving meeting."""
    if not meeting.bot_id:
        # A joining meeting whose dispatch never bound a bot (the process stopped mid call).
        if meeting.status == "joining" and _join_timed_out(meeting):
            return await _close_unstarted(deps, meeting.id, ("joining",), "failed", JOIN_TIMEOUT)
        if meeting.status in ENDABLE_STATUSES:
            await snapshot_transcript(meeting)
        return meeting

    try:
        view = await deps.get_bot(meeting.bot_id)
    except Exception as exc:
        logger.warning("agentar meeting %s: bot %s lookup failed: %r", meeting.id, meeting.bot_id, exc)
        if meeting.status == "joining" and _join_timed_out(meeting):
            return await _fail_stuck_join(meeting, deps)
        if meeting.status in ENDABLE_STATUSES:
            await snapshot_transcript(meeting)
        return meeting
    return await _apply_bot_view(meeting, view, deps)


def _join_timed_out(meeting: AgentarMeeting) -> bool:
    started = meeting.dispatched_at or meeting.updated_at
    return clock.utcnow() - started > timedelta(seconds=JOIN_TIMEOUT_SECONDS)


async def _fail_stuck_join(meeting: AgentarMeeting, deps: LifecycleDeps) -> Optional[AgentarMeeting]:
    closed = await _close_unstarted(deps, meeting.id, ("joining",), "failed", JOIN_TIMEOUT)
    if closed is not None and meeting.bot_id:
        try:
            await deps.remove_bot(meeting.bot_id)
        except Exception as exc:
            logger.warning("agentar meeting %s: removing stuck bot failed: %r", meeting.id, exc)
    return closed


async def _apply_bot_view(
    meeting: AgentarMeeting, view: Optional[dict[str, Any]], deps: LifecycleDeps
) -> Optional[AgentarMeeting]:
    state = view.get("state") if isinstance(view, dict) else None
    jitsi_id = view.get("meeting_id") if isinstance(view, dict) else None
    joined_once = isinstance(view, dict) and view.get("joined_at") is not None

    if meeting.status == "joining":
        if state in BOT_JOINED_STATES or (state == "left" and joined_once):
            fields: dict[str, Any] = {"status": "live", "joined_at": clock.utcnow()}
            if jitsi_id and not meeting.jitsi_meeting_id:
                fields["jitsi_meeting_id"] = jitsi_id
            live = await _transition(meeting.id, ("joining",), fields)
            if live is None:
                return await AgentarMeeting.find_by_id(meeting.id)
            meeting = live
        elif state is None or state in BOT_TERMINAL_STATES:
            reason = BOT_LOST if state in (None, "lost") else WORKER_ERROR
            return await _close_unstarted(deps, meeting.id, ("joining",), "failed", reason)
        else:
            if _join_timed_out(meeting):
                return await _fail_stuck_join(meeting, deps)
            return meeting

    if meeting.status not in ENDABLE_STATUSES:
        return meeting
    if jitsi_id and not meeting.jitsi_meeting_id:
        meeting = await _transition(
            meeting.id, ENDABLE_STATUSES, {"jitsi_meeting_id": jitsi_id},
            extra_filter={"jitsi_meeting_id": None},
        ) or meeting
    if state is None or state in BOT_TERMINAL_STATES:
        reason = END_BOT_LEFT if state == "left" else END_BOT_LOST
        await mark_agentar_ended(meeting.id, reason, deps=deps)
        return await AgentarMeeting.find_by_id(meeting.id)
    await snapshot_transcript(meeting)
    return meeting


# Transcript snapshot ----------------------------------------------------------------------------


def _entry_dict(entry: Any) -> dict[str, Any]:
    """One snapshot line. Same fields as ``core.session_manager.TranscriptEntry``."""
    return {
        "participant_id": entry.participant_id,
        "username": entry.username,
        "text": entry.text,
        "timestamp": entry.timestamp,
        "source": entry.source,
    }


def _session_entries(session: Any) -> list[dict[str, Any]]:
    if session is None or not session.transcript:
        return []
    return [_entry_dict(entry) for entry in session.transcript]


def _find_session(meeting: AgentarMeeting) -> Any:
    """The in-memory session of a meeting: by the bound Jitsi id, else by room name.

    Without a bound id, one session for the room is taken as it is; with several (a room reused
    before the idle sweep cleared the old one) the one with an open Jigasi socket wins, and
    otherwise none is taken.
    """
    from core.session_manager import SessionManager

    sessions = SessionManager._sessions
    if meeting.jitsi_meeting_id:
        return sessions.get(meeting.jitsi_meeting_id)
    room = meeting.room_name.casefold()
    candidates = [s for s in list(sessions.values()) if (s.room_name or "").casefold() == room]
    if len(candidates) > 1:
        candidates = [s for s in candidates if s.meeting_id in SessionManager._connections]
    return candidates[0] if len(candidates) == 1 else None


async def snapshot_transcript(meeting: AgentarMeeting) -> bool:
    """Copy the live transcript into ``transcript_snapshot`` when it changed. Returns True on a
    write. An empty or missing session never overwrites a stored snapshot."""
    session = _find_session(meeting)
    entries = _session_entries(session)
    fields: dict[str, Any] = {}
    if session is not None and not meeting.jitsi_meeting_id:
        fields["jitsi_meeting_id"] = session.meeting_id
    if entries and entries != (meeting.transcript_snapshot or []):
        fields["transcript_snapshot"] = entries
    if not fields:
        return False
    return await _transition(meeting.id, ENDABLE_STATUSES, fields) is not None


# End of a meeting -------------------------------------------------------------------------------


async def mark_agentar_ended(
    meeting_id: str,
    reason: str,
    *,
    entries: Optional[list[dict[str, Any]]] = None,
    jitsi_meeting_id: Optional[str] = None,
    deps: Optional[LifecycleDeps] = None,
) -> None:
    """End a live or leaving meeting and queue its report. Idempotent: a meeting in any other
    status is left as it is and no second report is queued.

    ``entries`` is a transcript copied by the caller; without it the in-memory session is read.
    Called by the tick, by the worker ``left`` event (B10) and by ``on_meeting_ended``.
    """
    deps = deps or default_deps()
    meeting = await AgentarMeeting.find_by_id(str(meeting_id))
    if meeting is None or meeting.status not in ENDABLE_STATUSES:
        return
    if entries is None:
        entries = _session_entries(_find_session(meeting))

    fields: dict[str, Any] = {
        "status": "ended",
        "status_reason": None,
        "ended_at": clock.utcnow(),
        "report_status": "pending",
    }
    if entries:
        fields["transcript_snapshot"] = entries
    if jitsi_meeting_id and not meeting.jitsi_meeting_id:
        fields["jitsi_meeting_id"] = jitsi_meeting_id
    ended = await _transition(meeting.id, ENDABLE_STATUSES, fields, unset=("live_lock",))
    if ended is None:
        return
    logger.info(
        "agentar meeting %s ended reason=%s snapshot_lines=%d",
        ended.id, reason, len(ended.transcript_snapshot or []),
    )
    try:
        await deps.enqueue_report(ended.id)
    except Exception:
        logger.exception("agentar meeting %s: report job not queued", ended.id)
        await _transition(ended.id, ("ended",), {"report_status": "failed"})


async def _end_for_jitsi_meeting(
    jitsi_meeting_id: str, room_name: Optional[str], entries: list[dict[str, Any]]
) -> None:
    """Jigasi reported the end of a Jitsi meeting: end the Agentar meeting bound to it, or the
    unbound live one in the same room."""
    query: dict[str, Any] = {"status": {"$in": list(ENDABLE_STATUSES)}}
    meeting = await AgentarMeeting.find_one({**query, "jitsi_meeting_id": jitsi_meeting_id})
    if meeting is None and room_name:
        meeting = await AgentarMeeting.find_one(
            {**query, "jitsi_meeting_id": None, "room_name": room_name.lower()}
        )
    if meeting is None:
        return
    await mark_agentar_ended(
        meeting.id, END_MEETING_ENDED, entries=entries, jitsi_meeting_id=jitsi_meeting_id
    )


# Strong references to running flush tasks, so the event loop does not drop them.
_flush_tasks: set[asyncio.Task] = set()


def on_meeting_ended(jitsi_meeting_id: str) -> Optional[asyncio.Task]:
    """Hook in ``SessionManager.mark_meeting_ended``, called before the session is cleared.

    Copies the transcript synchronously, then ends the matching Agentar meeting in a task so the
    Jigasi request never waits on Mongo. Never raises. Returns the task, for tests.
    """
    try:
        from core.session_manager import SessionManager

        session = SessionManager._sessions.get(jitsi_meeting_id)
        entries = _session_entries(session)
        room_name = session.room_name if session is not None else None
        task = asyncio.get_running_loop().create_task(
            _guarded("end hook", jitsi_meeting_id, _end_for_jitsi_meeting(jitsi_meeting_id, room_name, entries))
        )
        _flush_tasks.add(task)
        task.add_done_callback(_flush_tasks.discard)
        return task
    except Exception:
        logger.exception("agentar end hook failed for meeting %s", jitsi_meeting_id)
        return None
