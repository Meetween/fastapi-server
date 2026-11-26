"""Agentar lifecycle tick: claim, wait for a human, join, follow the bot, snapshot, end."""
import asyncio
from dataclasses import replace
from datetime import timedelta
from typing import Any, Optional

import pytest

from config import settings
from core import SessionManager
from src.database.models import ALL_MODELS, AgentarMeeting
from utils import agentar_lifecycle as lifecycle
from utils import clock

OWNER = "agentar-a@meetween.test"
OTHER = "agentar-b@meetween.test"
ROOM = "weekly"
JITSI_ID = "jitsi-meeting-1"


class FakeDeps:
    """Stands in for Prosody, the room owner check, the worker, the report job and the events."""

    def __init__(self) -> None:
        self.humans: dict[str, Optional[bool]] = {}
        self.owned = True
        self.dispatch_error: Optional[Exception] = None
        self.dispatch_state = "joining"
        self.bots: dict[str, Optional[dict[str, Any]]] = {}
        self.dispatched: list[str] = []
        self.removed: list[str] = []
        self.reports: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.ownership_checks = 0

    async def human_in_room(self, room: str) -> Optional[bool]:
        await asyncio.sleep(0)
        return self.humans.get(room, False)

    async def owns_room(self, meeting: AgentarMeeting) -> bool:
        self.ownership_checks += 1
        return self.owned

    async def dispatch_bot(self, meeting: AgentarMeeting) -> dict[str, Any]:
        # Yield so concurrent ticks interleave at the worker call.
        await asyncio.sleep(0)
        if self.dispatch_error is not None:
            raise self.dispatch_error
        bot_id = f"bot{len(self.dispatched) + 1}"
        self.dispatched.append(meeting.id)
        view = {"id": bot_id, "room": meeting.room_name, "display_name": "Alice Agentar", "state": self.dispatch_state}
        self.bots[bot_id] = view
        return dict(view)

    async def get_bot(self, bot_id: str) -> Optional[dict[str, Any]]:
        view = self.bots.get(bot_id)
        return dict(view) if view else None

    async def remove_bot(self, bot_id: str) -> None:
        self.removed.append(bot_id)

    async def enqueue_report(self, meeting_id: str) -> None:
        self.reports.append(meeting_id)

    async def record_event(self, **kwargs: Any) -> None:
        self.events.append(kwargs)

    def set_bot(self, bot_id: str, **fields: Any) -> None:
        self.bots[bot_id] = {**(self.bots.get(bot_id) or {"id": bot_id, "room": ROOM}), **fields}

    def deps(self) -> lifecycle.LifecycleDeps:
        return lifecycle.LifecycleDeps(
            human_in_room=self.human_in_room,
            owns_room=self.owns_room,
            dispatch_bot=self.dispatch_bot,
            get_bot=self.get_bot,
            remove_bot=self.remove_bot,
            enqueue_report=self.enqueue_report,
            record_event=self.record_event,
        )


@pytest.fixture(autouse=True)
def _indexes():
    for model in ALL_MODELS:
        asyncio.run(model.setup_indexes())


@pytest.fixture(autouse=True)
def _isolate_sessions(monkeypatch: pytest.MonkeyPatch):
    for name in ("_sessions", "_connections", "_caption_handlers", "_participant_locks", "_last_activity", "_ended_meetings"):
        monkeypatch.setattr(SessionManager, name, {})


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeDeps:
    fake = FakeDeps()
    monkeypatch.setattr(lifecycle, "default_deps", fake.deps)
    return fake


async def _meeting(**fields: Any) -> AgentarMeeting:
    defaults = dict(owner=OWNER, room_name=ROOM, mode="scheduled", status="scheduled", grace_minutes=15)
    defaults.update(fields)
    return await AgentarMeeting.create(**defaults)


async def _load(meeting: AgentarMeeting) -> AgentarMeeting:
    return await AgentarMeeting.find_by_id(meeting.id)


async def _live_meeting(fake: FakeDeps, **fields: Any) -> AgentarMeeting:
    fake.set_bot("bot9", state="joined", meeting_id=JITSI_ID, joined_at=1.0)
    defaults = dict(
        status="live", live_lock=True, bot_id="bot9", jitsi_meeting_id=JITSI_ID,
        start_at=clock.utcnow(), dispatched_at=clock.utcnow(), joined_at=clock.utcnow(),
    )
    defaults.update(fields)
    return await _meeting(**defaults)


def _say(text: str, meeting_id: str = JITSI_ID, who: str = "p1", at: float = 1.0) -> None:
    SessionManager.add_participants(meeting_id, ROOM, [{"participant_id": who, "username": "Bruno"}])
    SessionManager.update_transcript(meeting_id, ROOM, text, participant_id=who, timestamp=at)


# Scheduled to joining to live -------------------------------------------------------------------


async def test_scheduled_meeting_joins_at_start_and_goes_live(frozen_clock, fake):
    meeting = await _meeting(start_at=clock.utcnow() + timedelta(minutes=5))
    fake.humans[ROOM] = True

    await lifecycle.tick()
    assert (await _load(meeting)).status == "scheduled"
    assert fake.dispatched == []

    frozen_clock.advance(minutes=5)
    await lifecycle.tick()
    joining = await _load(meeting)
    assert joining.status == "joining"
    assert joining.bot_id == "bot1"
    assert joining.live_lock is True
    assert joining.dispatched_at == clock.utcnow()
    assert fake.dispatched == [meeting.id]
    assert fake.ownership_checks == 1

    frozen_clock.advance(seconds=30)
    fake.set_bot("bot1", state="joined", meeting_id=JITSI_ID, joined_at=2.0)
    await lifecycle.tick()
    live = await _load(meeting)
    assert live.status == "live"
    assert live.jitsi_meeting_id == JITSI_ID
    assert live.joined_at == clock.utcnow()
    assert fake.dispatched == [meeting.id]


async def test_meeting_not_yet_due_is_not_claimed(frozen_clock, fake):
    meeting = await _meeting(start_at=clock.utcnow() + timedelta(seconds=1))
    assert await lifecycle.claim_scheduled(meeting.id, fake.deps()) is None
    assert (await _load(meeting)).status == "scheduled"


# Nobody in the room ------------------------------------------------------------------------------


async def test_empty_room_waits_through_the_grace_window_then_missed(frozen_clock, fake):
    meeting = await _meeting(start_at=clock.utcnow(), grace_minutes=10)

    await lifecycle.tick()
    assert (await _load(meeting)).status == "waiting_for_human"

    frozen_clock.advance(minutes=10)
    await lifecycle.tick()
    assert (await _load(meeting)).status == "waiting_for_human"

    frozen_clock.advance(seconds=30)
    await lifecycle.tick()
    missed = await _load(meeting)
    assert missed.status == "missed"
    assert missed.status_reason == lifecycle.ROOM_STAYED_EMPTY
    assert missed.live_lock is None
    assert fake.dispatched == []
    assert fake.events == [{
        "owner": OWNER, "meeting_id": meeting.id, "type": "missed",
        "data": {"reason": "room_stayed_empty"},
        "event_id": f"{meeting.id}:missed:{int(meeting.start_at.timestamp())}",
    }]


async def test_human_arriving_inside_the_window_joins(frozen_clock, fake):
    meeting = await _meeting(start_at=clock.utcnow())
    await lifecycle.tick()
    frozen_clock.advance(minutes=7)
    fake.humans[ROOM] = True
    await lifecycle.tick()
    assert (await _load(meeting)).status == "joining"


async def test_unknown_presence_counts_as_empty(frozen_clock, fake):
    meeting = await _meeting(start_at=clock.utcnow())
    fake.humans[ROOM] = None
    await lifecycle.tick()
    assert (await _load(meeting)).status == "waiting_for_human"
    assert fake.dispatched == []


async def test_full_capacity_keeps_waiting_then_missed_with_capacity(frozen_clock, fake):
    assert settings.agentar.max_live_total == 1
    await _meeting(owner=OTHER, room_name="other", status="live", live_lock=True, bot_id="botx")
    fake.set_bot("botx", state="joined")
    meeting = await _meeting(start_at=clock.utcnow(), grace_minutes=1)
    fake.humans[ROOM] = True

    await lifecycle.tick()
    assert (await _load(meeting)).status == "waiting_for_human"
    frozen_clock.advance(minutes=2)
    await lifecycle.tick()
    missed = await _load(meeting)
    assert (missed.status, missed.status_reason) == ("missed", lifecycle.AGENTAR_CAPACITY)
    assert fake.dispatched == []


async def test_owner_with_another_live_agentar_is_missed_after_grace(frozen_clock, fake):
    await _meeting(room_name="first", status="live", live_lock=True, bot_id="botx")
    fake.set_bot("botx", state="joined")
    meeting = await _meeting(start_at=clock.utcnow(), grace_minutes=1)

    await lifecycle.tick()
    assert (await _load(meeting)).status == "scheduled"
    frozen_clock.advance(minutes=2)
    await lifecycle.tick()
    missed = await _load(meeting)
    assert (missed.status, missed.status_reason) == ("missed", lifecycle.ANOTHER_AGENTAR_LIVE)


# Ownership ---------------------------------------------------------------------------------------


async def test_room_owned_by_another_user_at_claim_fails(frozen_clock, fake):
    meeting = await _meeting(start_at=clock.utcnow())
    fake.humans[ROOM] = True
    fake.owned = False

    await lifecycle.tick()
    failed = await _load(meeting)
    assert (failed.status, failed.status_reason) == ("failed", lifecycle.ROOM_NOT_OWNED)
    assert failed.live_lock is None
    assert fake.dispatched == []
    assert [e["type"] for e in fake.events] == ["failed"]


async def test_ownership_check_error_hands_the_meeting_back(frozen_clock, fake):
    meeting = await _meeting(start_at=clock.utcnow())
    fake.humans[ROOM] = True

    async def broken(_meeting):
        raise RuntimeError("mongo down")

    deps = replace(fake.deps(), owns_room=broken)
    await lifecycle.tick(deps)
    back = await _load(meeting)
    assert back.status == "scheduled"
    assert back.live_lock is None
    assert fake.dispatched == []


# Worker failures ---------------------------------------------------------------------------------


async def test_worker_dispatch_error_fails_the_meeting(frozen_clock, fake):
    meeting = await _meeting(start_at=clock.utcnow())
    fake.humans[ROOM] = True
    fake.dispatch_error = RuntimeError("worker 502")

    await lifecycle.tick()
    failed = await _load(meeting)
    assert (failed.status, failed.status_reason) == ("failed", lifecycle.WORKER_ERROR)
    assert failed.live_lock is None
    assert failed.bot_id is None


@pytest.mark.parametrize(
    "bot_state, reason",
    [("error", lifecycle.WORKER_ERROR), ("lost", lifecycle.BOT_LOST), (None, lifecycle.BOT_LOST)],
)
async def test_bot_failing_while_joining_fails_the_meeting(frozen_clock, fake, bot_state, reason):
    meeting = await _meeting(start_at=clock.utcnow())
    fake.humans[ROOM] = True
    await lifecycle.tick()
    if bot_state is None:
        fake.bots["bot1"] = None  # the worker answers 404
    else:
        fake.set_bot("bot1", state=bot_state)

    await lifecycle.tick()
    failed = await _load(meeting)
    assert (failed.status, failed.status_reason) == ("failed", reason)
    assert failed.live_lock is None
    assert fake.reports == []


async def test_stuck_joining_times_out_and_removes_the_bot(frozen_clock, fake):
    meeting = await _meeting(start_at=clock.utcnow())
    fake.humans[ROOM] = True
    await lifecycle.tick()

    frozen_clock.advance(seconds=lifecycle.JOIN_TIMEOUT_SECONDS)
    await lifecycle.tick()
    assert (await _load(meeting)).status == "joining"

    frozen_clock.advance(seconds=1)
    await lifecycle.tick()
    failed = await _load(meeting)
    assert (failed.status, failed.status_reason) == ("failed", lifecycle.JOIN_TIMEOUT)
    assert fake.removed == ["bot1"]


async def test_recall_during_dispatch_removes_the_new_bot(frozen_clock, fake):
    meeting = await _meeting(status="waiting_for_human", live_lock=True, start_at=clock.utcnow())
    fake.humans[ROOM] = True
    real_dispatch = fake.dispatch_bot

    async def dispatch_then_recall(m):
        view = await real_dispatch(m)
        await AgentarMeeting.get_collection().update_one(
            {"status": "joining"}, {"$set": {"status": "leaving"}}
        )
        return view

    deps = replace(fake.deps(), dispatch_bot=dispatch_then_recall)
    await lifecycle.advance_waiting(meeting.id, deps)
    assert fake.removed == ["bot1"]
    assert (await _load(meeting)).bot_id == "bot1"


# Race --------------------------------------------------------------------------------------------


async def test_two_concurrent_ticks_send_one_bot(frozen_clock, fake):
    meeting = await _meeting(start_at=clock.utcnow())
    fake.humans[ROOM] = True

    await asyncio.gather(lifecycle.tick(), lifecycle.tick())

    assert fake.dispatched == [meeting.id]
    assert fake.ownership_checks == 1
    assert (await _load(meeting)).status == "joining"


async def test_claim_is_won_once(frozen_clock, fake):
    meeting = await _meeting(start_at=clock.utcnow())
    first, second = await asyncio.gather(
        lifecycle.claim_scheduled(meeting.id, fake.deps()),
        lifecycle.claim_scheduled(meeting.id, fake.deps()),
    )
    assert [m is not None for m in (first, second)].count(True) == 1


async def test_concurrent_joins_of_two_owners_respect_the_global_cap(frozen_clock, fake, monkeypatch):
    a = await _meeting(status="waiting_for_human", live_lock=True, start_at=clock.utcnow())
    b = await _meeting(owner=OTHER, room_name="other", status="waiting_for_human", live_lock=True, start_at=clock.utcnow())
    fake.humans.update({ROOM: True, "other": True})
    # Both processes read the count before either moved its meeting: the pre check passes twice.
    async def stale_count(exclude_id=None):
        return 0

    monkeypatch.setattr(lifecycle, "_live_total", stale_count)
    await lifecycle.advance_waiting(b.id, fake.deps())
    frozen_clock.advance(seconds=1)
    await lifecycle.advance_waiting(a.id, fake.deps())

    assert settings.agentar.max_live_total == 1
    assert fake.dispatched == [b.id]
    assert (await _load(b)).status == "joining"
    backed_off = await _load(a)
    assert backed_off.status == "waiting_for_human"
    assert backed_off.dispatched_at is None
    assert backed_off.live_lock is True


# Snapshot ----------------------------------------------------------------------------------------


async def test_snapshot_survives_clear_session(frozen_clock, fake):
    meeting = await _live_meeting(fake)
    _say("We ship on Friday.")
    _say("Bruno owns the release notes.", at=2.0)

    await lifecycle.tick()
    stored = await _load(meeting)
    assert [line["text"] for line in stored.transcript_snapshot] == [
        "We ship on Friday.", "Bruno owns the release notes.",
    ]
    assert stored.transcript_snapshot[0] == {
        "participant_id": "p1", "username": "Bruno", "text": "We ship on Friday.",
        "timestamp": 1.0, "source": "speech",
    }

    SessionManager.clear_session(JITSI_ID, ROOM, force=True)
    assert SessionManager.get_transcript(JITSI_ID, ROOM) is None

    fake.set_bot("bot9", state="left")
    await lifecycle.tick()
    ended = await _load(meeting)
    assert ended.status == "ended"
    assert ended.report_status == "pending"
    assert ended.live_lock is None
    assert len(ended.transcript_snapshot) == 2
    assert fake.reports == [meeting.id]


async def test_snapshot_binds_the_jitsi_id_by_room(frozen_clock, fake):
    meeting = await _live_meeting(fake, jitsi_meeting_id=None)
    fake.set_bot("bot9", state="joined", meeting_id=None)
    _say("hello")

    await lifecycle.tick()
    stored = await _load(meeting)
    assert stored.jitsi_meeting_id == JITSI_ID
    assert [line["text"] for line in stored.transcript_snapshot] == ["hello"]


async def test_bot_lost_while_live_ends_the_meeting(frozen_clock, fake):
    meeting = await _live_meeting(fake)
    _say("one line")
    fake.bots["bot9"] = None

    await lifecycle.tick()
    ended = await _load(meeting)
    assert ended.status == "ended"
    assert [line["text"] for line in ended.transcript_snapshot] == ["one line"]
    assert fake.reports == [meeting.id]


# End hook ----------------------------------------------------------------------------------------


async def test_jigasi_end_flushes_the_transcript_before_clearing(frozen_clock, fake):
    meeting = await _live_meeting(fake)
    _say("before the tick")
    await lifecycle.tick()
    _say("after the tick", at=2.0)

    tasks: list[asyncio.Task] = []
    real_hook = lifecycle.on_meeting_ended

    def spy(meeting_id):
        task = real_hook(meeting_id)
        tasks.append(task)
        return task

    lifecycle_module_hook = pytest.MonkeyPatch()
    lifecycle_module_hook.setattr(lifecycle, "on_meeting_ended", spy)
    try:
        assert await SessionManager.mark_meeting_ended(JITSI_ID) is True
        assert SessionManager.get_transcript(JITSI_ID, ROOM) is None
        await asyncio.gather(*tasks)
    finally:
        lifecycle_module_hook.undo()

    ended = await _load(meeting)
    assert ended.status == "ended"
    assert ended.ended_at == clock.utcnow()
    assert [line["text"] for line in ended.transcript_snapshot] == ["before the tick", "after the tick"]
    assert fake.reports == [meeting.id]

    # The bot leaves later and the tick sees it: nothing changes, no second report.
    fake.set_bot("bot9", state="left")
    await lifecycle.tick()
    assert fake.reports == [meeting.id]


async def test_end_hook_ends_an_unbound_meeting_by_room(frozen_clock, fake):
    meeting = await _live_meeting(fake, jitsi_meeting_id=None)
    _say("only line")
    task = lifecycle.on_meeting_ended(JITSI_ID)
    await task
    ended = await _load(meeting)
    assert ended.status == "ended"
    assert ended.jitsi_meeting_id == JITSI_ID


async def test_end_hook_without_an_agentar_meeting_is_harmless(frozen_clock, fake):
    _say("plain meeting")
    task = lifecycle.on_meeting_ended(JITSI_ID)
    await task
    assert fake.reports == []
    assert await AgentarMeeting.count_documents({}) == 0


async def test_mark_agentar_ended_is_idempotent(frozen_clock, fake):
    meeting = await _live_meeting(fake, status="leaving")
    _say("line")
    await lifecycle.mark_agentar_ended(meeting.id, "left")
    await lifecycle.mark_agentar_ended(meeting.id, "left")
    ended = await _load(meeting)
    assert ended.status == "ended"
    assert fake.reports == [meeting.id]


async def test_mark_agentar_ended_ignores_meetings_not_live(frozen_clock, fake):
    meeting = await _meeting(status="joining", live_lock=True, bot_id="bot1")
    await lifecycle.mark_agentar_ended(meeting.id, "left")
    await lifecycle.mark_agentar_ended("not-an-id", "left")
    assert (await _load(meeting)).status == "joining"
    assert fake.reports == []


async def test_report_queue_failure_marks_the_report_failed(frozen_clock, fake):
    meeting = await _live_meeting(fake)

    async def broken(_meeting_id):
        raise RuntimeError("queue down")

    deps = replace(fake.deps(), enqueue_report=broken)
    await lifecycle.mark_agentar_ended(meeting.id, "left", deps=deps)
    ended = await _load(meeting)
    assert (ended.status, ended.report_status) == ("ended", "failed")


async def test_one_failing_meeting_does_not_stop_the_tick(frozen_clock, fake):
    bad = await _meeting(room_name="bad", start_at=clock.utcnow())
    good = await _meeting(owner=OTHER, room_name="good", start_at=clock.utcnow())
    fake.humans["good"] = True

    async def human_in_room(room):
        if room == "bad":
            raise RuntimeError("boom")
        return fake.humans.get(room, False)

    deps = replace(fake.deps(), human_in_room=human_in_room)
    await lifecycle.tick(deps)
    assert (await _load(bad)).status == "waiting_for_human"
    assert (await _load(good)).status == "joining"


# Human in the room through the virtual participant route ---------------------------------------


@pytest.fixture
def vp_route(monkeypatch: pytest.MonkeyPatch):
    from api.routes import virtual_participants as route

    async def client():
        return object()

    monkeypatch.setattr(route, "get_vp_http_client", client)
    return route


def _require_human(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    vp = settings.virtual_participants.model_copy(update={"require_human": value})
    monkeypatch.setattr(settings, "_virtual_participants", vp)


@pytest.mark.parametrize("humans, expected", [(0, False), (2, True)])
async def test_human_in_room_reads_the_prosody_count(monkeypatch, vp_route, humans, expected):
    _require_human(monkeypatch, True)
    rooms: list[str] = []

    async def count(client, vp, room):
        rooms.append(room)
        return humans

    monkeypatch.setattr(vp_route, "_count_humans", count)
    assert await lifecycle._human_in_room(ROOM) is expected
    assert rooms == [ROOM]


async def test_human_in_room_is_unknown_when_prosody_fails(monkeypatch, vp_route):
    from fastapi import HTTPException

    _require_human(monkeypatch, True)

    async def count(client, vp, room):
        raise HTTPException(status_code=503, detail="room presence lookup unavailable")

    monkeypatch.setattr(vp_route, "_count_humans", count)
    assert await lifecycle._human_in_room(ROOM) is None


async def test_human_in_room_is_skipped_without_vp_require_human(monkeypatch, vp_route):
    _require_human(monkeypatch, False)

    async def count(client, vp, room):
        raise AssertionError("Prosody must not be asked")

    monkeypatch.setattr(vp_route, "_count_humans", count)
    assert await lifecycle._human_in_room(ROOM) is True
