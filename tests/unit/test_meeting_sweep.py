"""Idle sweep: clears meetings with no open websocket and no recent activity."""

import pytest

from config import settings
from core import SessionManager

MEETING_ID = "5506be1e-b870-4922-845d-d053b1d0b4db"
IDLE_TIMEOUT = 600
RETENTION = 100


class _Lock:
    """Stands in for asyncio.Lock: only `locked()` matters to the sweep."""

    def __init__(self, held: bool):
        self._held = held

    def locked(self) -> bool:
        return self._held


@pytest.fixture(autouse=True)
def _isolate_session_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "meeting_idle_timeout_seconds", IDLE_TIMEOUT)
    monkeypatch.setattr(settings, "ended_meeting_retention_seconds", RETENTION)
    monkeypatch.setattr(SessionManager, "_sessions", {})
    monkeypatch.setattr(SessionManager, "_connections", {})
    monkeypatch.setattr(SessionManager, "_caption_handlers", {})
    monkeypatch.setattr(SessionManager, "_participant_locks", {})
    monkeypatch.setattr(SessionManager, "_last_activity", {})
    monkeypatch.setattr(SessionManager, "_ended_meetings", {})


class TestSweepIdleMeetings:
    def test_idle_meeting_is_cleared(self):
        SessionManager.get_or_create_session(meeting_id=MEETING_ID)
        SessionManager._caption_handlers[MEETING_ID] = object()
        SessionManager._participant_locks[f"{MEETING_ID}::p1"] = _Lock(held=False)
        last_seen = SessionManager._last_activity[MEETING_ID]

        swept = SessionManager.sweep_idle_meetings(now=last_seen + IDLE_TIMEOUT)

        assert swept == [MEETING_ID]
        assert MEETING_ID not in SessionManager._sessions
        assert MEETING_ID not in SessionManager._caption_handlers
        assert f"{MEETING_ID}::p1" not in SessionManager._participant_locks
        assert MEETING_ID not in SessionManager._last_activity

    def test_recently_active_meeting_is_kept(self):
        SessionManager.get_or_create_session(meeting_id=MEETING_ID)
        last_seen = SessionManager._last_activity[MEETING_ID]

        swept = SessionManager.sweep_idle_meetings(now=last_seen + IDLE_TIMEOUT - 1)

        assert swept == []
        assert MEETING_ID in SessionManager._sessions

    def test_meeting_with_open_websocket_is_kept(self):
        SessionManager.get_or_create_session(meeting_id=MEETING_ID)
        SessionManager._connections[MEETING_ID] = object()
        last_seen = SessionManager._last_activity[MEETING_ID]

        swept = SessionManager.sweep_idle_meetings(now=last_seen + IDLE_TIMEOUT * 10)

        assert swept == []
        assert MEETING_ID in SessionManager._sessions

    def test_disconnect_restarts_idle_clock(self):
        SessionManager._connections[MEETING_ID] = object()
        SessionManager.remove_connection(MEETING_ID)
        last_seen = SessionManager._last_activity[MEETING_ID]

        assert SessionManager.sweep_idle_meetings(now=last_seen + IDLE_TIMEOUT - 1) == []
        assert SessionManager.sweep_idle_meetings(now=last_seen + IDLE_TIMEOUT) == [MEETING_ID]

    def test_held_lock_survives_idle_clear(self):
        SessionManager.get_or_create_session(meeting_id=MEETING_ID)
        SessionManager._participant_locks[f"{MEETING_ID}::p1"] = _Lock(held=True)
        SessionManager._participant_locks[f"{MEETING_ID}::p2"] = _Lock(held=False)
        last_seen = SessionManager._last_activity[MEETING_ID]

        SessionManager.sweep_idle_meetings(now=last_seen + IDLE_TIMEOUT)

        assert f"{MEETING_ID}::p1" in SessionManager._participant_locks
        assert f"{MEETING_ID}::p2" not in SessionManager._participant_locks

    def test_orphaned_unlocked_lock_is_pruned_and_held_one_kept(self):
        SessionManager._participant_locks["ghost::p1"] = _Lock(held=False)
        SessionManager._participant_locks["ghost::p2"] = _Lock(held=True)

        SessionManager.sweep_idle_meetings(now=0.0)

        assert "ghost::p1" not in SessionManager._participant_locks
        assert "ghost::p2" in SessionManager._participant_locks


class TestPruneEndedMeetings:
    def test_ended_meeting_is_forgotten_after_retention(self):
        SessionManager._ended_meetings[MEETING_ID] = 1000.0

        SessionManager.sweep_idle_meetings(now=1000.0 + RETENTION - 1)
        assert SessionManager.is_meeting_ended(MEETING_ID)

        SessionManager.sweep_idle_meetings(now=1000.0 + RETENTION)
        assert not SessionManager.is_meeting_ended(MEETING_ID)


class TestReadsDoNotCreateState:
    def test_get_transcript_on_unknown_meeting_creates_nothing(self):
        assert SessionManager.get_transcript(meeting_id=MEETING_ID, room_name=None) is None

        assert MEETING_ID not in SessionManager._sessions
        assert MEETING_ID not in SessionManager._last_activity

    def test_get_participant_history_on_unknown_meeting_creates_nothing(self):
        history = SessionManager.get_participant_history(
            meeting_id=MEETING_ID, room_name=None, participant_id="p1"
        )

        assert history == []
        assert MEETING_ID not in SessionManager._sessions
        assert MEETING_ID not in SessionManager._last_activity

    def test_read_after_end_does_not_revive_ended_meeting_state(self):
        SessionManager.update_transcript(
            meeting_id=MEETING_ID,
            room_name=None,
            transcript="hello",
            participant_id="p1",
            timestamp=1.0,
        )
        SessionManager._sessions.pop(MEETING_ID)
        SessionManager._last_activity.pop(MEETING_ID)

        assert SessionManager.get_transcript(meeting_id=MEETING_ID, room_name=None) is None
        assert MEETING_ID not in SessionManager._sessions
