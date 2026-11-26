"""Worker events intake and dashboard notifications.

The lifecycle module (B7) is not part of this branch. `lifecycle` installs a fake
`utils.agentar_lifecycle` with the assumed `mark_agentar_ended(meeting_id, reason)`, which records
the call and moves the meeting to ended as the real one will.
"""
import asyncio
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from config import settings
from src.database.models.agentar_event import AgentarEvent
from src.database.models.agentar_meeting import AgentarMeeting
from utils import agentar_events, clock

WORKER = {"Authorization": "Bearer worker-key-for-tests"}
BOT = "bot1"


@pytest.fixture(autouse=True)
def _worker_key(monkeypatch):
    monkeypatch.setattr(settings, "vp_meetween_api_key", "worker-key-for-tests")


class Lifecycle:
    """Stands in for B7's module: records calls, ends the meeting like the real hand over."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail_next = False

    async def mark_agentar_ended(self, *, meeting_id: str, reason: str) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("lifecycle down")
        self.calls.append({"meeting_id": meeting_id, "reason": reason})
        await AgentarMeeting.get_collection().update_one(
            {"_id": ObjectId(meeting_id)},
            {"$set": {"status": "ended", "ended_at": clock.utcnow()}, "$unset": {"live_lock": ""}},
        )


def set_agentar(monkeypatch, **fields) -> None:
    """`settings.agentar` is a frozen view built at start, so a field patch does not reach it."""
    monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update=fields))


@pytest.fixture
def lifecycle(monkeypatch) -> Lifecycle:
    fake = Lifecycle()
    module = types.ModuleType("utils.agentar_lifecycle")
    module.mark_agentar_ended = fake.mark_agentar_ended
    monkeypatch.setitem(sys.modules, "utils.agentar_lifecycle", module)
    return fake


def make_meeting(owner: str, status: str = "live", bot_id: str | None = BOT, lock: bool = True, **fields) -> str:
    """`lock=False` lets one owner hold two active meetings, which the live slot index forbids."""
    meeting = AgentarMeeting(owner=owner, room_name="weekly", status=status, bot_id=bot_id, title="Weekly", **fields)
    if lock and status in ("joining", "live", "leaving"):
        meeting.live_lock = True
    return asyncio.run(meeting.save()).id


def read_meeting(meeting_id: str) -> dict:
    return asyncio.run(AgentarMeeting.get_collection().find_one({"_id": ObjectId(meeting_id)}))


def stored_events(meeting_id: str | None = None) -> list[dict]:
    query = {"meeting_id": meeting_id} if meeting_id else {}
    return asyncio.run(AgentarEvent.get_collection().find(query).to_list(length=None))


def body(meeting_id: str, kind: str = "joined", data: dict | None = None, **overrides) -> dict:
    payload = {
        "event_id": str(uuid.uuid4()),
        "agentar_meeting_id": meeting_id,
        "bot_id": BOT,
        "type": kind,
        "at": clock.utcnow().isoformat(),
        "data": {"kind": kind, **(data or {})},
    }
    return {**payload, **overrides}


def question(meeting_id: str, **overrides) -> dict:
    return body(meeting_id, "question_asked", {"question_id": "q1", "text": "Who covers the leave?"}, **overrides)


def left(meeting_id: str, reason: str = "recalled", **overrides) -> dict:
    return body(meeting_id, "left", {"reason": reason}, **overrides)


def fatal_error(meeting_id: str) -> dict:
    return body(meeting_id, "error", {"message": "browser crashed", "fatal": True})


def post(test_client: TestClient, payload: dict, headers: dict | None = None):
    return test_client.post("/api/agentar/events", json=payload, headers=WORKER if headers is None else headers)


@pytest.fixture
def owner(user_a) -> str:
    return user_a.email


class TestAuth:
    def test_user_token_is_refused(self, test_client: TestClient, user_a, frozen_clock) -> None:
        meeting_id = make_meeting(user_a.email)
        response = post(test_client, question(meeting_id), headers=user_a.headers)
        assert response.status_code == 401
        assert stored_events() == []

    def test_no_header_and_wrong_key_are_refused(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner)
        assert post(test_client, question(meeting_id), headers={}).status_code == 401
        assert post(test_client, question(meeting_id), headers={"Authorization": "Bearer nope"}).status_code == 401
        assert stored_events() == []

    def test_key_is_required_with_auth_off(self, test_client: TestClient, owner, frozen_clock, monkeypatch) -> None:
        monkeypatch.setattr(settings, "enable_auth", False)
        meeting_id = make_meeting(owner)
        assert post(test_client, question(meeting_id), headers={}).status_code == 401
        assert post(test_client, question(meeting_id)).status_code == 202

    def test_unset_key_rejects_everything(self, test_client: TestClient, owner, frozen_clock, monkeypatch) -> None:
        monkeypatch.setattr(settings, "vp_meetween_api_key", "")
        meeting_id = make_meeting(owner)
        assert post(test_client, question(meeting_id), headers={"Authorization": "Bearer "}).status_code == 401


class TestPair:
    def test_unknown_meeting_is_404(self, test_client: TestClient, frozen_clock) -> None:
        response = post(test_client, question("0123456789abcdef01234567"))
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_malformed_meeting_id_is_a_validation_error(self, test_client: TestClient, frozen_clock) -> None:
        assert post(test_client, question("not-an-id")).status_code == 422

    def test_other_bot_is_404(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner)
        assert post(test_client, question(meeting_id, bot_id="bot2")).status_code == 404
        assert stored_events() == []

    def test_meeting_without_a_bot_is_404(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner, status="joining", bot_id=None)
        assert post(test_client, question(meeting_id)).status_code == 404

    def test_ack_shape(self, test_client: TestClient, owner, frozen_clock) -> None:
        response = post(test_client, question(make_meeting(owner)))
        assert response.status_code == 202
        assert response.json() == {"accepted": True, "duplicate": False}


class TestState:
    @pytest.mark.parametrize("status", ["joining", "live", "leaving"])
    def test_accepting_statuses(self, test_client: TestClient, owner, frozen_clock, status) -> None:
        assert post(test_client, question(make_meeting(owner, status))).status_code == 202

    @pytest.mark.parametrize("status", ["draft", "scheduled", "waiting_for_human", "missed", "failed", "ended"])
    def test_other_statuses_are_409(self, test_client: TestClient, owner, frozen_clock, status) -> None:
        meeting_id = make_meeting(owner, status, ended_at=clock.utcnow() - timedelta(hours=1))
        response = post(test_client, question(meeting_id))
        assert response.status_code == 409
        assert response.json()["code"] == "invalid_state"
        assert stored_events() == []

    def test_left_is_accepted_shortly_after_ended(self, test_client: TestClient, owner, frozen_clock, lifecycle) -> None:
        meeting_id = make_meeting(owner, "ended", ended_at=clock.utcnow())
        frozen_clock.advance(minutes=4, seconds=59)
        assert post(test_client, left(meeting_id)).status_code == 202
        # Already ended: stored and shown, nothing handed over again.
        assert lifecycle.calls == []
        assert [row["type"] for row in stored_events(meeting_id)] == ["left"]

    def test_left_is_refused_after_the_grace(self, test_client: TestClient, owner, frozen_clock, lifecycle) -> None:
        meeting_id = make_meeting(owner, "ended", ended_at=clock.utcnow())
        frozen_clock.advance(minutes=5, seconds=1)
        assert post(test_client, left(meeting_id)).status_code == 409

    def test_only_left_gets_the_grace(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner, "ended", ended_at=clock.utcnow())
        assert post(test_client, question(meeting_id)).status_code == 409

    def test_grace_follows_the_setting(self, test_client: TestClient, owner, frozen_clock, monkeypatch) -> None:
        set_agentar(monkeypatch, event_left_grace_seconds=10)
        meeting_id = make_meeting(owner, "ended", ended_at=clock.utcnow())
        frozen_clock.advance(seconds=11)
        assert post(test_client, left(meeting_id)).status_code == 409

    def test_ended_without_a_time_is_refused(self, test_client: TestClient, owner, frozen_clock) -> None:
        assert post(test_client, left(make_meeting(owner, "ended"))).status_code == 409


class TestClockSkew:
    @pytest.mark.parametrize("offset", [timedelta(minutes=-11), timedelta(minutes=11)])
    def test_beyond_ten_minutes_is_422(self, test_client: TestClient, owner, frozen_clock, offset) -> None:
        meeting_id = make_meeting(owner)
        response = post(test_client, question(meeting_id, at=(clock.utcnow() + offset).isoformat()))
        assert response.status_code == 422
        assert response.json()["code"] == "event_clock_skew"
        assert stored_events() == []

    @pytest.mark.parametrize("offset", [timedelta(minutes=-9), timedelta(minutes=9)])
    def test_within_ten_minutes_is_accepted(self, test_client: TestClient, owner, frozen_clock, offset) -> None:
        meeting_id = make_meeting(owner)
        assert post(test_client, question(meeting_id, at=(clock.utcnow() + offset).isoformat())).status_code == 202

    def test_a_zone_offset_is_honoured(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner)
        local = clock.utcnow().astimezone(timezone(timedelta(hours=1)))
        assert post(test_client, question(meeting_id, at=local.isoformat())).status_code == 202

    def test_a_naive_time_is_taken_as_utc(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner)
        naive = clock.utcnow().replace(tzinfo=None)
        assert post(test_client, question(meeting_id, at=naive.isoformat())).status_code == 202

    def test_limit_follows_the_setting(self, test_client: TestClient, owner, frozen_clock, monkeypatch) -> None:
        set_agentar(monkeypatch, event_max_skew_seconds=30)
        meeting_id = make_meeting(owner)
        late = (clock.utcnow() - timedelta(seconds=31)).isoformat()
        assert post(test_client, question(meeting_id, at=late)).status_code == 422


class TestIdempotency:
    def test_repeat_answers_200_duplicate(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner)
        payload = question(meeting_id)
        assert post(test_client, payload).status_code == 202
        again = post(test_client, payload)
        assert again.status_code == 200
        assert again.json() == {"accepted": True, "duplicate": True}
        assert len(stored_events(meeting_id)) == 1

    def test_effect_is_applied_once(self, test_client: TestClient, owner, frozen_clock, lifecycle) -> None:
        meeting_id = make_meeting(owner)
        payload = left(meeting_id)
        assert post(test_client, payload).status_code == 202
        assert post(test_client, payload).status_code == 200
        assert len(lifecycle.calls) == 1

    def test_a_late_retry_is_still_a_duplicate(self, test_client: TestClient, owner, frozen_clock) -> None:
        """The retry carries the original `at` and may arrive after the meeting ended."""
        meeting_id = make_meeting(owner)
        payload = question(meeting_id)
        assert post(test_client, payload).status_code == 202
        asyncio.run(
            AgentarMeeting.get_collection().update_one(
                {"_id": ObjectId(meeting_id)}, {"$set": {"status": "ended", "ended_at": clock.utcnow()}}
            )
        )
        frozen_clock.advance(minutes=30)
        assert post(test_client, payload).status_code == 200

    def test_an_event_id_of_another_meeting_is_unknown(self, test_client: TestClient, user_a, user_b, frozen_clock) -> None:
        first = make_meeting(user_a.email)
        second = make_meeting(user_b.email, bot_id="bot2")
        payload = question(first)
        assert post(test_client, payload).status_code == 202
        stolen = {**question(second, bot_id="bot2"), "event_id": payload["event_id"]}
        assert post(test_client, stolen).status_code == 404
        assert len(stored_events()) == 1

    def test_a_race_past_the_lookup_is_a_duplicate(self, test_client: TestClient, owner, frozen_clock, monkeypatch) -> None:
        """Two attempts pass the lookup together: the unique index decides."""
        asyncio.run(AgentarEvent.setup_indexes())
        meeting_id = make_meeting(owner)
        payload = question(meeting_id)
        assert post(test_client, payload).status_code == 202

        lookups = iter([False, True])

        async def racing(event_id: str, meeting: str) -> bool:
            return next(lookups)

        monkeypatch.setattr(agentar_events, "_is_duplicate", racing)
        again = post(test_client, payload)
        assert again.status_code == 200
        assert again.json()["duplicate"] is True
        assert len(stored_events(meeting_id)) == 1


class TestEffects:
    def test_joined_moves_joining_to_live(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner, "joining")
        assert post(test_client, body(meeting_id, "joined", {"join_ms": 4200})).status_code == 202
        meeting = read_meeting(meeting_id)
        assert meeting["status"] == "live"
        assert clock.as_utc(meeting["joined_at"]) == clock.utcnow()

    def test_joined_on_a_live_meeting_keeps_the_first_time(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner, "joining")
        post(test_client, body(meeting_id, "joined"))
        first = read_meeting(meeting_id)["joined_at"]
        frozen_clock.advance(minutes=1)
        assert post(test_client, body(meeting_id, "joined")).status_code == 202
        meeting = read_meeting(meeting_id)
        assert meeting["status"] == "live"
        assert meeting["joined_at"] == first

    def test_joined_does_not_revive_a_leaving_meeting(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner, "leaving")
        assert post(test_client, body(meeting_id, "joined")).status_code == 202
        assert read_meeting(meeting_id)["status"] == "leaving"

    def test_the_jitsi_meeting_id_is_bound_once(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner, "joining")
        post(test_client, body(meeting_id, "joined", jitsi_meeting_id="abc123"))
        post(test_client, question(meeting_id, jitsi_meeting_id="other"))
        assert read_meeting(meeting_id)["jitsi_meeting_id"] == "abc123"

    @pytest.mark.parametrize("status", ["live", "leaving", "joining"])
    def test_left_hands_over_to_the_lifecycle(self, test_client: TestClient, owner, frozen_clock, lifecycle, status) -> None:
        meeting_id = make_meeting(owner, status)
        assert post(test_client, left(meeting_id, "alone")).status_code == 202
        assert lifecycle.calls == [{"meeting_id": meeting_id, "reason": "alone"}]
        assert read_meeting(meeting_id)["status"] == "ended"

    def test_left_with_error_while_joining_fails_the_meeting(self, test_client: TestClient, owner, frozen_clock, lifecycle) -> None:
        meeting_id = make_meeting(owner, "joining")
        assert post(test_client, left(meeting_id, "error")).status_code == 202
        meeting = read_meeting(meeting_id)
        assert meeting["status"] == "failed"
        assert meeting["status_reason"] == "worker_error"
        assert "live_lock" not in meeting
        assert lifecycle.calls == []

    def test_fatal_error_while_joining_fails_the_meeting(self, test_client: TestClient, owner, frozen_clock, lifecycle) -> None:
        meeting_id = make_meeting(owner, "joining")
        assert post(test_client, fatal_error(meeting_id)).status_code == 202
        meeting = read_meeting(meeting_id)
        assert (meeting["status"], meeting["status_reason"]) == ("failed", "worker_error")
        assert "live_lock" not in meeting
        assert lifecycle.calls == []

    @pytest.mark.parametrize("status", ["live", "leaving"])
    def test_fatal_error_while_live_ends_it_so_a_report_is_made(
        self, test_client: TestClient, owner, frozen_clock, lifecycle, status
    ) -> None:
        meeting_id = make_meeting(owner, status)
        assert post(test_client, fatal_error(meeting_id)).status_code == 202
        assert lifecycle.calls == [{"meeting_id": meeting_id, "reason": "error"}]
        assert read_meeting(meeting_id)["status"] == "ended"

    def test_non_fatal_error_changes_nothing(self, test_client: TestClient, owner, frozen_clock, lifecycle) -> None:
        meeting_id = make_meeting(owner, "live")
        payload = body(meeting_id, "error", {"message": "tts retry", "fatal": False})
        assert post(test_client, payload).status_code == 202
        assert read_meeting(meeting_id)["status"] == "live"
        assert lifecycle.calls == []

    @pytest.mark.parametrize("kind", ["question_asked", "answer_given", "escalation"])
    def test_dialogue_events_only_store(self, test_client: TestClient, owner, frozen_clock, lifecycle, kind) -> None:
        data = {
            "question_asked": {"question_id": "q2", "text": "Who is the deputy?", "first_word_ms": 900},
            "answer_given": {"question": "Seats?", "answer": "Sixty", "asker": "Marta", "channel": "group"},
            "escalation": {"question": "Can you commit 5000 euros?", "asker": "Marta"},
        }[kind]
        meeting_id = make_meeting(owner, "live")
        assert post(test_client, body(meeting_id, kind, data)).status_code == 202
        assert read_meeting(meeting_id)["status"] == "live"
        assert lifecycle.calls == []

    def test_a_failed_hand_over_leaves_no_row_so_the_retry_applies_it(
        self, test_client: TestClient, owner, frozen_clock, lifecycle
    ) -> None:
        meeting_id = make_meeting(owner, "live")
        payload = left(meeting_id)
        lifecycle.fail_next = True
        with pytest.raises(RuntimeError):
            post(test_client, payload)
        assert stored_events() == []
        assert post(test_client, payload).status_code == 202
        assert len(lifecycle.calls) == 1


class TestStoredEvent:
    def test_row_fields(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner)
        payload = question(meeting_id)
        post(test_client, payload)
        (row,) = stored_events(meeting_id)
        assert row["event_id"] == payload["event_id"]
        assert row["owner"] == owner
        assert row["bot_id"] == BOT
        assert row["type"] == "question_asked"
        assert row["source"] == "worker"
        assert row["data"] == {"question_id": "q1", "text": "Who covers the leave?"}
        assert clock.as_utc(row["received_at"]) == clock.utcnow()

    def test_control_and_bidi_characters_are_removed(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner)
        data = {"question": "Budget?\x00\x07 \u202eevil\u202c\u2066", "asker": "Ma\x1brta\nR"}
        post(test_client, body(meeting_id, "escalation", data))
        (row,) = stored_events(meeting_id)
        assert row["data"] == {"question": "Budget? evil", "asker": "Marta\nR"}

    def test_markup_is_kept_as_plain_text(self, test_client: TestClient, owner, frozen_clock, user_a) -> None:
        meeting_id = make_meeting(owner)
        data = {"question": "<img src=x onerror=alert(1)>", "asker": None}
        post(test_client, body(meeting_id, "escalation", data))
        item = test_client.get("/api/agentar/notifications", headers=user_a.headers).json()["items"][0]
        assert item["data"] == {"question": "<img src=x onerror=alert(1)>", "asker": None}

    def test_extra_fields_are_refused(self, test_client: TestClient, owner, frozen_clock) -> None:
        meeting_id = make_meeting(owner)
        payload = question(meeting_id)
        payload["data"]["html"] = "<b>x</b>"
        assert post(test_client, payload).status_code == 422
        assert post(test_client, {**question(meeting_id), "type": "left"}).status_code == 422


class TestNotifications:
    URL = "/api/agentar/notifications"

    def get(self, test_client: TestClient, user, **params):
        response = test_client.get(self.URL, params=params, headers=user.headers)
        assert response.status_code == 200, response.text
        return response.json()

    def test_needs_a_token(self, test_client: TestClient) -> None:
        response = test_client.get(self.URL, headers={})
        assert response.status_code == 401

    def test_worker_key_is_not_a_user(self, test_client: TestClient) -> None:
        assert test_client.get(self.URL, headers=WORKER).status_code == 401

    def test_empty(self, test_client: TestClient, user_a) -> None:
        assert self.get(test_client, user_a) == {"items": [], "cursor": None}

    def test_item_shape_and_title(self, test_client: TestClient, user_a, frozen_clock) -> None:
        meeting_id = make_meeting(user_a.email)
        post(test_client, body(meeting_id, "escalation", {"question": "5000 euros?", "asker": "Marta"}))
        (item,) = self.get(test_client, user_a)["items"]
        assert set(item) == {"id", "at", "meeting_id", "meeting_title", "type", "severity", "data"}
        assert item["meeting_id"] == meeting_id
        assert item["meeting_title"] == "Weekly"
        assert item["type"] == "escalation"
        assert item["severity"] == "warning"
        assert item["data"] == {"question": "5000 euros?", "asker": "Marta"}
        assert datetime.fromisoformat(item["at"]) == clock.utcnow()

    def test_escalation_is_listed_on_the_next_poll(self, test_client: TestClient, user_a, frozen_clock) -> None:
        """S8: the dashboard polls every 2 s, so a stored escalation must show on the next call."""
        meeting_id = make_meeting(user_a.email)
        cursor = self.get(test_client, user_a)["cursor"]
        post(test_client, body(meeting_id, "escalation", {"question": "Budget?", "asker": None, "channel": "voice"}))
        page = self.get(test_client, user_a, since=cursor) if cursor else self.get(test_client, user_a)
        assert [item["type"] for item in page["items"]] == ["escalation"]

    def test_severity_by_type(self, test_client: TestClient, user_a, frozen_clock, lifecycle) -> None:
        meeting_id = make_meeting(user_a.email, "joining")
        post(test_client, body(meeting_id, "joined"))
        post(test_client, question(meeting_id))
        post(test_client, body(meeting_id, "answer_given", {"question": "q", "answer": "a", "asker": "M"}))
        post(test_client, body(meeting_id, "escalation", {"question": "q", "asker": "M"}))
        post(test_client, body(meeting_id, "error", {"message": "tts retry", "fatal": False}))
        post(test_client, left(meeting_id))
        for kind, reason in (("missed", "room_stayed_empty"), ("failed", "join_timeout"), ("report_ready", "ok")):
            asyncio.run(
                agentar_events.record_server_event(
                    owner=user_a.email, meeting_id=meeting_id, type=kind, data={"reason": reason}
                )
            )
        severities = {item["type"]: item["severity"] for item in self.get(test_client, user_a)["items"]}
        assert severities == {
            "joined": "info",
            "question_asked": "info",
            "answer_given": "info",
            "escalation": "warning",
            "error": "error",
            "left": "info",
            "missed": "warning",
            "failed": "error",
            "report_ready": "info",
        }

    def test_server_event_data_is_the_reason(self, test_client: TestClient, user_a, frozen_clock) -> None:
        meeting_id = make_meeting(user_a.email, "failed")
        asyncio.run(
            agentar_events.record_server_event(
                owner=user_a.email, meeting_id=meeting_id, type="failed", data={"reason": "join_timeout\x00"}
            )
        )
        (item,) = self.get(test_client, user_a)["items"]
        assert (item["type"], item["data"]) == ("failed", {"reason": "join_timeout"})
        assert stored_events()[0]["source"] == "server"

    def test_server_event_with_a_stable_id_is_stored_once(self, test_client: TestClient, user_a, frozen_clock) -> None:
        asyncio.run(AgentarEvent.setup_indexes())
        meeting_id = make_meeting(user_a.email, "ended")
        for _ in range(2):
            asyncio.run(
                agentar_events.record_server_event(
                    owner=user_a.email,
                    meeting_id=meeting_id,
                    type="report_ready",
                    data={},
                    event_id=f"report_ready:{meeting_id}",
                )
            )
        assert len(self.get(test_client, user_a)["items"]) == 1

    def test_unknown_server_type_is_refused(self, user_a) -> None:
        with pytest.raises(ValueError):
            asyncio.run(
                agentar_events.record_server_event(owner=user_a.email, meeting_id="0" * 24, type="bogus", data={})
            )

    def test_cursor_paging(self, test_client: TestClient, user_a, frozen_clock) -> None:
        meeting_id = make_meeting(user_a.email)
        ids = []
        for number in range(1, 8):
            post(test_client, body(meeting_id, "question_asked", {"question_id": f"q{number}", "text": f"t{number}"}))
        first = self.get(test_client, user_a, limit=3)
        # Without a cursor the newest come back, oldest first.
        assert [item["data"]["question_id"] for item in first["items"]] == ["q5", "q6", "q7"]
        assert first["cursor"] == first["items"][-1]["id"]

        meeting_two = make_meeting(user_a.email, "live", bot_id="bot2", lock=False)
        for number in range(8, 12):
            post(test_client, body(meeting_two, "question_asked", {"question_id": f"q{number}", "text": "x"}, bot_id="bot2"))
        page = self.get(test_client, user_a, since=first["cursor"], limit=3)
        assert [item["data"]["question_id"] for item in page["items"]] == ["q8", "q9", "q10"]
        ids += [item["id"] for item in page["items"]]
        page = self.get(test_client, user_a, since=page["cursor"], limit=3)
        assert [item["data"]["question_id"] for item in page["items"]] == ["q11"]
        ids += [item["id"] for item in page["items"]]
        assert ids == sorted(ids)

    def test_walking_from_the_start_never_repeats_or_skips(self, test_client: TestClient, user_a, frozen_clock) -> None:
        meeting_id = make_meeting(user_a.email)
        for number in range(1, 8):
            post(test_client, body(meeting_id, "question_asked", {"question_id": f"q{number}", "text": "x"}))
        oldest = test_client.get(self.URL, params={"limit": 100}, headers=user_a.headers).json()["items"][0]["id"]
        before = ObjectId(oldest)
        seen: list[str] = []
        cursor = str(ObjectId.from_datetime(before.generation_time - timedelta(seconds=1)))
        for _ in range(10):
            page = self.get(test_client, user_a, since=cursor, limit=2)
            if not page["items"]:
                break
            seen += [item["data"]["question_id"] for item in page["items"]]
            cursor = page["cursor"]
        assert seen == [f"q{n}" for n in range(1, 8)]

    def test_cursor_is_unchanged_when_nothing_is_new(self, test_client: TestClient, user_a, frozen_clock) -> None:
        post(test_client, question(make_meeting(user_a.email)))
        cursor = self.get(test_client, user_a)["cursor"]
        again = self.get(test_client, user_a, since=cursor)
        assert again == {"items": [], "cursor": cursor}

    def test_without_a_cursor_the_newest_twenty(self, test_client: TestClient, user_a, frozen_clock) -> None:
        meeting_id = make_meeting(user_a.email)
        for number in range(1, 26):
            post(test_client, body(meeting_id, "question_asked", {"question_id": f"q{number}", "text": "x"}))
        items = self.get(test_client, user_a)["items"]
        assert len(items) == 20
        assert items[0]["data"]["question_id"] == "q6"
        assert items[-1]["data"]["question_id"] == "q25"

    def test_without_a_cursor_only_the_last_24_hours(self, test_client: TestClient, user_a, frozen_clock) -> None:
        meeting_id = make_meeting(user_a.email)
        post(test_client, body(meeting_id, "question_asked", {"question_id": "q1", "text": "old"}))
        frozen_clock.advance(hours=24, minutes=1)
        post(test_client, body(meeting_id, "question_asked", {"question_id": "q2", "text": "new"}))
        assert [item["data"]["text"] for item in self.get(test_client, user_a)["items"]] == ["new"]

    def test_old_items_are_still_reachable_with_a_cursor(self, test_client: TestClient, user_a, frozen_clock) -> None:
        meeting_id = make_meeting(user_a.email)
        post(test_client, body(meeting_id, "question_asked", {"question_id": "q1", "text": "old"}))
        frozen_clock.advance(days=2)
        floor = str(ObjectId.from_datetime(datetime(2000, 1, 1, tzinfo=timezone.utc)))
        assert len(self.get(test_client, user_a, since=floor)["items"]) == 1

    def test_limit_below_twenty_applies_without_a_cursor(self, test_client: TestClient, user_a, frozen_clock) -> None:
        meeting_id = make_meeting(user_a.email)
        for number in range(1, 6):
            post(test_client, body(meeting_id, "question_asked", {"question_id": f"q{number}", "text": "x"}))
        assert len(self.get(test_client, user_a, limit=2)["items"]) == 2

    @pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"limit": "x"}, {"since": "x" * 65}])
    def test_bad_parameters_are_422(self, test_client: TestClient, user_a, params) -> None:
        assert test_client.get(self.URL, params=params, headers=user_a.headers).status_code == 422

    @pytest.mark.parametrize("since", ["nope", "0123456789ABCDEF01234567", "0123456789abcdef0123456", "g" * 24])
    def test_a_malformed_cursor_is_422(self, test_client: TestClient, user_a, since) -> None:
        response = test_client.get(self.URL, params={"since": since}, headers=user_a.headers)
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["query", "since"]

    def test_404_across_owners_means_nothing_is_visible(self, test_client: TestClient, user_a, user_b, frozen_clock) -> None:
        meeting_id = make_meeting(user_a.email)
        post(test_client, body(meeting_id, "escalation", {"question": "private", "asker": None}))
        mine = self.get(test_client, user_a)
        assert len(mine["items"]) == 1
        assert self.get(test_client, user_b) == {"items": [], "cursor": None}
        # Another owner's cursor value reveals nothing either.
        assert self.get(test_client, user_b, since=mine["items"][0]["id"]) == {
            "items": [],
            "cursor": mine["items"][0]["id"],
        }

    def test_owners_only_see_their_own_events(self, test_client: TestClient, user_a, user_b, frozen_clock) -> None:
        first = make_meeting(user_a.email)
        second = make_meeting(user_b.email, bot_id="bot2")
        post(test_client, question(first))
        post(test_client, question(second, bot_id="bot2"))
        assert [item["meeting_id"] for item in self.get(test_client, user_a)["items"]] == [first]
        assert [item["meeting_id"] for item in self.get(test_client, user_b)["items"]] == [second]

    def test_a_deleted_meeting_has_no_title(self, test_client: TestClient, user_a, frozen_clock) -> None:
        meeting_id = make_meeting(user_a.email)
        post(test_client, question(meeting_id))
        asyncio.run(AgentarMeeting.get_collection().delete_one({"_id": ObjectId(meeting_id)}))
        (item,) = self.get(test_client, user_a)["items"]
        assert item["meeting_title"] is None


class TestFullRun:
    def test_joined_question_escalation_left(self, test_client: TestClient, user_a, frozen_clock, lifecycle) -> None:
        meeting_id = make_meeting(user_a.email, "joining")
        for payload in (
            body(meeting_id, "joined", {"join_ms": 3100}, jitsi_meeting_id="jm1"),
            question(meeting_id),
            body(meeting_id, "escalation", {"question": "Budget?", "asker": "Marta", "channel": "group"}),
            left(meeting_id, "recalled"),
        ):
            frozen_clock.advance(seconds=5)
            payload["at"] = clock.utcnow().isoformat()
            assert post(test_client, payload).status_code == 202
        meeting = read_meeting(meeting_id)
        assert (meeting["status"], meeting["jitsi_meeting_id"]) == ("ended", "jm1")
        items = test_client.get("/api/agentar/notifications", headers=user_a.headers).json()["items"]
        assert [item["type"] for item in items] == ["joined", "question_asked", "escalation", "left"]
