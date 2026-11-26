"""Agentar collections: indexes exist, uniqueness holds, the user profile survives save()."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from pymongo.errors import DuplicateKeyError

from src.database.models import (
    ALL_MODELS,
    AgentarEvent,
    AgentarFile,
    AgentarMeeting,
    AgentarReport,
    User,
    UserProfile,
    VpBotOwner,
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _indexes():
    for model in ALL_MODELS:
        run(model.setup_indexes())


def _index_names(model) -> set[str]:
    return set(run(model.get_collection().index_information()))


def test_every_model_has_a_distinct_collection() -> None:
    names = [m.collection_name for m in ALL_MODELS]
    assert len(set(names)) == len(names)
    for expected in ("agentar_files", "agentar_meetings", "agentar_reports", "agentar_events", "vp_bot_owners"):
        assert expected in names


def test_meeting_indexes_follow_the_contract() -> None:
    info = run(AgentarMeeting.get_collection().index_information())
    keys = {tuple(v["key"]) for v in info.values()}
    assert (("owner", 1), ("start_at", -1), ("created_at", -1)) in keys
    assert (("status", 1), ("start_at", 1)) in keys
    assert info["owner_live_lock"]["unique"] is True
    assert info["owner_live_lock"]["partialFilterExpression"] == {"live_lock": True}
    assert info["bot_id_1"]["unique"] is True


def test_ttl_indexes() -> None:
    events = run(AgentarEvent.get_collection().index_information())
    assert events["received_at_1"]["expireAfterSeconds"] == 30 * 24 * 3600
    owners = run(VpBotOwner.get_collection().index_information())
    assert owners["created_at_1"]["expireAfterSeconds"] == 24 * 3600


def test_second_live_meeting_of_one_owner_is_refused() -> None:
    run(AgentarMeeting.create(owner="a@x.test", room_name="r1", status="live", live_lock=True))
    with pytest.raises(DuplicateKeyError):
        run(AgentarMeeting.create(owner="a@x.test", room_name="r2", status="joining", live_lock=True))
    # Another owner, and a meeting that holds no lock, are unaffected.
    run(AgentarMeeting.create(owner="b@x.test", room_name="r1", status="live", live_lock=True))
    run(AgentarMeeting.create(owner="a@x.test", room_name="r3", status="draft"))
    run(AgentarMeeting.create(owner="a@x.test", room_name="r4", status="ended", live_lock=None))


def test_bot_id_is_unique_only_when_set() -> None:
    run(AgentarMeeting.create(owner="a@x.test", room_name="r1"))
    run(AgentarMeeting.create(owner="a@x.test", room_name="r2"))
    run(AgentarMeeting.create(owner="a@x.test", room_name="r3", bot_id="bot1"))
    with pytest.raises(DuplicateKeyError):
        run(AgentarMeeting.create(owner="b@x.test", room_name="r4", bot_id="bot1"))


def test_event_id_is_unique() -> None:
    kwargs = dict(owner="a@x.test", meeting_id="m" * 24, type="joined")
    run(AgentarEvent.create(event_id="e1", **kwargs))
    with pytest.raises(DuplicateKeyError):
        run(AgentarEvent.create(event_id="e1", **kwargs))


def test_one_report_per_meeting() -> None:
    run(AgentarReport.create(meeting_id="m1", owner="a@x.test"))
    with pytest.raises(DuplicateKeyError):
        run(AgentarReport.create(meeting_id="m1", owner="a@x.test"))


def test_user_without_profile_round_trips_as_none() -> None:
    user = run(User.create(email="p@x.test", password_hash="h"))
    assert run(User.find_by_email("p@x.test")).profile is None
    assert user.profile is None


def test_unset_profile_is_absent_not_null_in_the_document() -> None:
    """A stored null would make `$set profile.agent_name` fail on a real MongoDB."""
    run(User.create(email="p@x.test", password_hash="h"))
    assert "profile" not in run(User.get_collection().find_one({"email": "p@x.test"}))


def test_user_profile_survives_save_of_a_loaded_copy() -> None:
    run(User.create(email="p@x.test", password_hash="h"))
    run(
        User.get_collection().update_one(
            {"email": "p@x.test"},
            {"$set": {"profile.agent_name": "Laura", "profile.voice": "af_heart"}},
        )
    )
    loaded = run(User.find_by_email("p@x.test"))
    assert loaded.profile == UserProfile(agent_name="Laura", voice="af_heart")
    loaded.first_name = "Laura"
    run(loaded.save())
    again = run(User.find_by_email("p@x.test"))
    assert again.profile.agent_name == "Laura"
    assert again.first_name == "Laura"


def test_undeclared_profile_fields_are_dropped_by_save() -> None:
    """Why every profile field is declared: save() replaces the document."""
    run(User.create(email="p@x.test", password_hash="h"))
    run(User.get_collection().update_one({"email": "p@x.test"}, {"$set": {"profile.agent_name": "L", "profile.nick": "x"}}))
    loaded = run(User.find_by_email("p@x.test"))
    run(loaded.save())
    stored = run(User.get_collection().find_one({"email": "p@x.test"}))
    assert "nick" not in stored["profile"]
    assert stored["profile"]["agent_name"] == "L"


def test_datetimes_are_aware_utc_after_a_round_trip() -> None:
    naive = datetime(2026, 11, 4, 9, 0)
    run(AgentarMeeting.create(owner="a@x.test", room_name="r1", start_at=naive))
    got = run(AgentarMeeting.find_one({"owner": "a@x.test"}))
    assert got.start_at.tzinfo is not None
    assert got.start_at == naive.replace(tzinfo=timezone.utc)
    assert got.created_at - datetime.now(timezone.utc) < timedelta(seconds=5)


def test_file_chunks_are_declared() -> None:
    f = run(
        AgentarFile.create(
            owner="a@x.test", name="n.txt", size=3, content_type="text/plain", sha256="0" * 64,
            storage_key="u/1/kb/2", text_status="ok", text_chars=3,
            chunks=[{"n": 0, "text": "abc", "page": None}],
        )
    )
    again = run(AgentarFile.find_by_id(f.id))
    assert again.chunks[0].text == "abc"
