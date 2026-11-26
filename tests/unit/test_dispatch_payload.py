"""The body fastapi sends to vp-worker for an Agentar meeting.

Every payload is checked against the worker's CreateBot schema (a copy under tests/unit/data),
then field by field: the display name, the greeting with the AI disclosure and the owner's name,
the opener, the questions, the escalation line, the farewell, the voice, the picture, the events
url and the bot token. The fallbacks of an owner who skipped onboarding have their own cases.
"""
import asyncio
import base64
import json
import logging
from urllib.parse import urlsplit

import httpx
import pytest
from bson import ObjectId

from config import settings
from src.database.models.agentar_file import AgentarFile
from src.database.models.agentar_meeting import AgentarMeeting, MeetingQuestion
from src.database.models.user import User, UserProfile
from tests.seams import install_bot_token
from tests.worker_schema import SCHEMA, check_create_bot, violations
from utils import agentar_worker as worker
from utils.agentar_worker import (
    WorkerDispatchError,
    build_worker_bot,
    dispatch_bot,
    events_url,
    get_bot,
    load_avatar,
    recall_bot,
)
from utils.object_store import MemoryObjectStore, ObjectStoreError

MEETING_ID = "0123456789abcdef01234567"
JPEG = b"\xff\xd8\xff\xe0" + b"j" * 64
EVENTS_BASE = "http://fastapi:8000"


@pytest.fixture(autouse=True)
def bot_token_seam(monkeypatch):
    install_bot_token(monkeypatch)


@pytest.fixture(autouse=True)
def events_base(monkeypatch):
    monkeypatch.setattr(
        settings, "_agentar", settings.agentar.model_copy(update={"worker_events_base_url": EVENTS_BASE})
    )


def make_user(**overrides) -> User:
    values = {
        "email": "laura@meetween.test",
        "password_hash": "x",
        "first_name": "Laura",
        "last_name": "Rossi",
        "profile": UserProfile(agent_name="Lia", voice="am_fenrir"),
    }
    values.update(overrides)
    return User(**values)


def make_meeting(**overrides) -> AgentarMeeting:
    values = {
        "id": MEETING_ID,
        "owner": "laura@meetween.test",
        "room_name": "weekly-sync",
        "purpose": "Agree the release date",
        "relevant_info": "PRIVATE budget is 40k",
        "questions": [MeetingQuestion(id="q1", text="Who owns the rollout?"), MeetingQuestion(id="q2", text="When?")],
    }
    values.update(overrides)
    return AgentarMeeting(**values)


def build(meeting=None, owner=None, **kwargs):
    kwargs.setdefault("jwt", "token.value.here")
    kwargs.setdefault("avatar_jpeg", None)
    return build_worker_bot(meeting=meeting or make_meeting(), owner=owner or make_user(), **kwargs)


def body_of(bot) -> dict:
    return bot.model_dump(mode="json", exclude_none=True)


class TestSchema:
    def test_full_payload_is_valid(self) -> None:
        bot = build(avatar_jpeg=JPEG)
        assert check_create_bot(body_of(bot)) == []

    def test_minimal_payload_is_valid(self) -> None:
        owner = make_user(first_name=None, last_name=None, profile=None)
        meeting = make_meeting(purpose="", questions=[])
        assert check_create_bot(body_of(build(meeting, owner))) == []

    def test_payload_with_unicode_name_is_valid(self) -> None:
        owner = make_user(profile=UserProfile(agent_name="Zoë O'Neil-Ünal"))
        assert check_create_bot(body_of(build(owner=owner))) == []

    def test_checker_rejects_what_the_schema_forbids(self) -> None:
        good = body_of(build())
        assert violations({**good, "unknown": 1}, SCHEMA)
        assert check_create_bot({**good, "display_name": "Laura (AI) Agentar"})
        assert check_create_bot({**good, "voice": "af_unknown"})
        assert check_create_bot({**good, "avatar_jpeg_b64": base64.b64encode(b"not a jpeg").decode()})
        assert check_create_bot({**good, "questions": [{"id": "x", "text": "t"}]})
        assert check_create_bot({**good, "room": "Bad Room"})

    def test_payload_never_sets_script(self) -> None:
        assert "script" not in body_of(build())


class TestFields:
    def test_display_name_is_name_and_agentar_without_suffix(self) -> None:
        assert build().display_name == "Lia Agentar"

    def test_greeting_discloses_ai_and_names_the_owner(self) -> None:
        greeting = build().greeting
        assert "Lia Agentar" in greeting
        assert "AI assistant" in greeting
        assert "Laura Rossi" in greeting
        assert greeting.rstrip(".").endswith("report back to Laura")

    def test_opener_is_built_from_the_purpose_only(self) -> None:
        opener = build().opener
        assert opener == "The purpose of this meeting, as Laura gave it to me, is: Agree the release date"
        assert "PRIVATE" not in opener

    def test_no_purpose_means_no_opener(self) -> None:
        assert build(make_meeting(purpose="   ")).opener is None

    def test_long_purpose_is_cut_at_a_word_and_stays_one_line(self) -> None:
        purpose = "alpha\nbeta " * 200
        opener = build(make_meeting(purpose=purpose)).opener
        assert "\n" not in opener
        assert len(opener) <= 1000
        assert not opener.endswith(" ")

    def test_questions_keep_ids_and_order(self) -> None:
        assert [(q.id, q.text) for q in build().questions] == [("q1", "Who owns the rollout?"), ("q2", "When?")]

    def test_escalation_line_and_farewell_name_the_owner(self) -> None:
        bot = build()
        assert bot.escalation_line == "I do not have that, I will check with Laura Rossi and report back"
        assert "Laura" in bot.farewell
        assert bot.farewell.endswith("Goodbye.")

    def test_meeting_and_token_and_room(self) -> None:
        bot = build(jwt="abc.def.ghi")
        assert bot.agentar_meeting_id == MEETING_ID
        assert bot.jwt == "abc.def.ghi"
        assert bot.room == "weekly-sync"

    def test_events_url_is_the_workers_backend_origin(self) -> None:
        url = build().events_url
        assert url == f"{EVENTS_BASE}/api/agentar/events"
        assert urlsplit(url)[:2] == urlsplit(EVENTS_BASE)[:2]

    def test_picture_is_inline_base64(self) -> None:
        bot = build(avatar_jpeg=JPEG)
        assert base64.b64decode(bot.avatar_jpeg_b64) == JPEG

    def test_meeting_voice_wins_over_the_profile(self) -> None:
        assert build(make_meeting(voice="af_bella")).voice == "af_bella"
        assert build().voice == "am_fenrir"

    def test_a_meeting_keeps_the_name_it_was_armed_with(self) -> None:
        meeting = make_meeting(display_name="Old Agentar (AI)")
        bot = build(meeting)
        assert bot.display_name == "Old Agentar"
        assert "Old Agentar" in bot.greeting


class TestSkippedOnboarding:
    def test_no_profile_uses_first_name_and_default_voice_and_no_picture(self) -> None:
        owner = make_user(profile=None)
        bot = build(owner=owner)
        assert bot.display_name == "Laura Agentar"
        assert bot.voice == settings.agentar.default_voice
        assert bot.avatar_jpeg_b64 is None

    def test_no_names_at_all_uses_meetween_and_the_meeting_owner(self) -> None:
        owner = make_user(first_name=None, last_name=None, profile=None)
        bot = build(owner=owner)
        assert bot.display_name == "Meetween Agentar"
        assert "attending for the meeting owner" in bot.greeting
        assert bot.escalation_line == "I do not have that, I will check with the meeting owner and report back"

    def test_a_voice_no_longer_offered_falls_back(self) -> None:
        owner = make_user(profile=UserProfile(voice="bf_old"))
        assert build(owner=owner).voice == settings.agentar.default_voice


class TestNames:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Lia", "Lia Agentar"),
            ("  Lia   Marie ", "Lia Marie Agentar"),
            ("Lia (AI)", "Lia AI Agentar"),
            ("Agentar Lia", "Lia Agentar"),
            ("Agenagentartar", "Meetween Agentar"),
            ("<b>Lia</b>", "bLiab Agentar"),
            ("--Lia--", "Lia Agentar"),
            ("Zoë", "Zoë Agentar"),
            ("😀😀", "Meetween Agentar"),
            ("L" * 80, "L" * 32 + " Agentar"),
        ],
    )
    def test_agent_name_is_reduced_to_a_valid_display_name(self, raw, expected) -> None:
        owner = make_user(first_name=None, profile=UserProfile(agent_name=raw))
        bot = build(owner=owner)
        assert bot.display_name == expected
        assert check_create_bot(body_of(bot)) == []

    def test_a_blank_agent_name_falls_through_to_the_first_name(self) -> None:
        owner = make_user(profile=UserProfile(agent_name="!!!"))
        assert build(owner=owner).display_name == "Laura Agentar"

    def test_owner_names_with_symbols_are_spoken_clean(self) -> None:
        owner = make_user(first_name="Laura\n(boss)", last_name="Rossi; DROP")
        bot = build(owner=owner)
        assert "\n" not in bot.greeting
        assert "(" not in bot.greeting


class TestEventsUrl:
    @pytest.mark.parametrize(
        "base",
        ["", "ftp://fastapi", "http://", "http://user:pass@fastapi:8000", "http://fastapi:8000?x=1", "http://fastapi#a", "fastapi:8000"],
    )
    def test_an_unusable_base_is_refused(self, monkeypatch, base) -> None:
        monkeypatch.setattr(
            settings, "_agentar", settings.agentar.model_copy(update={"worker_events_base_url": base})
        )
        with pytest.raises(WorkerDispatchError):
            events_url()

    def test_nothing_from_the_meeting_reaches_the_url(self) -> None:
        meeting = make_meeting(purpose="http://evil.example/x", title="https://evil.example")
        assert build(meeting).events_url == f"{EVENTS_BASE}/api/agentar/events"


def seed_picture(store: MemoryObjectStore, owner: str, kind="picture", data=JPEG, put=True) -> str:
    file_id = str(ObjectId())
    key = f"u/someone/avatar/{file_id}.jpg"
    record = AgentarFile(owner=owner, kind=kind, storage_key=key, content_type="image/jpeg")
    document = record.to_dict()
    document["_id"] = ObjectId(file_id)
    asyncio.run(AgentarFile.get_collection().insert_one(document))
    if put:
        asyncio.run(store.put(key, data))
    return file_id


class TestLoadAvatar:
    def test_meeting_picture_wins_over_the_profile_picture(self) -> None:
        store = MemoryObjectStore()
        owner = make_user()
        mine = seed_picture(store, owner.email, data=JPEG + b"meeting")
        theirs = seed_picture(store, owner.email, data=JPEG + b"profile")
        owner.profile.picture_id = theirs
        meeting = make_meeting(picture_id=mine)
        assert asyncio.run(load_avatar(meeting, owner, store)) == JPEG + b"meeting"

    def test_profile_picture_when_the_meeting_has_none(self) -> None:
        store = MemoryObjectStore()
        owner = make_user()
        owner.profile.picture_id = seed_picture(store, owner.email)
        assert asyncio.run(load_avatar(make_meeting(), owner, store)) == JPEG

    def test_none_when_neither_exists(self) -> None:
        assert asyncio.run(load_avatar(make_meeting(), make_user(), MemoryObjectStore())) is None

    def test_a_missing_object_falls_back_to_the_next_candidate(self) -> None:
        store = MemoryObjectStore()
        owner = make_user()
        meeting = make_meeting(picture_id=seed_picture(store, owner.email, put=False))
        owner.profile.picture_id = seed_picture(store, owner.email)
        assert asyncio.run(load_avatar(meeting, owner, store)) == JPEG

    def test_a_picture_of_another_owner_is_never_read(self) -> None:
        store = MemoryObjectStore()
        owner = make_user()
        meeting = make_meeting(picture_id=seed_picture(store, "other@meetween.test"))
        assert asyncio.run(load_avatar(meeting, owner, store)) is None

    def test_a_non_jpeg_or_oversized_object_is_skipped(self) -> None:
        store = MemoryObjectStore()
        owner = make_user()
        png = seed_picture(store, owner.email, data=b"\x89PNG" + b"0" * 20)
        big = seed_picture(store, owner.email, data=JPEG + b"0" * 409_700)
        assert asyncio.run(load_avatar(make_meeting(picture_id=png), owner, store)) is None
        assert asyncio.run(load_avatar(make_meeting(picture_id=big), owner, store)) is None

    def test_an_unreachable_store_is_an_error_not_a_default_face(self) -> None:
        class Down(MemoryObjectStore):
            async def get(self, key, *, chunk_size=0):
                raise ObjectStoreError("down")

        store = Down()
        owner = make_user()
        meeting = make_meeting(picture_id=seed_picture(store, owner.email))
        with pytest.raises(WorkerDispatchError) as raised:
            asyncio.run(load_avatar(meeting, owner, store))
        assert raised.value.reason == "storage_unavailable"


@pytest.fixture
def vp_configured(monkeypatch):
    from config import VirtualParticipantSettings

    monkeypatch.setattr(
        settings,
        "_virtual_participants",
        VirtualParticipantSettings(
            worker_url="http://worker:8000",
            worker_secret="worker-secret-value",
            require_human=True,
            max_bots_per_room=2,
            prosody_http_url="http://prosody:5280",
            xmpp_domain="meet.jitsi",
            request_timeout_s=5.0,
        ),
    )
    monkeypatch.setattr(settings, "jwt_jitsi_secret_key", "k" * 40)


def run_dispatch(handler, meeting=None, owner=None, store=None):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await dispatch_bot(
                meeting or make_meeting(), owner or make_user(), store=store, client=client
            )

    return asyncio.run(go())


def ok_handler(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        return httpx.Response(202, json={"id": "bot-1", "room": body["room"], "display_name": body["display_name"] + " (AI)", "state": "joining"})

    return handler


class TestDispatchBot:
    def test_posts_a_schema_valid_body_with_the_worker_secret(self, vp_configured) -> None:
        sent: list[httpx.Request] = []
        store = MemoryObjectStore()
        owner = make_user()
        owner.profile.picture_id = seed_picture(store, owner.email)
        bot = run_dispatch(ok_handler(sent), owner=owner, store=store)
        (request,) = sent
        assert request.method == "POST"
        assert str(request.url) == "http://worker:8000/bots"
        assert request.headers["X-VP-Secret"] == "worker-secret-value"
        body = json.loads(request.content)
        assert check_create_bot(body) == []
        assert base64.b64decode(body["avatar_jpeg_b64"]) == JPEG
        assert body["jwt"] and body["jwt"].count(".") == 2
        assert body["voice"] == "am_fenrir"
        assert bot["id"] == "bot-1"
        assert bot["display_name"] == "Lia Agentar (AI)"
        assert bot["state"] == "joining"

    def test_the_owner_is_read_from_the_meeting_when_not_given(self, vp_configured) -> None:
        owner = make_user()
        asyncio.run(owner.save())
        sent: list[httpx.Request] = []

        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(ok_handler(sent))) as client:
                return await dispatch_bot(make_meeting(), client=client)

        assert asyncio.run(go())["id"] == "bot-1"
        assert json.loads(sent[0].content)["display_name"] == "Lia Agentar"

    def test_an_owner_that_no_longer_exists_is_an_error(self, vp_configured) -> None:
        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(ok_handler([]))) as client:
                return await dispatch_bot(make_meeting(owner="gone@meetween.test"), client=client)

        with pytest.raises(WorkerDispatchError):
            asyncio.run(go())

    def test_without_a_picture_the_store_is_never_needed(self, vp_configured, monkeypatch) -> None:
        def unconfigured(cfg):
            raise AssertionError("the store was opened")

        monkeypatch.setattr(worker, "get_configured_object_store", unconfigured)
        owner = make_user(profile=None)
        asyncio.run(owner.save())

        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(ok_handler([]))) as client:
                return await dispatch_bot(make_meeting(), client=client)

        assert asyncio.run(go())["id"] == "bot-1"

    def test_a_picture_with_no_store_configured_is_an_error(self, vp_configured, monkeypatch) -> None:
        monkeypatch.setattr(worker, "get_configured_object_store", lambda cfg: None)
        owner = make_user()
        owner.profile.picture_id = seed_picture(MemoryObjectStore(), owner.email, put=False)
        with pytest.raises(WorkerDispatchError) as raised:
            run_dispatch(ok_handler([]), owner=owner, store=None)
        assert raised.value.reason == "storage_unavailable"

    def test_the_bot_token_is_room_bound_and_carries_no_name(self, vp_configured) -> None:
        from jose import jwt as jose_jwt

        sent: list[httpx.Request] = []
        run_dispatch(ok_handler(sent))
        token = json.loads(sent[0].content)["jwt"]
        claims = jose_jwt.get_unverified_claims(token)
        assert claims["room"] == "weekly-sync"
        assert claims["sub"] == "vp-bot"
        assert not claims.get("moderator")
        assert "context" not in claims

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, json={"id": "other-bot"}),
            httpx.Response(401),
            httpx.Response(422, json={"detail": [{"loc": ["body", "voice"], "msg": "bad", "input": "SECRET-ECHO"}]}),
            httpx.Response(429, headers={"Retry-After": "5"}),
            httpx.Response(500),
            httpx.Response(202, text="not json"),
            httpx.Response(202, json={"id": "bad id!"}),
            httpx.Response(202, json=["x"]),
        ],
        ids=["duplicate", "secret", "invalid", "full", "fault", "no-json", "bad-id", "not-object"],
    )
    def test_a_worker_that_does_not_accept_is_an_error(self, vp_configured, response) -> None:
        with pytest.raises(WorkerDispatchError) as raised:
            run_dispatch(lambda request: response)
        assert raised.value.reason == "worker_error"

    def test_an_unreachable_worker_is_an_error(self, vp_configured) -> None:
        def handler(request):
            raise httpx.ConnectError("refused")

        with pytest.raises(WorkerDispatchError) as raised:
            run_dispatch(handler)
        assert raised.value.reason == "worker_error"

    def test_an_empty_jitsi_key_stops_the_dispatch(self, vp_configured, monkeypatch) -> None:
        monkeypatch.setattr(settings, "jwt_jitsi_secret_key", "")
        with pytest.raises(WorkerDispatchError) as raised:
            run_dispatch(ok_handler([]))
        assert raised.value.reason == "jitsi_key_missing"

    def test_logs_never_carry_the_token_or_an_echoed_input(self, vp_configured, caplog) -> None:
        response = httpx.Response(
            422, json={"detail": [{"loc": ["body", "jwt"], "msg": "bad", "input": "TOKEN-ECHO-VALUE"}]}
        )
        with caplog.at_level(logging.DEBUG), pytest.raises(WorkerDispatchError) as raised:
            run_dispatch(lambda request: response)
        assert "TOKEN-ECHO-VALUE" not in caplog.text
        assert "TOKEN-ECHO-VALUE" not in str(raised.value)
        assert "body.jwt" in raised.value.detail


class TestRecallBot:
    def run(self, handler, **kwargs):
        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await recall_bot("bot-1", client=client, **kwargs)

        return asyncio.run(go())

    def test_farewell_is_a_flag_that_defaults_to_off(self, vp_configured) -> None:
        seen: list[httpx.Request] = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json={"id": "bot-1", "state": "leaving"})

        assert self.run(handler)["state"] == "leaving"
        assert self.run(handler, farewell=True)["state"] == "leaving"
        assert [dict(r.url.params) for r in seen] == [{}, {"farewell": "1"}]
        assert all(r.method == "DELETE" and r.url.path == "/bots/bot-1" for r in seen)
        assert all(r.headers["X-VP-Secret"] == "worker-secret-value" for r in seen)

    def test_an_unknown_bot_is_none_not_an_error(self, vp_configured) -> None:
        assert self.run(lambda request: httpx.Response(404)) is None

    @pytest.mark.parametrize("response", [httpx.Response(401), httpx.Response(500), httpx.Response(200, text="x")])
    def test_a_fault_is_an_error(self, vp_configured, response) -> None:
        with pytest.raises(WorkerDispatchError):
            self.run(lambda request: response)

    def test_an_unreachable_worker_is_an_error(self, vp_configured) -> None:
        def handler(request):
            raise httpx.ReadTimeout("slow")

        with pytest.raises(WorkerDispatchError):
            self.run(handler)


class TestGetBot:
    def run(self, handler):
        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await get_bot("bot-1", client=client)

        return asyncio.run(go())

    def test_returns_the_worker_view(self, vp_configured) -> None:
        seen: list[httpx.Request] = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json={"id": "bot-1", "state": "joined"})

        assert self.run(handler) == {"id": "bot-1", "state": "joined"}
        assert seen[0].method == "GET" and seen[0].url.path == "/bots/bot-1"
        assert seen[0].headers["X-VP-Secret"] == "worker-secret-value"

    def test_an_unknown_bot_is_none(self, vp_configured) -> None:
        assert self.run(lambda request: httpx.Response(404)) is None

    @pytest.mark.parametrize("response", [httpx.Response(401), httpx.Response(503), httpx.Response(200, json=[1])])
    def test_a_fault_is_an_error(self, vp_configured, response) -> None:
        with pytest.raises(WorkerDispatchError):
            self.run(lambda request: response)

    def test_the_id_is_quoted_into_the_path(self, vp_configured) -> None:
        seen: list[httpx.Request] = []

        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: (seen.append(r), httpx.Response(404))[1])) as client:
                return await get_bot("../x?y", client=client)

        asyncio.run(go())
        assert seen[0].url.raw_path == b"/bots/..%2Fx%3Fy"
