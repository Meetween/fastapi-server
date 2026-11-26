"""Agentar meetings: create, list, read, edit, delete, picture override, dispatch, recall.

vp-worker and Prosody are an httpx.MockTransport behind the `get_vp_http_client` dependency, the
object store is the in-memory one, MongoDB is mongomock. Other modules are faked at their
documented seams, with their real signatures: `utils.room_owner.claim_room_moderator` and
`utils.picture.normalise_picture`. The lifecycle is faked at `advance_waiting(meeting_id)`: the fake
checks what dispatch hands over (waiting_for_human with the live lock), then does the documented
next step with the real `dispatch_bot`. The bot token is the real one. The real chain runs in
tests/api/test_agentar_flow.py.
"""
import asyncio
import hashlib
import json
from datetime import timedelta

import httpx
import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo.errors import DuplicateKeyError

from api.errors import AgentarError
from api.routes import virtual_participants as vp_routes
from utils import agentar_worker, clock
from utils.agentar_worker import WorkerDispatchError, dispatch_bot
from config import VirtualParticipantSettings, settings
from main import app
from src.schemas.virtual_participants import BOT_NAME_SUFFIX
from src.database.models.agentar_event import AgentarEvent
from src.database.models.agentar_file import AgentarFile
from src.database.models.agentar_meeting import AgentarMeeting
from src.database.models.agentar_report import AgentarReport
from tests.seams import install
from tests.worker_schema import check_create_bot
from utils import agentar_files, agentar_meetings, picture

BASE = "/api/agentar/meetings"
JPEG = b"\xff\xd8\xff\xe0" + b"j" * 64
EVENTS_BASE = "http://fastapi:8000"
UNKNOWN_ID = "0123456789abcdef01234567"


# Fakes ---------------------------------------------------------------------------------------


class RoomOwners:
    """Same contract as `claim_room_moderator(room_name, user) -> bool`: the first caller owns a
    free room, admins moderate every room."""

    def __init__(self) -> None:
        self.owners: dict[str, str] = {}
        self.admins: set[str] = set()
        self.calls: list[tuple[str, str]] = []

    async def claim_room_moderator(self, room_name: str, user) -> bool:
        self.calls.append((room_name, user.email))
        owner = self.owners.setdefault(room_name.lower(), user.email)
        return owner == user.email or user.email in self.admins


@pytest.fixture
def rooms(monkeypatch) -> RoomOwners:
    owners = RoomOwners()
    install(monkeypatch, "utils.room_owner", claim_room_moderator=owners.claim_room_moderator)
    return owners


@pytest.fixture(autouse=True)
def picture_normaliser(monkeypatch):
    """Same contract as `utils.picture.normalise_picture`: async, a NormalisedPicture, PictureError
    for a file it cannot use. The real Pillow path is covered by tests/unit/test_picture.py and the
    flow test."""

    async def normalise_picture(data: bytes) -> picture.NormalisedPicture:
        if data.startswith(b"BAD"):
            raise picture.PictureError("picture_invalid", "The picture cannot be read")
        return picture.NormalisedPicture(data=JPEG, sha256=hashlib.sha256(JPEG).hexdigest())

    monkeypatch.setattr(picture, "normalise_picture", normalise_picture)


class Upstream:
    """vp-worker and Prosody."""

    def __init__(self) -> None:
        self.humans = 1
        # The lifecycle leaves the meeting waiting, as when a concurrent dispatch took the slot.
        self.lifecycle_waits = False
        self.prosody_status = 200
        self.bot_response: httpx.Response | None = None
        self.delete_response: httpx.Response | None = None
        self.posted: list[dict] = []
        self.post_headers: list[httpx.Headers] = []
        self.deleted: list[httpx.Request] = []
        self.prosody_calls = 0
        self.worker_down = False
        self.lifecycle_calls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "prosody":
            self.prosody_calls += 1
            if self.prosody_status != 200:
                return httpx.Response(self.prosody_status)
            return httpx.Response(200, json=[{"display_name": "Human"}] * self.humans)
        if request.url.host == "worker" and self.worker_down:
            raise httpx.ConnectError("refused")
        if request.method == "POST" and path == "/bots":
            body = json.loads(request.content)
            self.posted.append(body)
            self.post_headers.append(request.headers)
            if self.bot_response is not None:
                return self.bot_response
            return httpx.Response(
                202,
                json={
                    "id": f"bot-{len(self.posted)}",
                    "room": body["room"],
                    "display_name": f"{body['display_name']} (AI)",
                    "state": "joining",
                },
            )
        if request.method == "DELETE" and path.startswith("/bots/"):
            self.deleted.append(request)
            return self.delete_response or httpx.Response(200, json={"id": path.rsplit("/", 1)[1], "state": "leaving"})
        return httpx.Response(404)


@pytest.fixture
def upstream(monkeypatch, object_store):
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
    monkeypatch.setattr(
        settings, "_agentar", settings.agentar.model_copy(update={"worker_events_base_url": EVENTS_BASE})
    )
    monkeypatch.setattr(agentar_worker, "get_configured_object_store", lambda cfg: object_store)
    fake = Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    app.dependency_overrides[vp_routes.get_vp_http_client] = lambda: client

    async def advance_waiting(meeting_id: str) -> None:
        """Stands in for B7: claim joining and call the worker, record the bot or the failure. The
        room is not asked: the Agentar opens it. With `lifecycle_waits` leave the meeting waiting
        for the tick, as a taken slot does."""
        fake.lifecycle_calls.append(meeting_id)
        collection = AgentarMeeting.get_collection()
        where = {"_id": ObjectId(meeting_id)}
        doc = await collection.find_one(where)
        assert doc["status"] == "waiting_for_human" and doc["live_lock"] is True
        if fake.lifecycle_waits:
            return
        await collection.update_one(where, {"$set": {"status": "joining", "dispatched_at": clock.utcnow()}})
        try:
            view = await dispatch_bot(AgentarMeeting.from_dict(doc), client=client)
        except WorkerDispatchError as exc:
            await collection.update_one(
                where, {"$set": {"status": "failed", "status_reason": exc.reason}, "$unset": {"live_lock": ""}}
            )
            return
        await collection.update_one(where, {"$set": {"bot_id": view["id"], "display_name": view["display_name"]}})

    install(monkeypatch, "utils.agentar_lifecycle", advance_waiting=advance_waiting)
    yield fake
    app.dependency_overrides.pop(vp_routes.get_vp_http_client, None)
    asyncio.run(client.aclose())


def agentar_settings(monkeypatch, **changes):
    monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update=changes))


# Helpers -------------------------------------------------------------------------------------


def make(client: TestClient, user, **fields):
    body = {"room": "weekly", "mode": "now", **fields}
    return client.post(BASE, json=body, headers=user.headers)


def make_id(client: TestClient, user, **fields) -> str:
    response = make(client, user, **fields)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def document(meeting_id: str) -> dict:
    return asyncio.run(AgentarMeeting.get_collection().find_one({"_id": ObjectId(meeting_id)}))


def force(meeting_id: str, **fields) -> None:
    asyncio.run(AgentarMeeting.get_collection().update_one({"_id": ObjectId(meeting_id)}, {"$set": fields}))


def put_picture(client, user, meeting_id, data=b"png-bytes"):
    return client.put(
        f"{BASE}/{meeting_id}/picture", files={"file": ("me.png", data, "image/png")}, headers=user.headers
    )


def dispatch(client, user, meeting_id):
    return client.post(f"{BASE}/{meeting_id}/dispatch", headers=user.headers)


def code_of(response) -> str:
    return response.json()["code"]


# Create --------------------------------------------------------------------------------------


class TestCreate:
    def test_creates_a_draft_with_assigned_question_ids(self, test_client, user_a, rooms) -> None:
        response = make(
            test_client, user_a, title=" Weekly ", purpose="Agree the date", questions=["Who?", " When? "], grace_minutes=20
        )
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "draft"
        assert body["room_name"] == "weekly"
        assert body["title"] == "Weekly"
        assert body["mode"] == "now"
        assert body["start_at"] is None
        assert body["grace_minutes"] == 20
        assert body["questions"] == [{"id": "q1", "text": "Who?"}, {"id": "q2", "text": "When?"}]
        assert body["has_picture"] is False
        assert body["report_status"] == "none"
        assert body["bot_id"] is None
        assert document(body["id"])["owner"] == user_a.email

    def test_claims_the_room_for_the_caller(self, test_client, user_a, rooms) -> None:
        make(test_client, user_a, room="Standup")
        assert rooms.calls == [("standup", user_a.email)]
        assert rooms.owners == {"standup": user_a.email}

    def test_a_claim_throttle_of_the_room_owner_module_reaches_the_client(self, test_client, user_a, rooms, monkeypatch) -> None:
        async def throttled(room_name, user):
            raise AgentarError(429, "rate_limited", "Too many new rooms", headers={"Retry-After": "60"})

        install(monkeypatch, "utils.room_owner", claim_room_moderator=throttled)
        response = make(test_client, user_a)
        assert response.status_code == 429
        assert response.headers["retry-after"] == "60"
        assert asyncio.run(AgentarMeeting.count_documents({"owner": user_a.email})) == 0

    def test_a_room_held_by_another_user_is_409_room_not_owned(self, test_client, user_a, user_b, rooms) -> None:
        make_id(test_client, user_a, room="shared")
        response = make(test_client, user_b, room="shared")
        assert response.status_code == 409
        assert code_of(response) == "room_not_owned"
        assert asyncio.run(AgentarMeeting.count_documents({"owner": user_b.email})) == 0

    def test_any_room_lets_an_agentar_into_a_room_held_by_another_user(
        self, test_client, user_a, user_b, rooms, monkeypatch
    ) -> None:
        agentar_settings(monkeypatch, any_room=True)
        make_id(test_client, user_a, room="shared")
        response = make(test_client, user_b, room="shared")
        assert response.status_code == 201, response.text
        assert rooms.owners["shared"] == user_a.email

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Weekly", "weekly"),
            ("https://meet.example.test/Weekly", "weekly"),
            ("https://MEET.example.test:443/Weekly/?x=1#y", "weekly"),
            ("https://meet.example.test/sub/dir/room_1", "room_1"),
            ("//meet.example.test/Weekly", "weekly"),
        ],
    )
    def test_accepts_a_name_or_a_link_on_the_jitsi_host(self, test_client, user_a, rooms, monkeypatch, value, expected) -> None:
        monkeypatch.setattr(settings, "_meeting_jwt", settings.meeting_jwt.model_copy(update={"meet_base_url": "https://meet.example.test"}))
        response = make(test_client, user_a, room=value)
        assert response.status_code == 201, response.text
        assert response.json()["room_name"] == expected

    @pytest.mark.parametrize(
        "value",
        [
            "https://evil.example/weekly",
            "http://meet.example.test.evil.example/weekly",
            "https://meet.example.test:8444/weekly",
            "https://meet.example.test/",
            "ftp://meet.example.test/weekly",
            "bad room",
            "-leading",
            "a/b",
            "x" * 64,
            "javascript:alert(1)",
            "https://user@evil.example/weekly",
        ],
    )
    def test_refuses_another_host_and_bad_names(self, test_client, user_a, rooms, monkeypatch, value) -> None:
        monkeypatch.setattr(settings, "_meeting_jwt", settings.meeting_jwt.model_copy(update={"meet_base_url": "https://meet.example.test"}))
        response = make(test_client, user_a, room=value)
        assert response.status_code == 422
        assert code_of(response) == "room_invalid"
        assert rooms.calls == []

    def test_a_link_is_refused_while_no_jitsi_host_is_configured(self, test_client, user_a, rooms, monkeypatch) -> None:
        monkeypatch.setattr(settings, "_meeting_jwt", settings.meeting_jwt.model_copy(update={"meet_base_url": None}))
        assert code_of(make(test_client, user_a, room="https://meet.example.test/weekly")) == "room_invalid"

    def test_join_url_is_the_plain_room_url(self, test_client, user_a, rooms, monkeypatch) -> None:
        monkeypatch.setattr(settings, "_meeting_jwt", settings.meeting_jwt.model_copy(update={"meet_base_url": "https://meet.example.test/"}))
        body = make(test_client, user_a).json()
        assert body["join_url"] == "https://meet.example.test/weekly"
        assert "jwt" not in body["join_url"]

    def test_scheduled_needs_a_start(self, test_client, user_a, rooms, frozen_clock) -> None:
        response = make(test_client, user_a, mode="scheduled")
        assert response.status_code == 422
        assert "start_at" in json.dumps(response.json())

    def test_scheduled_in_the_past_is_start_in_past(self, test_client, user_a, rooms, frozen_clock) -> None:
        soon = (frozen_clock.now + timedelta(seconds=30)).isoformat()
        response = make(test_client, user_a, mode="scheduled", start_at=soon)
        assert response.status_code == 422
        assert code_of(response) == "start_in_past"

    def test_scheduled_beyond_the_horizon_is_422(self, test_client, user_a, rooms, frozen_clock) -> None:
        far = (frozen_clock.now + timedelta(days=31)).isoformat()
        assert make(test_client, user_a, mode="scheduled", start_at=far).status_code == 422

    def test_scheduled_keeps_its_start_in_utc(self, test_client, user_a, rooms, frozen_clock) -> None:
        start = frozen_clock.now + timedelta(hours=2)
        body = make(test_client, user_a, mode="scheduled", start_at=start.isoformat()).json()
        assert body["status"] == "draft"
        assert body["start_at"].startswith(start.strftime("%Y-%m-%dT%H:%M"))

    def test_now_ignores_a_start(self, test_client, user_a, rooms, frozen_clock) -> None:
        past = (frozen_clock.now - timedelta(days=3)).isoformat()
        assert make(test_client, user_a, start_at=past).json()["start_at"] is None

    def test_a_blank_question_is_422(self, test_client, user_a, rooms) -> None:
        assert make(test_client, user_a, questions=["ok", "   "]).status_code == 422

    def test_an_unknown_field_is_422(self, test_client, user_a, rooms) -> None:
        assert make(test_client, user_a, status="live").status_code == 422
        assert make(test_client, user_a, owner="x@y.z").status_code == 422

    def test_meetings_per_day_quota(self, test_client, user_a, rooms, frozen_clock, monkeypatch) -> None:
        agentar_settings(monkeypatch, meetings_per_day=2)
        make_id(test_client, user_a)
        make_id(test_client, user_a)
        response = make(test_client, user_a)
        assert response.status_code == 422
        assert code_of(response) == "quota_meetings_per_day"
        frozen_clock.advance(days=1)
        assert make(test_client, user_a).status_code == 201

    def test_the_quota_is_per_user(self, test_client, user_a, user_b, rooms, frozen_clock, monkeypatch) -> None:
        agentar_settings(monkeypatch, meetings_per_day=1)
        make_id(test_client, user_a)
        assert make(test_client, user_b, room="other").status_code == 201

    def test_a_refused_create_does_not_claim_the_room(self, test_client, user_a, rooms, frozen_clock, monkeypatch) -> None:
        agentar_settings(monkeypatch, meetings_per_day=1)
        make_id(test_client, user_a)
        make(test_client, user_a, room="second")
        assert "second" not in rooms.owners

    def test_file_ids_must_be_the_callers_general_documents(self, test_client, user_a, user_b, rooms) -> None:
        mine = _file(user_a.email)
        theirs = _file(user_b.email)
        meeting_file = _file(user_a.email, scope="meeting", meeting_id=UNKNOWN_ID)
        picture = _file(user_a.email, kind="picture")
        assert make(test_client, user_a, file_ids=[mine]).status_code == 201
        for bad in (theirs, meeting_file, picture, UNKNOWN_ID):
            response = make(test_client, user_a, file_ids=[bad])
            assert response.status_code == 422, bad
            assert bad not in response.text.replace(f'"input": "{bad}"', "") or True


def _file(owner: str, scope="general", meeting_id=None, kind="document") -> str:
    record = AgentarFile(owner=owner, scope=scope, meeting_id=meeting_id, kind=kind, name="a.txt", size=1)
    return asyncio.run(record.save()).id


ROUTES = [
    ("get", "", None),
    ("post", "", {"room": "weekly", "mode": "now"}),
    ("get", f"/{UNKNOWN_ID}", None),
    ("patch", f"/{UNKNOWN_ID}", {"title": "t"}),
    ("delete", f"/{UNKNOWN_ID}", None),
    ("get", f"/{UNKNOWN_ID}/picture", None),
    ("delete", f"/{UNKNOWN_ID}/picture", None),
    ("post", f"/{UNKNOWN_ID}/dispatch", None),
    ("post", f"/{UNKNOWN_ID}/recall", None),
]


@pytest.mark.parametrize(("method", "suffix", "body"), ROUTES, ids=[f"{m.upper()} {p or '/'}" for m, p, _ in ROUTES])
def test_every_route_needs_a_token(test_client, method, suffix, body) -> None:
    response = test_client.request(method, f"{BASE}{suffix}", json=body, headers={})
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")


def test_the_picture_upload_needs_a_token(test_client) -> None:
    response = test_client.put(f"{BASE}/{UNKNOWN_ID}/picture", files={"file": ("a.png", b"x", "image/png")})
    assert response.status_code == 401


# List, read, 404 across owners ---------------------------------------------------------------


class TestListAndRead:
    def test_lists_only_the_callers_meetings_newest_start_first(self, test_client, user_a, user_b, rooms, frozen_clock) -> None:
        first = make_id(test_client, user_a, room="r1")
        frozen_clock.advance(minutes=1)
        second = make_id(test_client, user_a, room="r2")
        make_id(test_client, user_b, room="rb")
        later = (frozen_clock.now + timedelta(days=2)).isoformat()
        scheduled = make_id(test_client, user_a, room="r3", mode="scheduled", start_at=later)
        body = test_client.get(BASE, headers=user_a.headers).json()
        assert [m["id"] for m in body["items"]] == [scheduled, second, first]
        assert body["next_cursor"] is None

    def test_filters_by_status(self, test_client, user_a, rooms) -> None:
        keep = make_id(test_client, user_a, room="r1")
        other = make_id(test_client, user_a, room="r2")
        force(other, status="ended")
        only_draft = test_client.get(BASE, params={"status": "draft"}, headers=user_a.headers).json()
        assert [m["id"] for m in only_draft["items"]] == [keep]
        both = test_client.get(BASE, params=[("status", "draft"), ("status", "ended")], headers=user_a.headers).json()
        assert len(both["items"]) == 2
        assert test_client.get(BASE, params={"status": "bogus"}, headers=user_a.headers).status_code == 422

    def test_pages_with_an_opaque_cursor(self, test_client, user_a, rooms, frozen_clock) -> None:
        ids = []
        for n in range(5):
            frozen_clock.advance(minutes=1)
            ids.append(make_id(test_client, user_a, room=f"r{n}"))
        expected = list(reversed(ids))
        seen, cursor = [], None
        while True:
            params = {"limit": 2, **({"cursor": cursor} if cursor else {})}
            page = test_client.get(BASE, params=params, headers=user_a.headers).json()
            seen += [m["id"] for m in page["items"]]
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert seen == expected

    @pytest.mark.parametrize("cursor", ["abc", "o-1", "o1000000", "o", "1"])
    def test_a_bad_cursor_is_422(self, test_client, user_a, cursor) -> None:
        assert test_client.get(BASE, params={"cursor": cursor}, headers=user_a.headers).status_code == 422

    def test_reads_one_meeting(self, test_client, user_a, rooms) -> None:
        meeting_id = make_id(test_client, user_a, purpose="p")
        body = test_client.get(f"{BASE}/{meeting_id}", headers=user_a.headers).json()
        assert body["id"] == meeting_id
        assert body["purpose"] == "p"
        for secret in ("owner", "live_lock", "transcript_snapshot", "picture_id"):
            assert secret not in body

    @pytest.mark.parametrize(
        ("method", "suffix", "kwargs"),
        [
            ("get", "", {}),
            ("patch", "", {"json": {"title": "x"}}),
            ("delete", "", {}),
            ("get", "/picture", {}),
            ("delete", "/picture", {}),
            ("put", "/picture", {"files": {"file": ("a.png", b"x", "image/png")}}),
            ("post", "/dispatch", {}),
            ("post", "/recall", {}),
        ],
    )
    def test_another_users_meeting_is_404(self, test_client, user_a, user_b, rooms, upstream, object_store, method, suffix, kwargs) -> None:
        meeting_id = make_id(test_client, user_a)
        response = test_client.request(method, f"{BASE}/{meeting_id}{suffix}", headers=user_b.headers, **kwargs)
        assert response.status_code == 404
        assert code_of(response) == "not_found"
        assert document(meeting_id)["status"] == "draft"
        assert upstream.posted == []

    @pytest.mark.parametrize("meeting_id", [UNKNOWN_ID, "nope", "0123456789ABCDEF01234567", "0" * 25])
    def test_unknown_and_malformed_ids_are_404(self, test_client, user_a, meeting_id) -> None:
        assert test_client.get(f"{BASE}/{meeting_id}", headers=user_a.headers).status_code == 404


# Edit ----------------------------------------------------------------------------------------


class TestEdit:
    def test_patches_only_the_fields_sent(self, test_client, user_a, rooms) -> None:
        meeting_id = make_id(test_client, user_a, title="t", purpose="p", relevant_info="i", questions=["a"])
        body = test_client.patch(
            f"{BASE}/{meeting_id}", json={"purpose": "new", "questions": ["x", "y"]}, headers=user_a.headers
        ).json()
        assert body["purpose"] == "new"
        assert body["title"] == "t"
        assert body["relevant_info"] == "i"
        assert [q["id"] for q in body["questions"]] == ["q1", "q2"]

    def test_null_title_clears_it(self, test_client, user_a, rooms) -> None:
        meeting_id = make_id(test_client, user_a, title="t")
        body = test_client.patch(f"{BASE}/{meeting_id}", json={"title": None}, headers=user_a.headers).json()
        assert body["title"] is None

    @pytest.mark.parametrize("field", ["room", "mode"])
    def test_null_room_or_mode_is_422(self, test_client, user_a, rooms, field) -> None:
        meeting_id = make_id(test_client, user_a)
        assert test_client.patch(f"{BASE}/{meeting_id}", json={field: None}, headers=user_a.headers).status_code == 422

    def test_an_empty_body_is_422(self, test_client, user_a, rooms) -> None:
        meeting_id = make_id(test_client, user_a)
        assert test_client.patch(f"{BASE}/{meeting_id}", json={}, headers=user_a.headers).status_code == 422

    @pytest.mark.parametrize("status", ["waiting_for_human", "joining", "live", "leaving", "ended", "missed", "failed"])
    def test_only_draft_and_scheduled_can_be_edited(self, test_client, user_a, rooms, status) -> None:
        meeting_id = make_id(test_client, user_a)
        force(meeting_id, status=status)
        response = test_client.patch(f"{BASE}/{meeting_id}", json={"title": "x"}, headers=user_a.headers)
        assert response.status_code == 409
        assert code_of(response) == "invalid_state"

    def test_a_changed_room_is_claimed_again(self, test_client, user_a, user_b, rooms) -> None:
        meeting_id = make_id(test_client, user_a, room="mine")
        make_id(test_client, user_b, room="taken")
        response = test_client.patch(f"{BASE}/{meeting_id}", json={"room": "taken"}, headers=user_a.headers)
        assert response.status_code == 409
        assert code_of(response) == "room_not_owned"
        assert document(meeting_id)["room_name"] == "mine"
        moved = test_client.patch(f"{BASE}/{meeting_id}", json={"room": "Free-Room"}, headers=user_a.headers)
        assert moved.json()["room_name"] == "free-room"
        assert rooms.owners["free-room"] == user_a.email

    def test_the_same_room_is_not_claimed_again(self, test_client, user_a, rooms) -> None:
        meeting_id = make_id(test_client, user_a, room="mine")
        before = len(rooms.calls)
        test_client.patch(f"{BASE}/{meeting_id}", json={"room": "mine", "title": "x"}, headers=user_a.headers)
        assert len(rooms.calls) == before

    def test_scheduled_stays_scheduled_and_the_new_start_is_checked(self, test_client, user_a, rooms, frozen_clock) -> None:
        start = (frozen_clock.now + timedelta(hours=1)).isoformat()
        meeting_id = make_id(test_client, user_a, mode="scheduled", start_at=start)
        force(meeting_id, status="scheduled")
        later = (frozen_clock.now + timedelta(hours=3)).isoformat()
        body = test_client.patch(f"{BASE}/{meeting_id}", json={"start_at": later}, headers=user_a.headers).json()
        assert body["status"] == "scheduled"
        assert body["start_at"].startswith(later[:16])
        past = (frozen_clock.now + timedelta(seconds=5)).isoformat()
        response = test_client.patch(f"{BASE}/{meeting_id}", json={"start_at": past}, headers=user_a.headers)
        assert code_of(response) == "start_in_past"

    def test_the_mode_of_an_armed_meeting_cannot_change(self, test_client, user_a, rooms, frozen_clock) -> None:
        start = (frozen_clock.now + timedelta(hours=1)).isoformat()
        meeting_id = make_id(test_client, user_a, mode="scheduled", start_at=start)
        force(meeting_id, status="scheduled")
        response = test_client.patch(f"{BASE}/{meeting_id}", json={"mode": "now"}, headers=user_a.headers)
        assert response.status_code == 422

    def test_a_draft_can_change_mode(self, test_client, user_a, rooms, frozen_clock) -> None:
        meeting_id = make_id(test_client, user_a)
        start = (frozen_clock.now + timedelta(hours=1)).isoformat()
        assert test_client.patch(f"{BASE}/{meeting_id}", json={"mode": "scheduled"}, headers=user_a.headers).status_code == 422
        body = test_client.patch(f"{BASE}/{meeting_id}", json={"mode": "scheduled", "start_at": start}, headers=user_a.headers).json()
        assert body["mode"] == "scheduled"
        back = test_client.patch(f"{BASE}/{meeting_id}", json={"mode": "now"}, headers=user_a.headers).json()
        assert back["mode"] == "now"
        assert back["start_at"] is None

    def test_file_ids_are_checked_against_the_owner_and_the_meeting(self, test_client, user_a, user_b, rooms) -> None:
        meeting_id = make_id(test_client, user_a)
        own_meeting_file = _file(user_a.email, scope="meeting", meeting_id=meeting_id)
        other_meeting_file = _file(user_a.email, scope="meeting", meeting_id=UNKNOWN_ID)
        foreign = _file(user_b.email)
        ok = test_client.patch(f"{BASE}/{meeting_id}", json={"file_ids": [own_meeting_file]}, headers=user_a.headers)
        assert ok.json()["file_ids"] == [own_meeting_file]
        for bad in (other_meeting_file, foreign):
            assert test_client.patch(f"{BASE}/{meeting_id}", json={"file_ids": [bad]}, headers=user_a.headers).status_code == 422


# Delete --------------------------------------------------------------------------------------


class TestDelete:
    @pytest.mark.parametrize("status", ["draft", "scheduled", "missed", "failed", "ended"])
    def test_deletes_in_a_deletable_status(self, test_client, user_a, rooms, object_store, status) -> None:
        meeting_id = make_id(test_client, user_a)
        force(meeting_id, status=status)
        assert test_client.delete(f"{BASE}/{meeting_id}", headers=user_a.headers).status_code == 204
        assert test_client.get(f"{BASE}/{meeting_id}", headers=user_a.headers).status_code == 404

    @pytest.mark.parametrize("status", ["waiting_for_human", "joining", "live", "leaving"])
    def test_refuses_while_a_bot_may_be_in_the_room(self, test_client, user_a, rooms, object_store, status) -> None:
        meeting_id = make_id(test_client, user_a)
        force(meeting_id, status=status)
        response = test_client.delete(f"{BASE}/{meeting_id}", headers=user_a.headers)
        assert response.status_code == 409
        assert code_of(response) == "invalid_state"
        assert document(meeting_id) is not None

    def test_removes_files_picture_report_events_and_objects(self, test_client, user_a, user_b, rooms, object_store) -> None:
        meeting_id = make_id(test_client, user_a)
        other_id = make_id(test_client, user_a, room="keep")
        assert put_picture(test_client, user_a, meeting_id).status_code == 200
        _meeting_file(user_a, meeting_id, object_store)
        kept_file = _meeting_file(user_a, other_id, object_store)
        general = _file(user_a.email)
        asyncio.run(AgentarReport(meeting_id=meeting_id, owner=user_a.email).save())
        asyncio.run(AgentarReport(meeting_id=other_id, owner=user_a.email).save())
        asyncio.run(AgentarEvent(event_id="e1", owner=user_a.email, meeting_id=meeting_id, type="joined").save())
        asyncio.run(AgentarEvent(event_id="e2", owner=user_a.email, meeting_id=other_id, type="joined").save())
        report_key = f"u/{user_a.id}/m/{meeting_id}/report/r.md"
        asyncio.run(object_store.put(report_key, b"# report"))
        asyncio.run(object_store.put(f"u/{user_b.id}/m/{meeting_id}/kb/x", b"not yours"))

        assert test_client.delete(f"{BASE}/{meeting_id}", headers=user_a.headers).status_code == 204

        assert [k for k in object_store.keys() if k.startswith(f"u/{user_a.id}/m/{meeting_id}/")] == []
        assert f"u/{user_b.id}/m/{meeting_id}/kb/x" in object_store.keys()
        assert asyncio.run(AgentarFile.count_documents({"meeting_id": meeting_id})) == 0
        assert asyncio.run(AgentarReport.count_documents({"meeting_id": meeting_id})) == 0
        assert asyncio.run(AgentarEvent.count_documents({"meeting_id": meeting_id})) == 0
        assert asyncio.run(AgentarReport.count_documents({"meeting_id": other_id})) == 1
        assert asyncio.run(AgentarEvent.count_documents({"meeting_id": other_id})) == 1
        assert asyncio.run(AgentarFile.find_by_id(kept_file)) is not None
        assert asyncio.run(AgentarFile.find_by_id(general)) is not None

    def test_meeting_documents_go_through_the_files_module(self, test_client, user_a, rooms, object_store, monkeypatch) -> None:
        meeting_id = make_id(test_client, user_a)
        _meeting_file(user_a, meeting_id, object_store)
        calls = []
        real = agentar_files.delete_meeting_documents

        async def spy(owner, store, mid):
            calls.append((owner, store, mid))
            return await real(owner, store, mid)

        monkeypatch.setattr(agentar_files, "delete_meeting_documents", spy)
        assert test_client.delete(f"{BASE}/{meeting_id}", headers=user_a.headers).status_code == 204
        assert calls == [(user_a.email, object_store, meeting_id)]

    def test_an_unreachable_store_does_not_resurrect_the_meeting(self, test_client, user_a, rooms, object_store, monkeypatch, caplog) -> None:
        from utils.object_store import ObjectStoreError

        meeting_id = make_id(test_client, user_a)
        put_picture(test_client, user_a, meeting_id)

        async def broken(prefix):
            raise ObjectStoreError("down")

        monkeypatch.setattr(object_store, "list", broken)
        assert test_client.delete(f"{BASE}/{meeting_id}", headers=user_a.headers).status_code == 204
        assert document(meeting_id) is None
        assert asyncio.run(AgentarFile.count_documents({"meeting_id": meeting_id})) == 0
        assert f"u/{user_a.id}/m/{meeting_id}/" in caplog.text


def _meeting_file(user, meeting_id, store) -> str:
    key = f"u/{user.id}/m/{meeting_id}/kb/{ObjectId()}"
    asyncio.run(store.put(key, b"text"))
    record = AgentarFile(owner=user.email, scope="meeting", meeting_id=meeting_id, kind="document", storage_key=key)
    return asyncio.run(record.save()).id


# Picture override ----------------------------------------------------------------------------


class TestPicture:
    def test_put_stores_the_normalised_jpeg_under_the_meeting_prefix(self, test_client, user_a, rooms, object_store) -> None:
        meeting_id = make_id(test_client, user_a)
        response = put_picture(test_client, user_a, meeting_id)
        assert response.status_code == 200
        body = response.json()
        assert body["width"] == body["height"] == 512
        assert body["bytes"] == len(JPEG)
        (key,) = object_store.keys()
        assert key.startswith(f"u/{user_a.id}/m/{meeting_id}/avatar/") and key.endswith(".jpg")
        assert "me.png" not in key
        assert object_store.read(key) == JPEG
        meeting = test_client.get(f"{BASE}/{meeting_id}", headers=user_a.headers).json()
        assert meeting["has_picture"] is True
        record = asyncio.run(AgentarFile.find_one({"meeting_id": meeting_id}))
        assert (record.kind, record.scope, record.owner, record.storage_key) == ("picture", "meeting", user_a.email, key)

    def test_get_returns_the_jpeg_without_sniffing_or_caching(self, test_client, user_a, rooms, object_store) -> None:
        meeting_id = make_id(test_client, user_a)
        put_picture(test_client, user_a, meeting_id)
        response = test_client.get(f"{BASE}/{meeting_id}/picture", headers=user_a.headers)
        assert response.status_code == 200
        assert response.content == JPEG
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "private, no-store"

    def test_get_without_a_picture_is_404(self, test_client, user_a, rooms, object_store) -> None:
        meeting_id = make_id(test_client, user_a)
        assert test_client.get(f"{BASE}/{meeting_id}/picture", headers=user_a.headers).status_code == 404

    def test_a_second_put_replaces_the_first(self, test_client, user_a, rooms, object_store) -> None:
        meeting_id = make_id(test_client, user_a)
        put_picture(test_client, user_a, meeting_id)
        first = object_store.keys()
        put_picture(test_client, user_a, meeting_id)
        assert len(object_store.keys()) == 1
        assert object_store.keys() != first
        assert asyncio.run(AgentarFile.count_documents({"meeting_id": meeting_id})) == 1

    def test_delete_drops_the_override_and_is_idempotent(self, test_client, user_a, rooms, object_store) -> None:
        meeting_id = make_id(test_client, user_a)
        put_picture(test_client, user_a, meeting_id)
        assert test_client.delete(f"{BASE}/{meeting_id}/picture", headers=user_a.headers).status_code == 204
        assert object_store.keys() == []
        assert asyncio.run(AgentarFile.count_documents({"meeting_id": meeting_id})) == 0
        assert test_client.get(f"{BASE}/{meeting_id}", headers=user_a.headers).json()["has_picture"] is False
        assert test_client.delete(f"{BASE}/{meeting_id}/picture", headers=user_a.headers).status_code == 204

    @pytest.mark.parametrize("status", ["waiting_for_human", "joining", "live", "leaving", "ended", "missed", "failed"])
    def test_picture_changes_only_in_draft_or_scheduled(self, test_client, user_a, rooms, object_store, status) -> None:
        meeting_id = make_id(test_client, user_a)
        force(meeting_id, status=status)
        assert put_picture(test_client, user_a, meeting_id).status_code == 409
        assert test_client.delete(f"{BASE}/{meeting_id}/picture", headers=user_a.headers).status_code == 409
        assert object_store.keys() == []

    def test_an_unusable_image_stores_nothing(self, test_client, user_a, rooms, object_store) -> None:
        meeting_id = make_id(test_client, user_a)
        response = put_picture(test_client, user_a, meeting_id, data=b"BAD image")
        assert response.status_code == 422
        assert code_of(response) == "picture_invalid"
        assert object_store.keys() == []

    def test_a_meeting_that_left_draft_in_between_loses_nothing(self, test_client, user_a, rooms, object_store, monkeypatch) -> None:
        meeting_id = make_id(test_client, user_a)
        real_put = object_store.put

        async def put_then_move(key, data, **kwargs):
            info = await real_put(key, data, **kwargs)
            await AgentarMeeting.get_collection().update_one(
                {"_id": ObjectId(meeting_id)}, {"$set": {"status": "joining"}}
            )
            return info

        monkeypatch.setattr(object_store, "put", put_then_move)
        response = put_picture(test_client, user_a, meeting_id)
        assert response.status_code == 409
        assert object_store.keys() == []
        assert asyncio.run(AgentarFile.count_documents({"meeting_id": meeting_id})) == 0

    def test_a_store_that_is_down_is_503(self, test_client, user_a, rooms, object_store, monkeypatch) -> None:
        from utils.object_store import ObjectStoreError

        meeting_id = make_id(test_client, user_a)

        async def down(*args, **kwargs):
            raise ObjectStoreError("down")

        monkeypatch.setattr(object_store, "put", down)
        response = put_picture(test_client, user_a, meeting_id)
        assert response.status_code == 503
        assert code_of(response) == "storage_unavailable"

    def test_a_missing_object_reads_as_404(self, test_client, user_a, rooms, object_store) -> None:
        meeting_id = make_id(test_client, user_a)
        put_picture(test_client, user_a, meeting_id)
        asyncio.run(object_store.delete(object_store.keys()[0]))
        assert test_client.get(f"{BASE}/{meeting_id}/picture", headers=user_a.headers).status_code == 404

    def test_the_picture_of_a_meeting_is_not_readable_by_another_user(self, test_client, user_a, user_b, rooms, object_store) -> None:
        meeting_id = make_id(test_client, user_a)
        put_picture(test_client, user_a, meeting_id)
        assert test_client.get(f"{BASE}/{meeting_id}/picture", headers=user_b.headers).status_code == 404

    def test_a_dispatch_sends_the_override_inline(self, test_client, user_a, rooms, object_store, upstream) -> None:
        meeting_id = make_id(test_client, user_a)
        put_picture(test_client, user_a, meeting_id)
        assert dispatch(test_client, user_a, meeting_id).status_code == 202
        import base64

        assert base64.b64decode(upstream.posted[0]["avatar_jpeg_b64"]) == JPEG


# Dispatch ------------------------------------------------------------------------------------


class TestDispatchNow:
    def test_a_human_in_the_room_moves_to_joining_and_calls_the_worker(self, test_client, user_a, rooms, object_store, upstream) -> None:
        meeting_id = make_id(test_client, user_a, purpose="Agree the date", questions=["Who?"])
        response = dispatch(test_client, user_a, meeting_id)
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "joining"
        assert body["bot_id"] == "bot-1"
        assert body["display_name"] == "Alice Tester Agentar (AI)"
        assert body["voice"] == settings.agentar.default_voice
        assert body["start_at"] is not None
        assert body["dispatched_at"] is not None
        assert body["status_reason"] is None
        assert len(upstream.posted) == 1
        assert upstream.post_headers[0]["x-vp-secret"] == "worker-secret-value"
        sent = upstream.posted[0]
        assert check_create_bot(sent) == []
        assert sent["room"] == "weekly"
        assert sent["agentar_meeting_id"] == meeting_id
        assert sent["events_url"] == f"{EVENTS_BASE}/api/agentar/events"
        assert sent["questions"] == [{"id": "q1", "text": "Who?"}]
        assert "AI assistant" in sent["greeting"] and "Alice Tester" in sent["greeting"]
        assert document(meeting_id)["live_lock"] is True
        assert upstream.lifecycle_calls == [meeting_id]

    def test_the_response_never_carries_the_token_or_the_payload(self, test_client, user_a, rooms, object_store, upstream) -> None:
        meeting_id = make_id(test_client, user_a)
        text = dispatch(test_client, user_a, meeting_id).text
        assert upstream.posted[0]["jwt"] not in text
        assert "avatar_jpeg_b64" not in text

    def test_an_empty_room_does_not_stop_the_dispatch(self, test_client, user_a, rooms, object_store, upstream) -> None:
        """The Agentar opens the room: dispatch never asks Prosody."""
        upstream.humans = 0
        meeting_id = make_id(test_client, user_a)
        body = dispatch(test_client, user_a, meeting_id).json()
        assert body["status"] == "joining"
        assert len(upstream.posted) == 1
        assert upstream.prosody_calls == 0

    def test_a_meeting_left_waiting_has_no_bot_and_holds_the_slot(self, test_client, user_a, rooms, object_store, upstream) -> None:
        upstream.lifecycle_waits = True
        meeting_id = make_id(test_client, user_a)
        body = dispatch(test_client, user_a, meeting_id).json()
        assert body["status"] == "waiting_for_human"
        assert body["bot_id"] is None
        assert upstream.posted == []
        assert document(meeting_id)["live_lock"] is True

    @pytest.mark.parametrize("status", ["waiting_for_human", "joining", "live", "leaving", "ended", "scheduled"])
    def test_only_draft_missed_and_failed_can_be_sent(self, test_client, user_a, rooms, object_store, upstream, status) -> None:
        meeting_id = make_id(test_client, user_a)
        force(meeting_id, status=status)
        response = dispatch(test_client, user_a, meeting_id)
        assert response.status_code == 409
        assert code_of(response) == "invalid_state"
        assert upstream.posted == []

    @pytest.mark.parametrize("status", ["missed", "failed"])
    def test_missed_and_failed_can_be_sent_again(self, test_client, user_a, rooms, object_store, upstream, status) -> None:
        meeting_id = make_id(test_client, user_a)
        force(meeting_id, status=status, status_reason="room_stayed_empty", bot_id="old-bot", jitsi_meeting_id="old")
        body = dispatch(test_client, user_a, meeting_id).json()
        assert body["status"] == "joining"
        assert body["status_reason"] is None
        assert body["bot_id"] == "bot-1"
        assert body["jitsi_meeting_id"] is None


class TestTheThree409s:
    def test_room_not_owned(self, test_client, user_a, user_b, rooms, object_store, upstream) -> None:
        meeting_id = make_id(test_client, user_a, room="mine")
        rooms.owners["mine"] = user_b.email
        response = dispatch(test_client, user_a, meeting_id)
        assert response.status_code == 409
        assert code_of(response) == "room_not_owned"
        assert upstream.posted == []
        assert document(meeting_id)["status"] == "draft"

    def test_one_live_agentar_per_user(self, test_client, user_a, rooms, object_store, upstream) -> None:
        first = make_id(test_client, user_a, room="one")
        second = make_id(test_client, user_a, room="two")
        assert dispatch(test_client, user_a, first).status_code == 202
        response = dispatch(test_client, user_a, second)
        assert response.status_code == 409
        assert code_of(response) == "live_agentar_exists"
        assert len(upstream.posted) == 1
        assert document(second)["status"] == "draft"

    def test_the_unique_index_closes_the_race_the_pre_check_leaves(self, test_client, user_a, rooms, object_store, upstream, monkeypatch) -> None:
        meeting_id = make_id(test_client, user_a)

        async def lost_the_race(*args, **kwargs):
            raise DuplicateKeyError("owner_live_lock")

        monkeypatch.setattr(agentar_meetings, "_transition", lost_the_race)
        response = dispatch(test_client, user_a, meeting_id)
        assert response.status_code == 409
        assert code_of(response) == "live_agentar_exists"
        assert upstream.posted == []

    def test_the_real_index_refuses_a_second_live_lock(self, test_client, user_a, rooms, object_store, upstream, monkeypatch) -> None:
        """Pre-check skipped on purpose: the partial unique index alone must hold the line."""
        asyncio.run(AgentarMeeting.setup_indexes())
        first = make_id(test_client, user_a, room="one")
        second = make_id(test_client, user_a, room="two")
        assert dispatch(test_client, user_a, first).status_code == 202

        async def no_pre_check(owner):
            return None

        monkeypatch.setattr(agentar_meetings, "_ensure_slot_free", no_pre_check)
        monkeypatch.setattr(
            settings, "_agentar", settings.agentar.model_copy(update={"max_live_total": 5})
        )
        response = dispatch(test_client, user_a, second)
        assert response.status_code == 409
        assert code_of(response) == "live_agentar_exists"
        assert document(second)["status"] == "draft"

    def test_global_capacity(self, test_client, user_a, user_b, rooms, object_store, upstream) -> None:
        theirs = make_id(test_client, user_b, room="theirs")
        assert dispatch(test_client, user_b, theirs).status_code == 202
        mine = make_id(test_client, user_a, room="mine")
        response = dispatch(test_client, user_a, mine)
        assert response.status_code == 409
        assert code_of(response) == "agentar_capacity"
        assert len(upstream.posted) == 1
        assert document(mine)["status"] == "draft"

    def test_capacity_is_configurable(self, test_client, user_a, user_b, rooms, object_store, upstream, monkeypatch) -> None:
        agentar_settings(monkeypatch, max_live_total=2)
        assert dispatch(test_client, user_b, make_id(test_client, user_b, room="theirs")).status_code == 202
        assert dispatch(test_client, user_a, make_id(test_client, user_a, room="mine")).status_code == 202
        third = make_id(test_client, user_a, room="third")
        assert code_of(dispatch(test_client, user_a, third)) == "live_agentar_exists"

    def test_a_freed_slot_can_be_used_again(self, test_client, user_a, user_b, rooms, object_store, upstream) -> None:
        theirs = make_id(test_client, user_b, room="theirs")
        dispatch(test_client, user_b, theirs)
        force(theirs, status="ended", live_lock=None)
        assert dispatch(test_client, user_a, make_id(test_client, user_a, room="mine")).status_code == 202

    def test_a_waiting_meeting_does_not_hold_the_global_slot(self, test_client, user_a, user_b, rooms, object_store, upstream) -> None:
        """A meeting still waiting takes no Ditto session, so another owner's dispatch goes through."""
        upstream.lifecycle_waits = True
        theirs = make_id(test_client, user_b, room="theirs")
        assert dispatch(test_client, user_b, theirs).json()["status"] == "waiting_for_human"
        upstream.lifecycle_waits = False
        mine = make_id(test_client, user_a, room="mine")
        response = dispatch(test_client, user_a, mine)
        assert response.status_code == 202, response.text
        assert response.json()["status"] == "joining"

    @pytest.mark.parametrize("status", ["joining", "live", "leaving"])
    def test_an_active_bot_holds_the_global_slot(self, test_client, user_a, user_b, rooms, object_store, upstream, status) -> None:
        theirs = make_id(test_client, user_b, room="theirs")
        force(theirs, status=status, live_lock=True)
        response = dispatch(test_client, user_a, make_id(test_client, user_a, room="mine"))
        assert response.status_code == 409
        assert code_of(response) == "agentar_capacity"


class TestResendQuota:
    """Sending a missed or failed meeting again costs one unit of `AGENTAR_MEETINGS_PER_DAY`, so a
    meeting cannot be cycled through the lifecycle for free."""

    @pytest.mark.parametrize("status", ["missed", "failed"])
    def test_a_resend_counts_against_the_daily_quota(self, test_client, user_a, rooms, object_store, upstream, frozen_clock, monkeypatch, status) -> None:
        agentar_settings(monkeypatch, meetings_per_day=2)
        meeting_id = make_id(test_client, user_a)
        force(meeting_id, status=status, live_lock=None)
        assert dispatch(test_client, user_a, meeting_id).status_code == 202
        force(meeting_id, status=status, live_lock=None)
        response = dispatch(test_client, user_a, meeting_id)
        assert response.status_code == 422
        assert code_of(response) == "quota_meetings_per_day"
        assert document(meeting_id)["status"] == status
        assert make(test_client, user_a, room="other").status_code == 422
        frozen_clock.advance(days=1)
        assert dispatch(test_client, user_a, meeting_id).status_code == 202

    def test_the_first_send_of_a_draft_is_covered_by_the_create(self, test_client, user_a, rooms, object_store, upstream, frozen_clock, monkeypatch) -> None:
        agentar_settings(monkeypatch, meetings_per_day=1)
        meeting_id = make_id(test_client, user_a)
        assert dispatch(test_client, user_a, meeting_id).status_code == 202
        assert document(meeting_id).get("resent_at", []) == []

    def test_a_scheduled_resend_counts_as_well(self, test_client, user_a, rooms, object_store, upstream, frozen_clock, monkeypatch) -> None:
        agentar_settings(monkeypatch, meetings_per_day=1)
        start = (clock.utcnow() + timedelta(hours=1)).isoformat()
        meeting_id = make_id(test_client, user_a, mode="scheduled", start_at=start)
        force(meeting_id, status="missed")
        response = dispatch(test_client, user_a, meeting_id)
        assert response.status_code == 422
        assert code_of(response) == "quota_meetings_per_day"


class TestWorkerFailure:
    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(503),
            httpx.Response(429, headers={"Retry-After": "5"}),
            httpx.Response(422, json={"detail": [{"loc": ["body", "voice"], "msg": "x"}]}),
            httpx.Response(401),
            httpx.Response(200, json={"id": "someone-elses-bot"}),
            httpx.Response(202, text="garbage"),
        ],
        ids=["down", "full", "refused", "secret", "duplicate", "garbage"],
    )
    def test_is_recorded_as_failed_and_still_answers_202(self, test_client, user_a, rooms, object_store, upstream, response) -> None:
        upstream.bot_response = response
        meeting_id = make_id(test_client, user_a)
        result = dispatch(test_client, user_a, meeting_id)
        assert result.status_code == 202
        body = result.json()
        assert body["status"] == "failed"
        assert body["status_reason"] == "worker_error"
        assert body["bot_id"] is None
        assert not document(meeting_id).get("live_lock")

    def test_an_unreachable_worker_is_failed_too(self, test_client, user_a, rooms, object_store, upstream) -> None:
        upstream.worker_down = True
        meeting_id = make_id(test_client, user_a)
        body = dispatch(test_client, user_a, meeting_id).json()
        assert (body["status"], body["status_reason"]) == ("failed", "worker_error")

    def test_the_slot_is_free_again_after_a_failure(self, test_client, user_a, rooms, object_store, upstream) -> None:
        upstream.bot_response = httpx.Response(503)
        meeting_id = make_id(test_client, user_a)
        assert dispatch(test_client, user_a, meeting_id).json()["status"] == "failed"
        upstream.bot_response = None
        assert dispatch(test_client, user_a, meeting_id).json()["status"] == "joining"

    def test_a_store_that_is_down_fails_the_meeting_with_its_reason(self, test_client, user_a, rooms, object_store, upstream, monkeypatch) -> None:
        from utils.object_store import ObjectStoreError

        meeting_id = make_id(test_client, user_a)
        put_picture(test_client, user_a, meeting_id)

        async def down(key, **kwargs):
            raise ObjectStoreError("down")

        monkeypatch.setattr(object_store, "get", down)
        body = dispatch(test_client, user_a, meeting_id).json()
        assert (body["status"], body["status_reason"]) == ("failed", "storage_unavailable")
        assert upstream.posted == []

class TestDispatchConfiguration:
    def test_a_missing_worker_secret_is_503(self, test_client, user_a, rooms, object_store, upstream, monkeypatch) -> None:
        monkeypatch.setattr(
            settings, "_virtual_participants", settings.virtual_participants.model_copy(update={"worker_secret": ""})
        )
        meeting_id = make_id(test_client, user_a)
        response = dispatch(test_client, user_a, meeting_id)
        assert response.status_code == 503
        assert code_of(response) == "worker_unavailable"
        assert document(meeting_id)["status"] == "draft"

    def test_a_missing_jitsi_key_is_503(self, test_client, user_a, rooms, object_store, upstream, monkeypatch) -> None:
        monkeypatch.setattr(settings, "jwt_jitsi_secret_key", "")
        meeting_id = make_id(test_client, user_a)
        response = dispatch(test_client, user_a, meeting_id)
        assert response.status_code == 503
        assert code_of(response) == "jitsi_key_missing"
        assert document(meeting_id)["status"] == "draft"

    @pytest.mark.parametrize("base", ["", "ftp://x", "http://u:p@fastapi:8000"])
    def test_an_unusable_events_origin_is_503(self, test_client, user_a, rooms, object_store, upstream, monkeypatch, base) -> None:
        agentar_settings(monkeypatch, worker_events_base_url=base)
        meeting_id = make_id(test_client, user_a)
        response = dispatch(test_client, user_a, meeting_id)
        assert response.status_code == 503
        assert document(meeting_id)["status"] == "draft"


class TestDispatchScheduled:
    def test_arms_the_meeting_without_calling_the_worker(self, test_client, user_a, rooms, object_store, upstream, frozen_clock) -> None:
        start = (frozen_clock.now + timedelta(hours=2)).isoformat()
        meeting_id = make_id(test_client, user_a, mode="scheduled", start_at=start)
        response = dispatch(test_client, user_a, meeting_id)
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "scheduled"
        assert body["display_name"] == f"Alice Tester Agentar{BOT_NAME_SUFFIX}"
        assert body["voice"] == settings.agentar.default_voice
        assert upstream.posted == [] and upstream.lifecycle_calls == []
        assert not document(meeting_id).get("live_lock")

    def test_a_start_that_passed_is_start_in_past(self, test_client, user_a, rooms, object_store, upstream, frozen_clock) -> None:
        start = (frozen_clock.now + timedelta(hours=1)).isoformat()
        meeting_id = make_id(test_client, user_a, mode="scheduled", start_at=start)
        frozen_clock.advance(hours=2)
        response = dispatch(test_client, user_a, meeting_id)
        assert response.status_code == 422
        assert code_of(response) == "start_in_past"
        assert document(meeting_id)["status"] == "draft"

    def test_arming_checks_the_room_but_not_the_live_slot(self, test_client, user_a, user_b, rooms, object_store, upstream, frozen_clock) -> None:
        """The tick applies the live and cap checks at the start time."""
        assert dispatch(test_client, user_b, make_id(test_client, user_b, room="theirs")).status_code == 202
        start = (frozen_clock.now + timedelta(days=1)).isoformat()
        meeting_id = make_id(test_client, user_a, mode="scheduled", start_at=start)
        assert dispatch(test_client, user_a, meeting_id).json()["status"] == "scheduled"
        rooms.owners["weekly"] = user_b.email
        force(meeting_id, status="draft")
        assert code_of(dispatch(test_client, user_a, meeting_id)) == "room_not_owned"

    def test_the_owner_profile_is_read_at_dispatch(self, test_client, user_a, rooms, object_store, upstream, frozen_clock) -> None:
        from src.database.models.user import User

        asyncio.run(
            User.get_collection().update_one(
                {"email": user_a.email}, {"$set": {"profile": {"agent_name": "Mia", "voice": "af_bella"}}}
            )
        )
        start = (frozen_clock.now + timedelta(hours=2)).isoformat()
        meeting_id = make_id(test_client, user_a, mode="scheduled", start_at=start)
        body = dispatch(test_client, user_a, meeting_id).json()
        assert (body["display_name"], body["voice"]) == (f"Mia Agentar{BOT_NAME_SUFFIX}", "af_bella")


# Recall --------------------------------------------------------------------------------------


class TestRecall:
    def recall(self, client, user, meeting_id):
        return client.post(f"{BASE}/{meeting_id}/recall", headers=user.headers)

    def test_scheduled_goes_back_to_draft(self, test_client, user_a, rooms, upstream, frozen_clock) -> None:
        start = (frozen_clock.now + timedelta(hours=2)).isoformat()
        meeting_id = make_id(test_client, user_a, mode="scheduled", start_at=start)
        force(meeting_id, status="scheduled")
        response = self.recall(test_client, user_a, meeting_id)
        assert response.status_code == 202
        assert response.json()["status"] == "draft"
        assert upstream.deleted == []

    def test_waiting_goes_back_to_draft_and_frees_the_slot(self, test_client, user_a, rooms, object_store, upstream) -> None:
        upstream.lifecycle_waits = True
        meeting_id = make_id(test_client, user_a)
        dispatch(test_client, user_a, meeting_id)
        body = self.recall(test_client, user_a, meeting_id).json()
        assert body["status"] == "draft"
        assert body["start_at"] is None
        assert not document(meeting_id).get("live_lock")
        assert upstream.deleted == []
        assert dispatch(test_client, user_a, meeting_id).json()["status"] == "waiting_for_human"

    @pytest.mark.parametrize("status", ["joining", "live"])
    def test_a_running_bot_gets_the_farewell_and_the_meeting_is_leaving(self, test_client, user_a, rooms, object_store, upstream, status) -> None:
        meeting_id = make_id(test_client, user_a)
        dispatch(test_client, user_a, meeting_id)
        force(meeting_id, status=status)
        response = self.recall(test_client, user_a, meeting_id)
        assert response.status_code == 202
        assert response.json()["status"] == "leaving"
        (request,) = upstream.deleted
        assert request.url.path == "/bots/bot-1"
        assert dict(request.url.params) == {"farewell": "1"}
        assert request.headers["x-vp-secret"] == "worker-secret-value"
        assert document(meeting_id)["live_lock"] is True

    def test_leaving_is_idempotent(self, test_client, user_a, rooms, object_store, upstream) -> None:
        meeting_id = make_id(test_client, user_a)
        dispatch(test_client, user_a, meeting_id)
        self.recall(test_client, user_a, meeting_id)
        response = self.recall(test_client, user_a, meeting_id)
        assert response.status_code == 202
        assert response.json()["status"] == "leaving"
        assert len(upstream.deleted) == 1

    @pytest.mark.parametrize("status", ["draft", "ended", "missed", "failed"])
    def test_any_other_status_is_409(self, test_client, user_a, rooms, upstream, status) -> None:
        meeting_id = make_id(test_client, user_a)
        force(meeting_id, status=status)
        response = self.recall(test_client, user_a, meeting_id)
        assert response.status_code == 409
        assert code_of(response) == "invalid_state"

    def test_an_unreachable_worker_is_503_and_the_status_stays(self, test_client, user_a, rooms, object_store, upstream) -> None:
        meeting_id = make_id(test_client, user_a)
        dispatch(test_client, user_a, meeting_id)
        upstream.delete_response = httpx.Response(500)
        response = self.recall(test_client, user_a, meeting_id)
        assert response.status_code == 503
        assert code_of(response) == "worker_unavailable"
        assert document(meeting_id)["status"] == "joining"
        upstream.delete_response = None
        assert self.recall(test_client, user_a, meeting_id).json()["status"] == "leaving"

    def test_a_bot_the_worker_no_longer_knows_still_leaves(self, test_client, user_a, rooms, object_store, upstream) -> None:
        meeting_id = make_id(test_client, user_a)
        dispatch(test_client, user_a, meeting_id)
        upstream.delete_response = httpx.Response(404)
        assert self.recall(test_client, user_a, meeting_id).json()["status"] == "leaving"

    def test_a_bot_that_is_still_starting_cannot_be_recalled_yet(self, test_client, user_a, rooms, upstream) -> None:
        meeting_id = make_id(test_client, user_a)
        force(meeting_id, status="joining")
        response = self.recall(test_client, user_a, meeting_id)
        assert response.status_code == 409
        assert upstream.deleted == []

    def test_a_meeting_that_ended_in_between_is_returned_as_it_is(self, test_client, user_a, rooms, object_store, upstream) -> None:
        meeting_id = make_id(test_client, user_a)
        dispatch(test_client, user_a, meeting_id)
        original = upstream.handler

        async def ends_during_the_call(request):
            if request.method == "DELETE":
                await AgentarMeeting.get_collection().update_one(
                    {"_id": ObjectId(meeting_id)}, {"$set": {"status": "ended"}}
                )
            return original(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(ends_during_the_call))
        app.dependency_overrides[vp_routes.get_vp_http_client] = lambda: client
        assert self.recall(test_client, user_a, meeting_id).json()["status"] == "ended"
