"""
Tests for the virtual participant control route.

Prosody and vp-worker are served by an httpx.MockTransport injected through the
get_vp_http_client dependency override; no network access and no global patching.
"""
import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from api.routes import virtual_participants as vp_routes
from config import VirtualParticipantSettings, settings
from main import app
from utils.auth import verify_jitsi_token

WORKER_HOST = "vp-worker.test"
PROSODY_HOST = "xmpp.test"
WORKER_SECRET = "test-vp-secret-0123456789"
XMPP_DOMAIN = "meet.jitsi"
ROOM = "team-sync"
BASE = "/api/virtual-participants"


def bot_view(
    bot_id: str = "bot-1",
    *,
    room: str = ROOM,
    display_name: str = "Agentar (bot)",
    state: str = "joining",
) -> dict[str, Any]:
    return {
        "id": bot_id,
        "room": room,
        "display_name": display_name,
        "state": state,
        "replica_id": "replica-a",
        "boot_id": "boot-a",
        "created_at": 1_790_000_000.0,
        "joined_at": None,
        "left_at": None,
        "error": None,
        "humans": None,
        "speech": {"state": "idle", "queue_position": None, "last": None},
        "media": None,
    }


def occupant(display_name: str, *, nick: str = "a1b2c3d4") -> dict[str, str]:
    """One occupant exactly as ``mod_muc_size`` reports it.

    The ``jid`` field is ``tostring(occupant.nick)``, that is the room JID with the MUC
    nickname as its resource. It is never the occupant's real JID, so it carries no
    information about who or what the occupant is. Verified against the running Prosody:
    ``{"jid": "room@muc.meet.jitsi/4d5f34aa", "display_name": "Probe (bot)", "email": ""}``.
    Do not add a parameter that lets a test supply a real JID here; that is what hid the
    transcriber from the human check.
    """
    return {
        "jid": f"{ROOM}@muc.{XMPP_DOMAIN}/{nick}",
        "email": "",
        "display_name": display_name,
    }


class FakeUpstream:
    """Scripted Prosody and vp-worker. Records every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        # None makes Prosody answer 404, as mod_muc_size does for a room that does not exist.
        self.occupants: list[dict[str, str]] | None = [occupant("Alice")]
        self.bots: list[dict[str, Any]] = []
        self.responses: dict[tuple[str, str], httpx.Response] = {}
        self.unreachable: set[tuple[str, str]] = set()
        self.prosody_unreachable = False
        self.prosody_response: httpx.Response | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == PROSODY_HOST:
            return self._prosody(request)
        assert request.url.host == WORKER_HOST
        key = (request.method, request.url.path)
        if key in self.unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if key in self.responses:
            return self.responses[key]
        if key == ("GET", "/bots"):
            room = request.url.params.get("room")
            return httpx.Response(
                200, json=[bot for bot in self.bots if room is None or bot["room"] == room]
            )
        if key == ("POST", "/bots"):
            return httpx.Response(202, json=bot_view())
        return httpx.Response(404, json={"detail": "bot not found"})

    def _prosody(self, request: httpx.Request) -> httpx.Response:
        if self.prosody_unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if self.prosody_response is not None:
            return self.prosody_response
        if request.url.path != "/room" or self.occupants is None:
            return httpx.Response(404)
        return httpx.Response(200, json=self.occupants)

    def calls(self, method: str, path: str, host: str = WORKER_HOST) -> list[httpx.Request]:
        return [
            r
            for r in self.requests
            if r.method == method and r.url.path == path and r.url.host == host
        ]

    def worker_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == WORKER_HOST]


def _vp_settings(**overrides: Any) -> VirtualParticipantSettings:
    values: dict[str, Any] = {
        "worker_url": f"http://{WORKER_HOST}:8000",
        "worker_secret": WORKER_SECRET,
        "require_human": True,
        "max_bots_per_room": 2,
        "prosody_http_url": f"http://{PROSODY_HOST}:5280",
        "xmpp_domain": XMPP_DOMAIN,
        "request_timeout_s": 5.0,
    }
    values.update(overrides)
    return VirtualParticipantSettings(**values)


@pytest.fixture
def vp_settings(monkeypatch):
    """Apply VirtualParticipantSettings overrides for one test."""

    def _apply(**overrides: Any) -> VirtualParticipantSettings:
        value = _vp_settings(**overrides)
        monkeypatch.setattr(settings, "_virtual_participants", value)
        return value

    _apply()
    return _apply


@pytest.fixture
def upstream(vp_settings):
    fake = FakeUpstream()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    app.dependency_overrides[vp_routes.get_vp_http_client] = lambda: http_client
    yield fake
    app.dependency_overrides.pop(vp_routes.get_vp_http_client, None)
    asyncio.run(http_client.aclose())


@pytest.mark.integration
class TestVirtualParticipantAuth:
    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("POST", BASE, {"room_name": ROOM}),
            ("GET", BASE, None),
            ("GET", f"{BASE}/bot-1", None),
            ("DELETE", f"{BASE}/bot-1", None),
            ("POST", f"{BASE}/bot-1/say", {"text": "Hello"}),
        ],
    )
    def test_requires_bearer_token(
        self, test_client: TestClient, upstream: FakeUpstream, method, path, body
    ):
        response = test_client.request(method, path, json=body)

        assert response.status_code == 401
        assert upstream.requests == []

    def test_rejects_invalid_bearer_token(self, test_client: TestClient, upstream: FakeUpstream):
        response = test_client.post(
            BASE,
            json={"room_name": ROOM},
            headers={"Authorization": "Bearer not-a-token"},
        )

        assert response.status_code == 401
        assert upstream.requests == []


@pytest.mark.integration
class TestCreateVirtualParticipant:
    @pytest.mark.parametrize(
        "payload",
        [
            {"room_name": "bad room"},
            {"room_name": "-leading-dash"},
            {"room_name": "a" * 64},
            {"room_name": "../cluster"},
            {"room_name": ""},
            {},
            {"room_name": ROOM, "display_name": "Bad/Name"},
            {"room_name": ROOM, "display_name": "x" * 49},
            {"room_name": ROOM, "script": "x" * 2001},
            {"room_name": ROOM, "jwt": "client-supplied"},
        ],
    )
    def test_rejects_invalid_body(self, client: TestClient, upstream: FakeUpstream, payload):
        response = client.post(BASE, json=payload)

        assert response.status_code == 422
        assert upstream.requests == []

    def test_empty_room_is_refused(self, client: TestClient, upstream: FakeUpstream):
        upstream.occupants = None

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == 409
        assert response.json()["detail"] == "no human participant in the room"
        assert len(upstream.calls("GET", "/room", host=PROSODY_HOST)) == 1
        assert upstream.worker_requests() == []

    def test_room_with_only_bots_is_refused(self, client: TestClient, upstream: FakeUpstream):
        upstream.occupants = [
            occupant("Agentar (bot)"),
            occupant("Helper (bot)", nick="e5f6a7b8"),
        ]

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == 409
        assert response.json()["detail"] == "no human participant in the room"
        assert upstream.worker_requests() == []

    def test_room_with_only_a_transcriber_is_refused(
        self, client: TestClient, upstream: FakeUpstream
    ):
        """A meeting whose last human left while transcription runs has no human in it.

        Jigasi joins as "Transcriber" (TranscriptionGatewaySession.DISPLAY_NAME) and its
        occupant record is indistinguishable from a human's except by that name.
        """
        upstream.occupants = [occupant("Transcriber", nick="jigasi01")]

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == 409
        assert response.json()["detail"] == "no human participant in the room"
        assert upstream.worker_requests() == []

    def test_service_name_match_ignores_case_and_padding(
        self, client: TestClient, upstream: FakeUpstream
    ):
        upstream.occupants = [occupant("  transcriber  "), occupant("RECORDER", nick="jibri001")]

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == 409
        assert upstream.worker_requests() == []

    def test_refused_at_max_bots_per_room(self, client: TestClient, upstream: FakeUpstream):
        upstream.bots = [
            bot_view("bot-1", state="joined"),
            bot_view("bot-2", display_name="Helper (bot)", state="speaking"),
            bot_view("bot-3", display_name="Old (bot)", state="left"),
        ]

        response = client.post(BASE, json={"room_name": ROOM, "display_name": "Third"})

        assert response.status_code == 409
        assert "limit is 2" in response.json()["detail"]
        assert upstream.calls("POST", "/bots") == []

    def test_terminal_bots_do_not_count_towards_limit(
        self, client: TestClient, upstream: FakeUpstream
    ):
        upstream.bots = [
            bot_view("bot-1", state="joining"),
            bot_view("bot-2", display_name="A (bot)", state="left"),
            bot_view("bot-3", display_name="B (bot)", state="error"),
            bot_view("bot-4", display_name="C (bot)", state="lost"),
            bot_view("bot-5", room="other-room", display_name="D (bot)", state="joined"),
        ]

        response = client.post(BASE, json={"room_name": ROOM, "display_name": "Second"})

        assert response.status_code == 202

    def test_relays_202_with_bot_token_for_room(
        self, client: TestClient, upstream: FakeUpstream
    ):
        upstream.occupants = [occupant("Agentar (bot)"), occupant("Alice")]

        response = client.post(
            BASE,
            json={"room_name": "Team-Sync", "script": "Welcome everyone."},
        )

        assert response.status_code == 202
        assert response.json() == bot_view()

        (lookup,) = upstream.calls("GET", "/room", host=PROSODY_HOST)
        assert lookup.url.params["room"] == ROOM
        assert lookup.url.params["domain"] == XMPP_DOMAIN
        assert lookup.headers["host"] == XMPP_DOMAIN

        (listing,) = upstream.calls("GET", "/bots")
        assert listing.url.params["room"] == ROOM
        assert listing.headers["x-vp-secret"] == WORKER_SECRET

        (create,) = upstream.calls("POST", "/bots")
        assert create.headers["x-vp-secret"] == WORKER_SECRET
        sent = json.loads(create.content)
        assert sent["room"] == ROOM
        assert sent["display_name"] == "Agentar"
        assert sent["script"] == "Welcome everyone."
        claims = verify_jitsi_token(sent["jwt"])
        assert claims["room"] == ROOM
        assert claims["sub"] == "vp-bot"
        assert claims["moderator"] is False

    def test_duplicate_active_bot_is_forwarded_at_limit(
        self, client: TestClient, upstream: FakeUpstream
    ):
        existing = bot_view("bot-1", state="joined")
        upstream.bots = [existing, bot_view("bot-2", display_name="Helper (bot)", state="joined")]
        upstream.responses[("POST", "/bots")] = httpx.Response(200, json=existing)

        response = client.post(BASE, json={"room_name": ROOM, "display_name": "Agentar"})

        assert response.status_code == 200
        assert response.json()["id"] == "bot-1"

    def test_relays_429_with_retry_after(self, client: TestClient, upstream: FakeUpstream):
        upstream.responses[("POST", "/bots")] = httpx.Response(
            429, json={"detail": "cluster at capacity"}, headers={"Retry-After": "7"}
        )

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == 429
        assert response.headers["retry-after"] == "7"
        assert response.json() == {"detail": "cluster at capacity"}

    @pytest.mark.parametrize("unreachable", [("GET", "/bots"), ("POST", "/bots")])
    def test_worker_unreachable_is_503(
        self, client: TestClient, upstream: FakeUpstream, unreachable
    ):
        upstream.unreachable.add(unreachable)

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == 503
        assert response.json()["detail"] == "virtual participant worker unreachable"

    def test_prosody_unreachable_is_503(self, client: TestClient, upstream: FakeUpstream):
        upstream.prosody_unreachable = True

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == 503
        assert upstream.worker_requests() == []

    @pytest.mark.parametrize(
        "prosody_response",
        [
            httpx.Response(403),
            httpx.Response(200, json={"participants": 1}),
            httpx.Response(200, content=b"not json"),
        ],
    )
    def test_unusable_prosody_answer_is_503(
        self, client: TestClient, upstream: FakeUpstream, prosody_response
    ):
        upstream.prosody_response = prosody_response

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == 503
        assert upstream.worker_requests() == []

    def test_occupant_without_name_counts_as_human(
        self, client: TestClient, upstream: FakeUpstream
    ):
        """A participant who never set a display name is a person, not a service.

        Nothing in the mod_muc_size payload separates them from a service participant, and
        refusing the request would be the worse error.
        """
        upstream.occupants = ["garbage", occupant("Agentar (bot)"), occupant("")]

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == 202

    @pytest.mark.parametrize(
        "listing,expected",
        [
            (httpx.Response(401, json={"detail": "bad secret"}), 502),
            (httpx.Response(503, json={"detail": "shutting down"}), 503),
            (httpx.Response(200, json={"unexpected": True}), 502),
        ],
    )
    def test_bot_count_failure_blocks_create(
        self, client: TestClient, upstream: FakeUpstream, listing, expected
    ):
        upstream.responses[("GET", "/bots")] = listing

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == expected
        assert upstream.calls("POST", "/bots") == []

    def test_human_check_can_be_disabled(
        self, client: TestClient, upstream: FakeUpstream, vp_settings
    ):
        vp_settings(require_human=False)
        upstream.occupants = None

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == 202
        assert upstream.calls("GET", "/room", host=PROSODY_HOST) == []

    @pytest.mark.parametrize("worker_status,expected", [(401, 502), (500, 502), (503, 503)])
    def test_worker_faults_are_not_relayed(
        self, client: TestClient, upstream: FakeUpstream, worker_status, expected
    ):
        upstream.responses[("POST", "/bots")] = httpx.Response(
            worker_status, json={"detail": "worker says no"}
        )

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == expected

    def test_missing_worker_secret_is_503(
        self, client: TestClient, upstream: FakeUpstream, vp_settings
    ):
        vp_settings(worker_secret="")

        response = client.post(BASE, json={"room_name": ROOM})

        assert response.status_code == 503
        assert upstream.requests == []


@pytest.mark.integration
class TestReadVirtualParticipants:
    def test_list_filters_by_normalized_room(self, client: TestClient, upstream: FakeUpstream):
        upstream.bots = [bot_view("bot-1"), bot_view("bot-2", room="other-room")]

        response = client.get(BASE, params={"room_name": "Team-Sync"})

        assert response.status_code == 200
        assert [bot["id"] for bot in response.json()] == ["bot-1"]
        (listing,) = upstream.calls("GET", "/bots")
        assert listing.url.params["room"] == ROOM
        assert listing.headers["x-vp-secret"] == WORKER_SECRET

    def test_list_without_room_returns_all(self, client: TestClient, upstream: FakeUpstream):
        upstream.bots = [bot_view("bot-1"), bot_view("bot-2", room="other-room")]

        response = client.get(BASE)

        assert response.status_code == 200
        assert len(response.json()) == 2
        (listing,) = upstream.calls("GET", "/bots")
        assert "room" not in listing.url.params

    def test_list_accepts_wrapped_payload(self, client: TestClient, upstream: FakeUpstream):
        upstream.responses[("GET", "/bots")] = httpx.Response(
            200, json={"bots": [bot_view("bot-1")]}
        )

        response = client.get(BASE, params={"room_name": ROOM})

        assert response.status_code == 200
        assert [bot["id"] for bot in response.json()] == ["bot-1"]

    def test_list_does_not_relay_worker_auth_failure(
        self, client: TestClient, upstream: FakeUpstream
    ):
        upstream.responses[("GET", "/bots")] = httpx.Response(401, json={"detail": "bad secret"})

        response = client.get(BASE)

        assert response.status_code == 502

    def test_list_rejects_invalid_room(self, client: TestClient, upstream: FakeUpstream):
        response = client.get(BASE, params={"room_name": "bad room"})

        assert response.status_code == 422
        assert upstream.requests == []

    def test_get_relays_view(self, client: TestClient, upstream: FakeUpstream):
        view = bot_view("bot-1", state="joined")
        upstream.responses[("GET", "/bots/bot-1")] = httpx.Response(200, json=view)

        response = client.get(f"{BASE}/bot-1")

        assert response.status_code == 200
        assert response.json() == view
        (call,) = upstream.calls("GET", "/bots/bot-1")
        assert call.headers["x-vp-secret"] == WORKER_SECRET

    def test_get_relays_404(self, client: TestClient, upstream: FakeUpstream):
        response = client.get(f"{BASE}/missing-bot")

        assert response.status_code == 404
        assert response.json() == {"detail": "bot not found"}

    def test_get_relays_404_without_json_body(self, client: TestClient, upstream: FakeUpstream):
        upstream.responses[("GET", "/bots/bot-1")] = httpx.Response(404, content=b"gone")

        response = client.get(f"{BASE}/bot-1")

        assert response.status_code == 404
        assert response.json() == {"detail": "Not Found"}

    @pytest.mark.parametrize("bot_id", ["bad.id", "x" * 65, "%2E%2E"])
    def test_rejects_invalid_bot_id(self, client: TestClient, upstream: FakeUpstream, bot_id):
        response = client.get(f"{BASE}/{bot_id}")

        assert response.status_code in (404, 422)
        assert upstream.requests == []


@pytest.mark.integration
class TestDeleteVirtualParticipant:
    def test_delete_relays_final_view(self, client: TestClient, upstream: FakeUpstream):
        final = bot_view("bot-1", state="left")
        upstream.responses[("DELETE", "/bots/bot-1")] = httpx.Response(200, json=final)

        response = client.delete(f"{BASE}/bot-1")

        assert response.status_code == 200
        assert response.json() == final
        (call,) = upstream.calls("DELETE", "/bots/bot-1")
        assert call.headers["x-vp-secret"] == WORKER_SECRET

    def test_delete_relays_404(self, client: TestClient, upstream: FakeUpstream):
        response = client.delete(f"{BASE}/missing-bot")

        assert response.status_code == 404


@pytest.mark.integration
class TestSayVirtualParticipant:
    def test_say_relays_202(self, client: TestClient, upstream: FakeUpstream):
        view = bot_view("bot-1", state="joined")
        view["speech"] = {"state": "queued", "queue_position": 0, "last": None}
        upstream.responses[("POST", "/bots/bot-1/say")] = httpx.Response(202, json=view)

        response = client.post(f"{BASE}/bot-1/say", json={"text": "Hello everyone."})

        assert response.status_code == 202
        assert response.json()["speech"]["state"] == "queued"
        (call,) = upstream.calls("POST", "/bots/bot-1/say")
        assert call.headers["x-vp-secret"] == WORKER_SECRET
        assert json.loads(call.content) == {"text": "Hello everyone."}

    def test_say_relays_409_when_busy(self, client: TestClient, upstream: FakeUpstream):
        upstream.responses[("POST", "/bots/bot-1/say")] = httpx.Response(
            409, json={"detail": "bot is busy"}
        )

        response = client.post(f"{BASE}/bot-1/say", json={"text": "Hello"})

        assert response.status_code == 409
        assert response.json() == {"detail": "bot is busy"}

    def test_say_relays_404(self, client: TestClient, upstream: FakeUpstream):
        response = client.post(f"{BASE}/missing-bot/say", json={"text": "Hello"})

        assert response.status_code == 404

    @pytest.mark.parametrize("payload", [{"text": ""}, {"text": "x" * 2001}, {}])
    def test_say_rejects_invalid_text(self, client: TestClient, upstream: FakeUpstream, payload):
        response = client.post(f"{BASE}/bot-1/say", json=payload)

        assert response.status_code == 422
        assert upstream.requests == []


@pytest.mark.unit
class TestVirtualParticipantHttpClient:
    def test_client_is_lazy_shared_and_closed_on_shutdown(self, monkeypatch, vp_settings):
        monkeypatch.setattr(vp_routes, "_http_client", None)

        async def scenario() -> None:
            first = await vp_routes.get_vp_http_client()
            assert await vp_routes.get_vp_http_client() is first
            await vp_routes.close_http_client()
            assert first.is_closed
            assert vp_routes._http_client is None
            await vp_routes.close_http_client()

        asyncio.run(scenario())
