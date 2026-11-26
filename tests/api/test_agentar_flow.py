"""One owner from sign up to the report, through the real modules wired together.

Fakes stop at the process edge: vp-worker and Prosody are an httpx.MockTransport behind the
process wide client, SpeechLMM is a scripted engine for chat and a scripted model for the report,
the object store is the in-memory one, MongoDB is mongomock. Everything between the routes, the
lifecycle, the worker payload, the prompt, the events and the report job is the real code.

The TestClient closes its event loop after each request, so a task scheduled inside a request
does not outlive it. The report job is therefore taken from what the lifecycle queued and run in
the test thread, with the same function the queue runs.
"""
import asyncio
import base64
import io
import json
import re
import time
import uuid
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.routes import virtual_participants as vp_route
from config import VirtualParticipantSettings, settings
from core import SessionManager
from models import TaskType
from src.database.models.user import User
from utils import agentar_report, agentar_worker, clock

KEY = "vp-flow-key-0123456789abcdef0123"
WORKER = {"Authorization": f"Bearer {KEY}"}
ROOM = "atlas-review"
JITSI_ID = "jm-flow-1"
PASSWORD = "flow-password-123"
KB_TEXT = "Atlas goes live on 1 December 2026 in Trento. The Bonn rollout follows in January."
QUESTION = "Who signs the data agreement?"
ANSWER_LINE = "Lyon signs the data agreement on 20 November."
ACTION_LINE = "Sara will send the training invitations by Friday."


# Fakes ---------------------------------------------------------------------------------------


class Upstream:
    """vp-worker and Prosody: one human in the room unless `room_open` is off, every bot accepted."""

    def __init__(self) -> None:
        self.bots: list[dict] = []
        self.deleted: list[str] = []
        self.delete_params: list[str] = []
        self.room_open = True
        self.prosody_calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "prosody":
            self.prosody_calls += 1
            if not self.room_open:
                # Guests waiting for a host under JWT auth are not in the MUC: muc_size has no room.
                return httpx.Response(404)
            return httpx.Response(200, json=[{"display_name": "Laura"}])
        if request.url.host != "worker":
            return httpx.Response(404)
        assert request.headers[vp_route.WORKER_SECRET_HEADER] == "worker-secret-value"
        if request.method == "POST" and request.url.path == "/bots":
            body = json.loads(request.content)
            self.bots.append(body)
            return httpx.Response(
                202,
                json={"id": f"{len(self.bots):032x}", "room": body["room"],  # uuid4 hex, as the worker
                      "display_name": f"{body['display_name']} (AI)", "state": "joining"},
            )
        if request.method == "DELETE" and request.url.path.startswith("/bots/"):
            self.deleted.append(request.url.path.rsplit("/", 1)[1])
            self.delete_params.append(str(request.url.params))
            return httpx.Response(200, json={"state": "leaving"})
        if request.method == "GET" and request.url.path.startswith("/bots/"):
            return httpx.Response(200, json={"id": request.url.path.rsplit("/", 1)[1], "state": "joined"})
        return httpx.Response(404)


class ChatEngine:
    """SpeechLMM for /api/chat: answers from the knowledge base chunk when the prompt holds it,
    else gives the sentinel the prompt asks for."""

    max_context_chars = 300_000
    max_history_chars = 600_000

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def _generate_text(self, **kwargs) -> str:
        prompt = kwargs["prompt"]
        self.prompts.append(prompt)
        question = re.search(r"<question>(.*?)</question>", prompt, flags=re.S).group(1)
        if "budget" in question.lower():
            sentinel = re.search(r"I do not have that, I will check with [^\n\"]+ and report back", prompt)
            return sentinel.group(0)
        if "1 December 2026" in prompt:
            return "Atlas goes live on 1 December 2026 in Trento."
        return "I am not sure."


class ReportModel:
    """SpeechLMM for the report job. It cites lines by the number the prompt gives them."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    @staticmethod
    def _line_of(prompt: str, quote: str) -> int:
        for match in re.finditer(r"^\[(\d+)\] [^\n]*$", prompt, flags=re.M):
            if quote in match.group(0):
                return int(match.group(1))
        raise AssertionError(f"line with {quote!r} is not in the prompt")

    async def __call__(self, prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        if prompt.startswith("You write short minutes") or prompt.startswith("Combine these partial"):
            return json.dumps({"summary": "Atlas go live confirmed; Lyon signs the data agreement."})
        if prompt.startswith("List the action items"):
            quote = "send the training invitations"
            return json.dumps({"items": [{"text": "Send the training invitations", "owner": "Sara",
                                          "due": "Friday", "line": self._line_of(prompt, quote), "quote": quote}]})
        if f"this question: {QUESTION}" in prompt:
            quote = "Lyon signs the data agreement"
            return json.dumps({"answered": True, "answer": "Lyon signs it on 20 November.",
                               "line": self._line_of(prompt, quote), "quote": quote})
        return json.dumps({"answered": False, "answer": None, "line": None, "quote": None})


# Setup ---------------------------------------------------------------------------------------


@pytest.fixture
def stack(monkeypatch, object_store):
    monkeypatch.setattr(settings, "enable_auth", True)
    monkeypatch.setattr(settings, "signup_enabled", True)
    monkeypatch.setattr(settings, "vp_meetween_api_key", KEY)
    monkeypatch.setattr(settings, "jwt_jitsi_secret_key", "j" * 40)
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
    monkeypatch.setattr(
        settings, "_agentar", settings.agentar.model_copy(update={"worker_events_base_url": "http://fastapi:8000"})
    )
    # Dispatch and the report job read the configured store; the routes get the fixture's.
    monkeypatch.setattr(agentar_worker, "get_configured_object_store", lambda cfg: object_store)
    monkeypatch.setattr(agentar_report, "get_configured_object_store", lambda cfg: object_store)
    queued: list[str] = []
    monkeypatch.setattr(agentar_report, "enqueue_report", queued.append)

    upstream = Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
    monkeypatch.setattr(vp_route, "_http_client", client)
    monkeypatch.setattr(SessionManager, "_sessions", {})
    monkeypatch.setattr(SessionManager, "_connections", {})
    monkeypatch.setattr(SessionManager, "_ended_meetings", {})
    monkeypatch.setattr(SessionManager, "_last_activity", {})
    yield upstream, queued
    asyncio.run(client.aclose())


def signup(client: TestClient, email: str, first: str, last: str) -> dict[str, str]:
    response = client.post(
        "/api/auth/signup", json={"email": email, "password": PASSWORD, "first_name": first, "last_name": last}
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def png(width: int = 800, height: int = 600) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (40, 120, 200)).save(out, format="PNG")
    return out.getvalue()


def event(meeting_id: str, bot_id: str, kind: str, data: dict | None = None, **extra) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "agentar_meeting_id": meeting_id,
        "bot_id": bot_id,
        "type": kind,
        "at": clock.utcnow().isoformat(),
        "data": {"kind": kind, **(data or {})},
        **extra,
    }


def say(speaker: str, who: str, text: str) -> None:
    SessionManager.add_participants(JITSI_ID, ROOM, [{"participant_id": who, "username": speaker}])
    SessionManager.update_transcript(JITSI_ID, ROOM, text, participant_id=who, timestamp=time.time())
    time.sleep(0.01)


def ask(client: TestClient, engine: ChatEngine, bot_id: str, question: str) -> dict:
    body = {
        "type": TaskType.SPOKEN_QUESTION_ANSWERING, "meeting_id": JITSI_ID, "room_name": ROOM,
        "participant_id": f"vp:{bot_id}", "language": "en-US", "prompt": question,
    }
    with patch("api.routes.chat.EngineManager.get_lmm", return_value=engine):
        response = client.post("/api/chat", json=body, headers=WORKER)
    assert response.status_code == 200, response.text
    return response.json()


# The flow ------------------------------------------------------------------------------------


def test_owner_flow_from_sign_up_to_the_report(test_client: TestClient, stack, object_store) -> None:
    upstream, queued = stack
    laura = signup(test_client, "laura@flow.test", "Laura", "Rossi")
    bruno = signup(test_client, "bruno@flow.test", "Bruno", "Verdi")

    # Profile and picture.
    me = test_client.get("/api/me", headers=laura).json()
    assert me["onboarding_state"] != "completed"
    patched = test_client.patch(
        "/api/me", json={"profile": {"agent_name": "Lia", "voice": "am_michael"}, "complete_onboarding": True},
        headers=laura,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["profile"]["agent_name"] == "Lia"
    picture = test_client.put("/api/me/picture", files={"file": ("me.png", png(), "image/png")}, headers=laura)
    assert picture.status_code == 200, picture.text
    avatar = test_client.get("/api/me/picture", headers=laura)
    assert avatar.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(avatar.content)).size == (512, 512)

    # Knowledge base file, general scope.
    upload = test_client.post(
        "/api/agentar/files", files={"file": ("atlas.txt", KB_TEXT.encode(), "text/plain")},
        data={"scope": "general"}, headers=laura,
    )
    assert upload.status_code == 201, upload.text
    file_id = upload.json()["id"]
    assert upload.json()["text_status"] == "ok"

    # Meeting, now.
    created = test_client.post(
        "/api/agentar/meetings",
        json={"room": ROOM, "mode": "now", "title": "Atlas review", "purpose": "Review the Atlas go live",
              "questions": [QUESTION], "use_general_kb": True},
        headers=laura,
    )
    assert created.status_code == 201, created.text
    meeting_id = created.json()["id"]
    assert test_client.post(
        "/api/agentar/meetings", json={"room": ROOM, "mode": "now"}, headers=bruno
    ).json()["code"] == "room_not_owned"

    # Dispatch: a human is in the room, so the bot is sent at once.
    dispatched = test_client.post(f"/api/agentar/meetings/{meeting_id}/dispatch", headers=laura)
    assert dispatched.status_code == 202, dispatched.text
    assert dispatched.json()["status"] == "joining"
    assert dispatched.json()["display_name"] == "Lia Agentar (AI)"
    [bot] = upstream.bots
    bot_id = dispatched.json()["bot_id"]
    assert bot_id == f"{1:032x}"
    assert (bot["room"], bot["display_name"], bot["voice"]) == (ROOM, "Lia Agentar", "am_michael")
    assert "Laura Rossi" in bot["greeting"] and "AI" in bot["greeting"]
    assert bot["escalation_line"] == "I do not have that, I will check with Laura Rossi and report back"
    assert bot["events_url"] == "http://fastapi:8000/api/agentar/events"
    assert bot["questions"] == [{"id": "q1", "text": QUESTION}]
    sent_avatar = Image.open(io.BytesIO(base64.b64decode(bot["avatar_jpeg_b64"])))
    assert (sent_avatar.format, sent_avatar.size) == ("JPEG", (512, 512))

    # The worker reports the join; the meeting is live and bound to the Jitsi meeting.
    joined = test_client.post(
        "/api/agentar/events", json=event(meeting_id, bot_id, "joined", {"join_ms": 3100}, jitsi_meeting_id=JITSI_ID),
        headers=WORKER,
    )
    assert joined.status_code == 202, joined.text
    live = test_client.get(f"/api/agentar/meetings/{meeting_id}", headers=laura).json()
    assert (live["status"], live["jitsi_meeting_id"]) == ("live", JITSI_ID)

    # A question answered from the knowledge base.
    say("Marco", "p-marco", "Good morning, let us start with Atlas.")
    engine = ChatEngine()
    answered = ask(test_client, engine, bot_id, "Agentar, when does Atlas go live?")
    assert answered["answer_kind"] == "answer"
    assert answered["result"] == "Atlas goes live on 1 December 2026 in Trento."
    assert "1 December 2026" in engine.prompts[-1] and "Laura Rossi" in engine.prompts[-1]

    # A question the brief does not answer: the exact line the worker got at dispatch.
    escalated = ask(test_client, engine, bot_id, "Agentar, what is the Bonn budget?")
    assert escalated["answer_kind"] == "escalate"
    assert escalated["result"] == bot["escalation_line"]

    # The worker asks the script question, reports the escalation, the room answers.
    asked = event(meeting_id, bot_id, "question_asked", {"question_id": "q1", "text": QUESTION})
    assert test_client.post("/api/agentar/events", json=asked, headers=WORKER).status_code == 202
    say("Marco", "p-marco", ANSWER_LINE)
    say("Sara", "p-sara", ACTION_LINE)
    escalation = event(meeting_id, bot_id, "escalation",
                       {"question": "What is the Bonn budget?", "asker": "Marco", "channel": "group"})
    assert test_client.post("/api/agentar/events", json=escalation, headers=WORKER).status_code == 202

    # The bot leaves; the meeting ends with the transcript and queues the report.
    left = test_client.post("/api/agentar/events", json=event(meeting_id, bot_id, "left", {"reason": "alone"}),
                            headers=WORKER)
    assert left.status_code == 202, left.text
    ended = test_client.get(f"/api/agentar/meetings/{meeting_id}", headers=laura).json()
    assert (ended["status"], ended["report_status"]) == ("ended", "pending")
    assert queued == [meeting_id]

    report_model = ReportModel()
    stored = asyncio.run(agentar_report.generate_report(meeting_id, model=report_model))
    assert stored is not None and stored.status == "ready", stored and stored.error

    report = test_client.get(f"/api/agentar/meetings/{meeting_id}/report", headers=laura)
    assert report.status_code == 200, report.text
    body = report.json()
    assert body["status"] == "ready"
    lines = {line["n"]: line["text"] for line in body["transcript"]}
    assert ANSWER_LINE in lines.values() and ACTION_LINE in lines.values()
    [item] = body["action_items"]
    assert item["quote"] == "send the training invitations"
    assert item["quote"] in lines[item["transcript_refs"][0]]
    [answer] = body["answers"]
    assert (answer["question_id"], answer["status"]) == ("q1", "answered")
    assert answer["quote"] in lines[answer["transcript_refs"][0]]
    assert "escalation" in [point["kind"] for point in body["open_points"]]

    markdown = test_client.get(f"/api/agentar/meetings/{meeting_id}/report.md", headers=laura)
    assert markdown.status_code == 200 and "Lyon signs" in markdown.text
    laura_id = asyncio.run(User.find_by_email("laura@flow.test")).id
    assert f"u/{laura_id}/m/{meeting_id}/report/{stored.id}.md" in object_store.keys()

    # Notifications: every step once, then nothing new after the cursor.
    notes = test_client.get("/api/agentar/notifications", headers=laura).json()
    types = [item["type"] for item in notes["items"]]
    assert types == ["joined", "question_asked", "escalation", "left", "report_ready"]
    severity = {item["type"]: item["severity"] for item in notes["items"]}
    assert severity["escalation"] == "warning"
    seen = test_client.get("/api/agentar/notifications", params={"since": notes["cursor"]}, headers=laura).json()
    assert seen["items"] == []

    # Nothing of Laura's is visible to Bruno, and no answer tells him it exists.
    for method, path in [
        ("get", f"/api/agentar/meetings/{meeting_id}"),
        ("get", f"/api/agentar/meetings/{meeting_id}/report"),
        ("get", f"/api/agentar/meetings/{meeting_id}/report.md"),
        ("post", f"/api/agentar/meetings/{meeting_id}/report/regenerate"),
        ("post", f"/api/agentar/meetings/{meeting_id}/dispatch"),
        ("delete", f"/api/agentar/meetings/{meeting_id}"),
        ("get", f"/api/agentar/files/{file_id}/download"),
        ("delete", f"/api/agentar/files/{file_id}"),
    ]:
        response = test_client.request(method, path, headers=bruno)
        assert (response.status_code, response.json()["code"]) == (404, "not_found"), f"{method} {path}"
    assert test_client.get("/api/agentar/notifications", headers=bruno).json()["items"] == []
    assert test_client.get("/api/agentar/files", headers=bruno).json()["items"] == []
    assert test_client.get("/api/agentar/meetings", headers=bruno).json()["items"] == []


def test_the_agentar_opens_a_room_of_waiting_guests(test_client: TestClient, stack, frozen_clock) -> None:
    """Owner decision A: the bot is sent at once into a room Prosody sees as empty, and with
    nobody inside the grace window after it went live it says goodbye and the meeting is missed."""
    from utils import agentar_lifecycle

    upstream, queued = stack
    upstream.room_open = False
    laura = signup(test_client, "laura@open.test", "Laura", "Rossi")
    created = test_client.post(
        "/api/agentar/meetings",
        json={"room": ROOM, "mode": "now", "purpose": "Review the Atlas go live", "grace_minutes": 5},
        headers=laura,
    )
    assert created.status_code == 201, created.text
    meeting_id = created.json()["id"]

    dispatched = test_client.post(f"/api/agentar/meetings/{meeting_id}/dispatch", headers=laura)
    assert dispatched.status_code == 202, dispatched.text
    assert dispatched.json()["status"] == "joining"
    assert len(upstream.bots) == 1
    assert upstream.prosody_calls == 0, "no human gate before the bot is in"
    bot_id = dispatched.json()["bot_id"]

    joined = event(meeting_id, bot_id, "joined", {"join_ms": 2100}, jitsi_meeting_id=JITSI_ID)
    assert test_client.post("/api/agentar/events", json=joined, headers=WORKER).status_code == 202

    asyncio.run(agentar_lifecycle.tick())
    assert test_client.get(f"/api/agentar/meetings/{meeting_id}", headers=laura).json()["status"] == "live"
    assert upstream.prosody_calls == 1
    assert upstream.deleted == []

    frozen_clock.advance(minutes=5, seconds=1)
    asyncio.run(agentar_lifecycle.tick())
    missed = test_client.get(f"/api/agentar/meetings/{meeting_id}", headers=laura).json()
    assert (missed["status"], missed["status_reason"]) == ("missed", "no_human_joined")
    assert upstream.deleted == [bot_id]
    assert upstream.delete_params == ["farewell=1"]
    assert queued == []
