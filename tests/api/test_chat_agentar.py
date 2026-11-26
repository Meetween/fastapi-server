"""POST /api/chat for a vp: bot bound to an Agentar meeting, and the Butler path left alone."""
import asyncio
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from config import settings
from core import SessionManager
from models import TaskType
from src.database.models import AgentarFile, AgentarMeeting, User
from src.database.models.agentar_file import FileChunk

KEY = "vp-test-key"
MEETING = "ws_agentar_chat_1"
ROOM = "atlas"
# Worker bot ids are uuid4 hex; the worker adds "-<n>" per question (see test_a_numbered_question_...).
BOT = "vp:b0a71a5"
SENTINEL = "I do not have that, I will check with Laura Bianchi and report back"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _worker_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "enable_auth", True)
    monkeypatch.setattr(settings, "vp_meetween_api_key", KEY)
    yield
    SessionManager.clear_session(meeting_id=MEETING, room_name=ROOM, force=True)


@pytest.fixture
def atlas():
    run(User.create(email="laura@x.test", password_hash="x", first_name="Laura", last_name="Bianchi"))
    meeting = run(AgentarMeeting.create(
        owner="laura@x.test", room_name=ROOM, status="live", bot_id="b0a71a5",
        purpose="Present Atlas", relevant_info="Deputy Marco Ferri",
    ))
    run(AgentarFile.create(
        owner="laura@x.test", scope="meeting", meeting_id=meeting.id, kind="document", text_status="ok",
        chunks=[FileChunk(n=1, text="Go live is planned on 1 December 2026 at Trento and Bonn.")],
    ))
    return meeting


class Engine:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.max_context_chars = 300_000
        self.max_history_chars = 600_000

    async def _generate_text(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        return self.reply


def ask(client: TestClient, engine: Engine, question: str, participant_id: str = BOT) -> dict:
    body = {
        "type": TaskType.SPOKEN_QUESTION_ANSWERING, "meeting_id": MEETING, "room_name": ROOM,
        "participant_id": participant_id, "language": "en-US", "prompt": question,
    }
    with patch("api.routes.chat.EngineManager.get_lmm", return_value=engine):
        response = client.post("/api/chat", json=body, headers={"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 200, response.text
    return response.json()


def test_a_bound_bot_gets_the_agentar_prompt_and_an_answer_kind(test_client, atlas):
    engine = Engine("Go live is planned on 1 December 2026.")
    body = ask(test_client, engine, "Agentar, when does Atlas go live?")
    assert body["result"] == "Go live is planned on 1 December 2026."
    assert body["answer_kind"] == "answer"
    prompt = engine.prompts[0]
    assert "attending a meeting for Laura Bianchi" in prompt
    assert "1 December 2026" in prompt and "Deputy Marco Ferri" in prompt


def test_a_bound_bot_answers_from_the_notes_before_anyone_has_spoken(test_client, atlas):
    assert SessionManager.get_transcript(MEETING, ROOM) is None
    body = ask(test_client, Engine("Go live is planned on 1 December 2026."), "Agentar, when does Atlas go live?")
    assert body["error"] is None and body["answer_kind"] == "answer"


def test_the_sentinel_comes_back_as_an_escalation(test_client, atlas):
    body = ask(test_client, Engine("I do not have that, I will check with Laura and report back."),
               "Agentar, who is the vendor?")
    assert body["result"] == SENTINEL and body["answer_kind"] == "escalate"


def test_a_commitment_never_reaches_the_model(test_client, atlas):
    engine = Engine("Yes, agreed.")
    body = ask(test_client, engine, "Agentar, can we move 5,000 euros to Lyon?")
    assert body["result"] == SENTINEL and body["answer_kind"] == "escalate"
    assert engine.prompts == []


def test_who_are_you_is_the_disclosure(test_client, atlas):
    body = ask(test_client, Engine("I am Laura."), "Agentar, who are you?")
    assert body["result"].startswith("I am Agentar, an AI assistant") and "Laura Bianchi" in body["result"]
    assert body["answer_kind"] == "answer"


def test_the_transcript_tail_reaches_the_prompt_and_old_lines_do_not(test_client, atlas):
    lines = "\n".join(f"Anil: filler line {i}" for i in range(30)) + "\nSara: Katrin Vogel runs the training."
    SessionManager.update_transcript(meeting_id=MEETING, room_name=ROOM, transcript=lines)
    engine = Engine("Katrin Vogel.")
    ask(test_client, engine, "Agentar, who runs the training?")
    assert "Katrin Vogel runs the training." in engine.prompts[0]
    assert "filler line 3\n" not in engine.prompts[0]


def test_a_numbered_question_from_the_worker_gets_the_agentar_prompt(test_client, atlas):
    # VP_CHAT_FRESH_HISTORY (the worker default) asks as vp:<bot id>-<n>. C2 run 2 got the Butler prompt.
    engine = Engine("Go live is planned on 1 December 2026.")
    body = ask(test_client, engine, "Laura, when does Atlas go live?", participant_id=BOT + "-2")
    assert body["answer_kind"] == "answer"
    assert "Present Atlas" in engine.prompts[0] and "Deputy Marco Ferri" in engine.prompts[0]


def test_the_turn_is_stored_under_the_bot_participant(test_client, atlas):
    ask(test_client, Engine("Go live is planned on 1 December 2026."), "Agentar, when does Atlas go live?")
    history = SessionManager.get_participant_history(MEETING, ROOM, BOT)
    assert [entry.role for entry in history] == ["user", "assistant"]


def test_a_bot_without_an_agentar_meeting_keeps_the_butler_path(test_client, atlas):
    SessionManager.update_transcript(meeting_id=MEETING, room_name=ROOM, transcript="Alice: the deadline is June 15.")
    engine = Engine("The deadline is June 15.")
    body = ask(test_client, engine, "When is the deadline?", participant_id="vp:butler-bot")
    assert body["answer_kind"] is None
    assert "will be read aloud" in engine.prompts[0] and "attending a meeting for" not in engine.prompts[0]


def test_butler_with_no_transcript_still_skips_the_model(test_client):
    engine = Engine("never used")
    body = ask(test_client, engine, "What did we decide?", participant_id="vp:butler-bot")
    assert body["error"] == "no_transcript" and body["result"] == "" and engine.prompts == []


def test_an_ended_meeting_does_not_steer_a_late_request(test_client, atlas):
    run(AgentarMeeting.get_collection().update_one({"bot_id": "b0a71a5"}, {"$set": {"status": "ended"}}))
    body = ask(test_client, Engine("x"), "Agentar, when does Atlas go live?")
    assert body["error"] == "no_transcript" and body["answer_kind"] is None


def test_a_web_user_gets_no_answer_kind(client, test_room_name, test_meeting_id):
    SessionManager.update_transcript(meeting_id=test_meeting_id, room_name=test_room_name,
                                     transcript="Alice: the deadline is June 15.")
    try:
        body = client.post("/api/chat", json={
            "type": TaskType.SPOKEN_QUESTION_ANSWERING, "meeting_id": test_meeting_id,
            "room_name": test_room_name, "participant_id": "user-web", "language": "en-US", "prompt": "When?",
        }).json()
    finally:
        SessionManager.clear_session(meeting_id=test_meeting_id, room_name=test_room_name, force=True)
    assert body["answer_kind"] is None
