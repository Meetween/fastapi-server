"""Shared data and fakes of the Agentar report tests.

The three transcripts under `tests/data/agentar_report/` are fictional (an invented steering
call) and are stored as the session keeps them: one entry per line, `username`, `text`,
`timestamp` in milliseconds, `source`. Line N of a transcript is entry N of the file.
"""
import asyncio
import copy
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from src.database.models.agentar_event import AgentarEvent
from src.database.models.agentar_meeting import AgentarMeeting, MeetingQuestion
from tests.utils.helpers import ensure_user_in_db
from utils import clock

DATA = Path(__file__).resolve().parents[1] / "data" / "agentar_report"

OWNER_EMAIL = "laura.bianchi@meetween.test"
PURPOSE = "Present the Atlas status and get answers to my two questions. Do not commit to budget or dates."
Q1 = "Can Lyon confirm the date they sign the data processing agreement?"
Q2 = "Who will run the user training at the Bonn site?"
QUESTIONS = [{"id": "q1", "text": Q1}, {"id": "q2", "text": Q2}]


def load_entries(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def load_expected(name: str) -> dict[str, Any]:
    return json.loads((DATA / f"{name}_expected_report.json").read_text(encoding="utf-8"))


def load_deferrals(name: str) -> list[dict[str, Any]]:
    return json.loads((DATA / f"{name}.deferrals.json").read_text(encoding="utf-8"))


class FakeModel:
    """Stands in for SpeechLMM. It answers by the kind of prompt with crafted JSON, records every
    prompt, and can be given raw replies to hand out first (to test the parse and retry paths)."""

    def __init__(
        self,
        *,
        summary: str = "A short meeting.",
        actions: Optional[list[dict[str, Any]]] = None,
        answers: Optional[dict[str, dict[str, Any]]] = None,
        raw: Optional[list[str]] = None,
        delay: float = 0.0,
    ) -> None:
        self.summary = summary
        self.actions = actions or []
        self.answers = answers or {}
        self.raw = list(raw or [])
        self.delay = delay
        self.prompts: list[str] = []

    async def __call__(self, prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raw:
            return self.raw.pop(0)
        if prompt.startswith("You write short minutes") or prompt.startswith("Combine these partial"):
            return json.dumps({"summary": self.summary})
        if prompt.startswith("List the action items"):
            return json.dumps({"items": self.actions})
        for qid, reply in self.answers.items():
            if f"this question: {_question_text(qid)}\n" in prompt:
                return json.dumps(reply)
        return json.dumps({"answered": False, "answer": None, "line": None, "quote": None})

    def kinds(self) -> list[str]:
        out = []
        for prompt in self.prompts:
            if prompt.startswith("You write short minutes"):
                out.append("summary")
            elif prompt.startswith("Combine these partial"):
                out.append("merge")
            elif prompt.startswith("List the action items"):
                out.append("actions")
            else:
                out.append("answer")
        return out


def _question_text(qid: str) -> str:
    return {q["id"]: q["text"] for q in QUESTIONS}[qid]


def answer(line: int, quote: str, text: str) -> dict[str, Any]:
    return {"answered": True, "answer": text, "line": line, "quote": quote}


def item(text: str, owner: str, due: Optional[str], line: int, quote: str) -> dict[str, Any]:
    return {"text": text, "owner": owner, "due": due, "line": line, "quote": quote}


# Crafted model replies per transcript ------------------------------------------------------------

T1 = dict(
    summary=(
        "Sara Neri will send the training invitations to the Trento group by Friday 13 November. "
        "Lyon will sign the data processing agreement on 20 November. "
        "Katrin Vogel will run the user training at Bonn, starting 2 December. "
        "The Bonn dry run room still needs to be booked by Laura."
    ),
    actions=[
        item(
            "Send the training invitations to the Trento group",
            "Sara Neri",
            "Friday 13 November",
            4,
            "I can send the training invitations to the Trento group by Friday 13 November",
        ),
        item(
            "Ask Pierre Duval to confirm the Lyon signing date in writing and send it to the group",
            "Anil Kapoor",
            None,
            9,
            "I will ask Pierre to confirm the signing date in writing",
        ),
        item("Book the Bonn dry run room", "Laura", None, 15, "Laura should book the room for the dry run"),
        # Hallucinated: nobody says this anywhere in the transcript.
        item("Book the hotel in Lyon for the steering group", "Anil Kapoor", None, 3, "I will book the hotel in Lyon"),
    ],
    answers={
        "q1": answer(7, "They will sign the agreement on 20 November", "Lyon will sign the data processing agreement on 20 November"),
        "q2": answer(11, "Katrin Vogel will run the training at Bonn", "Katrin Vogel will run the user training at Bonn"),
    },
)

T2 = dict(
    summary=(
        "Lyon will sign the data processing agreement on 20 November. "
        "Training material is ready in English and Italian, and the Trento room is booked for 25 November. "
        "The usage survey goes out on Friday. The next steering call is on Tuesday 24 November."
    ),
    actions=[
        item(
            "Send the usage survey to the group",
            "Anil Kapoor",
            "Friday",
            10,
            "The usage survey goes out on Friday, I will send it to the group",
        )
    ],
    answers={
        "q1": answer(5, "they sign on 20 November", "Lyon will sign on 20 November"),
        # The trap: line 8 talks about Bonn training material and names no person. This reply
        # invents the trainer and cites that line.
        "q2": answer(8, "The Bonn training material is ready", "Katrin Vogel will run the user training at Bonn"),
    },
)

T3 = dict(
    summary=(
        "Sara Neri will send the training invitations to the Trento group by Friday 13 November. "
        "Lyon will sign the data processing agreement on 20 November. "
        "Katrin Vogel will run the user training at Bonn."
    ),
    actions=[
        item(
            "Send the training invitations to the Trento group",
            "Sara Neri",
            "Friday 13 November",
            5,
            "I can send them to, uh, the Trento group by, by Friday",
        )
    ],
    answers={
        "q1": answer(8, "they will sign, uh, on the twentieth of November", "Lyon will sign on 20 November"),
        "q2": answer(11, "Katrin Vogel will, will run the training in, in Bonn", "Katrin Vogel will run the user training at Bonn"),
    },
)

SCRIPTS = {"t1": T1, "t2": T2, "t3": T3}


def fake_for(name: str) -> FakeModel:
    """A fresh model with its own copy of the script, so a test may change its replies."""
    return FakeModel(**copy.deepcopy(SCRIPTS[name]))


# Seeding -----------------------------------------------------------------------------------------


def seed_owner(email: str = OWNER_EMAIL, first_name: str = "Laura", last_name: str = "Bianchi") -> None:
    ensure_user_in_db(email=email, password="pytest-secret-123", first_name=first_name, last_name=last_name)


def seed_meeting(
    name: Optional[str] = "t1",
    *,
    owner: str = OWNER_EMAIL,
    status: str = "ended",
    entries: Optional[list[dict[str, Any]]] = None,
    questions: Optional[list[dict[str, str]]] = None,
    ended_ago_s: Optional[float] = 20,
    with_events: bool = True,
) -> AgentarMeeting:
    """An ended meeting with the transcript snapshot of `name`, the deferral events of that
    transcript and a `question_asked` event per marker line."""
    snapshot = entries if entries is not None else (load_entries(name) if name else [])
    now = clock.utcnow()
    meeting = AgentarMeeting(
        owner=owner,
        title="Atlas steering",
        room_name="atlas-steering",
        purpose=PURPOSE,
        questions=[MeetingQuestion(**q) for q in (questions if questions is not None else QUESTIONS)],
        status=status,
        joined_at=now - timedelta(minutes=30),
        ended_at=(now - timedelta(seconds=ended_ago_s)) if ended_ago_s is not None else None,
        report_status="pending" if status == "ended" else "none",
        transcript_snapshot=snapshot,
    )
    meeting = asyncio.run(meeting.save())
    if with_events and name:
        asyncio.run(_seed_events(meeting, name, snapshot))
    return meeting


async def _seed_events(meeting: AgentarMeeting, name: str, snapshot: list[dict[str, Any]]) -> None:
    for i, deferral in enumerate(load_deferrals(name)):
        await AgentarEvent(
            event_id=f"{meeting.id}-esc-{i}",
            owner=meeting.owner,
            meeting_id=meeting.id,
            type="escalation",
            data={"kind": "escalation", "question": deferral["question"], "asker": deferral["asker"]},
        ).save()
    for entry in snapshot:
        if entry.get("source") == "agentar_marker":
            qid = entry["username"].rsplit(" ", 1)[-1].rstrip(")").lower()
            await AgentarEvent(
                event_id=f"{meeting.id}-ask-{qid}",
                owner=meeting.owner,
                meeting_id=meeting.id,
                type="question_asked",
                data={"kind": "question_asked", "question_id": qid, "text": entry["text"]},
            ).save()


def line_text(entries: list[dict[str, Any]], n: int) -> str:
    return entries[n - 1]["text"]


def ts(value: datetime) -> float:
    return value.timestamp()
