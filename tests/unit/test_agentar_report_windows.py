"""Answer windows on fragmented captions (S10, C2 run 5).

In C2 run 5 the captions arrived as 0.5 s fragments, so the answer to Q1 ("On the 20th," then
"Of November, 2013.") sat on two lines and no single line held a quote: the report said "no
answer captured" although the answer was spoken. The fixture `c2_run5.jsonl` is the transcript of
that report (K/C2/logs/run5-report.txt). Consecutive fragments of one speaker within a short gap
are now merged before the answer call; the window starts at the marker and ends at the next
marker or after ANSWER_WINDOW_S. The model here is honest: it answers only with a quote that one
rendered line of its prompt holds.
"""
import asyncio
import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from src.database.models.agentar_report import AgentarReport
from tests.utils.agentar_report_helpers import DATA, Q1, Q2, seed_meeting, seed_owner
from utils import agentar_report as ar

RUN5 = [json.loads(line) for line in (DATA / "c2_run5.jsonl").read_text(encoding="utf-8").splitlines()]
# C2 replay of the Q2 answer call (K/C2/logs/replay-answer-q2.json), with the key it lacked: the
# quote is the question, which no line of the window holds.
Q2_REPLAY = {"answered": True, "answer": "Catherine Deneuve!", "line": 13, "quote": Q2}

_RENDERED = re.compile(r"^\[(\d+)\] ([^:]+): (.*)$")


def run(coro):
    return asyncio.run(coro)


def fold(text: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", text.casefold()).split())


class WindowModel:
    """Answers Q1 from the prompt it gets: the line holding "20th of november", quoted from that
    line only. Without such a line it says not answered. Q2 gets the C2 replay."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def __call__(self, prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        if prompt.startswith("You write short minutes") or prompt.startswith("Combine these partial"):
            return json.dumps({"summary": "Lyon will sign the data processing agreement."})
        if prompt.startswith("List the action items"):
            return json.dumps({"items": []})
        if f"this question: {Q2}\n" in prompt:
            return json.dumps(Q2_REPLAY)
        for row in prompt.splitlines():
            match = _RENDERED.match(row)
            if not match:
                continue
            n, _, text = match.groups()
            start = text.find("On the 20th")
            if start != -1 and "november" in fold(text[start:]).split()[:5]:
                quote = text[start:].split("2013")[0].strip(" ,")
                return json.dumps({
                    "answered": True,
                    "answer": "Lyon will sign the data processing agreement on the 20th of November.",
                    "line": int(n),
                    "quote": quote,
                })
        return json.dumps({"answered": False, "answer": None, "line": None, "quote": None})


@pytest.fixture(autouse=True)
def _owner():
    seed_owner()


def report_of(entries, model):
    meeting = seed_meeting(None, entries=entries)
    run(ar.generate_report(meeting.id, model=model))
    return AgentarReport.from_dict(run(AgentarReport.get_collection().find_one({"meeting_id": meeting.id})))


def answers_by_id(report) -> dict:
    return {a["question_id"]: a for a in (report.answers if isinstance(report.answers[0], dict) else [a.model_dump() for a in report.answers])}


class TestC2Run5:
    def test_q1_is_captured_from_the_merged_fragments(self) -> None:
        report = report_of(RUN5, WindowModel())

        assert report.status == "ready", report.error
        q1 = answers_by_id(report)["q1"]
        assert q1["status"] == "answered"
        assert q1["speaker"] == "Anil Kapoor"
        assert fold(q1["quote"]) == "on the 20th of november"
        # The quote spans the two caption lines it was spoken on.
        assert q1["transcript_refs"] == [9, 10]

    def test_q2_with_nothing_grounded_stays_no_answer_captured(self) -> None:
        q2 = answers_by_id(report_of(RUN5, WindowModel()))["q2"]
        assert q2["status"] == "no_answer_captured"
        assert q2["answer"] is None and q2["transcript_refs"] == []

    def test_the_transcript_keeps_every_caption_line(self) -> None:
        report = report_of(RUN5, WindowModel())
        assert [line["n"] if isinstance(line, dict) else line.n for line in report.transcript] == list(range(1, 20))

    def test_the_answer_prompt_reads_one_line_per_speaker_turn(self) -> None:
        model = WindowModel()
        report_of(RUN5, model)
        [q1_prompt] = [p for p in model.prompts if f"this question: {Q1}\n" in p]
        rendered = [row for row in q1_prompt.splitlines() if _RENDERED.match(row)]
        assert rendered == [
            "[2] Anil Kapoor: Yeah! Yes! Leon. We'll sign the... The data Processing, agreement On the 20th, "
            "Of November, 2013. November!"
        ]


T0 = datetime(2026, 11, 4, 13, 48, 0, tzinfo=timezone.utc)


def line(n, speaker, text, seconds, source="speech", **extra):
    return {"n": n, "speaker": speaker, "text": text, "at": T0 + timedelta(seconds=seconds), "source": source, **extra}


def marker(n, qid, seconds):
    return line(n, "Agentar", f"Question {qid}?", seconds, source="agentar", question_id=qid)


class TestMerge:
    def test_fragments_merge_only_within_the_gap_and_the_speaker(self) -> None:
        lines = [
            line(2, "Ana", "We sign", 1.0),
            line(3, "Ana", "on Friday.", 2.5),
            line(4, "Ana", "Later point.", 2.5 + ar.MERGE_GAP_S + 0.1),
            line(5, "Bo", "Agreed.", 9.0),
        ]
        units = ar.merge_fragments(lines)
        assert [(u["n"], u["refs"], u["speaker"], u["text"]) for u in units] == [
            (2, [2, 3], "Ana", "We sign on Friday."),
            (4, [4], "Ana", "Later point."),
            (5, [5], "Bo", "Agreed."),
        ]

    def test_chat_lines_are_never_merged(self) -> None:
        lines = [line(2, "Ana", "We sign", 1.0), line(3, "Ana", "on Friday.", 1.5, source="chat")]
        assert [u["refs"] for u in ar.merge_fragments(lines)] == [[2], [3]]

    def test_a_merged_turn_is_capped(self) -> None:
        lines = [line(i, "Ana", "word " * 40, i * 0.5) for i in range(2, 12)]
        units = ar.merge_fragments(lines)
        assert all(len(u["text"]) <= ar.MERGE_MAX_CHARS for u in units)
        assert [n for u in units for n in u["refs"]] == list(range(2, 12))

    def test_lines_without_a_time_merge_by_speaker(self) -> None:
        lines = [dict(line(2, "Ana", "We sign", 0), at=None), dict(line(3, "Ana", "on Friday.", 0), at=None)]
        assert [u["refs"] for u in ar.merge_fragments(lines)] == [[2, 3]]


class TestWindowBounds:
    def test_the_window_ends_at_the_next_marker(self) -> None:
        lines = [marker(1, "q1", 0), line(2, "Ana", "First answer.", 5), marker(3, "q2", 10), line(4, "Bo", "Second.", 12)]
        assert [w["n"] for w in ar._answer_window(lines[0], lines)] == [2]

    def test_the_window_ends_after_the_bounded_time(self) -> None:
        lines = [
            marker(1, "q1", 0),
            line(2, "Ana", "First answer.", 5),
            line(3, "Bo", "Much later chatter.", ar.ANSWER_WINDOW_S + 1),
        ]
        assert [w["n"] for w in ar._answer_window(lines[0], lines)] == [2]

    def test_refs_follow_the_quote_inside_a_merged_turn(self) -> None:
        units = ar.merge_fragments([line(2, "Ana", "Yes.", 1), line(3, "Ana", "We sign", 2), line(4, "Ana", "on Friday.", 3)])
        assert ar.quote_refs(units[0], "we sign on friday") == [3, 4]
        assert ar.quote_refs(units[0], "Yes. We sign") == [2, 3]


# K/R1 run 3 (2026-11-04): the transcript of the report and the replay of its Q1 answer call
# (K/R1/logs/replay-answer-run3-q1.json). The guest said "Marco will run the user training at the Bonn
# site"; the captions came as fragments. SpeechLMM v2 L copied the rendered speaker label into the
# quote, and the one word answer is shorter than MIN_QUOTE_CHARS, so the report said "no answer
# captured" while its summary said Marco runs the training.
R1_Q1 = {"id": "q1", "text": "Who will run the user training at the Bonn site?"}
R1_RUN3 = [
    {"n": 1, "speaker": "Agentar", "text": R1_Q1["text"], "at": "2026-11-04T17:28:02.417000", "source": "agentar"},
    {"n": 2, "speaker": "Anil Kapoor", "text": "Marco.", "at": "2026-11-04T17:28:16.520000", "source": "speech"},
    {"n": 3, "speaker": "Anil Kapoor", "text": "the user", "at": "2026-11-04T17:28:18.740000", "source": "speech"},
    {"n": 4, "speaker": "Anil Kapoor", "text": "Training.", "at": "2026-11-04T17:28:19.376000", "source": "speech"},
    {"n": 5, "speaker": "Anil Kapoor", "text": "the bonds", "at": "2026-11-04T17:28:20.019000", "source": "speech"},
    {"n": 6, "speaker": "Anil Kapoor", "text": "Side.", "at": "2026-11-04T17:28:20.657000", "source": "speech"},
]
R1_REPLY = {"answered": True, "answer": "Marco", "line": 2, "quote": "Anil Kapoor: Marco."}


def _r1_verdict(reply: dict) -> dict:
    lines = ar.normalise_transcript(R1_RUN3)
    window = ar.merge_fragments(ar._answer_window(ar._marker_for(R1_Q1, lines), lines))
    return ar.validate_answer(R1_Q1, ar._AnswerOut.model_validate(reply), window, ar._evidence(lines))


class TestR1Run3:
    def test_the_speaker_label_the_model_copies_is_not_part_of_the_quote(self) -> None:
        got = _r1_verdict(R1_REPLY)
        assert got["status"] == "answered"
        assert (got["answer"], got["speaker"], got["quote"]) == ("Marco", "Anil Kapoor", "Marco.")
        assert got["transcript_refs"] == [2]

    def test_a_short_quote_counts_when_it_is_a_whole_caption_line(self) -> None:
        assert _r1_verdict({**R1_REPLY, "quote": "Marco."})["status"] == "answered"

    def test_a_short_quote_inside_a_line_is_still_no_evidence(self) -> None:
        got = _r1_verdict({"answered": True, "answer": "Marco", "line": 3, "quote": "user"})
        assert got["status"] == "no_answer_captured"

    def test_a_label_of_someone_else_is_not_removed(self) -> None:
        got = _r1_verdict({**R1_REPLY, "quote": "Sofia Rossi: Marco."})
        assert got["status"] == "no_answer_captured"


class TestAgentarChatLines:
    """The server writes the chat question and the Agentar's chat answer into the transcript with
    a `vp:` participant id (source chat). Both are relayed by the bot: never a guest's answer to
    a script question, whatever the text says, because the worker strips the wake word. An
    escalation cites the asker's relayed line, never the Agentar's reply."""

    BOT = "vp:bot1"

    def entries(self):
        return [
            {"username": "Agentar (asked q1)", "text": "Question q1?", "source": "agentar_marker", "participant_id": self.BOT, "timestamp": 0.0},
            {"username": "Anil", "text": "when does Atlas go live?", "source": "chat", "participant_id": self.BOT, "timestamp": 5.0},
            {"username": "Agentar", "text": "Atlas goes live on the 20th.", "source": "chat", "participant_id": self.BOT, "timestamp": 5.0},
            {"username": "Ana", "text": "Yes, we sign on Friday.", "source": "speech", "participant_id": "abc123", "timestamp": 8.0},
        ]

    def test_relayed_chat_lines_are_not_in_the_answer_window(self) -> None:
        lines = ar.normalise_transcript(self.entries())
        assert [w["n"] for w in ar._answer_window(lines[0], lines)] == [4]

    def test_the_flag_survives_a_second_normalisation(self) -> None:
        lines = ar.normalise_transcript(ar.normalise_transcript(self.entries()))
        assert [w["n"] for w in ar._answer_window(lines[0], lines)] == [4]

    def test_a_guest_chat_line_with_the_question_text_stays_an_answer_candidate(self) -> None:
        entries = self.entries()
        entries[1] = {**entries[1], "participant_id": "abc123"}
        entries[2] = {**entries[2], "participant_id": "def456"}
        lines = ar.normalise_transcript(entries)
        assert [w["n"] for w in ar._answer_window(lines[0], lines)] == [2, 3, 4]

    def test_a_chat_message_of_a_guest_named_agentar_cannot_be_a_marker(self) -> None:
        [line_] = ar.normalise_transcript([{"username": "Agentar", "text": "Atlas goes live.", "source": "chat", "timestamp": 1.0}])
        assert (line_["source"], line_["speaker"]) == ("chat", "Agentar")
        assert "question_id" not in line_

    def test_the_escalation_cites_the_askers_line_not_the_reply(self) -> None:
        entries = [
            {"username": "Agentar", "text": "Can we move 5,000 euros to Lyon? is for Laura.", "source": "chat", "participant_id": self.BOT, "timestamp": 1.0},
            {"username": "Anil", "text": "can we move 5,000 euros to Lyon?", "source": "chat", "participant_id": self.BOT, "timestamp": 1.0},
        ]
        lines = ar.normalise_transcript(entries)
        assert ar._escalation_refs("can we move 5,000 euros to Lyon?", "Anil", lines) == [2]
        assert ar._escalation_refs("can we move 5,000 euros to Lyon?", None, lines) == [2]
