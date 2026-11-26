"""Offline evaluation of the Agentar prompt on the Atlas chat cases.

The harness builds the context of the Atlas meeting from `data/`, runs every case through
`answer_agentar_question` and checks the result against the case. It takes any engine with an async
`_generate_text`: the tests use `ExtractiveReader`, a fake that reads the prompt like a perfect
extractive model, and `run_real.py` uses the SpeechLMM client. With the fake the run proves
structure: the right chunk reaches the prompt, the data stays inside its tags, the rules that do not
need a model fire, the answer is cleaned. It says nothing about the quality of a real model.

Case fields: id, question, asker, expected_kind (answer, escalate, not_in_brief), expected_facts
(`a|b` is any of the variants), forbidden (whole words or phrases), source, optional
transcript_tail and tag. Matching is case insensitive.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from utils.agentar_prompt import (
    AgentarContext,
    AgentarTurn,
    Chunk,
    answer_agentar_question,
    build_agentar_prompt,
    escalation_line,
    tokenize,
)

DATA_DIR = Path(os.environ.get("AGENTAR_EVAL_DATA", Path(__file__).parent / "data"))
PAGE_BREAK = "\f"


def data_available() -> bool:
    return (DATA_DIR / "chat-eval.jsonl").is_file() and (DATA_DIR / "brief.json").is_file()


def load_cases(path: Path | None = None) -> list[dict]:
    path = path or DATA_DIR / "chat-eval.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_brief(path: Path | None = None) -> dict:
    return json.loads((path or DATA_DIR / "brief.json").read_text(encoding="utf-8"))


def pack(text: str, page: int | None, max_chars: int) -> list[Chunk]:
    """Pack lines into chunks of at most `max_chars`, a stand in for the real chunker: no overlap,
    a line is never split."""
    chunks: list[Chunk] = []
    current: list[str] = []
    size = 0
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        if current and size + len(line) > max_chars:
            chunks.append(Chunk(text="\n".join(current), page=page))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append(Chunk(text="\n".join(current), page=page))
    return chunks


def load_kb(chunk_chars: int = 600, kb_dir: Path | None = None) -> list[Chunk]:
    kb_dir = kb_dir or DATA_DIR / "kb"
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.iterdir()):
        pages = path.read_text(encoding="utf-8").split(PAGE_BREAK)
        for number, page_text in enumerate(pages, 1):
            chunks.extend(pack(page_text, number if len(pages) > 1 else None, chunk_chars))
    return chunks


def make_context(chunk_chars: int = 600) -> AgentarContext:
    brief = load_brief()
    return AgentarContext(
        owner_name=brief["owner_name"],
        purpose=brief["purpose"],
        relevant_info=brief["relevant_info"],
        chunks=tuple(load_kb(chunk_chars)),
    )


# --- Fake model --------------------------------------------------------------------------------


class ExtractiveReader:
    """A fake model. It reads <notes>, <transcript> and <question> of the prompt and answers with
    the line that shares the most words with the question (a digit breaks ties for a when or how many
    question), or with the escalation line when no line
    shares at least half of them (two at the least, words match on their first four letters). It counts its calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    @staticmethod
    def _block(prompt: str, tag: str) -> str:
        match = re.search(rf"^<{tag}>\n?(.*?)\n?</{tag}>$", prompt, re.DOTALL | re.MULTILINE)
        return match.group(1) if match else ""

    async def _generate_text(self, prompt: str, **_kwargs) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        question = tokenize(self._block(prompt, "question"))
        owner_match = re.search(r"reply exactly: (I do not have that, .*? and report back)", prompt)
        sentinel = owner_match.group(1) if owner_match else "I do not have that"
        background = re.search(r"^Background from [^:\n]+: (.*)$", prompt, re.MULTILINE)
        raw_lines = [background.group(1)] if background else []
        for block in ("notes", "transcript"):
            raw_lines.extend(self._block(prompt, block).splitlines())
        lines = [
            sentence.replace("&lt;", "<").replace("&gt;", ">")
            for line in raw_lines
            if line.strip() and line.strip() not in ("---", "(none)", "(empty)")
            for sentence in re.split(r"(?<=[.!?])\s+", line.strip())
        ]
        if not question or not lines:
            return sentinel
        wanted = max(2, math.ceil(len(question) / 2))
        asks_number = bool(re.search(r"\b(?:when|how many|how long|how much|what date)\b", self._block(prompt, "question"), re.I))
        best, best_score, best_overlap = "", 0.0, 0
        for line in lines:
            overlap = sum(1 for q in set(question) if any(t[:4] == q[:4] for t in tokenize(line)))
            score = overlap + (0.5 if asks_number and re.search(r"\d", line) else 0.0)
            if score > best_score:
                best, best_score, best_overlap = line, score, overlap
        return best if best_overlap >= wanted else sentinel


# --- Checks ------------------------------------------------------------------------------------


def _variants(fact: str) -> list[str]:
    return [v.strip().lower() for v in fact.split("|") if v.strip()]


def _contains_fact(text: str, fact: str) -> bool:
    haystack = text.lower()
    return any(variant in haystack for variant in _variants(fact))


def _contains_forbidden(text: str, forbidden: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(forbidden.lower())}(?!\w)", text.lower()) is not None


@dataclass
class CaseResult:
    case_id: str
    expected_kind: str
    kind: str
    text: str
    decided_by: str
    prompt_chars: int
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def transcript_of(case: dict) -> str | None:
    tail = case.get("transcript_tail")
    return "\n".join(tail) if tail else None


def check_turn(case: dict, turn: AgentarTurn, owner: str) -> list[str]:
    """Problems of one answer against its case. Empty list means the case passes."""
    problems: list[str] = []
    wants_escalation = case["expected_kind"] in ("escalate", "not_in_brief")
    if wants_escalation and turn.kind != "escalate":
        problems.append(f"expected escalate, got {turn.kind}: {turn.text!r}")
    if not wants_escalation and turn.kind != "answer":
        problems.append(f"expected answer, got {turn.kind}: {turn.text!r}")
    for fact in case.get("expected_facts", []):
        if not _contains_fact(turn.text, fact):
            problems.append(f"fact missing: {fact}")
    for word in case.get("forbidden", []):
        if _contains_forbidden(turn.text, word):
            problems.append(f"forbidden present: {word}")
    if wants_escalation and turn.text != escalation_line(owner):
        problems.append("escalation text is not the fixed line")
    return problems


async def run_case(engine, context: AgentarContext, case: dict) -> CaseResult:
    transcript = transcript_of(case)
    turn = await answer_agentar_question(engine, context, case["question"], transcript)
    prompt = build_agentar_prompt(context, case["question"], transcript)
    return CaseResult(
        case_id=case["id"],
        expected_kind=case["expected_kind"],
        kind=turn.kind,
        text=turn.text,
        decided_by=turn.decided_by,
        prompt_chars=len(prompt),
        problems=check_turn(case, turn, context.owner_name),
    )


async def run_all(engine, context: AgentarContext, cases: list[dict]) -> list[CaseResult]:
    return [await run_case(engine, context, case) for case in cases]


def retrieval_recall(context: AgentarContext, cases: list[dict]) -> tuple[int, int, list[str]]:
    """For each answer case whose facts live in a file chunk, are the facts in the notes of the
    prompt? Independent of any model. Returns hits, total and the ids of the misses."""
    hits, total, missed = 0, 0, []
    for case in cases:
        sources = case.get("source", [])
        if (
            case["expected_kind"] != "answer"
            or case.get("transcript_tail")
            or not any(s.startswith("atlas-") for s in sources)
        ):
            continue
        prompt = build_agentar_prompt(context, case["question"], transcript_of(case))
        notes = re.search(r"^<notes>\n(.*?)\n</notes>$", prompt, re.DOTALL | re.MULTILINE).group(1)
        total += 1
        if all(_contains_fact(notes, fact) for fact in case["expected_facts"]):
            hits += 1
        else:
            missed.append(case["id"])
    return hits, total, missed
