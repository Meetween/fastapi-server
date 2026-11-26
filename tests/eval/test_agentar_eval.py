"""The Atlas cases through the Agentar pipeline with a fake model: structure, not model quality."""
import asyncio
import re

import pytest

from tests.eval import agentar_eval as ev
from utils.agentar_prompt import MAX_PROMPT_CHARS, build_agentar_prompt, escalation_line

pytestmark = pytest.mark.skipif(not ev.data_available(), reason="eval data folder absent")

CASES = ev.load_cases() if ev.data_available() else []
OWNER = "Laura Bianchi"


@pytest.fixture(scope="module")
def context():
    return ev.make_context(600)


def test_the_data_holds_the_24_cases():
    assert [c["id"] for c in CASES] == [f"c{i:02d}" for i in range(1, 25)]


# c15 needs the meaning of "next call" against "meet again", which a word overlap reader lacks. The
# case is covered by the retrieval check and by the real model run.
FAKE_CANNOT = {"c15"}


@pytest.mark.parametrize("chunk_chars", [600, 300])
def test_every_case_passes_with_the_extractive_reader(chunk_chars):
    context = ev.make_context(chunk_chars)
    reader = ev.ExtractiveReader()
    results = asyncio.run(ev.run_all(reader, context, [c for c in CASES if c["id"] not in FAKE_CANNOT]))
    failures = {r.case_id: r.problems for r in results if not r.ok}
    assert failures == {}


def test_commitment_cases_and_who_are_you_never_reach_the_model(context):
    reader = ev.ExtractiveReader()
    by_id = {c["id"]: c for c in CASES}
    for case_id in ("c12", "c17", "c18", "c19", "c20"):
        result = asyncio.run(ev.run_case(reader, context, by_id[case_id]))
        assert result.ok, (case_id, result.problems)
        assert result.decided_by != "model"
    assert reader.calls == 0


def test_the_model_sees_the_fixed_line_for_the_other_deferrals(context):
    reader = ev.ExtractiveReader()
    by_id = {c["id"]: c for c in CASES}
    for case_id in ("c21", "c22", "c23", "c24"):
        result = asyncio.run(ev.run_case(reader, context, by_id[case_id]))
        assert result.kind == "escalate", (case_id, result.text)
        assert result.text == escalation_line(OWNER)
    assert reader.calls == 4
    assert all(escalation_line(OWNER) in prompt for prompt in reader.prompts)


@pytest.mark.parametrize("chunk_chars", [600, 300, 150])
def test_the_facts_of_file_cases_reach_the_prompt(chunk_chars):
    context = ev.make_context(chunk_chars)
    hits, total, missed = ev.retrieval_recall(context, CASES)
    assert total == 10
    assert hits == total, f"chunk size {chunk_chars}, missed {missed}"


def test_the_injection_case_gets_no_notes(context):
    case = next(c for c in CASES if c["id"] == "c24")
    prompt = build_agentar_prompt(context, case["question"], None)
    assert re.search(r"^<notes>\n\(none\)\n</notes>", prompt, re.MULTILINE)


def test_no_case_prompt_passes_the_length_cap(context):
    for case in CASES:
        prompt = build_agentar_prompt(context, case["question"], ev.transcript_of(case))
        assert len(prompt) <= MAX_PROMPT_CHARS
