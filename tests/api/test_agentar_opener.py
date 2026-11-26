"""Opener of an Agentar meeting: the `opener` field and the draft route.

The draft is three short spoken sentences built from the purpose and the BM25 top chunks of the
meeting's files. The model is a fake with the engine's `_generate_text` signature. Every figure
and name of the draft must stand in the purpose or the chunks: a sentence with an invented date
or name is removed, and so is a sentence that commits the owner. The draft is never saved.
"""
import asyncio
from unittest.mock import patch

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from config import settings
from src.database.models.agentar_file import AgentarFile, FileChunk
from src.database.models.agentar_meeting import AgentarMeeting
from utils import agentar_opener

BASE = "/api/agentar/meetings"
UNKNOWN_ID = "0123456789abcdef01234567"
PURPOSE = "Present the Atlas status and get answers to my two questions. Do not commit to budget or dates."
STATUS_PAGE = (
    "Atlas pilot, status for the steering call\n"
    "- Go live planned on 1 December 2026 at two sites, Trento and Bonn\n"
    "- Lyon joins in January, once its data processing agreement is signed\n"
    "- Training material is ready in English and Italian\n"
    "- Open risk: the Lyon agreement is still unsigned"
)
UNRELATED = "Canteen menu for the spring: pasta on Mondays, soup on Fridays, coffee machine on floor 3."


class Engine:
    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.prompts: list[str] = []
        self.calls: list[dict] = []

    async def _generate_text(self, **kwargs):
        self.calls.append(kwargs)
        self.prompts.append(kwargs["prompt"])
        if self.error is not None:
            raise self.error
        return self.reply


GOOD = (
    "Atlas goes live on 1 December 2026 in Trento and Bonn. "
    "Lyon joins in January, once its data processing agreement is signed. "
    "Training material is ready in English and Italian."
)


def run(coro):
    return asyncio.run(coro)


def make(client: TestClient, user, **fields):
    body = {"room": "atlas-steering", "mode": "now", "purpose": PURPOSE, **fields}
    return client.post(BASE, json=body, headers=user.headers)


def make_id(client: TestClient, user, **fields) -> str:
    response = make(client, user, **fields)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def add_file(owner: str, meeting_id: str | None, *texts: str, scope: str = "meeting") -> str:
    record = AgentarFile(
        owner=owner,
        scope=scope,
        meeting_id=meeting_id if scope == "meeting" else None,
        kind="document",
        name="atlas.md",
        size=1,
        text_status="ok",
        chunks=[FileChunk(n=i, text=t) for i, t in enumerate(texts, start=1)],
    )
    return run(record.save()).id


def draft(client: TestClient, user, meeting_id: str, engine: Engine | None):
    with patch("utils.agentar_opener.EngineManager.get_lmm", return_value=engine):
        return client.post(f"{BASE}/{meeting_id}/opener-draft", headers=user.headers)


@pytest.fixture(autouse=True)
def rooms(monkeypatch):
    async def claim_room_moderator(room_name: str, user) -> bool:
        return True

    from tests.seams import install

    install(monkeypatch, "utils.room_owner", claim_room_moderator=claim_room_moderator)


# The field ------------------------------------------------------------------------------------


class TestOpenerField:
    def test_create_read_and_patch_keep_the_opener(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a, opener="  Atlas goes live on 1 December 2026.  ")
        read = test_client.get(f"{BASE}/{meeting_id}", headers=user_a.headers).json()
        assert read["opener"] == "Atlas goes live on 1 December 2026."

        patched = test_client.patch(f"{BASE}/{meeting_id}", json={"opener": "Lyon joins in January."}, headers=user_a.headers)
        assert patched.status_code == 200, patched.text
        assert patched.json()["opener"] == "Lyon joins in January."

    def test_default_is_empty_and_an_empty_patch_clears_it(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a)
        assert test_client.get(f"{BASE}/{meeting_id}", headers=user_a.headers).json()["opener"] == ""
        test_client.patch(f"{BASE}/{meeting_id}", json={"opener": "One."}, headers=user_a.headers)
        cleared = test_client.patch(f"{BASE}/{meeting_id}", json={"opener": ""}, headers=user_a.headers)
        assert cleared.json()["opener"] == ""

    def test_opener_is_plain_text_on_one_line(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a, opener="First line.\nSecond\tline.​")
        assert test_client.get(f"{BASE}/{meeting_id}", headers=user_a.headers).json()["opener"] == "First line. Second line."

    def test_over_500_characters_is_422(self, test_client, user_a) -> None:
        assert make(test_client, user_a, opener="a" * 501).status_code == 422
        assert make(test_client, user_a, opener="a" * 500).status_code == 201


# The draft ------------------------------------------------------------------------------------


class TestOpenerDraft:
    def test_draft_from_purpose_and_files_is_not_saved(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a)
        add_file(user_a.email, meeting_id, STATUS_PAGE)
        engine = Engine(GOOD)

        response = draft(test_client, user_a, meeting_id, engine)

        assert response.status_code == 200, response.text
        assert response.json() == {"opener": GOOD}
        stored = run(AgentarMeeting.get_collection().find_one({"_id": ObjectId(meeting_id)}))
        assert stored.get("opener", "") == ""
        assert len(engine.calls) == 1
        assert engine.calls[0]["audio"] is None and engine.calls[0]["do_sample"] is False

    def test_prompt_carries_the_purpose_and_the_best_chunks_only(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a)
        add_file(user_a.email, meeting_id, STATUS_PAGE, UNRELATED)
        engine = Engine(GOOD)
        draft(test_client, user_a, meeting_id, engine)

        prompt = engine.prompts[0]
        assert "Atlas status" in prompt
        assert "1 December 2026" in prompt
        assert "Canteen" not in prompt
        assert "three" in prompt.lower() and "sentences" in prompt.lower()

    def test_prompt_asks_for_the_status_facts_as_a_question_to_answer(self, test_client, user_a) -> None:
        # Measured on SpeechLMM v2 L (K/R1 logs/replay-opener-*.json): a prompt that ends in "Opener:" gets
        # "Good morning, everyone." and nothing else, so the draft fell back to the purpose sentence. The same
        # notes asked as a question ending in "Answer:" get three status facts.
        meeting_id = make_id(test_client, user_a)
        add_file(user_a.email, meeting_id, STATUS_PAGE)
        engine = Engine(GOOD)
        draft(test_client, user_a, meeting_id, engine)

        prompt = engine.prompts[0]
        assert "Opener:" not in prompt
        assert prompt.rstrip().endswith("Answer:")
        question = prompt[prompt.rindex("Question:"):]
        assert "status facts" in question and "three" in question and "No greeting" in question
        assert prompt.index("</notes>") < prompt.index("Question:"), "the question comes after the data"

    def test_prompt_stays_far_under_the_8k_token_budget(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a, purpose="Atlas status " * 70)
        add_file(user_a.email, meeting_id, *[f"Atlas status note {i}. " + "word " * 600 for i in range(40)])
        engine = Engine(GOOD)
        draft(test_client, user_a, meeting_id, engine)

        prompt = engine.prompts[0]
        assert len(prompt) <= agentar_opener.MAX_PROMPT_CHARS
        # About 4 characters per token, with room for the reply.
        assert len(prompt) / 4 + engine.calls[0]["max_new_tokens"] < 8000

    def test_a_hallucinated_date_is_removed(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a)
        add_file(user_a.email, meeting_id, STATUS_PAGE)
        reply = (
            "Atlas goes live on 1 December 2026 in Trento and Bonn. "
            "Lyon signs its agreement on 14 March 2027. "
            "Training material is ready in English and Italian."
        )

        opener = draft(test_client, user_a, meeting_id, Engine(reply)).json()["opener"]

        assert "March" not in opener and "14" not in opener and "2027" not in opener
        assert opener == (
            "Atlas goes live on 1 December 2026 in Trento and Bonn. "
            "Training material is ready in English and Italian."
        )

    def test_an_invented_name_and_a_commitment_are_removed(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a)
        add_file(user_a.email, meeting_id, STATUS_PAGE)
        reply = (
            "Atlas goes live on 1 December 2026 in Trento and Bonn. "
            "Katrin Vogel will run the training. "
            "Alice agrees to approve the extra budget today."
        )

        opener = draft(test_client, user_a, meeting_id, Engine(reply)).json()["opener"]

        assert opener == "Atlas goes live on 1 December 2026 in Trento and Bonn."

    def test_markdown_and_a_fourth_sentence_are_cut(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a)
        add_file(user_a.email, meeting_id, STATUS_PAGE)
        reply = "**Opener:** " + GOOD + " Training material is ready in Italian."

        opener = draft(test_client, user_a, meeting_id, Engine(reply)).json()["opener"]

        assert opener == GOOD
        assert len(opener) <= 500

    def test_nothing_grounded_left_falls_back_to_the_purpose(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a)
        add_file(user_a.email, meeting_id, STATUS_PAGE)

        opener = draft(test_client, user_a, meeting_id, Engine("Everything closes on 9 June 2031.")).json()["opener"]

        assert "2031" not in opener
        assert opener.startswith("The purpose of this meeting, as Alice gave it to me, is: Present the Atlas status")
        assert len(opener) <= 500

    def test_general_files_named_by_the_meeting_are_used(self, test_client, user_a) -> None:
        general = add_file(user_a.email, None, STATUS_PAGE, scope="general")
        meeting_id = make_id(test_client, user_a, file_ids=[general])
        engine = Engine(GOOD)
        draft(test_client, user_a, meeting_id, engine)
        assert "Trento and Bonn" in engine.prompts[0]

    def test_files_of_another_owner_never_reach_the_prompt(self, test_client, user_a, user_b) -> None:
        meeting_id = make_id(test_client, user_a)
        add_file(user_b.email, meeting_id, "Atlas status secret of Bruno: 42 seats.")
        engine = Engine(GOOD)
        draft(test_client, user_a, meeting_id, engine)
        assert "Bruno" not in engine.prompts[0]

    def test_no_purpose_and_no_files_is_422(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a, purpose="")
        engine = Engine(GOOD)
        assert draft(test_client, user_a, meeting_id, engine).status_code == 422
        assert engine.calls == []


class TestOpenerDraftErrors:
    def test_another_owner_and_an_unknown_id_are_404(self, test_client, user_a, user_b) -> None:
        meeting_id = make_id(test_client, user_a)
        engine = Engine(GOOD)
        assert draft(test_client, user_b, meeting_id, engine).status_code == 404
        assert draft(test_client, user_a, UNKNOWN_ID, engine).status_code == 404
        assert draft(test_client, user_a, "not-an-id", engine).status_code == 404
        assert engine.calls == []

    def test_no_token_is_401(self, test_client) -> None:
        assert test_client.post(f"{BASE}/{UNKNOWN_ID}/opener-draft").status_code == 401

    def test_model_not_loaded_is_503(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a)
        response = draft(test_client, user_a, meeting_id, None)
        assert response.status_code == 503
        assert response.json()["code"] == "model_unavailable"

    def test_model_error_is_503(self, test_client, user_a) -> None:
        meeting_id = make_id(test_client, user_a)
        response = draft(test_client, user_a, meeting_id, Engine(error=RuntimeError("Triton 500")))
        assert response.status_code == 503
        assert response.json()["code"] == "model_unavailable"
        assert "Triton" not in response.text

    def test_model_timeout_is_503(self, test_client, user_a, monkeypatch) -> None:
        monkeypatch.setattr(agentar_opener, "CALL_TIMEOUT_S", 0.05)

        class Slow(Engine):
            async def _generate_text(self, **kwargs):
                await asyncio.sleep(1)
                return GOOD

        meeting_id = make_id(test_client, user_a)
        assert draft(test_client, user_a, meeting_id, Slow()).status_code == 503

    def test_over_the_limit_is_429_with_retry_after(self, test_client, user_a, monkeypatch) -> None:
        monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update={"opener_drafts_per_minute": 2}))
        meeting_id = make_id(test_client, user_a)
        engine = Engine(GOOD)
        assert draft(test_client, user_a, meeting_id, engine).status_code == 200
        assert draft(test_client, user_a, meeting_id, engine).status_code == 200

        refused = draft(test_client, user_a, meeting_id, engine)

        assert refused.status_code == 429
        assert refused.json()["code"] == "rate_limited"
        assert int(refused.headers["Retry-After"]) >= 1
        assert len(engine.calls) == 2

    def test_the_limit_is_per_owner(self, test_client, user_a, user_b, monkeypatch) -> None:
        monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update={"opener_drafts_per_minute": 1}))
        a = make_id(test_client, user_a)
        b = make_id(test_client, user_b, room="atlas-other")
        engine = Engine(GOOD)
        assert draft(test_client, user_a, a, engine).status_code == 200
        assert draft(test_client, user_b, b, engine).status_code == 200
        assert draft(test_client, user_a, a, engine).status_code == 429


def test_default_limit_is_six_per_minute() -> None:
    from config import Settings

    assert Settings().agentar.opener_drafts_per_minute == 6
