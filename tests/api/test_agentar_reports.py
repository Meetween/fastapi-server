"""Report routes: read, regenerate, Markdown download. Owner scoped, 404 across owners, 409 states."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from api.routes import agentar_reports as routes
from main import app
from src.database.models.agentar_meeting import AgentarMeeting
from src.database.models.agentar_report import AgentarReport
from src.schemas.agentar import Report
from tests.utils.agentar_report_helpers import fake_for, seed_meeting
from utils import agentar_report as ar
from utils.object_store import MemoryObjectStore

UNKNOWN = "0123456789abcdef01234567"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def store():
    memory = MemoryObjectStore()
    app.dependency_overrides[routes._optional_store] = lambda: memory
    yield memory
    app.dependency_overrides.pop(routes._optional_store, None)


@pytest.fixture
def model(monkeypatch):
    fake = fake_for("t1")
    monkeypatch.setattr(ar, "_default_model", lambda: fake)
    return fake


def ready_meeting(owner, name="t1"):
    meeting = seed_meeting(name, owner=owner.email)
    report = run(ar.generate_report(meeting.id, model=fake_for(name)))
    assert report.status == "ready"
    return meeting, report


def urls(meeting_id):
    base = f"/api/agentar/meetings/{meeting_id}"
    return {"get": base + "/report", "regen": base + "/report/regenerate", "md": base + "/report.md"}


def set_report(meeting_id, **fields):
    run(AgentarReport.get_collection().update_one({"meeting_id": meeting_id}, {"$set": fields}))


class TestGetReport:
    def test_ready_report_has_the_contract_shape(self, test_client: TestClient, user_a) -> None:
        meeting, report = ready_meeting(user_a)
        response = test_client.get(urls(meeting.id)["get"], headers=user_a.headers)
        assert response.status_code == 200
        body = response.json()
        parsed = Report.model_validate(body)  # the contract model accepts it
        assert (body["meeting_id"], body["status"], body["error"]) == (meeting.id, "ready", None)
        assert (body["room_name"], body["title"]) == ("atlas-steering", "Atlas steering")
        assert body["generated_at"] and isinstance(body["generation_ms"], int)
        assert "Lyon will sign" in body["summary"]
        assert [a["status"] for a in body["answers"]] == ["answered", "answered"]
        assert body["answers"][0]["transcript_refs"] == [7]
        assert set(body["action_items"][0]) == {"text", "owner", "due", "quote", "transcript_refs"}
        for item in body["action_items"]:
            cited = body["transcript"][item["transcript_refs"][0] - 1]["text"]
            assert item["quote"] and item["quote"].casefold() in cited.casefold()
        # The owner of this account is Alice, so "Laura" in the transcript is not the meeting owner.
        assert [p["kind"] for p in body["open_points"]] == ["escalation"]
        assert len(parsed.transcript) == 18 and body["transcript"][5]["source"] == "agentar"
        assert "owner" not in body and "md_key" not in body

    def test_pending_and_failed_are_200_with_their_status(self, test_client: TestClient, user_a) -> None:
        meeting, _ = ready_meeting(user_a)
        set_report(meeting.id, status="pending")
        assert test_client.get(urls(meeting.id)["get"], headers=user_a.headers).json()["status"] == "pending"
        set_report(meeting.id, status="failed", error="report_timeout")
        body = test_client.get(urls(meeting.id)["get"], headers=user_a.headers).json()
        assert (body["status"], body["error"]) == ("failed", "report_timeout")

    def test_an_ended_meeting_without_a_record_yet_reads_as_pending(self, test_client: TestClient, user_a) -> None:
        meeting = seed_meeting("t1", owner=user_a.email)
        response = test_client.get(urls(meeting.id)["get"], headers=user_a.headers)
        assert response.status_code == 200
        body = response.json()
        assert (body["status"], body["summary"], body["action_items"]) == ("pending", None, [])
        assert body["room_name"] == "atlas-steering" and body["ended_at"]

    def test_a_meeting_that_never_ended_has_no_report(self, test_client: TestClient, user_a) -> None:
        meeting = seed_meeting("t1", owner=user_a.email, status="live")
        assert test_client.get(urls(meeting.id)["get"], headers=user_a.headers).status_code == 404

    @pytest.mark.parametrize("bad_id", [UNKNOWN, "nope", "0123456789ABCDEF01234567", "../x"])
    def test_unknown_and_malformed_ids_are_404(self, test_client: TestClient, user_a, bad_id) -> None:
        for route in ("get", "md"):
            response = test_client.get(urls(bad_id)[route], headers=user_a.headers)
            assert response.status_code == 404
        assert test_client.post(urls(bad_id)["regen"], headers=user_a.headers).status_code == 404

    def test_404_across_owners_on_every_route(self, test_client: TestClient, user_a, user_b, model, store) -> None:
        meeting, _ = ready_meeting(user_a)
        u = urls(meeting.id)
        for response in (
            test_client.get(u["get"], headers=user_b.headers),
            test_client.get(u["md"], headers=user_b.headers),
            test_client.post(u["regen"], headers=user_b.headers),
        ):
            assert response.status_code == 404
            assert response.json() == {"detail": "Not found", "code": "not_found"}
        assert model.prompts == []  # the foreign regenerate started nothing
        assert test_client.get(u["get"], headers=user_a.headers).status_code == 200

    def test_a_report_row_of_another_owner_is_never_returned(self, test_client: TestClient, user_a, user_b) -> None:
        meeting, _ = ready_meeting(user_a)
        set_report(meeting.id, owner=user_b.email)
        # The meeting is A's, the report row says B: A gets a pending view, never B's data.
        body = test_client.get(urls(meeting.id)["get"], headers=user_a.headers).json()
        assert body["summary"] is None

    def test_token_is_required(self, test_client: TestClient, user_a) -> None:
        meeting, _ = ready_meeting(user_a)
        for response in (
            test_client.get(urls(meeting.id)["get"], headers={}),
            test_client.get(urls(meeting.id)["md"], headers={}),
            test_client.post(urls(meeting.id)["regen"], headers={}),
        ):
            assert response.status_code == 401


class TestDownload:
    def test_markdown_attachment(self, test_client: TestClient, user_a) -> None:
        meeting, report = ready_meeting(user_a)
        response = test_client.get(urls(meeting.id)["md"], headers=user_a.headers)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert response.headers["x-content-type-options"] == "nosniff"
        disposition = response.headers["content-disposition"]
        assert disposition.startswith('attachment; filename="agentar-report-atlas-steering-')
        assert disposition.endswith('.md"')
        assert response.text == ar.render_markdown(report.to_dict())
        assert "## Answers to the script questions" in response.text

    def test_rendered_from_the_record_not_from_the_store(self, test_client: TestClient, user_a, store) -> None:
        meeting = seed_meeting("t1", owner=user_a.email)
        report = run(ar.generate_report(meeting.id, model=fake_for("t1"), store=store))
        assert report.md_key in store.keys()
        run(store.delete(report.md_key))
        assert test_client.get(urls(meeting.id)["md"], headers=user_a.headers).status_code == 200

    @pytest.mark.parametrize("status", ["pending", "failed"])
    def test_not_ready_is_409(self, test_client: TestClient, user_a, status) -> None:
        meeting, _ = ready_meeting(user_a)
        set_report(meeting.id, status=status)
        response = test_client.get(urls(meeting.id)["md"], headers=user_a.headers)
        assert response.status_code == 409
        assert response.json()["code"] == "report_not_ready"

    def test_ended_without_a_record_is_409_and_not_ended_is_404(self, test_client: TestClient, user_a) -> None:
        ended = seed_meeting("t1", owner=user_a.email)
        assert test_client.get(urls(ended.id)["md"], headers=user_a.headers).json()["code"] == "report_not_ready"
        live = seed_meeting("t1", owner=user_a.email, status="live")
        assert test_client.get(urls(live.id)["md"], headers=user_a.headers).status_code == 404


class TestRegenerate:
    def test_queues_the_job_and_it_runs_on_the_stored_transcript(
        self, test_client: TestClient, user_a, model, store
    ) -> None:
        meeting, first = ready_meeting(user_a)
        model.summary = "Sara Neri will send the invitations."
        response = test_client.post(urls(meeting.id)["regen"], headers=user_a.headers)
        assert response.status_code == 202
        body = response.json()
        assert (body["status"], body["meeting_id"]) == ("pending", meeting.id)
        assert body["generated_at"] is None and len(body["transcript"]) == 18

        # The background job ran after the response, on the transcript kept in the report.
        after = test_client.get(urls(meeting.id)["get"], headers=user_a.headers).json()
        assert (after["status"], after["summary"]) == ("ready", "Sara Neri will send the invitations.")
        assert run(AgentarReport.get_collection().count_documents({})) == 1
        stored = AgentarReport.from_dict(run(AgentarReport.get_collection().find_one({})))
        assert stored.id == first.id and stored.md_key in store.keys()
        assert run(AgentarMeeting.find_by_id(meeting.id)).report_status == "ready"

    def test_a_failed_regeneration_is_visible(self, test_client: TestClient, user_a, monkeypatch, store) -> None:
        meeting, _ = ready_meeting(user_a)
        monkeypatch.setattr(ar, "_default_model", lambda: _broken)
        assert test_client.post(urls(meeting.id)["regen"], headers=user_a.headers).status_code == 202
        body = test_client.get(urls(meeting.id)["get"], headers=user_a.headers).json()
        assert (body["status"], body["error"]) == ("failed", "model_unavailable")
        # Failed can be regenerated again.
        monkeypatch.setattr(ar, "_default_model", lambda: fake_for("t1"))
        assert test_client.post(urls(meeting.id)["regen"], headers=user_a.headers).status_code == 202
        assert test_client.get(urls(meeting.id)["get"], headers=user_a.headers).json()["status"] == "ready"

    def test_409_while_generating(self, test_client: TestClient, user_a, model) -> None:
        meeting, _ = ready_meeting(user_a)
        set_report(meeting.id, status="pending", timestamp=ar.clock.utcnow())
        response = test_client.post(urls(meeting.id)["regen"], headers=user_a.headers)
        assert response.status_code == 409
        assert response.json() == {"detail": "The report is being generated", "code": "invalid_state"}
        assert model.prompts == []

    def test_409_when_the_meeting_has_not_ended(self, test_client: TestClient, user_a, model) -> None:
        meeting = seed_meeting("t1", owner=user_a.email, status="live")
        response = test_client.post(urls(meeting.id)["regen"], headers=user_a.headers)
        assert (response.status_code, response.json()["code"]) == (409, "invalid_state")

    def test_first_report_of_an_ended_meeting_can_be_started_here(
        self, test_client: TestClient, user_a, model, store
    ) -> None:
        meeting = seed_meeting("t1", owner=user_a.email)  # snapshot only, no record yet
        assert test_client.post(urls(meeting.id)["regen"], headers=user_a.headers).status_code == 202
        body = test_client.get(urls(meeting.id)["get"], headers=user_a.headers).json()
        assert body["status"] == "ready" and len(body["transcript"]) == 18

    def test_works_without_an_object_store(self, test_client: TestClient, user_a, model, monkeypatch) -> None:
        monkeypatch.setattr(routes, "get_configured_object_store", lambda cfg: None)
        meeting, _ = ready_meeting(user_a)
        assert test_client.post(urls(meeting.id)["regen"], headers=user_a.headers).status_code == 202
        assert test_client.get(urls(meeting.id)["get"], headers=user_a.headers).json()["status"] == "ready"


async def _broken(prompt: str, max_new_tokens: int) -> str:
    raise RuntimeError("Triton inference failed")
