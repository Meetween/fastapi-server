"""The app declares every path and method of the Agentar contract.

`tests/contract/openapi.yaml` is a copy of the Agentar API contract. It is
checked against the app's own OpenAPI document, so the test does not depend on how FastAPI
nests routers internally.
"""
from pathlib import Path

import pytest
import yaml

from main import app

CONTRACT = yaml.safe_load((Path(__file__).parent / "openapi.yaml").read_text(encoding="utf-8"))
HTTP_METHODS = {"get", "put", "post", "delete", "patch"}

def _is_new_route(path: str) -> bool:
    return path.startswith(("/api/agentar", "/api/me")) or path in {"/api/auth/config", "/api/auth/signup"}


OPERATIONS = sorted(
    (path, method)
    for path, item in CONTRACT["paths"].items()
    for method in item
    if method in HTTP_METHODS
)


@pytest.fixture(scope="module")
def app_paths() -> dict:
    return app.openapi()["paths"]


def test_contract_copy_has_the_expected_size() -> None:
    assert len(OPERATIONS) == 37


@pytest.mark.parametrize(("path", "method"), OPERATIONS, ids=[f"{m.upper()} {p}" for p, m in OPERATIONS])
def test_operation_is_declared(app_paths: dict, path: str, method: str) -> None:
    assert path in app_paths, f"{path} is not served"
    assert method in app_paths[path], f"{method.upper()} {path} is not served"


@pytest.mark.parametrize(("path", "method"), OPERATIONS, ids=[f"{m.upper()} {p}" for p, m in OPERATIONS])
def test_documented_statuses_match(app_paths: dict, path: str, method: str) -> None:
    """Every status of the contract is documented by the app, for the new routes.

    422 is left out: FastAPI adds it by itself where a route has parameters or a body.
    Existing routes (Jitsi token, login, bots) change when their own work lands.
    """
    if not _is_new_route(path):
        pytest.skip("existing route, changed separately")
    wanted = {str(code) for code in CONTRACT["paths"][path][method]["responses"]} - {"422"}
    served = set(app_paths[path][method]["responses"])
    assert wanted <= served, f"{method.upper()} {path}: missing {sorted(wanted - served)}"


# Contract schema name -> pydantic model. The property names must be identical.
def _schema_models() -> dict:
    from src.schemas import agentar as a
    from src.schemas import auth as au

    return {
        "AuthConfig": au.AuthConfig,
        "SignupRequest": au.SignupRequest,
        "Error": a.Error,
        "PlainError": a.PlainError,
        "Profile": a.Profile,
        "Me": a.Me,
        "MePatch": a.MePatch,
        "Picture": a.Picture,
        "VoiceInfo": a.VoiceInfo,
        "AgentarFile": a.AgentarFile,
        "Usage": a.Usage,
        "FileList": a.FileList,
        "Question": a.Question,
        "Meeting": a.Meeting,
        "MeetingList": a.MeetingList,
        "WorkerEvent": a.WorkerEvent,
        "EventJoined": a.EventJoined,
        "EventQuestionAsked": a.EventQuestionAsked,
        "EventAnswerGiven": a.EventAnswerGiven,
        "EventEscalation": a.EventEscalation,
        "EventLeft": a.EventLeft,
        "EventError": a.EventError,
        "EventAck": a.EventAck,
        "Notification": a.Notification,
        "NotificationList": a.NotificationList,
        "TranscriptLine": a.TranscriptLine,
        "ActionItem": a.ActionItem,
        "Answer": a.Answer,
        "OpenPoint": a.OpenPoint,
        "Report": a.Report,
    }


SCHEMA_MODELS = _schema_models()


@pytest.mark.parametrize("name", sorted(SCHEMA_MODELS))
def test_schema_properties_match_the_contract(name: str) -> None:
    contract = set(CONTRACT["components"]["schemas"][name]["properties"])
    assert set(SCHEMA_MODELS[name].model_fields) == contract


def test_meeting_create_and_patch_share_the_contract_field_set() -> None:
    from src.schemas.agentar import MeetingCreate, MeetingPatch

    contract = set(CONTRACT["components"]["schemas"]["MeetingFields"]["properties"])
    assert set(MeetingCreate.model_fields) == contract
    assert set(MeetingPatch.model_fields) == contract


def test_enums_match_the_contract() -> None:
    from typing import get_args

    from src.schemas import agentar as a

    schemas = CONTRACT["components"]["schemas"]
    assert set(get_args(a.ErrorCode)) == set(schemas["ErrorCode"]["enum"])
    assert set(get_args(a.MeetingStatus)) == set(schemas["MeetingStatus"]["enum"])
    assert set(get_args(a.VoiceId)) == set(schemas["VoiceId"]["enum"])
    assert set(get_args(a.EventType)) == set(schemas["EventType"]["enum"])
    assert set(get_args(a.NotificationType)) == set(schemas["NotificationType"]["enum"])
    assert set(get_args(a.FileContentType)) == set(schemas["AgentarFile"]["properties"]["content_type"]["enum"])
