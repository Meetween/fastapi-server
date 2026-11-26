"""Scaffold behaviour: auth on every new route, 501 bodies, fixtures, error model.

Real tests replace the 501 expectations of a route when it is built; delete its rows below.
"""
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from api.errors import AgentarError, agentar_error_handler, object_id_or_404
from config import settings
from main import app

ID = "0123456789abcdef01234567"

# (method, path, body). Routes that need a user token.
USER_ROUTES = [
    ("get", "/api/me", None),
    ("patch", "/api/me", {"first_name": "A"}),
    ("post", "/api/me/onboarding/skip", None),
    ("get", "/api/me/picture", None),
    ("delete", "/api/me/picture", None),
    ("get", "/api/agentar/voices", None),
    ("get", "/api/agentar/files", None),
    ("delete", f"/api/agentar/files/{ID}", None),
    ("get", f"/api/agentar/files/{ID}/download", None),
    ("get", "/api/agentar/meetings", None),
    ("post", "/api/agentar/meetings", {"room": "weekly", "mode": "now"}),
    ("get", f"/api/agentar/meetings/{ID}", None),
    ("patch", f"/api/agentar/meetings/{ID}", {"title": "t"}),
    ("delete", f"/api/agentar/meetings/{ID}", None),
    ("get", f"/api/agentar/meetings/{ID}/picture", None),
    ("delete", f"/api/agentar/meetings/{ID}/picture", None),
    ("post", f"/api/agentar/meetings/{ID}/dispatch", None),
    ("post", f"/api/agentar/meetings/{ID}/recall", None),
    ("get", f"/api/agentar/meetings/{ID}/report", None),
    ("post", f"/api/agentar/meetings/{ID}/report/regenerate", None),
    ("get", f"/api/agentar/meetings/{ID}/report.md", None),
    ("get", "/api/agentar/notifications", None),
]
PUBLIC_ROUTES = [
    ("get", "/api/auth/config", None),
    ("post", "/api/auth/signup", {"email": "a@b.co", "password": "x" * 10, "first_name": "A", "last_name": "B"}),
    ("get", "/api/agentar/voices/af_heart/sample", None),
]


def _ids(routes):
    return [f"{m.upper()} {p}" for m, p, _ in routes]


@pytest.mark.parametrize(("method", "path", "body"), USER_ROUTES, ids=_ids(USER_ROUTES))
def test_user_routes_need_a_token(test_client: TestClient, method, path, body) -> None:
    response = test_client.request(method, path, json=body, headers={})
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")


@pytest.mark.parametrize(("method", "path", "body"), USER_ROUTES, ids=_ids(USER_ROUTES))
def test_user_routes_are_stubs(test_client: TestClient, user_a, object_store, method, path, body) -> None:
    response = test_client.request(method, path, json=body, headers=user_a.headers)
    assert response.status_code == 501
    assert response.json() == {"detail": "not implemented"}


@pytest.mark.parametrize(("method", "path", "body"), PUBLIC_ROUTES, ids=_ids(PUBLIC_ROUTES))
def test_public_routes_are_stubs_without_a_token(test_client: TestClient, method, path, body) -> None:
    assert test_client.request(method, path, json=body, headers={}).status_code == 501


def test_uploads_are_declared_as_multipart(test_client: TestClient, user_a, object_store) -> None:
    files = {"file": ("a.txt", b"hello", "text/plain")}
    assert test_client.post("/api/agentar/files", files=files, data={"scope": "general"}, headers=user_a.headers).status_code == 501
    assert test_client.put("/api/me/picture", files=files, headers=user_a.headers).status_code == 501
    assert test_client.put(f"/api/agentar/meetings/{ID}/picture", files=files, headers=user_a.headers).status_code == 501


@pytest.mark.parametrize(
    ("method", "path"),
    [("post", "/api/agentar/files"), ("put", "/api/me/picture"), ("put", f"/api/agentar/meetings/{ID}/picture")],
)
def test_chunked_upload_is_411(test_client: TestClient, user_a, object_store, method, path) -> None:
    def body():
        yield b"--x\r\n"
        yield b"--x--\r\n"

    headers = {**user_a.headers, "Content-Type": "multipart/form-data; boundary=x"}
    response = test_client.request(method, path, content=body(), headers=headers)
    assert response.status_code == 411
    assert response.json()["code"] == "length_required"


@pytest.mark.parametrize(
    ("method", "path"),
    [("post", "/api/agentar/files"), ("put", "/api/me/picture"), ("put", "/api/agentar/meetings/{meeting_id}/picture")],
)
def test_upload_routes_document_a_multipart_body(method, path) -> None:
    body = app.openapi()["paths"][path][method]["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert "file" in body["required"]
    assert body["properties"]["file"] == {"type": "string", "format": "binary"}


def test_file_routes_answer_503_while_storage_is_unconfigured(test_client: TestClient, user_a) -> None:
    """No object_store fixture: the default settings have no bucket."""
    response = test_client.get(f"/api/agentar/files/{ID}/download", headers=user_a.headers)
    assert response.status_code == 503
    assert response.json() == {"detail": "Object storage is not configured", "code": "storage_unavailable"}


class TestWorkerEvents:
    BODY = {
        "event_id": "6f1c2f1e-0f3a-4c52-9a43-0a6f4e3d9b11",
        "agentar_meeting_id": ID,
        "bot_id": "bot1",
        "type": "joined",
        "at": "2026-11-10T09:00:00Z",
        "data": {"kind": "joined"},
    }
    WORKER = {"Authorization": "Bearer worker-key-for-tests"}

    @pytest.fixture(autouse=True)
    def _worker_key(self, monkeypatch):
        monkeypatch.setattr(settings, "vp_meetween_api_key", "worker-key-for-tests")

    def test_user_token_is_refused(self, test_client: TestClient, user_a) -> None:
        assert test_client.post("/api/agentar/events", json=self.BODY, headers=user_a.headers).status_code == 401

    def test_worker_key_reaches_the_stub(self, test_client: TestClient) -> None:
        assert test_client.post("/api/agentar/events", json=self.BODY, headers=self.WORKER).status_code == 501

    def test_worker_key_is_checked_even_with_auth_off(self, test_client: TestClient, monkeypatch) -> None:
        monkeypatch.setattr(settings, "enable_auth", False)
        assert test_client.post("/api/agentar/events", json=self.BODY).status_code == 401
        assert test_client.post("/api/agentar/events", json=self.BODY, headers=self.WORKER).status_code == 501

    def test_unset_worker_key_rejects_everything(self, test_client: TestClient, monkeypatch) -> None:
        monkeypatch.setattr(settings, "vp_meetween_api_key", "")
        assert test_client.post("/api/agentar/events", json=self.BODY, headers={"Authorization": "Bearer "}).status_code == 401

    def test_type_and_data_kind_must_agree(self, test_client: TestClient) -> None:
        body = {**self.BODY, "type": "left"}
        assert test_client.post("/api/agentar/events", json=body, headers=self.WORKER).status_code == 422


class TestFixtures:
    def test_two_users_are_distinct_and_both_valid(self, test_client: TestClient, user_a, user_b) -> None:
        assert user_a.email != user_b.email
        assert user_a.id != user_b.id
        for user in (user_a, user_b):
            # 501, not 401: the token resolves to an existing account.
            assert test_client.get("/api/me", headers=user.headers).status_code == 501

    def test_object_store_override_is_removed_after_the_test(self) -> None:
        from api.deps.agentar import get_object_store

        assert get_object_store not in app.dependency_overrides


class TestErrorModel:
    @pytest.fixture
    def mini(self) -> TestClient:
        mini_app = FastAPI()
        mini_app.add_exception_handler(AgentarError, agentar_error_handler)
        router = APIRouter()

        @router.get("/boom")
        async def boom():
            raise AgentarError(429, "rate_limited", "Slow down", headers={"Retry-After": "7"})

        @router.get("/item/{item_id}")
        async def item(item_id: str):
            object_id_or_404(item_id)
            return {"ok": True}

        mini_app.include_router(router)
        return TestClient(mini_app)

    def test_body_has_detail_and_code_and_keeps_headers(self, mini: TestClient) -> None:
        response = mini.get("/boom")
        assert response.status_code == 429
        assert response.json() == {"detail": "Slow down", "code": "rate_limited"}
        assert response.headers["retry-after"] == "7"

    @pytest.mark.parametrize("value", ["short", "z" * 24, "0123456789ABCDEF01234567", "0123456789ABCDEF01234567x", "0123456789abcdef0123456é"])
    def test_malformed_ids_are_404_not_found(self, mini: TestClient, value: str) -> None:
        response = mini.get(f"/item/{value}")
        assert response.status_code == 404
        assert response.json() == {"detail": "Not found", "code": "not_found"}

    def test_a_valid_id_passes(self, mini: TestClient) -> None:
        assert mini.get(f"/item/{ID}").json() == {"ok": True}
