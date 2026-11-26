"""Scaffold behaviour: auth on every Agentar route, upload framing, fixtures, error model."""
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from api.errors import AgentarError, agentar_error_handler, object_id_or_404
from main import app

ID = "0123456789abcdef01234567"

# Routes of the contract that do not take a user token: the worker key, and a static sample.
NOT_USER_ROUTES = {("POST", "/api/agentar/events"), ("GET", "/api/agentar/voices/{voice_id}/sample")}


def _user_routes() -> list[tuple[str, str]]:
    """Every /api/me and /api/agentar operation of the app, so a new route is covered by default."""
    found = []
    for path, operations in app.openapi()["paths"].items():
        if not (path == "/api/me" or path.startswith(("/api/me/", "/api/agentar"))):
            continue
        for method in operations:
            if (method.upper(), path) not in NOT_USER_ROUTES:
                found.append((method.upper(), path))
    return found


USER_ROUTES = _user_routes()


def test_the_route_table_has_the_agentar_routes() -> None:
    assert len(USER_ROUTES) >= 20


@pytest.mark.parametrize(("method", "path"), USER_ROUTES, ids=[f"{m} {p}" for m, p in USER_ROUTES])
def test_user_routes_need_a_token(test_client: TestClient, method, path) -> None:
    url = path.replace("{meeting_id}", ID).replace("{file_id}", ID).replace("{voice_id}", "af_heart")
    response = test_client.request(method, url, json={}, headers={})
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")


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


class TestFixtures:
    def test_two_users_are_distinct_and_both_valid(self, test_client: TestClient, user_a, user_b) -> None:
        assert user_a.email != user_b.email
        assert user_a.id != user_b.id
        for user in (user_a, user_b):
            # 200, not 401: the token resolves to an existing account.
            assert test_client.get("/api/me", headers=user.headers).status_code == 200

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
