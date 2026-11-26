"""read_single_upload: bounded, one file part, a few small fields."""
import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from api.deps.agentar import require_content_length
from api.errors import AgentarError, agentar_error_handler
from utils.uploads import read_single_upload

MAX = 100


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.add_exception_handler(AgentarError, agentar_error_handler)

    @app.post("/up", dependencies=[Depends(require_content_length)])
    async def up(request: Request):
        got = await read_single_upload(request, max_bytes=MAX, fields=("scope", "meeting_id"))
        return {"size": len(got.data), "name": got.filename, "type": got.content_type, "fields": dict(got.fields)}

    return TestClient(app)


def test_one_file_and_fields(client: TestClient) -> None:
    response = client.post(
        "/up", files={"file": ("../../evil.txt", b"hello", "text/plain")}, data={"scope": "general"}
    )
    assert response.status_code == 200
    assert response.json() == {"size": 5, "name": "../../evil.txt", "type": "text/plain", "fields": {"scope": "general"}}


def test_exactly_the_cap_is_accepted(client: TestClient) -> None:
    assert client.post("/up", files={"file": ("a", b"x" * MAX)}).json()["size"] == MAX


def test_one_byte_over_the_cap_is_413(client: TestClient) -> None:
    response = client.post("/up", files={"file": ("a", b"x" * (MAX + 1))})
    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"


def test_declared_length_far_over_the_cap_is_refused_before_parsing(client: TestClient) -> None:
    response = client.post(
        "/up", content=b"x" * 200_000, headers={"Content-Type": "multipart/form-data; boundary=zz"}
    )
    assert response.status_code == 413


def test_chunked_body_is_411(client: TestClient) -> None:
    def body():
        yield b"--x--\r\n"

    response = client.post("/up", content=body(), headers={"Content-Type": "multipart/form-data; boundary=x"})
    assert response.status_code == 411
    assert response.json()["code"] == "length_required"


def test_two_files_are_refused(client: TestClient) -> None:
    response = client.post("/up", files=[("file", ("a", b"1")), ("file", ("b", b"2"))])
    assert response.status_code == 400


def test_missing_file_part_is_422(client: TestClient) -> None:
    assert client.post("/up", data={"scope": "general"}).status_code == 422


def test_file_under_another_name_is_422(client: TestClient) -> None:
    assert client.post("/up", files={"upload": ("a", b"1")}).status_code == 422


def test_unexpected_field_is_422(client: TestClient) -> None:
    response = client.post("/up", files={"file": ("a", b"1")}, data={"role": "admin"})
    assert response.status_code == 422


def test_oversized_text_field_is_refused(client: TestClient) -> None:
    response = client.post("/up", files={"file": ("a", b"1")}, data={"scope": "x" * 2000})
    assert response.status_code in {400, 413, 422}
