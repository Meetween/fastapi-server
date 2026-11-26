"""read_single_upload: bounded, one file part, a few small fields."""
import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from api.deps.agentar import require_content_length
from api.errors import AgentarError, agentar_error_handler
from utils.text_extract import MARKDOWN, PDF, TEXT
from utils.uploads import clean_filename, content_disposition, read_single_upload, sniff_document

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


def test_two_files_are_422_not_the_parsers_400(client: TestClient) -> None:
    """Starlette answers 400 for a second file part. The contract lists no 400: it is 422 in the
    FastAPI validation shape, like every other form problem."""
    response = client.post("/up", files=[("file", ("a", b"1")), ("file", ("b", b"2"))])
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert detail[0]["loc"] == ["body", "file"]
    assert "Too many files" in detail[0]["msg"]


def test_two_parts_named_file_with_one_a_text_field_are_422(client: TestClient) -> None:
    response = client.post("/up", files={"file": ("a", b"1")}, data={"file": "text"})
    assert response.status_code == 422


def test_too_many_fields_are_422(client: TestClient) -> None:
    response = client.post(
        "/up", files={"file": ("a", b"1")}, data={"scope": "a", "meeting_id": "b", "x": "c", "y": "d"}
    )
    assert response.status_code == 422


def test_malformed_multipart_is_422(client: TestClient) -> None:
    response = client.post(
        "/up", content=b"not a form at all", headers={"Content-Type": "multipart/form-data"}
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_json_body_has_no_file_part(client: TestClient) -> None:
    assert client.post("/up", json={"file": "x"}).status_code == 422


def test_missing_file_part_is_422(client: TestClient) -> None:
    assert client.post("/up", data={"scope": "general"}).status_code == 422


def test_file_under_another_name_is_422(client: TestClient) -> None:
    assert client.post("/up", files={"upload": ("a", b"1")}).status_code == 422


def test_unexpected_field_is_422(client: TestClient) -> None:
    response = client.post("/up", files={"file": ("a", b"1")}, data={"role": "admin"})
    assert response.status_code == 422


def test_oversized_text_field_is_refused(client: TestClient) -> None:
    response = client.post("/up", files={"file": ("a", b"1")}, data={"scope": "x" * 2000})
    assert response.status_code == 422


class TestSniff:
    def test_pdf_by_magic_bytes(self) -> None:
        assert sniff_document(b"%PDF-1.7\n...", "x.txt") == PDF

    def test_text_and_markdown_by_extension_only(self) -> None:
        assert sniff_document("caf\u00e9\n".encode(), "notes.txt") == TEXT
        assert sniff_document(b"# t", "README.MD") == MARKDOWN
        assert sniff_document(b"# t", "a.markdown") == MARKDOWN
        assert sniff_document(b"plain", None) == TEXT

    def test_a_text_that_starts_with_pk_is_text(self) -> None:
        assert sniff_document(b"PKI notes", "pki.txt") == TEXT

    @pytest.mark.parametrize(
        "data",
        [
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 8,
            b"MZ\x90\x00\x03",
            "hi".encode("utf-16"),
            b"text\x00with a nul",
            b"\xff\xfe\xfa invalid utf8 \xc3\x28",
            b"PK\x03\x04 not really a zip",
        ],
    )
    def test_everything_else_is_415(self, data: bytes) -> None:
        with pytest.raises(AgentarError) as caught:
            sniff_document(data, "file.txt")
        assert caught.value.status_code == 415
        assert caught.value.code == "unsupported_type"


class TestFilenames:
    @pytest.mark.parametrize(
        ("raw", "clean"),
        [
            ("../../etc/passwd", "passwd"),
            ("C:\\Users\\x\\notes.txt", "notes.txt"),
            ("a\r\nX-Evil: 1.txt", "aX-Evil: 1.txt"),
            ("re\u202etxt.exe", "retxt.exe"),
            ("zero\u200bwidth.md", "zerowidth.md"),
            ("", "file"),
            (None, "file"),
            ("...", "file"),
            ("  spaced.txt  ", "spaced.txt"),
            ("cafe\u0301.txt", "caf\u00e9.txt"),
        ],
    )
    def test_clean(self, raw, clean) -> None:
        assert clean_filename(raw) == clean

    def test_long_names_keep_the_extension(self) -> None:
        name = clean_filename("n" * 400 + ".docx")
        assert len(name) == 255
        assert name.endswith(".docx")

    def test_disposition_is_one_ascii_header_value(self) -> None:
        value = content_disposition('r\u00e9sum\u00e9 "final".pdf')
        assert value.startswith("attachment; ")
        assert value.isascii()
        assert "\r" not in value and "\n" not in value
        assert "filename*=UTF-8''r%C3%A9sum%C3%A9%20%22final%22.pdf" in value
        assert 'filename="r_sum_ _final_.pdf"' in value
