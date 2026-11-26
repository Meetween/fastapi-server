"""Bounded reading of one uploaded file, shared by the file, profile picture and meeting picture routes.

Why not `UploadFile` parameters: FastAPI parses the body before it runs dependencies, so a chunked
body would be spooled before any size check. These routes declare no body parameters. The
`require_content_length` dependency runs first (411), then `read_single_upload` bounds the form.

Order of checks: Content-Length over `max_bytes` + `OVERHEAD_BYTES` answers 413 before parsing;
the form allows one file and three small fields; the file read is capped at `max_bytes`.

Every form problem (a second file part, too many fields, an oversized field, a malformed body, a
missing file, an unknown field) answers 422 in the FastAPI validation shape. Starlette reports
the parser's own complaints as 400, a status the contract does not list; they are mapped to 422.

`sniff_document` decides the type of a knowledge base file from its bytes, never from the client's
content type: `%PDF-` is a pdf, a zip that passes the docx pre check is a docx, anything that decodes
as UTF-8 without a NUL byte is text. Only the extension tells `.md` from `.txt`.
"""
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.errors import AgentarError
from utils.text_extract import DOCX, MARKDOWN, PDF, TEXT, DocxArchiveError, check_docx_archive

logger = logging.getLogger(__name__)

# Multipart framing and the small text fields (scope, meeting_id) on top of the file bytes.
OVERHEAD_BYTES = 64 * 1024
MAX_FIELDS = 3
MAX_FIELD_BYTES = 1024


@dataclass(frozen=True)
class UploadedFile:
    data: bytes
    filename: str | None
    content_type: str | None
    fields: Mapping[str, str] = field(default_factory=dict)


def _too_large(max_bytes: int) -> AgentarError:
    return AgentarError(413, "file_too_large", f"The file is larger than {max_bytes} bytes")


def invalid_form(field_name: str, message: str, kind: str = "value_error") -> RequestValidationError:
    """A 422 in the default FastAPI validation body (`detail` is an array), for a form problem."""
    return RequestValidationError([{"type": kind, "loc": ("body", field_name), "msg": message, "input": None}])


async def read_single_upload(
    request: Request, *, max_bytes: int, fields: Sequence[str] = ()
) -> UploadedFile:
    """Read the `file` part and the named text fields, within the caps.

    Raises 413 `file_too_large`; 422 for too many parts, a malformed body, a missing or repeated
    file part, or an unexpected field. The filename is returned as sent; it is display metadata
    and must never reach an object key (see `clean_filename`).
    """
    declared = int(request.headers.get("content-length", "0") or 0)
    if declared > max_bytes + OVERHEAD_BYTES:
        raise _too_large(max_bytes)

    try:
        async with request.form(max_files=1, max_fields=MAX_FIELDS, max_part_size=MAX_FIELD_BYTES) as form:
            parts = form.getlist("file")
            if len(parts) != 1 or not isinstance(parts[0], UploadFile):
                raise invalid_form("file", "Exactly one file part named file is required", "missing")
            upload = parts[0]
            values: dict[str, str] = {}
            for key, value in form.multi_items():
                if key == "file":
                    continue
                if key not in fields or not isinstance(value, str):
                    raise invalid_form(key, f"Unexpected form field {key}", "extra_forbidden")
                values[key] = value
            data = await upload.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise _too_large(max_bytes)
            return UploadedFile(
                data=data, filename=upload.filename, content_type=upload.content_type, fields=values
            )
    except StarletteHTTPException as exc:
        if exc.status_code == 400:
            raise invalid_form("file", str(exc.detail), "multipart_invalid") from exc
        raise
    except AssertionError as exc:  # python-multipart missing: a deployment error, not a client one
        raise HTTPException(status_code=503, detail="Multipart parsing is not available") from exc


_UNSUPPORTED = "Only txt, md, pdf and docx files are accepted"


def sniff_document(data: bytes, filename: str | None) -> str:
    """Content type of a knowledge base file from its bytes. Raises 415 `unsupported_type`.

    The client's Content-Type header is ignored. A zip that is not a docx (no `word/document.xml`,
    too many entries, a bomb) is refused, `PK` alone is not a docx.
    """
    refuse = AgentarError(415, "unsupported_type", _UNSUPPORTED)
    if data.startswith(b"%PDF-"):
        return PDF
    if data.startswith(b"PK\x03\x04"):
        try:
            check_docx_archive(data)
        except DocxArchiveError as exc:
            logger.info("Upload refused as docx: %s", exc)
            raise refuse from exc
        return DOCX
    if b"\x00" in data:
        raise refuse
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise refuse from exc
    name = (filename or "").lower()
    return MARKDOWN if name.endswith((".md", ".markdown")) else TEXT


_SEPARATORS = re.compile(r"[\\/]")
_DROPPED_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"}
MAX_NAME_CHARS = 255


def clean_filename(name: str | None) -> str:
    """Display name: no path, no control, format (bidi, zero width) or private use characters, at most
    255 characters with the extension kept, `file` when nothing is left. Never used in a key."""
    text = unicodedata.normalize("NFC", name or "")
    text = _SEPARATORS.split(text)[-1]
    text = "".join(ch for ch in text if unicodedata.category(ch) not in _DROPPED_CATEGORIES).strip()
    if not text.strip("."):
        return "file"
    if len(text) > MAX_NAME_CHARS:
        stem, dot, ext = text.rpartition(".")
        if dot and 0 < len(ext) <= 12:
            text = stem[: MAX_NAME_CHARS - len(ext) - 1] + "." + ext
        else:
            text = text[:MAX_NAME_CHARS]
    return text


def content_disposition(name: str) -> str:
    """`attachment` with an ASCII fallback and the RFC 5987 UTF-8 form of the display name."""
    fallback = re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:100] or "file"
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(name, safe='')}"


def multipart_openapi(properties: dict[str, Any], required: Sequence[str]) -> dict[str, Any]:
    """`openapi_extra` for a route that reads the form itself, so the schema still shows the body."""
    return {
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {"type": "object", "required": list(required), "properties": properties}
                }
            },
        }
    }


FILE_PART = {"type": "string", "format": "binary"}
