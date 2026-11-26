"""Bounded reading of one uploaded file, shared by the file, profile picture and meeting picture routes.

Why not `UploadFile` parameters: FastAPI parses the body before it runs dependencies, so a chunked
body would be spooled before any size check. These routes declare no body parameters. The
`require_content_length` dependency runs first (411), then `read_single_upload` bounds the form.

Order of checks: Content-Length over `max_bytes` + `OVERHEAD_BYTES` answers 413 before parsing;
the form allows one file and three small fields; the file read is capped at `max_bytes`.
"""
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from fastapi import HTTPException, Request
from starlette.datastructures import UploadFile

from api.errors import AgentarError

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


async def read_single_upload(
    request: Request, *, max_bytes: int, fields: Sequence[str] = ()
) -> UploadedFile:
    """Read the `file` part and the named text fields, within the caps.

    Raises 413 `file_too_large`, 400 for too many parts (from the parser), 422 for a missing
    file part or an unexpected field. The filename is returned as sent; it is display metadata
    and must never reach an object key.
    """
    declared = int(request.headers.get("content-length", "0") or 0)
    if declared > max_bytes + OVERHEAD_BYTES:
        raise _too_large(max_bytes)

    try:
        async with request.form(max_files=1, max_fields=MAX_FIELDS, max_part_size=MAX_FIELD_BYTES) as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(status_code=422, detail="A file part named file is required")
            values: dict[str, str] = {}
            for key, value in form.multi_items():
                if key == "file":
                    continue
                if key not in fields or not isinstance(value, str):
                    raise HTTPException(status_code=422, detail=f"Unexpected form field {key}")
                values[key] = value
            data = await upload.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise _too_large(max_bytes)
            return UploadedFile(
                data=data, filename=upload.filename, content_type=upload.content_type, fields=values
            )
    except AssertionError as exc:  # python-multipart missing: a deployment error, not a client one
        raise HTTPException(status_code=503, detail="Multipart parsing is not available") from exc


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
