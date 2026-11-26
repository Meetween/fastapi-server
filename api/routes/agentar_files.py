"""Knowledge base files: upload, list, delete, download. Owner scoped, 404 across owners.

Routes stay thin: the work is in `utils.agentar_files`. Uploads declare no body parameters so that
`require_content_length` (411) and the size caps run before any byte is parsed, see `utils.uploads`.
"""
from typing import Annotated, AsyncIterator, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response, StreamingResponse

from api.deps.agentar import CurrentUser, ObjectStoreDep, require_content_length
from api.errors import error_responses
from config import settings
from src.schemas.agentar import AgentarFile, FileList, FileScope, ObjectId
from utils import agentar_files
from utils.object_store import ObjectStream
from utils.uploads import FILE_PART, content_disposition, multipart_openapi, read_single_upload

router = APIRouter(prefix="/api/agentar/files", tags=["files"])

_BYTES = {200: {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}}


@router.post(
    "",
    status_code=201,
    response_model=AgentarFile,
    dependencies=[Depends(require_content_length)],
    responses=error_responses(401, 404, 409, 411, 413, 415, 422, 429, 503),
    openapi_extra=multipart_openapi(
        {
            "file": FILE_PART,
            "scope": {"type": "string", "enum": ["general", "meeting"]},
            "meeting_id": {"type": "string", "pattern": "^[a-f0-9]{24}$"},
        },
        ["file", "scope"],
    ),
)
async def upload_file(request: Request, user: CurrentUser, store: ObjectStoreDep):
    """Form fields: `file` (one part), `scope`, `meeting_id` (with scope meeting). The type is
    sniffed from the bytes, the text is extracted once and chunked."""
    upload = await read_single_upload(
        request, max_bytes=settings.agentar.file_max_bytes, fields=("scope", "meeting_id")
    )
    return await agentar_files.create_document(user, store, upload)


@router.get("", response_model=FileList, responses=error_responses(401))
async def list_files(
    user: CurrentUser,
    scope: Annotated[Optional[FileScope], Query()] = None,
    meeting_id: Annotated[Optional[ObjectId], Query()] = None,
):
    return await agentar_files.list_documents(user, scope, meeting_id)


@router.delete("/{file_id}", status_code=204, responses=error_responses(401, 404, 409, 503))
async def delete_file(file_id: str, user: CurrentUser, store: ObjectStoreDep):
    await agentar_files.delete_document(user, store, file_id)
    return Response(status_code=204)


async def _body(stream: ObjectStream) -> AsyncIterator[bytes]:
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await stream.aclose()


@router.get(
    "/{file_id}/download",
    response_class=Response,
    responses={**_BYTES, **error_responses(401, 404, 503)},
)
async def download_file(file_id: str, user: CurrentUser, store: ObjectStoreDep):
    """The original bytes, streamed. Always an attachment, never sniffed or cached."""
    doc, stream = await agentar_files.open_document(user, store, file_id)
    headers = {
        "Content-Disposition": content_disposition(doc["name"]),
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, no-store",
        "Content-Length": str(doc["size"]),
    }
    return StreamingResponse(_body(stream), media_type="application/octet-stream", headers=headers)
