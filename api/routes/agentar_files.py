"""Knowledge base files: upload, list, delete, download. Owner scoped, 404 across owners.

Stubs until built: every body answers 501.
"""
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response

from api.deps.agentar import CurrentUser, ObjectStoreDep, require_content_length
from api.errors import error_responses, not_implemented
from src.schemas.agentar import AgentarFile, FileList, FileScope, ObjectId
from utils.uploads import FILE_PART, multipart_openapi

router = APIRouter(prefix="/api/agentar/files", tags=["files"])

_BYTES = {200: {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}}


@router.post(
    "",
    status_code=201,
    response_model=AgentarFile,
    dependencies=[Depends(require_content_length)],
    responses=error_responses(401, 404, 409, 411, 413, 415, 422, 503),
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
    """Form fields: file, scope, meeting_id. Read with `utils.uploads.read_single_upload`."""
    not_implemented()


@router.get("", response_model=FileList, responses=error_responses(401))
async def list_files(
    user: CurrentUser,
    scope: Annotated[Optional[FileScope], Query()] = None,
    meeting_id: Annotated[Optional[ObjectId], Query()] = None,
):
    not_implemented()


@router.delete("/{file_id}", status_code=204, responses=error_responses(401, 404, 409))
async def delete_file(file_id: str, user: CurrentUser, store: ObjectStoreDep):
    not_implemented()


@router.get(
    "/{file_id}/download",
    response_class=Response,
    responses={**_BYTES, **error_responses(401, 404)},
)
async def download_file(file_id: str, user: CurrentUser, store: ObjectStoreDep):
    not_implemented()
