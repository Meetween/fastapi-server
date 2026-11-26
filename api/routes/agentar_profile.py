"""Profile, onboarding state, default picture and voices of the signed in user.

Stubs until built: every body answers 501.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from api.deps.agentar import CurrentUser, ObjectStoreDep, require_content_length
from api.errors import error_responses, not_implemented
from src.schemas.agentar import Me, MePatch, Picture, VoiceId, VoiceInfo
from utils.uploads import FILE_PART, multipart_openapi

router = APIRouter(prefix="/api", tags=["profile"])

_JPEG = {200: {"content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}}}
_WAV = {200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}}}


@router.get("/me", response_model=Me, responses=error_responses(401))
async def get_me(user: CurrentUser):
    not_implemented()


@router.patch("/me", response_model=Me, responses=error_responses(401))
async def update_me(body: MePatch, user: CurrentUser):
    not_implemented()


@router.post("/me/onboarding/skip", response_model=Me, responses=error_responses(401))
async def skip_onboarding(user: CurrentUser):
    not_implemented()


@router.put(
    "/me/picture",
    response_model=Picture,
    dependencies=[Depends(require_content_length)],
    responses=error_responses(401, 411, 413, 415, 422, 503),
    openapi_extra=multipart_openapi({"file": FILE_PART}, ["file"]),
)
async def put_my_picture(request: Request, user: CurrentUser, store: ObjectStoreDep):
    """Reads the form with `utils.uploads.read_single_upload`, not with an UploadFile parameter."""
    not_implemented()


@router.get(
    "/me/picture",
    response_class=Response,
    responses={**_JPEG, **error_responses(401, 404)},
)
async def get_my_picture(user: CurrentUser, store: ObjectStoreDep):
    not_implemented()


@router.delete("/me/picture", status_code=204, responses=error_responses(401))
async def delete_my_picture(user: CurrentUser, store: ObjectStoreDep):
    not_implemented()


@router.get("/agentar/voices", response_model=list[VoiceInfo], responses=error_responses(401))
async def list_voices(user: CurrentUser):
    not_implemented()


@router.get(
    "/agentar/voices/{voice_id}/sample",
    response_class=Response,
    responses={**_WAV, **error_responses(404)},
)
async def get_voice_sample(voice_id: VoiceId):
    """Public, no user data, so an audio element can load it without a header."""
    not_implemented()
