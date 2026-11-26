"""Profile, onboarding state, default picture and voices of the signed in user.

Every route is scoped to the token's user: the picture is found through the caller's own profile
and owner, never by an id from the client. Voice samples are public static files.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, Response

from api.deps.agentar import CurrentUser, ObjectStoreDep, require_content_length
from api.errors import AgentarError, error_responses
from config import settings
from src.schemas.agentar import Me, MePatch, Picture, VoiceId, VoiceInfo
from utils import agentar_profile as profiles
from utils.picture import PictureError, normalise_picture
from utils.uploads import FILE_PART, multipart_openapi, read_single_upload

router = APIRouter(prefix="/api", tags=["profile"])

_JPEG = {200: {"content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}}}
_WAV = {200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}}}

_PICTURE_HEADERS = {
    "Cache-Control": "private, no-store",
    "X-Content-Type-Options": "nosniff",
}
_SAMPLE_HEADERS = {
    "Cache-Control": "public, max-age=86400",
    "X-Content-Type-Options": "nosniff",
}


@router.get("/me", response_model=Me, responses=error_responses(401))
async def get_me(user: CurrentUser):
    return profiles.build_me(user)


@router.patch("/me", response_model=Me, responses=error_responses(401))
async def update_me(body: MePatch, user: CurrentUser):
    return profiles.build_me(await profiles.update_profile(user, body))


@router.post("/me/onboarding/skip", response_model=Me, responses=error_responses(401))
async def skip_onboarding(user: CurrentUser):
    return profiles.build_me(await profiles.skip_onboarding(user))


@router.put(
    "/me/picture",
    response_model=Picture,
    dependencies=[Depends(require_content_length)],
    responses=error_responses(401, 411, 413, 415, 422, 429, 503),
    openapi_extra=multipart_openapi({"file": FILE_PART}, ["file"]),
)
async def put_my_picture(request: Request, user: CurrentUser, store: ObjectStoreDep):
    """Reads the form with `utils.uploads.read_single_upload`, not with an UploadFile parameter."""
    upload = await read_single_upload(request, max_bytes=settings.agentar.picture_max_bytes)
    try:
        normalised = await normalise_picture(upload.data)
    except PictureError as exc:
        raise AgentarError(exc.status_code, exc.code, exc.message) from exc
    return await profiles.set_picture(user, store, normalised)


@router.get(
    "/me/picture",
    response_class=Response,
    responses={**_JPEG, **error_responses(401, 404, 503)},
)
async def get_my_picture(user: CurrentUser, store: ObjectStoreDep):
    data, sha256 = await profiles.read_picture(user, store)
    return Response(content=data, media_type="image/jpeg", headers={**_PICTURE_HEADERS, "ETag": f'"{sha256}"'})


@router.delete("/me/picture", status_code=204, responses=error_responses(401))
async def delete_my_picture(user: CurrentUser, store: ObjectStoreDep):
    await profiles.clear_picture(user, store)
    return Response(status_code=204)


@router.get("/agentar/voices", response_model=list[VoiceInfo], responses=error_responses(401))
async def list_voices(user: CurrentUser):
    return profiles.list_voices()


@router.get(
    "/agentar/voices/{voice_id}/sample",
    response_class=Response,
    responses={**_WAV, **error_responses(404)},
)
async def get_voice_sample(voice_id: VoiceId):
    """Public, no user data, so an audio element can load it without a header."""
    path = profiles.voice_sample_path(voice_id)
    if path is None:
        raise AgentarError(404, "not_found", "No sample for this voice")
    return FileResponse(path, media_type="audio/wav", headers=_SAMPLE_HEADERS)
