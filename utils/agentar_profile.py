"""Profile reads and writes, onboarding state, voices and fallbacks of the Agentar.

Writes to `users` are `$set` on declared fields, filtered by `_id`. `User.save()` replaces the whole
document and is never used here, so a concurrent login (password rehash) is not undone. The
`profile` key is absent until the first `$set`.

Fallbacks for a profile that is not filled in (skipped onboarding), used at dispatch:
`effective_agent_name`, `effective_voice`. `Me.profile` shows what is stored, so the UI can tell
"not chosen" from "chosen".
"""
from __future__ import annotations

import hashlib
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi.exceptions import RequestValidationError
from pymongo import ReturnDocument

from api.errors import AgentarError
from config import settings
from src.database.models.agentar_file import AgentarFile
from src.database.models.user import User, UserProfile
from src.schemas.agentar import Me, MePatch, Picture, Profile, VoiceInfo
from src.schemas.names import agent_name
from utils import clock
from utils.object_store import ObjectNotFound, ObjectStore, ObjectStoreError
from utils.picture import NormalisedPicture

logger = logging.getLogger(__name__)

OnboardingState = Literal["pending", "done", "skipped"]

# Label and gender of the Kokoro voices the contract allows (VoiceId). The ids offered at run time
# are the ones of AGENTAR_VOICES, in that order; the first is the default.
VOICE_CATALOG: dict[str, tuple[str, Literal["female", "male"]]] = {
    "af_heart": ("Heart", "female"),
    "af_bella": ("Bella", "female"),
    "am_fenrir": ("Fenrir", "male"),
    "am_michael": ("Michael", "male"),
}

# Rendered once with the worker TTS (task W1) and shipped with the image. A missing file is a 404.
VOICE_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "assets" / "voices"

# Used when a user has neither an agent name nor a usable first and last name.
FALLBACK_AGENT_NAME = "Meetween"


# Voices --------------------------------------------------------------------------------------


def list_voices() -> list[VoiceInfo]:
    default = settings.agentar.default_voice
    voices: list[VoiceInfo] = []
    for voice_id in settings.agentar.voices:
        known = VOICE_CATALOG.get(voice_id)
        if known is None:
            logger.warning("AGENTAR_VOICES lists %r, which the contract does not know; skipped", voice_id)
            continue
        label, gender = known
        voices.append(
            VoiceInfo(
                id=voice_id,  # type: ignore[arg-type]
                label=label,
                gender=gender,
                sample_url=f"/api/agentar/voices/{voice_id}/sample",
                default=voice_id == default,
            )
        )
    return voices


def voice_offered(voice_id: str) -> bool:
    return voice_id in VOICE_CATALOG and voice_id in settings.agentar.voices


def voice_sample_path(voice_id: str) -> Optional[Path]:
    """The sample file of an offered voice, or None while it is not rendered."""
    if not voice_offered(voice_id):
        return None
    path = VOICE_SAMPLE_DIR / f"{voice_id}.wav"
    return path if path.is_file() else None


# Reads ---------------------------------------------------------------------------------------


def onboarding_state(user: User) -> OnboardingState:
    profile = user.profile
    if profile is not None and profile.onboarded_at is not None:
        return "done"
    if profile is not None and profile.onboarding_skipped_at is not None:
        return "skipped"
    return "pending"


def build_me(user: User) -> Me:
    stored = user.profile or UserProfile()
    return Me(
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        is_admin=user.is_admin,
        profile=Profile(
            agent_name=stored.agent_name,
            voice=stored.voice,  # type: ignore[arg-type]
            language=stored.language,  # type: ignore[arg-type]
            onboarded_at=stored.onboarded_at,
            onboarding_skipped_at=stored.onboarding_skipped_at,
            has_picture=bool(stored.picture_id),
        ),
        onboarding_state=onboarding_state(user),
    )


def _sanitise_agent_name(raw: str) -> Optional[str]:
    """Reduce free text to something `agent_name` accepts, or None when nothing is left.

    Keeps Unicode letters, marks and digits, plus space and `_ . ' -`; drops the rest, collapses
    spaces, cuts at 32 characters. A name that holds the word "agentar" is refused as a whole.
    """
    kept = []
    for ch in raw:
        if unicodedata.category(ch)[0] in {"L", "M"} or ch in "0123456789 _.'-":
            kept.append(ch)
    text = " ".join("".join(kept).split())[:32].strip()
    try:
        return agent_name(text)
    except ValueError:
        return None


def effective_agent_name(user: User) -> str:
    """Name for the bot: the stored agent name, else the first and last name, else the first
    name, else "Meetween". The bot is shown as "<name> Agentar (AI)"."""
    stored = user.profile.agent_name if user.profile else None
    if stored:
        return stored
    full = " ".join(part for part in (user.first_name, user.last_name) if part)
    for candidate in (full, user.first_name or ""):
        cleaned = _sanitise_agent_name(candidate)
        if cleaned:
            return cleaned
    return FALLBACK_AGENT_NAME


def effective_voice(user: User) -> str:
    """The stored voice while it is still offered, else the default voice."""
    stored = user.profile.voice if user.profile else None
    if stored and voice_offered(stored):
        return stored
    return settings.agentar.default_voice


# Writes --------------------------------------------------------------------------------------


def _users():
    return User.get_collection()


def _user_oid(user: User) -> ObjectId:
    try:
        return ObjectId(user.id)
    except (InvalidId, TypeError) as exc:  # a user without an id cannot reach a route
        raise AgentarError(404, "not_found", "Not found") from exc


def _null_not_allowed(field: str) -> RequestValidationError:
    return RequestValidationError(
        [{"type": "string_type", "loc": ("body", *field.split(".")), "msg": "Input should not be null", "input": None}]
    )


async def reload_user(user: User) -> User:
    fresh = await User.find_one({"_id": _user_oid(user)})
    if fresh is None:  # deleted while the request ran
        raise AgentarError(404, "not_found", "Not found")
    return fresh


async def update_profile(user: User, patch: MePatch) -> User:
    """Apply a PATCH /api/me body and return the stored user.

    Only fields the client sent are written. `profile.agent_name: null` clears the name; null for
    any other field is a 422, as the contract types them as strings.
    """
    sent = patch.model_fields_set
    changes: dict[str, object] = {}

    for name in ("first_name", "last_name"):
        if name in sent:
            value = getattr(patch, name)
            if value is None:
                raise _null_not_allowed(name)
            changes[name] = value

    if "profile" in sent:
        if patch.profile is None:
            raise _null_not_allowed("profile")
        for name in patch.profile.model_fields_set:
            value = getattr(patch.profile, name)
            if value is None and name != "agent_name":
                raise _null_not_allowed(f"profile.{name}")
            if name == "voice" and not voice_offered(value):
                raise RequestValidationError(
                    [
                        {
                            "type": "literal_error",
                            "loc": ("body", "profile", "voice"),
                            "msg": "Voice is not offered by this server",
                            "input": value,
                        }
                    ]
                )
            changes[f"profile.{name}"] = value

    if "complete_onboarding" in sent and patch.complete_onboarding is None:
        raise _null_not_allowed("complete_onboarding")

    oid = _user_oid(user)
    if changes:
        await _users().update_one({"_id": oid}, {"$set": changes})
    if patch.complete_onboarding:
        # Once: the filter matches a missing profile and a null onboarded_at only.
        await _users().update_one(
            {"_id": oid, "profile.onboarded_at": None},
            {"$set": {"profile.onboarded_at": clock.utcnow()}},
        )
    return await reload_user(user)


async def skip_onboarding(user: User) -> User:
    """Idempotent. A completed onboarding is left as it is, and the first skip time is kept."""
    await _users().update_one(
        {"_id": _user_oid(user), "profile.onboarded_at": None, "profile.onboarding_skipped_at": None},
        {"$set": {"profile.onboarding_skipped_at": clock.utcnow()}},
    )
    return await reload_user(user)


# Picture -------------------------------------------------------------------------------------


def picture_key(user: User, file_id: ObjectId | str) -> str:
    """`u/{user_id}/avatar/{file_id}.jpg`. Built from ids only, never from a client filename."""
    return f"u/{user.id}/avatar/{file_id}.jpg"


def _picture_view(record: AgentarFile) -> Picture:
    return Picture(bytes=record.size, sha256=record.sha256, updated_at=record.created_at)


def _storage_down() -> AgentarError:
    return AgentarError(503, "storage_unavailable", "Object storage is not reachable")


async def _drop_picture(store: ObjectStore, owner: str, picture_id: Optional[str]) -> None:
    """Delete a replaced or removed picture, record and object. Failures are logged, not raised:
    the profile no longer points at it, so the worst case is an orphan the purge finds by prefix."""
    if not picture_id:
        return
    try:
        oid = ObjectId(picture_id)
    except (InvalidId, TypeError):
        return
    collection = AgentarFile.get_collection()
    record = await collection.find_one({"_id": oid, "owner": owner, "kind": "picture"})
    if record is None:
        return
    try:
        await store.delete(record["storage_key"])
    except ObjectStoreError:
        logger.warning("Could not delete picture object %s", record["storage_key"], exc_info=True)
        return
    await collection.delete_one({"_id": oid})


async def set_picture(user: User, store: ObjectStore, picture: NormalisedPicture) -> Picture:
    """Store the normalised picture, point the profile at it, delete the previous one.

    Order: object, record, profile, then the old object. A failure before the profile is written
    removes what this call created, so a failed upload leaves the old picture in place.
    """
    file_id = ObjectId()
    key = picture_key(user, file_id)
    try:
        await store.put(key, picture.data, content_type="image/jpeg")
    except ObjectStoreError as exc:
        raise _storage_down() from exc

    record = AgentarFile(
        owner=user.email,
        scope="general",
        kind="picture",
        name="avatar.jpg",
        size=len(picture.data),
        content_type="image/jpeg",
        sha256=picture.sha256,
        storage_key=key,
        created_at=clock.utcnow(),
    )
    collection = AgentarFile.get_collection()
    try:
        await collection.insert_one({**record.to_dict(), "_id": file_id})
        previous = await _users().find_one_and_update(
            {"_id": _user_oid(user)},
            {"$set": {"profile.picture_id": str(file_id), "profile.picture_sha256": picture.sha256}},
            projection={"profile.picture_id": 1},
            return_document=ReturnDocument.BEFORE,
        )
        if previous is None:
            raise AgentarError(404, "not_found", "Not found")
    except BaseException:
        await collection.delete_one({"_id": file_id})
        try:
            await store.delete(key)
        except ObjectStoreError:
            logger.warning("Could not clean up picture object %s", key, exc_info=True)
        raise

    old_id = (previous.get("profile") or {}).get("picture_id")
    await _drop_picture(store, user.email, old_id)
    return _picture_view(record)


async def clear_picture(user: User, store: ObjectStore) -> None:
    """Remove the picture. Also fine when none is set."""
    previous = await _users().find_one_and_update(
        {"_id": _user_oid(user), "profile.picture_id": {"$ne": None}},
        {"$set": {"profile.picture_id": None, "profile.picture_sha256": None}},
        projection={"profile.picture_id": 1},
        return_document=ReturnDocument.BEFORE,
    )
    if previous is not None:
        await _drop_picture(store, user.email, (previous.get("profile") or {}).get("picture_id"))


async def read_picture(user: User, store: ObjectStore) -> tuple[bytes, str]:
    """The stored JPEG and its sha256. 404 when none is set or the object is gone."""
    picture_id = user.profile.picture_id if user.profile else None
    if not picture_id:
        raise AgentarError(404, "not_found", "No picture is set")
    try:
        oid = ObjectId(picture_id)
    except (InvalidId, TypeError) as exc:
        raise AgentarError(404, "not_found", "No picture is set") from exc
    record = await AgentarFile.get_collection().find_one({"_id": oid, "owner": user.email, "kind": "picture"})
    if record is None:
        raise AgentarError(404, "not_found", "No picture is set")
    try:
        stream = await store.get(record["storage_key"])
    except ObjectNotFound as exc:
        raise AgentarError(404, "not_found", "No picture is set") from exc
    except ObjectStoreError as exc:
        raise _storage_down() from exc
    try:
        data = b"".join([chunk async for chunk in stream])
    except ObjectStoreError as exc:
        raise _storage_down() from exc
    finally:
        await stream.aclose()
    return data, hashlib.sha256(data).hexdigest()
