"""Knowledge base files: quota, create, list, delete, open for download.

Documents only. Pictures live in the same collection with `kind: "picture"` (profile and meeting
picture routes); every query here filters on `kind: "document"`, so a picture never shows in the
knowledge base list, counts against the document quota, or downloads through these routes.

Every query filters by `owner` (the token subject, lower cased email). An id that is malformed,
unknown or owned by someone else is `not_found()`. Object keys use `user.id` and the new file id,
never the client filename.

Create order: sniff, quota, extract and chunk, `put_object`, Mongo insert (the object is deleted when
the insert fails), quota check again. The second check closes the gap between the first check and the
insert for parallel uploads of one user: if the user is over quota after the insert, the record and the
object are removed and the request fails.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any, Optional

from bson import ObjectId
from bson.errors import InvalidId

from api.errors import AgentarError, not_found, object_id_or_404
from config import settings
from src.database.models.agentar_file import AgentarFile as AgentarFileDoc
from src.database.models.agentar_meeting import BOT_PRESENT_STATUSES, AgentarMeeting
from src.schemas.agentar import AgentarFile, FileList, Usage
from utils import clock
from utils.chunking import chunk_blocks
from utils.object_store import ObjectNotFound, ObjectStore, ObjectStoreError, ObjectStream
from utils.text_extract import extract_text
from utils.uploads import UploadedFile, clean_filename, invalid_form, sniff_document

logger = logging.getLogger(__name__)

EDITABLE_MEETING_STATUSES = ("draft", "scheduled")
_OBJECT_ID = re.compile(r"[a-f0-9]{24}")


def _owner(user) -> str:
    return user.email.lower()


def _documents(owner: str, **extra: Any) -> dict[str, Any]:
    return {"owner": owner, "kind": "document", **extra}


def storage_unavailable() -> AgentarError:
    return AgentarError(503, "storage_unavailable", "Object storage is unavailable")


def storage_key(user_id: str, file_id: ObjectId, meeting_id: Optional[str]) -> str:
    """`u/{user_id}/kb/{file_id}` or `u/{user_id}/m/{meeting_id}/kb/{file_id}`."""
    if meeting_id:
        return f"u/{user_id}/m/{meeting_id}/kb/{file_id}"
    return f"u/{user_id}/kb/{file_id}"


# Usage and quota ------------------------------------------------------------------------------


async def get_usage(owner: str) -> Usage:
    cursor = AgentarFileDoc.get_collection().find(_documents(owner), {"size": 1})
    sizes = [int(doc.get("size", 0)) async for doc in cursor]
    cfg = settings.agentar
    return Usage(
        files=len(sizes),
        files_max=cfg.user_max_files,
        bytes=sum(sizes),
        bytes_max=cfg.user_max_bytes,
        file_max_bytes=cfg.file_max_bytes,
    )


def _check_quota(usage: Usage, adding: int) -> None:
    if usage.files + 1 > usage.files_max:
        raise AgentarError(422, "quota_files", f"You can keep at most {usage.files_max} files. Delete one first")
    if usage.bytes + adding > usage.bytes_max:
        raise AgentarError(
            422, "quota_bytes", f"Your files may use at most {usage.bytes_max} bytes in total. Delete one first"
        )


# Form fields ----------------------------------------------------------------------------------


def _parse_target(fields: dict[str, str]) -> tuple[str, Optional[str]]:
    scope = fields.get("scope")
    if scope not in ("general", "meeting"):
        raise invalid_form("scope", "scope must be general or meeting", "enum")
    meeting_id = fields.get("meeting_id") or None
    if scope == "general":
        if meeting_id:
            raise invalid_form("meeting_id", "meeting_id is only allowed with scope meeting")
        return scope, None
    if meeting_id is None or not _OBJECT_ID.fullmatch(meeting_id):
        raise invalid_form("meeting_id", "meeting_id must be 24 lower case hex characters when scope is meeting")
    return scope, meeting_id


async def _require_editable_meeting(owner: str, meeting_id: str) -> None:
    doc = await AgentarMeeting.get_collection().find_one(
        {"_id": ObjectId(meeting_id), "owner": owner}, {"status": 1}
    )
    if doc is None:
        raise not_found()
    if doc.get("status") not in EDITABLE_MEETING_STATUSES:
        raise AgentarError(
            409, "invalid_state", "Files can only be added to a meeting that is a draft or scheduled"
        )


# Create ---------------------------------------------------------------------------------------


def _to_api(doc: dict[str, Any]) -> AgentarFile:
    return AgentarFile(
        id=str(doc["_id"]),
        scope=doc["scope"],
        meeting_id=doc.get("meeting_id"),
        name=doc["name"],
        size=doc["size"],
        content_type=doc["content_type"],
        sha256=doc["sha256"],
        text_status=doc["text_status"],
        text_chars=doc.get("text_chars", 0),
        chunk_count=doc["chunk_count"] if "chunk_count" in doc else len(doc.get("chunks", [])),
        error=doc.get("error"),
        created_at=clock.as_utc(doc["created_at"]),
    )


async def _discard(store: ObjectStore, key: str) -> None:
    try:
        await store.delete(key)
    except Exception:  # best effort: the purge script removes a user prefix
        logger.exception("Could not delete object %s while undoing an upload", key)


async def create_document(user, store: ObjectStore, upload: UploadedFile) -> AgentarFile:
    owner = _owner(user)
    scope, meeting_id = _parse_target(dict(upload.fields))
    if not upload.data:
        raise invalid_form("file", "The file is empty", "too_short")
    if meeting_id:
        await _require_editable_meeting(owner, meeting_id)

    content_type = sniff_document(upload.data, upload.filename)
    cfg = settings.agentar
    _check_quota(await get_usage(owner), len(upload.data))

    extraction = await extract_text(content_type, upload.data, max_chars=cfg.text_max_chars)
    chunks = chunk_blocks(extraction.blocks, target_tokens=cfg.chunk_tokens)

    file_id = ObjectId()
    key = storage_key(user.id, file_id, meeting_id)
    document = AgentarFileDoc(
        owner=owner,
        scope=scope,
        meeting_id=meeting_id,
        kind="document",
        name=clean_filename(upload.filename),
        size=len(upload.data),
        content_type=content_type,
        sha256=hashlib.sha256(upload.data).hexdigest(),
        storage_key=key,
        text_status=extraction.status,
        text_chars=extraction.chars,
        chunks=chunks,
        error=extraction.error,
        created_at=clock.utcnow(),
    )
    try:
        await store.put(key, upload.data, content_type=content_type)
    except ObjectStoreError as exc:
        logger.error("Object store put failed: %s", exc)
        raise storage_unavailable() from exc

    collection = AgentarFileDoc.get_collection()
    record = {**document.to_dict(), "_id": file_id}
    try:
        await collection.insert_one(record)
    except BaseException:
        await asyncio.shield(_discard(store, key))
        raise

    try:
        _check_quota_after_insert(await get_usage(owner))
    except AgentarError:
        await asyncio.shield(collection.delete_one({"_id": file_id}))
        await asyncio.shield(_discard(store, key))
        raise
    logger.info("Stored knowledge base file %s (%s, %d bytes, text %s)", file_id, content_type, document.size, extraction.status)
    return _to_api({**record, "chunk_count": len(chunks)})


def _check_quota_after_insert(usage: Usage) -> None:
    """The new record is already counted."""
    if usage.files > usage.files_max:
        raise AgentarError(422, "quota_files", f"You can keep at most {usage.files_max} files. Delete one first")
    if usage.bytes > usage.bytes_max:
        raise AgentarError(
            422, "quota_bytes", f"Your files may use at most {usage.bytes_max} bytes in total. Delete one first"
        )


# List -----------------------------------------------------------------------------------------

_LIST_FIELDS = {
    name: 1
    for name in (
        "scope",
        "meeting_id",
        "name",
        "size",
        "content_type",
        "sha256",
        "text_status",
        "text_chars",
        "error",
        "created_at",
    )
}


async def list_documents(user, scope: Optional[str], meeting_id: Optional[str]) -> FileList:
    """Newest first. `usage` counts all of the owner's documents, whatever the filter."""
    owner = _owner(user)
    flt = _documents(owner)
    if scope:
        flt["scope"] = scope
    if meeting_id:
        flt["meeting_id"] = meeting_id
    # Chunk text stays in Mongo: only its count is computed.
    pipeline = [
        {"$match": flt},
        {"$sort": {"created_at": -1, "_id": -1}},
        {"$project": {**_LIST_FIELDS, "chunk_count": {"$size": {"$ifNull": ["$chunks", []]}}}},
    ]
    docs = await AgentarFileDoc.get_collection().aggregate(pipeline).to_list(length=None)
    return FileList(items=[_to_api(doc) for doc in docs], usage=await get_usage(owner))


# Delete ---------------------------------------------------------------------------------------


async def _find_document(owner: str, file_id: str) -> dict[str, Any]:
    oid = object_id_or_404(file_id)
    doc = await AgentarFileDoc.get_collection().find_one(_documents(owner, _id=oid))
    if doc is None:
        raise not_found()
    return doc


def _object_id_or_none(value: Optional[str]) -> Optional[ObjectId]:
    try:
        return ObjectId(value) if value else None
    except (InvalidId, TypeError):
        return None


async def delete_document(user, store: ObjectStore, file_id: str) -> None:
    """409 `file_in_use` while a meeting with a bot in the room uses the file: it lists the id, owns
    the file as a meeting file, or (for a general file) uses the whole general knowledge base. Otherwise
    the id is pulled from the owner's other meetings, then the record and the object are removed."""
    owner = _owner(user)
    doc = await _find_document(owner, file_id)
    fid = str(doc["_id"])
    meetings = AgentarMeeting.get_collection()

    uses = [{"file_ids": fid}]
    meeting_oid = _object_id_or_none(doc.get("meeting_id"))
    if meeting_oid is not None:
        uses.append({"_id": meeting_oid})
    if doc.get("scope") == "general":
        uses.append({"use_general_kb": True})
    live = await meetings.find_one({"owner": owner, "status": {"$in": list(BOT_PRESENT_STATUSES)}, "$or": uses}, {"_id": 1})
    if live is not None:
        raise AgentarError(409, "file_in_use", "A meeting that is running or about to start uses this file")

    await meetings.update_many(
        {"owner": owner, "file_ids": fid, "status": {"$nin": list(BOT_PRESENT_STATUSES)}},
        {"$pull": {"file_ids": fid}},
    )
    await AgentarFileDoc.get_collection().delete_one({"_id": doc["_id"], "owner": owner})
    try:
        await store.delete(doc["storage_key"])
    except ObjectStoreError:
        # The record is gone and the user's intent is done. The orphan sits under the user prefix
        # and the purge script removes it.
        logger.exception("Object %s could not be deleted after its record was removed", doc["storage_key"])


async def delete_meeting_documents(owner: str, store: ObjectStore, meeting_id: str) -> int:
    """Remove every meeting scoped document of one meeting, records first, then objects. For the
    meeting delete of B6, which has already refused a meeting with a bot in the room. Returns the number
    of files removed. An object that cannot be deleted is logged and left to the purge script."""
    collection = AgentarFileDoc.get_collection()
    flt = _documents(owner.lower(), scope="meeting", meeting_id=meeting_id)
    docs = await collection.find(flt, {"storage_key": 1}).to_list(length=None)
    await collection.delete_many({**flt, "_id": {"$in": [doc["_id"] for doc in docs]}})
    for doc in docs:
        try:
            await store.delete(doc["storage_key"])
        except ObjectStoreError:
            logger.exception("Object %s could not be deleted with its meeting", doc["storage_key"])
    return len(docs)


# Download -------------------------------------------------------------------------------------


async def open_document(user, store: ObjectStore, file_id: str) -> tuple[dict[str, Any], ObjectStream]:
    doc = await _find_document(_owner(user), file_id)
    try:
        stream = await store.get(doc["storage_key"])
    except ObjectNotFound as exc:
        logger.error("Record %s points to a missing object", doc["_id"])
        raise not_found() from exc
    except ObjectStoreError as exc:
        logger.error("Object store get failed: %s", exc)
        raise storage_unavailable() from exc
    return doc, stream
