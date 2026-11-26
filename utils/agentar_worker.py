"""Agentar calls to vp-worker: the bot payload, the dispatch call, recall and room presence.

`build_worker_bot` turns a meeting and its owner's profile into the `AgentarWorkerBot` body of
`POST /bots`. It is a pure function: the token and the picture bytes come in as arguments, so the
mapping is testable against the worker's schema without a network. `dispatch_bot` reads the
picture, mints the bot token, posts to the worker and returns the worker's bot view. The
lifecycle (`utils/agentar_lifecycle.py`) is the one caller: it claims the meeting, calls
`dispatch_bot`, `get_bot` and `recall_bot`, and records the result.

Nothing here logs the payload: it carries the bot token and the picture. A worker error body can
echo the request, so only the status and the field locations of a schema 422 are logged, and
for a policy 422 the worker's own sentence.
"""
import base64
import logging
import re
import unicodedata
from typing import Any, Optional
from urllib.parse import quote, urlsplit

import httpx

from api.errors import AgentarError
from api.routes.virtual_participants import WORKER_SECRET_HEADER, get_vp_http_client
from config import settings
from src.database.models.agentar_file import AgentarFile
from src.database.models.agentar_meeting import AgentarMeeting
from src.database.models.user import User
from src.schemas.agentar import OWNER_FIRST_NAME_CHARS, AgentarWorkerBot, AgentarWorkerQuestion
from src.schemas.virtual_participants import BOT_ID_PATTERN
from utils import agentar_profile, auth
from utils.agentar_names import clean_name, escalation_line, owner_names
from utils.object_store import (
    ObjectNotFound,
    ObjectStore,
    ObjectStoreError,
    get_configured_object_store,
)

logger = logging.getLogger(__name__)

BOT_SUBJECT = "vp-bot"
AGENTAR_WORD = "Agentar"

EVENTS_PATH = "/api/agentar/events"
MAX_AVATAR_BYTES = 409_600
_JPEG_MAGIC = b"\xff\xd8\xff"
_BOT_ID = re.compile(BOT_ID_PATTERN)
_ARMED_NAME = re.compile(r"(?P<name>.+) Agentar(?: \([^()]*\))?")

# Spoken text. Placeholders: agent_name, owner_full_name, owner_first_name, purpose. The greeting
# carries the AI disclosure and the owner's name (AI Act, booklet Objective 5). The opener leaves
# the owner's private notes out on purpose: only the purpose is spoken.
GREETING_TEMPLATE = (
    "Hello, I am {agent_name} Agentar, an AI assistant attending for {owner_full_name}. "
    "I will listen, answer from the notes I was given where I can, and report back to {owner_first_name}."
)
OPENER_TEMPLATE = "The purpose of this meeting, as {owner_first_name} gave it to me, is: {purpose}"
# The escalation line comes from utils.agentar_names, shared with the chat path.
FAREWELL_TEMPLATE = "Thank you all, I will pass this on to {owner_first_name}. Goodbye."
OPENER_PURPOSE_CHARS = 500
OPENER_MAX_CHARS = 500


class WorkerDispatchError(Exception):
    """The bot could not be sent. `reason` becomes `status_reason` of the meeting, `detail` is
    for the log only and never reaches a client."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# Names ---------------------------------------------------------------------------------------



def resolve_display_name(owner: User) -> str:
    """"<name> Agentar": the name the profile page shows (`effective_agent_name`)."""
    return f"{agentar_profile.effective_agent_name(owner)} {AGENTAR_WORD}"


def resolve_voice(owner: User) -> str:
    """The profile voice while it is still offered, else the default voice."""
    return agentar_profile.effective_voice(owner)


def _one_line(text: str, limit: int) -> str:
    """Single line without control or format characters, cut at a word boundary."""
    cleaned = "".join(
        " " if unicodedata.category(ch) in {"Cc", "Cf", "Zl", "Zp"} else ch for ch in text
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit]
    boundary = cut.rfind(" ")
    return (cut[:boundary] if boundary > limit // 2 else cut).rstrip(" ,;:-")


def purpose_opener(owner_first_name: str, purpose: str, limit: Optional[int] = None) -> Optional[str]:
    """The opener spoken when the meeting has none of its own: the purpose sentence, or None
    without a purpose. `limit` caps the whole sentence, the purpose is cut at a word."""
    prefix = OPENER_TEMPLATE.format(owner_first_name=owner_first_name, purpose="")
    room = OPENER_PURPOSE_CHARS if limit is None else min(OPENER_PURPOSE_CHARS, limit - len(prefix))
    text = _one_line(purpose, max(room, 0))
    return f"{prefix}{text}" if text else None


# Payload -------------------------------------------------------------------------------------


def events_url() -> str:
    """Where the worker posts events: the worker's own view of this API plus the events path.

    The worker refuses an events_url outside the origin of its VP_MEETWEEN_API_URL, so the value
    is built from that one setting and nothing a user can write. A base with credentials, a query
    or a fragment, or without http(s) and a host, is a configuration error.
    """
    base = settings.agentar.worker_events_base_url
    parts = urlsplit(base)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise WorkerDispatchError("worker_error", "VP_MEETWEEN_API_URL is not a usable http(s) origin")
    url = f"{base}{EVENTS_PATH}"
    if len(url) > 512:
        raise WorkerDispatchError("worker_error", "events url is longer than 512 characters")
    return url


def build_worker_bot(
    *,
    meeting: AgentarMeeting,
    owner: User,
    jwt: Optional[str],
    avatar_jpeg: Optional[bytes],
) -> AgentarWorkerBot:
    """The body of `POST /bots` for an Agentar meeting. Built only here, never from a client body.

    Fallbacks for an owner who skipped onboarding: the name is the one the profile page shows
    (`effective_agent_name`: first and last name, then first name, then "Meetween"), the voice to the default voice, the picture to none (the worker then uses its
    bundled avatar), the greeting to "the meeting owner" when no names are stored.
    """
    if not meeting.id:
        raise ValueError("the meeting has no id")
    full_name, first_name = owner_names(owner.first_name, owner.last_name)
    # A meeting armed earlier keeps the name it was armed with, whatever the profile says now.
    armed = _ARMED_NAME.fullmatch(meeting.display_name or "")
    agent_name = (clean_name(armed["name"], 32) if armed else None) or agentar_profile.effective_agent_name(owner)
    display_name = f"{agent_name} {AGENTAR_WORD}"

    opener = _one_line(meeting.opener, OPENER_MAX_CHARS) or purpose_opener(first_name, meeting.purpose)
    voice = meeting.voice if meeting.voice in settings.agentar.voices else resolve_voice(owner)

    return AgentarWorkerBot(
        room=meeting.room_name,
        display_name=display_name,
        jwt=jwt,
        greeting=GREETING_TEMPLATE.format(
            agent_name=agent_name, owner_full_name=full_name, owner_first_name=first_name
        ),
        opener=opener,
        questions=[AgentarWorkerQuestion(id=q.id, text=q.text) for q in meeting.questions],
        escalation_line=escalation_line(full_name),
        farewell=FAREWELL_TEMPLATE.format(owner_first_name=first_name),
        voice=voice,
        avatar_jpeg_b64=base64.b64encode(avatar_jpeg).decode("ascii") if avatar_jpeg else None,
        agentar_meeting_id=meeting.id,
        events_url=events_url(),
        owner_first_name=clean_name(owner.first_name, OWNER_FIRST_NAME_CHARS),
    )


def mint_bot_token(room: str) -> str:
    """Room bound token for the bot, from B2's `create_jitsi_bot_token`: signed with the Jitsi key,
    no moderator claim and no name claim (a name would replace the "(AI)" display name). It raises
    a 503 AgentarError itself when the key is empty; that becomes a dispatch error with its code."""
    try:
        return auth.create_jitsi_bot_token(room)
    except AgentarError as exc:
        raise WorkerDispatchError(exc.code, str(exc.detail)) from exc


async def read_picture_bytes(store: ObjectStore, key: str) -> Optional[bytes]:
    stream = await store.get(key)
    data = bytearray()
    try:
        async for chunk in stream:
            data.extend(chunk)
            if len(data) > MAX_AVATAR_BYTES:
                return None
    finally:
        await stream.aclose()
    return bytes(data)


async def load_avatar(
    meeting: AgentarMeeting, owner: User, store: Optional[ObjectStore] = None
) -> Optional[bytes]:
    """JPEG bytes of the meeting picture, else the profile picture, else None.

    A record or object that is gone, not a JPEG or over the cap is skipped with a warning and the
    next candidate is tried. A store that cannot be reached, or is not configured while a picture
    is referenced, is an error: the bot would otherwise join with a face the owner did not choose.
    Without a picture the store is never touched.
    """
    candidates = [meeting.picture_id, owner.profile.picture_id if owner.profile else None]
    for picture_id in candidates:
        if not picture_id:
            continue
        record = await AgentarFile.find_by_id(picture_id)
        if record is None or record.owner != owner.email or record.kind != "picture":
            logger.warning(f"agentar picture record missing meeting={meeting.id} picture={picture_id}")
            continue
        store = store or get_configured_object_store(settings.object_store)
        if store is None:
            raise WorkerDispatchError("storage_unavailable", "object store is not configured")
        try:
            data = await read_picture_bytes(store, record.storage_key)
        except ObjectNotFound:
            logger.warning(f"agentar picture object missing meeting={meeting.id} picture={picture_id}")
            continue
        except ObjectStoreError as exc:
            raise WorkerDispatchError("storage_unavailable", type(exc).__name__) from exc
        if data is None or not data.startswith(_JPEG_MAGIC):
            logger.warning(f"agentar picture unusable meeting={meeting.id} picture={picture_id}")
            continue
        return data
    return None


# Worker calls --------------------------------------------------------------------------------


def _worker_url(path: str) -> str:
    return f"{settings.virtual_participants.worker_url.rstrip('/')}{path}"


def _worker_headers() -> dict[str, str]:
    return {WORKER_SECRET_HEADER: settings.virtual_participants.worker_secret}


WORKER_REFUSAL_MAX_CHARS = 200


def _worker_fault(response: httpx.Response) -> str:
    """Log line for a refused call: status, and for a 422 only where the problem was, never the
    echoed input.

    A schema 422 carries a list of errors that can echo the body, so only their field locations
    are kept. A policy 422 from the worker's own checks (missing AI label, voice not allowed, events
    url refused) carries one sentence the worker wrote: that sentence is kept, on one line and cut
    to WORKER_REFUSAL_MAX_CHARS, because it is the only clue to the cause."""
    detail = f"status {response.status_code}"
    if response.status_code == 422:
        try:
            errors = response.json().get("detail", [])
            if isinstance(errors, str):
                reason = " ".join(errors.split())[:WORKER_REFUSAL_MAX_CHARS]
                if reason:
                    detail += f" refused: {reason}"
            else:
                locs = [".".join(str(p) for p in e.get("loc", [])) for e in errors if isinstance(e, dict)]
                detail += f" fields {locs[:8]}"
        except (ValueError, AttributeError, TypeError):
            pass
    return detail


async def dispatch_bot(
    meeting: AgentarMeeting,
    owner: Optional[User] = None,
    *,
    store: Optional[ObjectStore] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> dict[str, Any]:
    """Send the bot for this meeting. Returns the worker's bot view (`id`, `display_name`, `state`).

    The owner is read from `meeting.owner` and the store from the settings unless given. Raises
    WorkerDispatchError for anything that stops the bot: no secret, an unusable events origin,
    an unknown owner, a token or store fault, a worker that is unreachable, full or refuses the
    body, and a worker that answers 200 because a bot with this name is already in the room (that
    bot is not this meeting's, so it is not adopted). The caller records the failure.
    """
    if not settings.virtual_participants.worker_secret:
        raise WorkerDispatchError("worker_error", "VP_WORKER_SECRET is empty")
    if owner is None:
        owner = await User.find_by_email(meeting.owner)
        if owner is None:
            raise WorkerDispatchError("worker_error", "the meeting owner no longer exists")
    http = client or await get_vp_http_client()
    avatar = await load_avatar(meeting, owner, store)
    payload = build_worker_bot(
        meeting=meeting, owner=owner, jwt=mint_bot_token(meeting.room_name), avatar_jpeg=avatar
    )
    try:
        response = await http.post(
            _worker_url("/bots"),
            json=payload.model_dump(mode="json", exclude_none=True),
            headers=_worker_headers(),
        )
    except httpx.HTTPError as exc:
        raise WorkerDispatchError("worker_error", f"unreachable {type(exc).__name__}") from exc
    if response.status_code == 200:
        raise WorkerDispatchError("worker_error", "a bot with this name is already in the room")
    if response.status_code != 202:
        raise WorkerDispatchError("worker_error", _worker_fault(response))
    view = _view(response)
    bot_id = view.get("id")
    if not isinstance(bot_id, str) or not _BOT_ID.fullmatch(bot_id):
        raise WorkerDispatchError("worker_error", "worker answered 202 without a usable bot id")
    return view


def _view(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise WorkerDispatchError("worker_error", f"worker answered {response.status_code} without JSON") from exc
    if not isinstance(body, dict):
        raise WorkerDispatchError("worker_error", f"worker answered {response.status_code} with a non-object")
    return body


async def get_bot(bot_id: str, *, client: Optional[httpx.AsyncClient] = None) -> Optional[dict[str, Any]]:
    """The worker's view of one bot, or None when the worker does not know it (404). Raises
    WorkerDispatchError when the worker cannot be reached or answers with a fault."""
    http = client or await get_vp_http_client()
    try:
        response = await http.get(_worker_url(f"/bots/{quote(bot_id, safe='')}"), headers=_worker_headers())
    except httpx.HTTPError as exc:
        raise WorkerDispatchError("worker_error", f"unreachable {type(exc).__name__}") from exc
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise WorkerDispatchError("worker_error", _worker_fault(response))
    return _view(response)


async def recall_bot(
    bot_id: str, farewell: bool = False, *, client: Optional[httpx.AsyncClient] = None
) -> Optional[dict[str, Any]]:
    """Make the bot leave, after its farewell line when `farewell` is set. Returns the worker's
    view, or None when the worker does not know the bot any more. Raises WorkerDispatchError when
    the worker cannot be reached or answers with a fault."""
    http = client or await get_vp_http_client()
    try:
        response = await http.delete(
            _worker_url(f"/bots/{quote(bot_id, safe='')}"),
            params={"farewell": "1"} if farewell else None,
            headers=_worker_headers(),
        )
    except httpx.HTTPError as exc:
        raise WorkerDispatchError("worker_error", f"unreachable {type(exc).__name__}") from exc
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise WorkerDispatchError("worker_error", _worker_fault(response))
    return _view(response)
