"""
Virtual participant control route.

Authorises the caller, requires a human in the room, caps bots per room, mints the
bot's room token and forwards to vp-worker with the shared secret. Browsers, media
and speech live in vp-worker; this module keeps no bot state.
"""
import logging
import re
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from api.deps.auth import REQUIRE_AUTH, get_current_subject
from config import VirtualParticipantSettings, settings
from src.schemas.virtual_participants import (
    BOT_ID_PATTERN,
    BOT_NAME_SUFFIX,
    ROOM_PATTERN,
    VirtualParticipantCreateRequest,
    VirtualParticipantError,
    VirtualParticipantSayRequest,
    VirtualParticipantView,
)
from utils.auth import create_jitsi_token

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["virtual-participants"],
    dependencies=REQUIRE_AUTH,
)

WORKER_SECRET_HEADER = "X-VP-Secret"
BOT_SUBJECT = "vp-bot"
ACTIVE_STATES = frozenset({"joining", "joined", "speaking"})
NO_HUMAN_DETAIL = "no human participant in the room"

# Worker statuses returned to the caller as they are. Anything else is a fault
# between this service and the worker: a 401 means the secrets differ, and must
# not reach a client that would read it as its own session expiring.
_RELAYED_STATUSES = frozenset({200, 202, 404, 409, 422, 429})
_ROOM_RE = re.compile(ROOM_PATTERN)

_http_client: httpx.AsyncClient | None = None


async def get_vp_http_client() -> httpx.AsyncClient:
    """Process wide client for vp-worker and Prosody, created on first use.

    Tests replace it through ``app.dependency_overrides``.
    """
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.virtual_participants.request_timeout_s),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            # Internal calls only; a proxy set for other traffic must not apply.
            trust_env=False,
        )
    return _http_client


async def close_http_client() -> None:
    """Close the shared client. Called on application shutdown."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": VirtualParticipantError, "description": "Unknown bot"},
    422: {"description": "Invalid room name, bot id or body"},
    502: {"model": VirtualParticipantError, "description": "vp-worker rejected the request or failed"},
    503: {"model": VirtualParticipantError, "description": "vp-worker or Prosody unreachable, or route not configured"},
}


def _vp_settings() -> VirtualParticipantSettings:
    vp = settings.virtual_participants
    if not vp.worker_secret:
        logger.error("VP_WORKER_SECRET is not set; virtual participant route disabled")
        raise HTTPException(status_code=503, detail="virtual participant worker is not configured")
    return vp


def _normalize_room(room_name: str) -> str:
    room = room_name.lower()
    if not _ROOM_RE.fullmatch(room):
        raise HTTPException(status_code=422, detail=f"room_name must match {ROOM_PATTERN}")
    return room


def _bot_path(bot_id: str, suffix: str = "") -> str:
    return f"/bots/{quote(bot_id, safe='')}{suffix}"


def _json_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"detail": response.reason_phrase or "invalid worker response"}


def _bots_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Accept a JSON array of bot views, or an object holding one under ``bots``."""
    if isinstance(payload, dict):
        payload = payload.get("bots")
    if not isinstance(payload, list):
        logger.error("vp-worker returned an unexpected bot list payload")
        raise HTTPException(status_code=502, detail="virtual participant worker error")
    return [bot for bot in payload if isinstance(bot, dict)]


async def _worker_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
) -> httpx.Response:
    vp = _vp_settings()
    try:
        return await client.request(
            method,
            f"{vp.worker_url.rstrip('/')}{path}",
            params=params,
            json=json,
            headers={WORKER_SECRET_HEADER: vp.worker_secret},
        )
    except httpx.HTTPError as exc:
        logger.warning(f"vp-worker {method} {path} failed: {exc!r}")
        raise HTTPException(
            status_code=503, detail="virtual participant worker unreachable"
        ) from exc


def _raise_worker_fault(response: httpx.Response) -> None:
    status = response.status_code
    path = response.request.url.path
    if status in (401, 403):
        logger.error(f"vp-worker rejected the control secret on {path} ({status})")
    else:
        logger.error(f"vp-worker returned {status} on {path}")
    if status == 503:
        raise HTTPException(status_code=503, detail="virtual participant worker unavailable")
    raise HTTPException(status_code=502, detail="virtual participant worker error")


def _relay(response: httpx.Response) -> JSONResponse:
    status = response.status_code
    if status not in _RELAYED_STATUSES:
        _raise_worker_fault(response)

    headers = {}
    if status == 429 and "retry-after" in response.headers:
        headers["Retry-After"] = response.headers["retry-after"]
    return JSONResponse(status_code=status, content=_json_body(response), headers=headers)


def _is_human(occupant: Any, xmpp_domain: str) -> bool:
    if not isinstance(occupant, dict):
        return False
    display_name = str(occupant.get("display_name") or "")
    if display_name.endswith(BOT_NAME_SUFFIX):
        return False
    bare_jid = str(occupant.get("jid") or "").split("/", 1)[0]
    jid_domain = bare_jid.rsplit("@", 1)[-1].lower()
    return jid_domain not in {f"hidden.{xmpp_domain}", f"recorder.{xmpp_domain}"}


async def _count_humans(
    client: httpx.AsyncClient, vp: VirtualParticipantSettings, room: str
) -> int:
    """Count human occupants through the Prosody ``mod_muc_size`` HTTP API."""
    try:
        response = await client.get(
            f"{vp.prosody_http_url.rstrip('/')}/room",
            params={"room": room, "domain": vp.xmpp_domain},
            # Prosody routes HTTP by virtual host; muc_size is loaded on the XMPP domain.
            headers={"Host": vp.xmpp_domain},
        )
    except httpx.HTTPError as exc:
        logger.warning(f"prosody room lookup failed for {room}: {exc!r}")
        raise HTTPException(status_code=503, detail="room presence lookup unavailable") from exc

    if response.status_code == 404:
        return 0
    if response.status_code != 200:
        logger.error(f"prosody room lookup returned {response.status_code} for {room}")
        raise HTTPException(status_code=503, detail="room presence lookup unavailable")
    occupants = _json_body(response)
    if not isinstance(occupants, list):
        logger.error(f"prosody room lookup returned an unexpected payload for {room}")
        raise HTTPException(status_code=503, detail="room presence lookup unavailable")
    return sum(1 for occupant in occupants if _is_human(occupant, vp.xmpp_domain))


async def _active_bots(client: httpx.AsyncClient, room: str) -> list[dict[str, Any]]:
    response = await _worker_request(client, "GET", "/bots", params={"room": room})
    if response.status_code != 200:
        _raise_worker_fault(response)
    return [
        bot
        for bot in _bots_from_payload(_json_body(response))
        if bot.get("state") in ACTIVE_STATES and bot.get("room", room) == room
    ]


@router.post(
    "/virtual-participants",
    status_code=202,
    response_model=VirtualParticipantView,
    responses={
        200: {"model": VirtualParticipantView, "description": "An active bot with this name is already in the room"},
        409: {"model": VirtualParticipantError, "description": "No human in the room, or bot limit reached"},
        429: {"model": VirtualParticipantError, "description": "All worker replicas are full; see Retry-After"},
        **_ERROR_RESPONSES,
    },
)
async def create_virtual_participant(
    request: VirtualParticipantCreateRequest,
    current_subject: str = Depends(get_current_subject),
    client: httpx.AsyncClient = Depends(get_vp_http_client),
):
    """Send a bot into a room that already has a human participant."""
    vp = _vp_settings()
    room = request.room_name

    if vp.require_human and await _count_humans(client, vp, room) == 0:
        logger.info(
            f"virtual participant create refused subject={current_subject} room={room} reason=no_human"
        )
        raise HTTPException(status_code=409, detail=NO_HUMAN_DETAIL)

    active = await _active_bots(client, room)
    names = {request.display_name, f"{request.display_name}{BOT_NAME_SUFFIX}"}
    duplicate = any(bot.get("display_name") in names for bot in active)
    # A duplicate is forwarded so the worker returns the existing bot with 200.
    if not duplicate and len(active) >= vp.max_bots_per_room:
        logger.info(
            f"virtual participant create refused subject={current_subject} room={room} reason=room_limit"
        )
        raise HTTPException(
            status_code=409,
            detail=f"room already has {len(active)} virtual participants, limit is {vp.max_bots_per_room}",
        )

    bot_token = create_jitsi_token(sub=BOT_SUBJECT, room=room, moderator=False)
    response = await _worker_request(
        client,
        "POST",
        "/bots",
        json={
            "room": room,
            "display_name": request.display_name,
            "jwt": bot_token,
            "script": request.script,
        },
    )
    body = _json_body(response)
    bot_id = body.get("id") if isinstance(body, dict) else None
    logger.info(
        f"virtual participant create subject={current_subject} room={room} "
        f"bot_id={bot_id} worker_status={response.status_code}"
    )
    return _relay(response)


@router.get(
    "/virtual-participants",
    response_model=list[VirtualParticipantView],
    responses=_ERROR_RESPONSES,
)
async def list_virtual_participants(
    room_name: str | None = Query(default=None, description="Filter by room. Lowercased."),
    current_subject: str = Depends(get_current_subject),
    client: httpx.AsyncClient = Depends(get_vp_http_client),
):
    """List bots known to the worker cluster, optionally for one room."""
    params = {"room": _normalize_room(room_name)} if room_name is not None else None
    response = await _worker_request(client, "GET", "/bots", params=params)
    if response.status_code != 200:
        return _relay(response)
    logger.debug(f"virtual participant list subject={current_subject} params={params}")
    return JSONResponse(content=_bots_from_payload(_json_body(response)))


@router.get(
    "/virtual-participants/{bot_id}",
    response_model=VirtualParticipantView,
    responses=_ERROR_RESPONSES,
)
async def get_virtual_participant(
    bot_id: str = Path(..., pattern=BOT_ID_PATTERN),
    current_subject: str = Depends(get_current_subject),
    client: httpx.AsyncClient = Depends(get_vp_http_client),
):
    """Return one bot's state."""
    response = await _worker_request(client, "GET", _bot_path(bot_id))
    logger.debug(f"virtual participant get subject={current_subject} bot_id={bot_id}")
    return _relay(response)


@router.delete(
    "/virtual-participants/{bot_id}",
    response_model=VirtualParticipantView,
    responses=_ERROR_RESPONSES,
)
async def delete_virtual_participant(
    bot_id: str = Path(..., pattern=BOT_ID_PATTERN),
    current_subject: str = Depends(get_current_subject),
    client: httpx.AsyncClient = Depends(get_vp_http_client),
):
    """Make a bot leave its room. Returns the final view."""
    response = await _worker_request(client, "DELETE", _bot_path(bot_id))
    body = _json_body(response)
    room = body.get("room") if isinstance(body, dict) else None
    logger.info(
        f"virtual participant delete subject={current_subject} room={room} "
        f"bot_id={bot_id} worker_status={response.status_code}"
    )
    return _relay(response)


@router.post(
    "/virtual-participants/{bot_id}/say",
    status_code=202,
    response_model=VirtualParticipantView,
    responses={
        409: {"model": VirtualParticipantError, "description": "The bot is already speaking"},
        **_ERROR_RESPONSES,
    },
)
async def say_virtual_participant(
    request: VirtualParticipantSayRequest,
    bot_id: str = Path(..., pattern=BOT_ID_PATTERN),
    current_subject: str = Depends(get_current_subject),
    client: httpx.AsyncClient = Depends(get_vp_http_client),
):
    """Queue one utterance for the bot to speak."""
    response = await _worker_request(
        client, "POST", _bot_path(bot_id, "/say"), json={"text": request.text}
    )
    body = _json_body(response)
    room = body.get("room") if isinstance(body, dict) else None
    logger.info(
        f"virtual participant say subject={current_subject} room={room} bot_id={bot_id} "
        f"chars={len(request.text)} worker_status={response.status_code}"
    )
    return _relay(response)
