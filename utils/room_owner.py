"""Room ownership: who moderates a Jitsi room.

The first user who claims a room owns it. Claims come from `POST /api/auth/jitsi/issue`, from
Agentar meeting create and dispatch, and from an operator through `cli.py room-owner set`. The
owner and admins moderate; everyone else joins as a guest. The owner key is the token subject,
the lower-cased email.

A first claim of a free room counts against `ROOM_CLAIMS_PER_DAY` per user and UTC day, so one
account cannot squat on every likely room name. Admins are exempt from both the gating and the
cap. Claims are counted from the `room_owner` collection, so the cap survives a restart.
"""
import logging
import math
from datetime import datetime, timedelta

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from api.errors import AgentarError
from config import settings
from src.database.models.room_owner import RoomOwner, room_key
from src.database.models.user import User
from utils import clock

logger = logging.getLogger(__name__)


def _admin(user: User) -> bool:
    return bool(user.is_active and user.is_admin)


async def is_admin(user_id: str) -> bool:
    """Whether the token subject is an active admin. Admins moderate every room."""
    user = await User.find_by_email(user_id)
    return user is not None and _admin(user)


def _utc_day_bounds() -> tuple[datetime, datetime, datetime]:
    """Now, the start of the current UTC day and the start of the next."""
    now = clock.utcnow()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return now, start, start + timedelta(days=1)


async def _claims_today(user_id: str) -> int:
    _, start, _ = _utc_day_bounds()
    return await RoomOwner.count_documents({"owner_user_id": user_id, "created_at": {"$gte": start}})


def _claims_exhausted() -> AgentarError:
    now, _, next_day = _utc_day_bounds()
    retry_after = max(1, math.ceil((next_day - now).total_seconds()))
    return AgentarError(
        429,
        "rate_limited",
        f"Too many new rooms today, the limit is {settings.agentar.room_claims_per_day} per day",
        headers={"Retry-After": str(retry_after)},
    )


async def _claim_free_room(room: str, user_id: str, *, admin: bool) -> RoomOwner:
    """Insert ``user_id`` as owner of a room that had none. Returns the owner record.

    The unique index on ``room_name`` settles concurrent first claims: the losing insert reads
    the winner. The cap is checked before the insert and again after it, so two concurrent
    claims at the limit cannot both pass; the second check removes the extra record and fails
    closed.
    """
    cap = settings.agentar.room_claims_per_day
    if not admin and await _claims_today(user_id) >= cap:
        raise _claims_exhausted()

    record = RoomOwner(room_name=room, owner_user_id=user_id)
    try:
        await record.save()
    except DuplicateKeyError:
        winner = await RoomOwner.find_by_room(room)
        if winner is None:
            raise
        return winner

    if not admin and await _claims_today(user_id) > cap:
        await record.delete()
        raise _claims_exhausted()
    logger.info("room claimed room=%s", room)
    return record


async def claim_room_moderator(room_name: str, user: User) -> bool:
    """Claim the room for ``user`` if it is free, and return whether ``user`` moderates it.

    True when the user owns the room (before or by this call) or is an active admin. The owner
    key is ``user.email``, the token subject. A first claim beyond `ROOM_CLAIMS_PER_DAY` raises
    429 `rate_limited` with Retry-After; claiming a room the user already owns, or looking at a
    room someone else owns, never counts.
    """
    room = room_key(room_name)
    owner = await RoomOwner.find_by_room(room)
    if owner is None:
        owner = await _claim_free_room(room, user.email, admin=_admin(user))
    return owner.owner_user_id == user.email or _admin(user)


async def is_room_moderator(*, room_name: str, user_id: str) -> bool:
    """Whether ``user_id`` owns the room or is an admin. Never creates an owner record."""
    owner = await RoomOwner.find_by_room(room_name)
    if owner is not None and owner.owner_user_id == user_id:
        return True
    return await is_admin(user_id)


async def get_room_owner(room_name: str) -> str | None:
    owner = await RoomOwner.find_by_room(room_name)
    return owner.owner_user_id if owner else None


async def set_room_owner(*, room_name: str, user_id: str) -> RoomOwner:
    """Operator action: make ``user_id`` the owner, replacing any current owner."""
    doc = await RoomOwner.get_collection().find_one_and_update(
        {"room_name": room_key(room_name)},
        {
            "$set": {"owner_user_id": user_id, "created_at": clock.utcnow()},
            "$setOnInsert": {"room_name": room_key(room_name), "timestamp": clock.utcnow()},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return RoomOwner.from_dict(doc)


async def release_room(room_name: str) -> bool:
    """Operator action: remove the owner record. The next claim takes the room."""
    return await RoomOwner.delete_many({"room_name": room_key(room_name)}) > 0
