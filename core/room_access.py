"""
Room access rules for the meeting domain (see GLOSSARY.md).

MVP: the room owner is the only moderator and the only user who can control the room's
virtual participants. The owner is the first moderator to request access to an unclaimed room,
recorded once. The owner-managed moderator set is deferred: see docs/deferred/room-moderators.md.
"""
from src.database.models.room_owner import RoomOwner


class RoomAccessError(Exception):
    """Base for room-access failures the API maps to HTTP status codes."""


class NotOwnerError(RoomAccessError):
    pass


def _normalise(email: str) -> str:
    return email.strip().lower()


async def grants_moderator(room_name: str, email: str) -> bool:
    """Decide whether a `moderator=true` token request is honored.

    An unclaimed room is claimed by the caller, who becomes its owner. Otherwise the request
    is honored only for the owner.
    """
    email = _normalise(email)
    owner = await RoomOwner.claim_if_unowned(room_name=room_name, owner_email=email)
    return owner.owner_email == email


async def is_owner(room_name: str, email: str) -> bool:
    owner = await RoomOwner.find_by_room(room_name)
    return owner is not None and owner.owner_email == _normalise(email)


async def require_owner(room_name: str, email: str) -> None:
    """MVP gate for virtual participants: only the room owner may control them."""
    if not await is_owner(room_name, email):
        raise NotOwnerError("Only the room owner can control virtual participants")
