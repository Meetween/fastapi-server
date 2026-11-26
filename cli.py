"""Operational CLI for the Meetween FastAPI server."""

import argparse
import asyncio
import getpass
import logging
import re
import secrets
import string
import sys
from typing import Callable, TextIO

from bson import ObjectId

from src.database.models.user import User
from src.database.mongo import cli_connection
from src.schemas.virtual_participants import ROOM_PATTERN
from utils import passwords
from utils.auth import create_user
from utils.room_owner import get_room_owner, release_room, set_room_owner

# Same bounds as sign up (SignupRequest.password).
PASSWORD_MIN_LENGTH = 10
PASSWORD_MAX_LENGTH = 128

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_random_password(length: int = 20) -> str:
    """Generate a cryptographically secure random password."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


async def create_admin(
    email: str,
    password: str | None = None,
    *,
    first_name: str | None = None,
    last_name: str | None = None,
) -> int:
    if not email:
        print("Email is required", file=sys.stderr)
        return 1

    async with cli_connection():
        if await User.find_by_email(email) is not None:
            print(f"Email {email} already exists", file=sys.stderr)
            return 1

        resolved_password = password or generate_random_password()

        try:
            user = await create_user(
                email=email,
                password=resolved_password,
                first_name=first_name,
                last_name=last_name,
                is_admin=True,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    logger.info(
        "Admin user %s created successfully with id %s",
        user.email,
        user.id,
    )
    print(f"Email: {user.email}")
    print(f"Password: {resolved_password}")
    return 0


def read_new_password(
    stdin: TextIO | None = None,
    prompt: Callable[[str], str] = getpass.getpass,
) -> str:
    """Prompt twice without echo on a terminal; otherwise read the first line of stdin.

    Raises ValueError with a message that never contains the password.
    """
    stdin = stdin if stdin is not None else sys.stdin
    if stdin.isatty():
        first = prompt("New password: ")
        if prompt("Repeat new password: ") != first:
            raise ValueError("The two passwords differ")
        password = first
    else:
        password = stdin.readline().rstrip("\r\n")
    if not PASSWORD_MIN_LENGTH <= len(password) <= PASSWORD_MAX_LENGTH:
        raise ValueError(
            f"The password must be {PASSWORD_MIN_LENGTH} to {PASSWORD_MAX_LENGTH} characters"
        )
    return password


async def reset_password(email: str, password: str) -> int:
    """Store a new argon2id hash for `email`. Prints the email only, never the password."""
    async with cli_connection():
        user = await User.find_by_email(email)
        if user is None:
            print(f"No account with email {email.lower()}", file=sys.stderr)
            return 1
        new_hash = await passwords.hash_password_async(password)
        await User.get_collection().update_one(
            {"_id": ObjectId(user.id)},
            {"$set": {"password_hash": new_hash}},
        )
    print(f"Password updated for {user.email}")
    # Refresh tokens carry no version yet (backlog), so a reset does not end open sessions.
    print("Sessions opened before the reset stay valid until their refresh token expires")
    return 0


async def migrate_hashes(*, dry_run: bool = False) -> int:
    """Wrap every legacy sha256 digest as `legacy$` + argon2id(digest). Idempotent.

    The update is conditional on the old value, so a login that replaced the hash meanwhile
    wins. Each user is then upgraded to a plain argon2id hash at the next successful login.
    """
    wrapped = skipped = unknown = 0
    async with cli_connection():
        collection = User.get_collection()
        async for doc in collection.find({}, {"password_hash": 1}):
            stored = doc.get("password_hash") or ""
            if stored.startswith("$argon2") or stored.startswith(passwords.LEGACY_PREFIX):
                skipped += 1
                continue
            if not passwords.is_legacy_digest(stored):
                unknown += 1
                logger.warning("User %s has a password hash in an unknown format", doc["_id"])
                continue
            if not dry_run:
                new_value = await passwords.wrap_legacy_async(stored)
                result = await collection.update_one(
                    {"_id": doc["_id"], "password_hash": stored},
                    {"$set": {"password_hash": new_value}},
                )
                if result.modified_count != 1:
                    skipped += 1
                    continue
            wrapped += 1
    verb = "would wrap" if dry_run else "wrapped"
    print(f"Legacy hashes {verb}: {wrapped}, already current: {skipped}, unknown format: {unknown}")
    return 0


async def _run_reset_password(args: argparse.Namespace) -> int:
    try:
        password = read_new_password()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return await reset_password(args.email, password)


async def _run_migrate_hashes(args: argparse.Namespace) -> int:
    return await migrate_hashes(dry_run=args.dry_run)


async def _run_create_admin(args: argparse.Namespace) -> int:
    return await create_admin(
        email=args.email,
        password=args.password,
        first_name=args.first_name,
        last_name=args.last_name,
    )


def _room_or_none(room: str) -> str | None:
    room = room.lower()
    return room if re.fullmatch(ROOM_PATTERN, room) else None


async def room_owner(action: str, room: str, email: str | None = None) -> int:
    """Show, set or release the owner of a room.

    `set` pre-claims a room for an account, for example the presenter's demo room, and replaces
    any current owner. `release` frees a squatted room; the next claim takes it.
    """
    room_name = _room_or_none(room)
    if room_name is None:
        print(f"Room must match {ROOM_PATTERN} after lower-casing", file=sys.stderr)
        return 1

    async with cli_connection():
        if action == "show":
            owner = await get_room_owner(room_name)
            print(f"{room_name}: {owner or 'free'}")
            return 0
        if action == "release":
            released = await release_room(room_name)
            print(f"{room_name}: {'released' if released else 'was free'}")
            return 0
        if action == "set":
            user = await User.find_by_email(email or "")
            if user is None:
                print(f"No account with email {email}", file=sys.stderr)
                return 1
            await set_room_owner(room_name=room_name, user_id=user.email)
            print(f"{room_name}: {user.email}")
            return 0
    print(f"Unknown action {action}", file=sys.stderr)
    return 1


async def _run_room_owner(args: argparse.Namespace) -> int:
    return await room_owner(args.action, args.room, getattr(args, "email", None))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Meetween FastAPI server operational commands",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_admin_parser = subparsers.add_parser(
        "create-admin",
        help="Create an admin user in MongoDB",
    )
    create_admin_parser.add_argument("email", help="Admin email address")
    create_admin_parser.add_argument(
        "--password",
        help="Account password (a random password is generated if omitted)",
    )
    create_admin_parser.add_argument("--first-name", dest="first_name")
    create_admin_parser.add_argument("--last-name", dest="last_name")
    create_admin_parser.set_defaults(func=_run_create_admin)

    reset_parser = subparsers.add_parser(
        "reset-password",
        help="Set a new password for an account (prompts, or reads one line from stdin)",
    )
    reset_parser.add_argument("email", help="Account email address")
    reset_parser.set_defaults(func=_run_reset_password)

    migrate_parser = subparsers.add_parser(
        "migrate-hashes",
        help="Wrap legacy sha256 password digests in argon2id (run once at deploy)",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count the legacy digests without writing",
    )
    migrate_parser.set_defaults(func=_run_migrate_hashes)

    room_owner_parser = subparsers.add_parser(
        "room-owner",
        help="Show, set or release the owner of a Jitsi room",
    )
    room_owner_actions = room_owner_parser.add_subparsers(dest="action", required=True)
    show_parser = room_owner_actions.add_parser("show", help="Print the room owner")
    show_parser.add_argument("room")
    set_parser = room_owner_actions.add_parser(
        "set", help="Make an account the room owner, replacing any current owner"
    )
    set_parser.add_argument("room")
    set_parser.add_argument("email", help="Email of an existing account")
    release_parser = room_owner_actions.add_parser(
        "release", help="Free the room; the next claim takes it"
    )
    release_parser.add_argument("room")
    room_owner_parser.set_defaults(func=_run_room_owner)

    args = parser.parse_args(argv)
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
