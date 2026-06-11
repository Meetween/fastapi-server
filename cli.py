"""Operational CLI for the Meetween FastAPI server."""

import argparse
import asyncio
import logging
import secrets
import string
import sys

from src.database.models.user import User
from src.database.mongo import cli_connection
from utils.auth import create_user

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


async def _run_create_admin(args: argparse.Namespace) -> int:
    return await create_admin(
        email=args.email,
        password=args.password,
        first_name=args.first_name,
        last_name=args.last_name,
    )


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

    args = parser.parse_args(argv)
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
