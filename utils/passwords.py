"""Password hashing: argon2id, legacy digests, and the dummy hash for unknown accounts.

Stored formats in `users.password_hash`:

- `$argon2id$v=19$m=19456,t=2,p=1$...`: the PHC string of argon2-cffi. Current format.
- `legacy$` + PHC string: a legacy digest wrapped by `cli.py migrate-hashes`. The inner hash is
  argon2id of `sha256(<pepper>:password)` as lower case hex, so no fast unsalted digest stays in
  the database. Verified once, then replaced with a plain argon2id hash.
- 64 lower case hex characters: a legacy digest that `migrate-hashes` has not wrapped yet.
  Accepted so a missed deploy step locks nobody out, and replaced at the next login.

The pepper of both legacy formats is `LEGACY_PASSWORD_PEPPER` when set, else `JWT_SECRET_KEY`.
Rotating `JWT_SECRET_KEY` therefore keeps legacy users in when the old key moves to
`LEGACY_PASSWORD_PEPPER` (see deploy/fastapi.env.example).

Hashing and verification run in a pool of `HASH_CONCURRENCY` threads, never on the event loop
that also carries captions. The pool is the concurrency bound: at most that many hashes run at
once (each holds `MEMORY_COST` KiB), further calls queue. argon2-cffi releases the GIL inside
the C call, so the threads run in parallel.
"""
import asyncio
import functools
import hashlib
import hmac
import logging
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, TypeVar

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from config import settings

logger = logging.getLogger(__name__)

# OWASP Password Storage Cheat Sheet, argon2id minimum: 19 MiB, 2 iterations, 1 lane.
TIME_COST = 2
MEMORY_COST = 19456
PARALLELISM = 1
HASH_LEN = 32
SALT_LEN = 16
HASH_CONCURRENCY = 4

LEGACY_PREFIX = "legacy$"
_LEGACY_HEX = re.compile(r"[0-9a-f]{64}")

_hasher = PasswordHasher(
    time_cost=TIME_COST,
    memory_cost=MEMORY_COST,
    parallelism=PARALLELISM,
    hash_len=HASH_LEN,
    salt_len=SALT_LEN,
    type=Type.ID,
)
_pool = ThreadPoolExecutor(max_workers=HASH_CONCURRENCY, thread_name_prefix="pwhash")

T = TypeVar("T")


@dataclass(frozen=True)
class Verification:
    """Outcome of a check. `new_hash` is set when the stored value must be replaced."""

    ok: bool
    new_hash: str | None = None


def hash_password(password: str) -> str:
    """argon2id PHC string with the current parameters. Blocking: call from a thread."""
    return _hasher.hash(password)


def _legacy_pepper() -> str:
    return settings.legacy_password_pepper or settings.app_jwt.secret_key


def legacy_digest(password: str) -> str:
    """The pre-argon2 format: `sha256(pepper:password)` as lower case hex, the pepper being
    `LEGACY_PASSWORD_PEPPER` when set, else `JWT_SECRET_KEY`."""
    payload = f"{_legacy_pepper()}:{password}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_legacy_digest(stored: str) -> bool:
    return bool(_LEGACY_HEX.fullmatch(stored))


def wrap_legacy(stored: str) -> str:
    """`legacy$` + argon2id(hex digest). Blocking: call from a thread."""
    if not is_legacy_digest(stored):
        raise ValueError("not a legacy digest")
    return LEGACY_PREFIX + _hasher.hash(stored)


@functools.lru_cache(maxsize=1)
def _dummy_hash() -> str:
    """argon2id of a random secret nobody knows, with the current parameters."""
    return _hasher.hash(secrets.token_urlsafe(32))


def _argon2_matches(phc: str, secret: str) -> bool:
    try:
        return _hasher.verify(phc, secret)
    except VerifyMismatchError:
        return False
    except (VerificationError, InvalidHashError):
        logger.warning("Stored password hash could not be verified (malformed hash)")
        return False


def burn_dummy(password: str) -> None:
    """Spend the cost of one verification, so a missing account answers as slowly as a real one."""
    _argon2_matches(_dummy_hash(), password)


def verify_password(stored: str, password: str) -> Verification:
    """Check `password` against a stored value of any supported format. Blocking: call from a
    thread. Every format costs one argon2id verification, so timing does not tell them apart."""
    if stored.startswith("$argon2"):
        if not _argon2_matches(stored, password):
            return Verification(ok=False)
        if _hasher.check_needs_rehash(stored):
            return Verification(ok=True, new_hash=_hasher.hash(password))
        return Verification(ok=True)

    if stored.startswith(LEGACY_PREFIX):
        if not _argon2_matches(stored[len(LEGACY_PREFIX):], legacy_digest(password)):
            return Verification(ok=False)
        return Verification(ok=True, new_hash=_hasher.hash(password))

    if is_legacy_digest(stored):
        burn_dummy(password)
        if not hmac.compare_digest(stored, legacy_digest(password)):
            return Verification(ok=False)
        return Verification(ok=True, new_hash=_hasher.hash(password))

    logger.warning("Stored password hash has an unknown format")
    burn_dummy(password)
    return Verification(ok=False)


async def _off_loop(fn: Callable[..., T], *args) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pool, functools.partial(fn, *args))


async def hash_password_async(password: str) -> str:
    return await _off_loop(hash_password, password)


async def verify_password_async(stored: str, password: str) -> Verification:
    return await _off_loop(verify_password, stored, password)


async def burn_dummy_async(password: str) -> None:
    await _off_loop(burn_dummy, password)


async def wrap_legacy_async(stored: str) -> str:
    return await _off_loop(wrap_legacy, stored)
