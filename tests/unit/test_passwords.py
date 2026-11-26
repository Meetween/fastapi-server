"""argon2id hashing, the legacy formats, and the thread pool that keeps hashing off the loop."""
import asyncio
import hashlib
import threading
import time

import pytest
from argon2 import PasswordHasher, Type

from utils import passwords

PASSWORD = "correct horse battery"


def test_hash_is_argon2id_with_the_owasp_parameters() -> None:
    stored = passwords.hash_password(PASSWORD)
    assert stored.startswith("$argon2id$v=19$m=19456,t=2,p=1$")
    assert passwords.hash_password(PASSWORD) != stored, "each hash has its own salt"


def test_current_hash_verifies_without_a_rehash() -> None:
    stored = passwords.hash_password(PASSWORD)
    assert passwords.verify_password(stored, PASSWORD) == passwords.Verification(ok=True)
    assert passwords.verify_password(stored, PASSWORD + "x") == passwords.Verification(ok=False)


def test_outdated_parameters_are_rehashed_on_success() -> None:
    weak = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1, type=Type.ID).hash(PASSWORD)
    result = passwords.verify_password(weak, PASSWORD)
    assert result.ok
    assert result.new_hash.startswith("$argon2id$v=19$m=19456,t=2,p=1$")
    assert passwords.verify_password(weak, "wrong password").new_hash is None


def test_argon2i_is_upgraded_to_argon2id() -> None:
    other = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1, type=Type.I).hash(PASSWORD)
    assert passwords.verify_password(other, PASSWORD).new_hash.startswith("$argon2id$")


def test_raw_legacy_digest_verifies_once_and_asks_for_argon2id() -> None:
    legacy = passwords.legacy_digest(PASSWORD)
    assert passwords.is_legacy_digest(legacy)
    result = passwords.verify_password(legacy, PASSWORD)
    assert result.ok and result.new_hash.startswith("$argon2id$")
    assert passwords.verify_password(legacy, "wrong password") == passwords.Verification(ok=False)


def test_wrapped_legacy_digest_verifies_inside_argon2() -> None:
    legacy = passwords.legacy_digest(PASSWORD)
    wrapped = passwords.wrap_legacy(legacy)
    assert wrapped.startswith("legacy$$argon2id$v=19$m=19456,t=2,p=1$")
    assert legacy not in wrapped, "the fast digest must not stay in the database"
    result = passwords.verify_password(wrapped, PASSWORD)
    assert result.ok and result.new_hash.startswith("$argon2id$")
    assert passwords.verify_password(wrapped, "wrong password") == passwords.Verification(ok=False)


def test_legacy_digest_is_the_old_peppered_sha256() -> None:
    pepper = passwords.settings.app_jwt.secret_key
    expected = hashlib.sha256(f"{pepper}:{PASSWORD}".encode("utf-8")).hexdigest()
    assert passwords.legacy_digest(PASSWORD) == expected


@pytest.mark.parametrize("value", ["", "plain", "A" * 64, "$argon2id$broken", "legacy$nothing"])
def test_unknown_or_malformed_hashes_never_verify(value: str) -> None:
    assert passwords.verify_password(value, PASSWORD) == passwords.Verification(ok=False)


def test_wrap_refuses_anything_but_a_legacy_digest() -> None:
    with pytest.raises(ValueError):
        passwords.wrap_legacy(passwords.hash_password(PASSWORD))


@pytest.mark.parametrize("make_stored", [
    lambda: passwords.hash_password(PASSWORD),
    lambda: passwords.legacy_digest(PASSWORD),
    lambda: passwords.wrap_legacy(passwords.legacy_digest(PASSWORD)),
    lambda: "unknown-format",
], ids=["argon2id", "raw-legacy", "wrapped-legacy", "unknown"])
def test_every_format_costs_one_argon2_verification(monkeypatch, make_stored) -> None:
    stored = make_stored()
    calls = []
    real = passwords._argon2_matches
    monkeypatch.setattr(passwords, "_argon2_matches", lambda h, p: calls.append(h) or real(h, p))
    passwords.verify_password(stored, "wrong password")
    assert len(calls) == 1


def test_dummy_hash_uses_the_current_parameters() -> None:
    assert passwords._dummy_hash().startswith("$argon2id$v=19$m=19456,t=2,p=1$")
    passwords.burn_dummy(PASSWORD)  # never raises, never matches


async def test_async_helpers_run_in_the_hash_pool() -> None:
    seen = []
    original = passwords.hash_password

    def spy(password: str) -> str:
        seen.append(threading.current_thread().name)
        return original(password)

    passwords.hash_password = spy
    try:
        stored = await passwords.hash_password_async(PASSWORD)
    finally:
        passwords.hash_password = original
    assert seen and seen[0].startswith("pwhash")
    assert (await passwords.verify_password_async(stored, PASSWORD)).ok


async def test_no_more_than_four_hashes_run_at_once() -> None:
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def busy() -> None:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1

    await asyncio.gather(*(passwords._off_loop(busy) for _ in range(10)))
    assert state["peak"] == passwords.HASH_CONCURRENCY == 4
