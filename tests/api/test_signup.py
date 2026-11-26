"""Open sign up, feature flags, login hardening, and the two CLI commands of B3.

The throttle is process wide; every test here starts from an empty limiter. TestClient's peer is
the string "testclient", so the client address key is `unknown` unless a test passes an address.
"""
import asyncio
import io

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

import cli
from config import settings
from main import app
from src.database.models.user import User
from utils import passwords
from utils import auth as auth_utils
from utils.rate_limit import limiter

PASSWORD = "pytest-signup-pass"


@pytest.fixture
def signup_on(monkeypatch):
    monkeypatch.setattr(settings, "signup_enabled", True)


def _agentar(monkeypatch, **changes) -> None:
    monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update=changes))


def _body(email: str = "new.user@meetween.test", **over) -> dict:
    return {"email": email, "password": PASSWORD, "first_name": "Laura", "last_name": "Bianchi", **over}


def _client(ip: str) -> TestClient:
    return TestClient(app, client=(ip, 50000))


def _stored_hash(email: str) -> str:
    return asyncio.run(User.find_by_email(email)).password_hash


def _insert_user(email: str, password_hash: str, *, is_active: bool = True) -> None:
    asyncio.run(User.create(email=email, password_hash=password_hash, first_name="Old", last_name="User",
                            is_active=is_active))


def _login(client: TestClient, email: str, password: str):
    return client.post("/api/auth/login", json={"email": email, "password": password})


# --- GET /api/auth/config -------------------------------------------------------------------

@pytest.mark.parametrize("flags", [(False, False, False), (True, False, True), (True, True, False)])
def test_config_serves_the_three_flags_without_auth(test_client, monkeypatch, flags) -> None:
    signup, email, calendar = flags
    monkeypatch.setattr(settings, "signup_enabled", signup)
    monkeypatch.setattr(settings, "email_enabled", email)
    monkeypatch.setattr(settings, "calendar_enabled", calendar)
    response = test_client.get("/api/auth/config", headers={})
    assert response.status_code == 200
    assert response.json() == {"signup_enabled": signup, "email_enabled": email, "calendar_enabled": calendar}


# --- POST /api/auth/signup ------------------------------------------------------------------

@pytest.mark.parametrize("body", [_body(), {"junk": True}], ids=["valid", "invalid"])
def test_signup_off_is_403_before_validation(test_client, monkeypatch, body) -> None:
    monkeypatch.setattr(settings, "signup_enabled", False)
    response = test_client.post("/api/auth/signup", json=body)
    assert response.status_code == 403
    assert response.json() == {"detail": "Sign up is disabled", "code": "signup_disabled"}
    assert asyncio.run(User.find_by_email("new.user@meetween.test")) is None


def test_signup_creates_an_active_user_and_logs_in(test_client, signup_on) -> None:
    response = test_client.post("/api/auth/signup", json=_body(email="New.User@Meetween.TEST"))
    assert response.status_code == 201
    tokens = response.json()
    assert tokens["token_type"] == "bearer" and tokens["access_token"] and tokens["refresh_token"]

    user = asyncio.run(User.find_by_email("new.user@meetween.test"))
    assert user.email == "new.user@meetween.test"
    assert (user.first_name, user.last_name, user.is_active, user.is_admin) == ("Laura", "Bianchi", True, False)
    assert user.password_hash.startswith("$argon2id$v=19$m=19456,t=2,p=1$")

    # The access token works on a protected route (404: authenticated, no context stored).
    me = test_client.get("/api/auth/context/any-room", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert me.status_code == 404
    assert _login(test_client, "new.user@meetween.test", PASSWORD).status_code == 200


def test_signup_with_a_registered_email_is_409(test_client, signup_on) -> None:
    assert test_client.post("/api/auth/signup", json=_body()).status_code == 201
    response = test_client.post("/api/auth/signup", json=_body(email="NEW.user@meetween.test"))
    assert response.status_code == 409
    assert response.json()["code"] == "email_taken"
    assert response.json()["detail"] == "An account with this email exists. Log in"


def test_signup_race_on_the_unique_index_is_409(test_client, signup_on, monkeypatch) -> None:
    async def nobody(email):
        return None

    _insert_user("new.user@meetween.test", passwords.hash_password("other-password"))
    monkeypatch.setattr(User, "find_by_email", staticmethod(nobody))
    response = test_client.post("/api/auth/signup", json=_body())
    assert response.status_code == 409
    assert response.json()["code"] == "email_taken"


@pytest.mark.parametrize(
    "over",
    [
        {"first_name": "Laura (AI)"},
        {"last_name": "Agentar"},
        {"first_name": "my aGeNtAr"},
        {"first_name": "La​ura"},
        {"last_name": "Bi\x07anchi"},
        {"first_name": ""},
        {"first_name": "x" * 65},
        {"password": "short"},
        {"password": "x" * 129},
        {"email": "not-an-email"},
        {"email": "a@b"},
        {"email": "a@" + "b" * 250 + ".it"},
        {"is_admin": True},
    ],
    ids=lambda o: next(iter(o)) + "=" + repr(next(iter(o.values())))[:20],
)
def test_signup_validation_is_422(test_client, signup_on, over) -> None:
    response = test_client.post("/api/auth/signup", json=_body(**over))
    assert response.status_code == 422
    assert asyncio.run(User.find_by_email("new.user@meetween.test")) is None


def test_signup_is_throttled_per_ip(signup_on) -> None:
    client = _client("198.51.100.7")
    for i in range(settings.agentar.signup_rate_per_ip):
        assert client.post("/api/auth/signup", json=_body(email=f"u{i}@meetween.test")).status_code == 201
    response = client.post("/api/auth/signup", json=_body(email="late@meetween.test"))
    assert response.status_code == 429
    assert response.json()["code"] == "rate_limited"
    assert 1 <= int(response.headers["retry-after"]) <= 3600
    assert asyncio.run(User.find_by_email("late@meetween.test")) is None
    # Another address is not affected.
    assert _client("198.51.100.8").post("/api/auth/signup", json=_body(email="late@meetween.test")).status_code == 201


def test_signup_is_throttled_per_email_across_addresses(signup_on, monkeypatch) -> None:
    calls = []
    real = passwords.hash_password_async

    async def spy(password):
        calls.append(password)
        return await real(password)

    monkeypatch.setattr(passwords, "hash_password_async", spy)
    limit = settings.agentar.signup_rate_per_email
    statuses = [
        _client(f"198.51.100.{i}").post("/api/auth/signup", json=_body()).status_code for i in range(limit + 1)
    ]
    assert statuses == [201] + [409] * (limit - 1) + [429]
    assert len(calls) == 1, "only the first attempt reached the hasher"


def test_signup_key_follows_x_real_ip_from_a_trusted_proxy(signup_on, monkeypatch) -> None:
    _agentar(monkeypatch, trusted_proxy_ips=("172.18.0.0/16",), signup_rate_per_ip=1)
    proxy = _client("172.18.0.3")
    first = proxy.post("/api/auth/signup", json=_body(email="a@meetween.test"), headers={"X-Real-IP": "203.0.113.1"})
    second = proxy.post("/api/auth/signup", json=_body(email="b@meetween.test"), headers={"X-Real-IP": "203.0.113.2"})
    third = proxy.post("/api/auth/signup", json=_body(email="c@meetween.test"), headers={"X-Real-IP": "203.0.113.1"})
    assert (first.status_code, second.status_code, third.status_code) == (201, 201, 429)


def test_x_real_ip_from_an_untrusted_peer_is_ignored(signup_on, monkeypatch) -> None:
    _agentar(monkeypatch, trusted_proxy_ips=("172.18.0.0/16",), signup_rate_per_ip=1)
    client = _client("198.51.100.7")
    assert client.post("/api/auth/signup", json=_body(email="a@meetween.test"),
                       headers={"X-Real-IP": "203.0.113.1"}).status_code == 201
    assert client.post("/api/auth/signup", json=_body(email="b@meetween.test"),
                       headers={"X-Real-IP": "203.0.113.2"}).status_code == 429


# --- POST /api/auth/login -------------------------------------------------------------------

def test_existing_legacy_user_logs_in_and_is_rehashed(test_client) -> None:
    """A user created by the old code (raw peppered sha256) keeps working and is upgraded."""
    _insert_user("legacy@meetween.test", passwords.legacy_digest(PASSWORD))
    assert _login(test_client, "legacy@meetween.test", "wrong-password").status_code == 401
    assert passwords.is_legacy_digest(_stored_hash("legacy@meetween.test")), "a failure changes nothing"

    response = _login(test_client, "Legacy@Meetween.test", PASSWORD)
    assert response.status_code == 200
    stored = _stored_hash("legacy@meetween.test")
    assert stored.startswith("$argon2id$v=19$m=19456,t=2,p=1$")
    assert _login(test_client, "legacy@meetween.test", PASSWORD).status_code == 200
    assert _stored_hash("legacy@meetween.test") == stored, "a current hash is not rewritten"


def test_wrapped_legacy_user_logs_in_and_is_rehashed(test_client) -> None:
    _insert_user("wrapped@meetween.test", passwords.wrap_legacy(passwords.legacy_digest(PASSWORD)))
    assert _login(test_client, "wrapped@meetween.test", PASSWORD).status_code == 200
    assert _stored_hash("wrapped@meetween.test").startswith("$argon2id$")


@pytest.mark.parametrize("wrapped", [False, True], ids=["raw", "wrapped"])
def test_old_user_logs_in_after_the_jwt_key_rotation(test_client, monkeypatch, wrapped) -> None:
    """Gap G: a short JWT_SECRET_KEY is rotated. LEGACY_PASSWORD_PEPPER keeps the old key for the
    legacy digests, the user keeps the password and is upgraded to argon2id at the login."""
    old_key, new_key = "short-old-key", "r" * 40
    monkeypatch.setattr(settings, "legacy_password_pepper", "")
    monkeypatch.setattr(settings, "_app_jwt", settings.app_jwt.model_copy(update={"secret_key": old_key}))
    digest = passwords.legacy_digest(PASSWORD)
    _insert_user("rotated@meetween.test", passwords.wrap_legacy(digest) if wrapped else digest)

    monkeypatch.setattr(settings, "_app_jwt", settings.app_jwt.model_copy(update={"secret_key": new_key}))
    assert _login(test_client, "rotated@meetween.test", PASSWORD).status_code == 401, "no pepper, locked out"

    monkeypatch.setattr(settings, "legacy_password_pepper", old_key)
    response = _login(test_client, "rotated@meetween.test", PASSWORD)
    assert response.status_code == 200, response.text
    assert _stored_hash("rotated@meetween.test").startswith("$argon2id$v=19$m=19456,t=2,p=1$")
    claims = auth_utils.verify_jwt_token(response.json()["access_token"], settings.app_jwt.audience_access)
    assert claims["sub"] == "rotated@meetween.test", "the new token is signed with the new key"

    # Once rehashed the pepper is no longer needed.
    monkeypatch.setattr(settings, "legacy_password_pepper", "")
    assert _login(test_client, "rotated@meetween.test", PASSWORD).status_code == 200


def test_rehash_does_not_overwrite_a_concurrent_change(monkeypatch) -> None:
    """The upgrade is conditional on the old value: a hash changed meanwhile wins."""
    _insert_user("race@meetween.test", passwords.legacy_digest(PASSWORD))

    async def verify_while_someone_resets(stored, password):
        await User.get_collection().update_one({"email": "race@meetween.test"},
                                               {"$set": {"password_hash": "$argon2id$reset-meanwhile"}})
        return passwords.Verification(ok=True, new_hash="$argon2id$from-this-login")

    monkeypatch.setattr(passwords, "verify_password_async", verify_while_someone_resets)
    assert asyncio.run(auth_utils.verify_user("race@meetween.test", PASSWORD)) is True
    assert _stored_hash("race@meetween.test") == "$argon2id$reset-meanwhile"


def test_unknown_email_costs_a_dummy_verification(test_client, monkeypatch) -> None:
    burned = []
    real = passwords.burn_dummy_async

    async def spy(password):
        burned.append(1)
        await real(password)

    monkeypatch.setattr(passwords, "burn_dummy_async", spy)
    response = _login(test_client, "nobody@meetween.test", PASSWORD)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid credentials"}
    assert burned == [1]


def test_inactive_user_is_401_even_with_the_right_password(test_client) -> None:
    _insert_user("inactive@meetween.test", passwords.hash_password(PASSWORD), is_active=False)
    response = _login(test_client, "inactive@meetween.test", PASSWORD)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid credentials"}


@pytest.mark.parametrize("body", [{"email": "a" * 255, "password": "x"}, {"email": "a@b.it", "password": "x" * 129}])
def test_login_length_caps_are_422(test_client, body) -> None:
    assert test_client.post("/api/auth/login", json=body).status_code == 422


def test_failed_logins_are_throttled_per_address_and_email(monkeypatch) -> None:
    _agentar(monkeypatch, login_fail_limit=3)
    _insert_user("target@meetween.test", passwords.hash_password(PASSWORD))
    attacker = _client("198.51.100.7")
    for _ in range(3):
        assert _login(attacker, "target@meetween.test", "wrong-password").status_code == 401

    verified = []
    real = passwords.verify_password_async

    async def spy(stored, password):
        verified.append(1)
        return await real(stored, password)

    monkeypatch.setattr(passwords, "verify_password_async", spy)
    blocked = _login(attacker, "target@meetween.test", PASSWORD)
    assert blocked.status_code == 429
    assert blocked.json()["code"] == "rate_limited"
    assert 1 <= int(blocked.headers["retry-after"]) <= settings.agentar.login_fail_window_minutes * 60
    assert verified == [], "a throttled login never reaches the hasher"

    # The owner on another address, and another account on the same address, still log in.
    assert _login(_client("198.51.100.8"), "target@meetween.test", PASSWORD).status_code == 200
    _insert_user("other@meetween.test", passwords.hash_password(PASSWORD))
    assert _login(attacker, "other@meetween.test", PASSWORD).status_code == 200


def test_failed_logins_are_throttled_per_address_over_all_emails(monkeypatch) -> None:
    _agentar(monkeypatch, login_fail_ip_limit=4)
    attacker = _client("198.51.100.7")
    for i in range(4):
        assert _login(attacker, f"spray{i}@meetween.test", "wrong-password").status_code == 401
    assert _login(attacker, "spray9@meetween.test", "wrong-password").status_code == 429
    assert _login(_client("198.51.100.8"), "spray9@meetween.test", "wrong-password").status_code == 401


def test_a_success_clears_the_address_and_email_counter(monkeypatch) -> None:
    _agentar(monkeypatch, login_fail_limit=3)
    _insert_user("typo@meetween.test", passwords.hash_password(PASSWORD))
    client = _client("198.51.100.7")
    for _ in range(2):
        assert _login(client, "typo@meetween.test", "wrong-password").status_code == 401
    assert _login(client, "typo@meetween.test", PASSWORD).status_code == 200
    for _ in range(2):
        assert _login(client, "typo@meetween.test", "wrong-password").status_code == 401
    assert _login(client, "typo@meetween.test", PASSWORD).status_code == 200


def test_a_concurrent_burst_reaches_the_hasher_at_most_limit_times(monkeypatch) -> None:
    """Every attempt is recorded before the hash, so parallel requests cannot all pass the check
    while the first ones are still hashing."""
    import httpx

    _agentar(monkeypatch, login_fail_limit=10, login_fail_ip_limit=50)
    _insert_user("burst@meetween.test", passwords.hash_password(PASSWORD))
    verified = []

    async def slow_verify(stored, password):
        verified.append(1)
        await asyncio.sleep(0.05)
        return passwords.Verification(ok=False, new_hash=None)

    monkeypatch.setattr(passwords, "verify_password_async", slow_verify)

    async def burst() -> list[int]:
        transport = httpx.ASGITransport(app=app, client=("198.51.100.7", 50000))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
            body = {"email": "burst@meetween.test", "password": "wrong-password"}
            responses = await asyncio.gather(*(http.post("/api/auth/login", json=body) for _ in range(30)))
        return [r.status_code for r in responses]

    codes = asyncio.run(burst())
    assert len(verified) <= 10, f"{len(verified)} verifications for a limit of 10"
    assert codes.count(401) == len(verified)
    assert codes.count(429) == 30 - len(verified)


def test_successful_logins_do_not_use_the_address_budget(monkeypatch) -> None:
    _agentar(monkeypatch, login_fail_limit=2, login_fail_ip_limit=3)
    _insert_user("office@meetween.test", passwords.hash_password(PASSWORD))
    shared = _client("198.51.100.7")
    for _ in range(5):
        assert _login(shared, "office@meetween.test", PASSWORD).status_code == 200
    assert _login(shared, "office@meetween.test", "wrong-password").status_code == 401


def test_the_throttle_window_slides(monkeypatch) -> None:
    clock = {"t": 5000.0}
    monkeypatch.setattr(limiter, "_now", lambda: clock["t"])
    _agentar(monkeypatch, login_fail_limit=1, login_fail_window_minutes=15)
    _insert_user("slide@meetween.test", passwords.hash_password(PASSWORD))
    client = _client("198.51.100.7")
    assert _login(client, "slide@meetween.test", "wrong-password").status_code == 401
    blocked = _login(client, "slide@meetween.test", PASSWORD)
    assert (blocked.status_code, blocked.headers["retry-after"]) == (429, "900")
    clock["t"] += 900
    assert _login(client, "slide@meetween.test", PASSWORD).status_code == 200


# --- cli.py migrate-hashes and reset-password -----------------------------------------------

async def test_migrate_hashes_wraps_legacy_digests_once(capsys) -> None:
    legacy = passwords.legacy_digest(PASSWORD)
    await User.create(email="m1@meetween.test", password_hash=legacy)
    await User.create(email="m2@meetween.test", password_hash=await passwords.hash_password_async(PASSWORD))
    await User.create(email="m3@meetween.test", password_hash="something-else")

    assert await cli.migrate_hashes(dry_run=True) == 0
    assert "would wrap: 1, already current: 1, unknown format: 1" in capsys.readouterr().out
    assert (await User.find_by_email("m1@meetween.test")).password_hash == legacy

    assert await cli.migrate_hashes() == 0
    assert "wrapped: 1, already current: 1, unknown format: 1" in capsys.readouterr().out
    wrapped = (await User.find_by_email("m1@meetween.test")).password_hash
    assert wrapped.startswith("legacy$$argon2id$") and legacy not in wrapped

    assert await cli.migrate_hashes() == 0
    assert "wrapped: 0, already current: 2, unknown format: 1" in capsys.readouterr().out
    assert (await User.find_by_email("m1@meetween.test")).password_hash == wrapped


def test_migrated_user_logs_in_and_gets_a_plain_argon2id_hash(test_client) -> None:
    _insert_user("migrated@meetween.test", passwords.legacy_digest(PASSWORD))
    assert asyncio.run(cli.migrate_hashes()) == 0
    assert _stored_hash("migrated@meetween.test").startswith(passwords.LEGACY_PREFIX)
    assert _login(test_client, "migrated@meetween.test", "wrong-password").status_code == 401
    assert _login(test_client, "migrated@meetween.test", PASSWORD).status_code == 200
    assert _stored_hash("migrated@meetween.test").startswith("$argon2id$")


def test_main_dispatches_migrate_hashes(capsys) -> None:
    assert cli.main(["migrate-hashes", "--dry-run"]) == 0
    assert "Legacy hashes would wrap" in capsys.readouterr().out


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_read_new_password_from_a_pipe() -> None:
    assert cli.read_new_password(io.StringIO("a-new-password-1\nignored\n")) == "a-new-password-1"


def test_read_new_password_prompts_twice_on_a_terminal() -> None:
    answers = iter(["a-new-password-1", "a-new-password-1"])
    assert cli.read_new_password(_Tty(), prompt=lambda _: next(answers)) == "a-new-password-1"
    answers = iter(["a-new-password-1", "a-new-password-2"])
    with pytest.raises(ValueError, match="differ"):
        cli.read_new_password(_Tty(), prompt=lambda _: next(answers))


@pytest.mark.parametrize("line", ["short\n", "x" * 129 + "\n", "\n"])
def test_read_new_password_enforces_the_signup_bounds(line: str) -> None:
    with pytest.raises(ValueError) as info:
        cli.read_new_password(io.StringIO(line))
    assert line.strip() == "" or line.strip() not in str(info.value)


def test_reset_password_sets_a_new_hash_and_prints_no_secret(test_client, capsys) -> None:
    _insert_user("forgot@meetween.test", passwords.legacy_digest(PASSWORD))
    new_password = "brand-new-password-42"
    assert asyncio.run(cli.reset_password("Forgot@Meetween.test", new_password)) == 0
    captured = capsys.readouterr()
    assert "Password updated for forgot@meetween.test" in captured.out
    assert new_password not in captured.out + captured.err
    assert _stored_hash("forgot@meetween.test").startswith("$argon2id$")
    assert _login(test_client, "forgot@meetween.test", PASSWORD).status_code == 401
    assert _login(test_client, "forgot@meetween.test", new_password).status_code == 200


def test_reset_password_for_an_unknown_email_fails(capsys) -> None:
    assert asyncio.run(cli.reset_password("ghost@meetween.test", "brand-new-password-42")) == 1
    assert "No account with email ghost@meetween.test" in capsys.readouterr().err


def test_main_reset_password_reads_stdin(monkeypatch, capsys) -> None:
    _insert_user("piped@meetween.test", passwords.hash_password(PASSWORD))
    monkeypatch.setattr("sys.stdin", io.StringIO("piped-new-password\n"))
    assert cli.main(["reset-password", "piped@meetween.test"]) == 0
    out = capsys.readouterr()
    assert "piped-new-password" not in out.out + out.err
    stored = _stored_hash("piped@meetween.test")
    assert passwords.verify_password(stored, "piped-new-password").ok


def test_main_reset_password_refuses_a_short_password(monkeypatch, capsys) -> None:
    _insert_user("piped@meetween.test", passwords.hash_password(PASSWORD))
    monkeypatch.setattr("sys.stdin", io.StringIO("short\n"))
    assert cli.main(["reset-password", "piped@meetween.test"]) == 1
    assert "10 to 128 characters" in capsys.readouterr().err
    assert passwords.verify_password(_stored_hash("piped@meetween.test"), PASSWORD).ok


def test_user_ids_are_object_ids() -> None:
    """verify_user and reset_password address the user by `_id`; keep that assumption tested."""
    _insert_user("id@meetween.test", passwords.hash_password(PASSWORD))
    user = asyncio.run(User.find_by_email("id@meetween.test"))
    assert ObjectId.is_valid(user.id)
