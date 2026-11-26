"""Jitsi room tokens: key split, room owner gating, claim cap, the `#jwt=` fragment, bot tokens.

Tokens are checked the way Prosody checks them, with python-jose and the raw shared key, not
with a helper of this service, so a test passes only if Prosody would accept the token.
"""
import asyncio
import base64
import json
from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml
from fastapi.testclient import TestClient
from jose import JWTError, jwt

import cli
from api.errors import AgentarError
from config import MeetingJwtSettings, Settings, settings
from main import app
from src.database.models.meeting_context import MeetingContext
from src.database.models.room_owner import RoomOwner
from src.database.models.user import User
from src.schemas.auth import JitsiIssueRequest, JitsiIssueResponse
from tests.utils.helpers import TEST_ADMIN_EMAIL
from utils import room_owner as room_owner_module
from utils.auth import (
    JITSI_ALGORITHM,
    create_jitsi_bot_token,
    create_jitsi_token,
    create_user,
    verify_jitsi_token,
)
from utils.startup_checks import unsafe_settings

ISSUE = "/api/auth/jitsi/issue"
JITSI_KEY = "jitsi-shared-key-" + "j" * 32
BASE_URL = "https://meet.example.test"
ROOM = "weekly-sync"


@pytest.fixture(autouse=True)
def jitsi_key(monkeypatch) -> str:
    monkeypatch.setattr(settings, "jwt_jitsi_secret_key", JITSI_KEY)
    return JITSI_KEY


@pytest.fixture
def meet_base(monkeypatch) -> str:
    monkeypatch.setattr(
        settings,
        "_meeting_jwt",
        MeetingJwtSettings(
            issuer=settings.meeting_jwt.issuer,
            expire_minutes=settings.meeting_jwt.expire_minutes,
            meet_base_url=BASE_URL,
        ),
    )
    return BASE_URL


@pytest.fixture
def claim_cap(monkeypatch):
    def _set(cap: int) -> None:
        monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update={"room_claims_per_day": cap}))

    return _set


def _prosody_decode(token: str, key: str = JITSI_KEY) -> dict:
    """What Prosody does: signature with JWT_APP_SECRET, then audience and issuer."""
    return jwt.decode(token, key, algorithms=[JITSI_ALGORITHM], audience="jitsi", issuer="meetween")


def _payload_text(token: str) -> str:
    segment = token.split(".")[1]
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)).decode("utf-8")


def _issue(client: TestClient, headers: dict, room: str = ROOM, **extra):
    return client.post(ISSUE, json={"room_name": room, **extra}, headers=headers)


def _user(email: str) -> User:
    return asyncio.run(User.find_by_email(email))


def _owner_of(room: str) -> str | None:
    record = asyncio.run(RoomOwner.find_by_room(room))
    return record.owner_user_id if record else None


@pytest.mark.integration
class TestJitsiKey:
    def test_empty_key_is_503_and_nothing_is_claimed(self, test_client, user_a, monkeypatch):
        monkeypatch.setattr(settings, "jwt_jitsi_secret_key", "")

        response = _issue(test_client, user_a.headers)

        assert response.status_code == 503
        assert response.json()["code"] == "jitsi_key_missing"
        assert asyncio.run(RoomOwner.count_documents()) == 0
        assert asyncio.run(MeetingContext.count_documents()) == 0

    def test_token_verifies_with_the_shared_key_and_not_with_the_api_key(self, test_client, user_a):
        """X0 proved Prosody accepts JWT_JITSI_SECRET_KEY tokens and refuses JWT_SECRET_KEY ones."""
        token = _issue(test_client, user_a.headers).json()["jitsi_token"]

        assert _prosody_decode(token)["room"] == ROOM
        with pytest.raises(JWTError):
            _prosody_decode(token, key=settings.jwt_secret_key)

    def test_a_token_signed_with_the_api_key_is_refused(self, user_a):
        claims = _prosody_decode(
            create_jitsi_token(room=ROOM, user_id=user_a.id, name=None, moderator=True)
        )
        forged = jwt.encode(claims, settings.jwt_secret_key, algorithm=JITSI_ALGORITHM)

        with pytest.raises(JWTError):
            _prosody_decode(forged)
        with pytest.raises(JWTError):
            verify_jitsi_token(forged)

    def test_the_api_key_is_never_a_fallback(self, monkeypatch, user_a):
        monkeypatch.setattr(settings, "jwt_jitsi_secret_key", "")

        for mint in (
            lambda: create_jitsi_token(room=ROOM, user_id=user_a.id, name=None, moderator=True),
            lambda: create_jitsi_bot_token(ROOM),
        ):
            with pytest.raises(AgentarError) as raised:
                mint()
            assert raised.value.status_code == 503
            assert raised.value.code == "jitsi_key_missing"


@pytest.mark.integration
class TestOwnerToken:
    def test_first_caller_owns_the_room_and_gets_a_moderator_token(self, test_client, user_a):
        response = _issue(test_client, user_a.headers)

        assert response.status_code == 200
        body = response.json()
        assert body["room"] == ROOM
        assert body["moderator"] is True
        assert _owner_of(ROOM) == user_a.email

        claims = _prosody_decode(body["jitsi_token"])
        assert claims["iss"] == "meetween"
        assert claims["aud"] == "jitsi"
        assert claims["sub"] == "meet.jitsi"
        assert claims["room"] == ROOM
        assert claims["moderator"] is True
        assert claims["context"] == {"user": {"id": user_a.id, "name": "Alice Tester", "moderator": True}}
        assert claims["exp"] - claims["iat"] == 15 * 60
        assert claims["jti"]

    def test_no_claim_carries_the_email(self, test_client, user_a):
        token = _issue(test_client, user_a.headers).json()["jitsi_token"]

        payload = _payload_text(token)
        assert "@" not in payload
        assert user_a.email.split("@")[0] not in payload

    def test_every_token_has_its_own_jti(self, test_client, user_a):
        first = _issue(test_client, user_a.headers).json()["jitsi_token"]
        second = _issue(test_client, user_a.headers).json()["jitsi_token"]

        assert _prosody_decode(first)["jti"] != _prosody_decode(second)["jti"]

    def test_reissue_for_an_owned_room_claims_nothing_new(self, test_client, user_a):
        _issue(test_client, user_a.headers)
        _issue(test_client, user_a.headers)

        assert asyncio.run(RoomOwner.count_documents()) == 1

    def test_room_name_is_lowercased(self, test_client, user_a):
        response = _issue(test_client, user_a.headers, room="Weekly-Sync")

        assert response.status_code == 200
        assert response.json()["room"] == ROOM
        assert _prosody_decode(response.json()["jitsi_token"])["room"] == ROOM
        assert _owner_of(ROOM) == user_a.email

    @pytest.mark.parametrize("room", ["bad room", "-dash", "a" * 64, "../x", "a/b", "", "r%2Fx"])
    def test_room_outside_the_pattern_is_422_and_claims_nothing(self, test_client, user_a, room):
        response = _issue(test_client, user_a.headers, room=room)

        assert response.status_code == 422
        assert asyncio.run(RoomOwner.count_documents()) == 0

    def test_requires_a_bearer_token(self, test_client):
        assert test_client.post(ISSUE, json={"room_name": ROOM}).status_code == 401

    def test_a_name_that_breaks_the_rule_is_left_out(self, test_client):
        """An account made before the name rule must not show "(AI)" through its token."""
        from tests.utils.helpers import login_headers

        asyncio.run(create_user("legacy@meetween.test", "pytest-secret-123", "Bob (AI)", None))
        headers = login_headers(test_client, email="legacy@meetween.test", password="pytest-secret-123")

        token = _issue(test_client, headers).json()["jitsi_token"]

        assert "name" not in _prosody_decode(token)["context"]["user"]


@pytest.mark.integration
class TestJoinUrl:
    def test_owner_url_carries_the_token_in_the_fragment(self, test_client, user_a, meet_base):
        body = _issue(test_client, user_a.headers).json()

        url = body["meet_join_url"]
        assert url == f"{meet_base}/{ROOM}#jwt=%22{body['jitsi_token']}%22"
        assert "?" not in url
        assert json.loads(unquote(url.split("#jwt=", 1)[1])) == body["jitsi_token"]

    def test_url_is_null_without_a_base(self, test_client, user_a):
        assert _issue(test_client, user_a.headers).json()["meet_join_url"] is None


@pytest.mark.integration
class TestGuests:
    def test_another_user_gets_no_token_and_a_plain_url(self, test_client, user_a, user_b, meet_base):
        _issue(test_client, user_a.headers)

        response = _issue(test_client, user_b.headers, moderator=True)

        assert response.status_code == 200
        body = response.json()
        assert body["jitsi_token"] is None
        assert body["moderator"] is False
        assert body["meet_join_url"] == f"{meet_base}/{ROOM}"
        assert body["meeting_context"]["user_id"] == user_b.email
        assert _owner_of(ROOM) == user_a.email

    def test_admin_moderates_a_room_owned_by_someone_else(self, test_client, user_a, admin_headers):
        _issue(test_client, user_a.headers)

        body = _issue(test_client, admin_headers).json()

        assert body["moderator"] is True
        assert _prosody_decode(body["jitsi_token"])["context"]["user"]["moderator"] is True
        assert _owner_of(ROOM) == user_a.email

    def test_inactive_admin_is_not_an_admin(self, test_client, user_a, admin_headers):
        _issue(test_client, user_a.headers)
        asyncio.run(User.get_collection().update_one({"email": TEST_ADMIN_EMAIL}, {"$set": {"is_active": False}}))

        assert asyncio.run(room_owner_module.is_admin(TEST_ADMIN_EMAIL)) is False
        assert _issue(test_client, admin_headers).status_code == 401

    def test_losing_a_concurrent_claim_reads_the_winner(self, user_a, user_b):
        """The unique index settles two first claims; the loser reads the winner, not an error."""
        asyncio.run(RoomOwner(room_name=ROOM, owner_user_id=user_a.email).save())

        winner = asyncio.run(room_owner_module._claim_free_room(ROOM, user_b.email, admin=False))

        assert winner.owner_user_id == user_a.email
        assert asyncio.run(RoomOwner.count_documents()) == 1


@pytest.mark.integration
class TestClaimInterface:
    """`claim_room_moderator(room_name, user) -> bool`, as B6 and B7 call it."""

    def test_free_room_is_claimed_and_true(self, user_a):
        assert asyncio.run(room_owner_module.claim_room_moderator("Free-Room", _user(user_a.email))) is True
        assert _owner_of("free-room") == user_a.email

    def test_room_of_another_user_is_false_and_unchanged(self, user_a, user_b):
        asyncio.run(room_owner_module.claim_room_moderator(ROOM, _user(user_a.email)))

        assert asyncio.run(room_owner_module.claim_room_moderator(room_name=ROOM, user=_user(user_b.email))) is False
        assert _owner_of(ROOM) == user_a.email

    def test_admin_is_true_for_a_room_of_another_user(self, user_a, admin_headers):
        asyncio.run(room_owner_module.claim_room_moderator(ROOM, _user(user_a.email)))

        assert asyncio.run(room_owner_module.claim_room_moderator(ROOM, _user(TEST_ADMIN_EMAIL))) is True
        assert _owner_of(ROOM) == user_a.email


@pytest.mark.integration
class TestClaimCap:
    def test_new_rooms_beyond_the_cap_are_429_until_the_next_utc_day(
        self, test_client, user_a, user_b, claim_cap, frozen_clock
    ):
        claim_cap(2)
        _issue(test_client, user_b.headers, room="bruno-room")
        assert _issue(test_client, user_a.headers, room="room-1").status_code == 200
        assert _issue(test_client, user_a.headers, room="room-2").status_code == 200

        refused = _issue(test_client, user_a.headers, room="room-3")

        assert refused.status_code == 429
        assert refused.json()["code"] == "rate_limited"
        # Frozen at 09:00 UTC: the next UTC day starts in 15 hours.
        assert refused.headers["retry-after"] == str(15 * 3600)
        assert _owner_of("room-3") is None
        # An owned room and someone else's room claim nothing, so they still work.
        assert _issue(test_client, user_a.headers, room="room-1").json()["moderator"] is True
        assert _issue(test_client, user_a.headers, room="bruno-room").json()["jitsi_token"] is None

        frozen_clock.advance(hours=15)
        assert _issue(test_client, user_a.headers, room="room-3").status_code == 200

    def test_admins_are_exempt(self, test_client, admin_headers, claim_cap):
        claim_cap(1)

        for room in ("admin-1", "admin-2", "admin-3"):
            assert _issue(test_client, admin_headers, room=room).status_code == 200

    def test_a_claim_that_raced_past_the_cap_is_undone(self, user_a, claim_cap, monkeypatch):
        """Two claims at the limit both pass the first count; the second count removes the extra."""
        claim_cap(2)
        counts = iter([1, 3])

        async def fake_count(user_id: str) -> int:
            return next(counts)

        monkeypatch.setattr(room_owner_module, "_claims_today", fake_count)

        with pytest.raises(AgentarError) as raised:
            asyncio.run(room_owner_module.claim_room_moderator("race", _user(user_a.email)))

        assert raised.value.status_code == 429
        assert _owner_of("race") is None


@pytest.mark.integration
class TestValidateRouteRemoved:
    def test_validate_is_gone(self, test_client, user_a):
        token = _issue(test_client, user_a.headers).json()["jitsi_token"]

        assert test_client.post("/api/auth/jitsi/validate", json={"token": token}).status_code == 404
        assert "/api/auth/jitsi/validate" not in app.openapi()["paths"]


@pytest.mark.unit
class TestContract:
    CONTRACT = yaml.safe_load(
        (Path(__file__).parents[1] / "contract" / "openapi.yaml").read_text(encoding="utf-8")
    )

    @pytest.mark.parametrize(
        ("name", "model"),
        [("JitsiIssueRequest", JitsiIssueRequest), ("JitsiIssueResponse", JitsiIssueResponse)],
    )
    def test_schema_properties_match(self, name, model):
        assert set(model.model_fields) == set(self.CONTRACT["components"]["schemas"][name]["properties"])

    def test_documented_statuses_match(self):
        wanted = set(self.CONTRACT["paths"][ISSUE]["post"]["responses"]) - {"422"}
        served = set(app.openapi()["paths"][ISSUE]["post"]["responses"])
        assert wanted <= served


@pytest.mark.unit
class TestBotToken:
    def test_bot_token_is_room_bound_nameless_and_lives_as_long_as_the_bot(self, monkeypatch):
        monkeypatch.setattr(settings, "vp_max_lifetime_s", 1800.0)

        token = create_jitsi_bot_token(ROOM)

        claims = _prosody_decode(token)
        assert claims["sub"] == "vp-bot"
        assert claims["room"] == ROOM
        assert claims["exp"] - claims["iat"] == 1800
        assert claims["jti"]
        # A name claim would replace the "(AI)" display name and get the bot transcribed.
        assert "context" not in claims
        assert "moderator" not in claims
        with pytest.raises(JWTError):
            _prosody_decode(token, key=settings.jwt_secret_key)


@pytest.mark.unit
class TestJitsiStartupRules:
    K = "s" * 40

    def _problems(self, **kw) -> list[str]:
        base = dict(jwt_secret_key="a" * 40, enable_auth=True)
        return unsafe_settings(Settings(_env_file=None, **{**base, **kw}))

    def test_agentar_needs_the_jitsi_key(self):
        problems = self._problems(agentar_enabled=True, vp_meetween_api_key="m" * 40)
        assert any("AGENTAR_ENABLED=true needs JWT_JITSI_SECRET_KEY" in p for p in problems)
        assert not self._problems(agentar_enabled=True, vp_meetween_api_key="m" * 40, jwt_jitsi_secret_key=self.K)

    @pytest.mark.parametrize("other", ["jigasi_meetween_api_key", "vp_meetween_api_key", "vp_worker_secret"])
    def test_jitsi_key_must_not_be_another_service_secret(self, other):
        problems = self._problems(jwt_jitsi_secret_key=self.K, **{other: self.K})
        assert any(f"JWT_JITSI_SECRET_KEY equals {other.upper()}" in p for p in problems)
        assert self.K not in " ".join(problems)


@pytest.mark.unit
class TestRoomOwnerCli:
    def test_set_show_release(self, user_a, user_b, capsys):
        asyncio.run(RoomOwner(room_name="demo", owner_user_id=user_b.email).save())

        assert cli.main(["room-owner", "set", "Demo", user_a.email]) == 0
        assert _owner_of("demo") == user_a.email
        assert cli.main(["room-owner", "show", "demo"]) == 0
        assert f"demo: {user_a.email}" in capsys.readouterr().out

        assert cli.main(["room-owner", "release", "demo"]) == 0
        assert _owner_of("demo") is None
        assert cli.main(["room-owner", "show", "demo"]) == 0
        assert "demo: free" in capsys.readouterr().out

    def test_set_refuses_an_unknown_account_and_a_bad_room(self, user_a):
        assert cli.main(["room-owner", "set", "demo", "nobody@meetween.test"]) == 1
        assert cli.main(["room-owner", "set", "bad room", user_a.email]) == 1
        assert _owner_of("demo") is None

    def test_pre_claimed_room_gives_the_presenter_the_token(self, test_client, user_a, user_b):
        assert cli.main(["room-owner", "set", ROOM, user_a.email]) == 0

        assert _issue(test_client, user_b.headers).json()["jitsi_token"] is None
        assert _issue(test_client, user_a.headers).json()["moderator"] is True
