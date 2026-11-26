"""Startup refuses unsafe or reused keys. Messages name the setting and never print a value."""
import asyncio

import pytest

import main
from config import Settings, settings
from utils.startup_checks import DEFAULT_JWT_SECRET, unsafe_settings

K1 = "a" * 40
K2 = "b" * 40


def _s(**kw) -> Settings:
    base = dict(jwt_secret_key=K1, enable_auth=True)
    return Settings(_env_file=None, **{**base, **kw})


def test_a_sound_configuration_has_no_problem() -> None:
    assert unsafe_settings(_s(jwt_jitsi_secret_key=K2, vp_meetween_api_key=K1 + "x", vp_worker_secret=K2 + "x")) == []


def test_empty_optional_secrets_are_allowed() -> None:
    """Empty means the feature answers 503, not that startup fails."""
    assert unsafe_settings(_s()) == []


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"jwt_secret_key": DEFAULT_JWT_SECRET}, "JWT_SECRET_KEY is the built in default"),
        ({"jwt_secret_key": "short"}, "JWT_SECRET_KEY is shorter"),
        ({"jwt_jitsi_secret_key": K1}, "JWT_JITSI_SECRET_KEY equals JWT_SECRET_KEY"),
        ({"jwt_jitsi_secret_key": "short"}, "JWT_JITSI_SECRET_KEY is shorter"),
        ({"signup_enabled": True, "enable_auth": False, "jwt_secret_key": "x"}, "SIGNUP_ENABLED=true needs ENABLE_AUTH"),
        ({"vp_meetween_api_key": K2, "vp_worker_secret": K2}, "VP_MEETWEEN_API_KEY equals VP_WORKER_SECRET"),
        ({"agentar_enabled": True}, "AGENTAR_ENABLED=true needs VP_MEETWEEN_API_KEY"),
    ],
)
def test_unsafe_configuration_is_reported(overrides: dict, needle: str) -> None:
    problems = unsafe_settings(_s(**overrides))
    assert any(needle in p for p in problems), problems


def test_no_message_contains_a_secret_value() -> None:
    bad = _s(jwt_secret_key=DEFAULT_JWT_SECRET, jwt_jitsi_secret_key=DEFAULT_JWT_SECRET, vp_meetween_api_key=K2, vp_worker_secret=K2)
    text = " ".join(unsafe_settings(bad))
    assert DEFAULT_JWT_SECRET not in text and K2 not in text


def test_default_key_is_tolerated_with_auth_off() -> None:
    """ENABLE_AUTH=false is a development switch; the key is not used to protect anything."""
    assert unsafe_settings(_s(enable_auth=False, jwt_secret_key=DEFAULT_JWT_SECRET)) == []


def test_startup_event_stops_before_touching_anything(monkeypatch) -> None:
    monkeypatch.setattr(settings, "jwt_secret_key", DEFAULT_JWT_SECRET)
    monkeypatch.setattr(settings, "enable_auth", True)
    with pytest.raises(RuntimeError, match="refusing to start"):
        asyncio.run(main.startup_event())
