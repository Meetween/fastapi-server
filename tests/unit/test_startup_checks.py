"""Startup refuses unsafe or reused keys. Messages name the setting and never print a value."""
import asyncio

import pytest

import main
from config import Settings, settings
from utils.startup_checks import DEFAULT_JWT_SECRET, settings_warnings, unsafe_settings

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


def test_signup_behind_no_trusted_proxy_is_refused() -> None:
    """Behind nginx every client would share the proxy address as its throttle key (A3 M2)."""
    problems = unsafe_settings(_s(signup_enabled=True, trusted_proxy_ips=""))
    assert any("SIGNUP_ENABLED=true needs TRUSTED_PROXY_IPS" in p for p in problems), problems


@pytest.mark.parametrize("value", ["172.18.0.0/16", "172.18.0.5", "10.0.0.2,fd00::/64"])
def test_signup_with_a_trusted_proxy_is_accepted(value: str) -> None:
    assert unsafe_settings(_s(signup_enabled=True, trusted_proxy_ips=value)) == []


def test_signup_off_needs_no_trusted_proxy() -> None:
    assert unsafe_settings(_s(signup_enabled=False, trusted_proxy_ips="")) == []


def test_no_message_contains_a_secret_value() -> None:
    bad = _s(jwt_secret_key=DEFAULT_JWT_SECRET, jwt_jitsi_secret_key=DEFAULT_JWT_SECRET, vp_meetween_api_key=K2, vp_worker_secret=K2)
    text = " ".join(unsafe_settings(bad))
    assert DEFAULT_JWT_SECRET not in text and K2 not in text


@pytest.mark.parametrize(
    ("key", "needle"),
    [(DEFAULT_JWT_SECRET, "JWT_SECRET_KEY is the built in default"), ("short", "JWT_SECRET_KEY is shorter")],
)
def test_a_weak_key_is_refused_with_auth_off(key: str, needle: str) -> None:
    """ENABLE_AUTH=false does not stop the Agentar routes, `get_current_user` and
    `/api/auth/jitsi/issue` from trusting tokens signed with this key (A3 H6)."""
    problems = unsafe_settings(_s(enable_auth=False, jwt_secret_key=key))
    assert any(needle in p for p in problems), problems


def test_startup_event_stops_on_the_default_key_with_auth_off(monkeypatch) -> None:
    monkeypatch.setattr(settings, "jwt_secret_key", DEFAULT_JWT_SECRET)
    monkeypatch.setattr(settings, "enable_auth", False)
    with pytest.raises(RuntimeError, match="refusing to start"):
        asyncio.run(main.startup_event())


def test_startup_event_stops_before_touching_anything(monkeypatch) -> None:
    monkeypatch.setattr(settings, "jwt_secret_key", DEFAULT_JWT_SECRET)
    monkeypatch.setattr(settings, "enable_auth", True)
    with pytest.raises(RuntimeError, match="refusing to start"):
        asyncio.run(main.startup_event())


# LEGACY_PASSWORD_PEPPER: a warning, never a refusal. Equal to JWT_SECRET_KEY it changes nothing
# (legacy digests are verified with that key anyway), so it weakens nothing either.

def test_a_legacy_pepper_equal_to_the_jwt_key_warns_without_refusing() -> None:
    s = _s(legacy_password_pepper=K1)
    assert unsafe_settings(s) == []
    warnings = settings_warnings(s)
    assert any("LEGACY_PASSWORD_PEPPER equals JWT_SECRET_KEY" in w for w in warnings), warnings
    assert K1 not in " ".join(warnings)


@pytest.mark.parametrize("pepper", ["", "short-old-key"])
def test_an_empty_or_short_legacy_pepper_is_fine(pepper: str) -> None:
    """The pepper is the old key, short by definition of the rotation; it signs nothing."""
    s = _s(legacy_password_pepper=pepper)
    assert unsafe_settings(s) == []
    assert settings_warnings(s) == []


def test_startup_event_logs_the_warnings(monkeypatch, caplog) -> None:
    monkeypatch.setattr(settings, "jwt_secret_key", DEFAULT_JWT_SECRET)
    monkeypatch.setattr(settings, "legacy_password_pepper", DEFAULT_JWT_SECRET)
    with pytest.raises(RuntimeError, match="refusing to start"):
        asyncio.run(main.startup_event())
    assert "LEGACY_PASSWORD_PEPPER equals JWT_SECRET_KEY" in caplog.text
