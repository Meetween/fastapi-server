"""Agentar settings: defaults are safe, empty secrets leave the features unconfigured."""

import pytest
from pydantic import ValidationError

from config import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_flags_and_secrets_are_off_by_default() -> None:
    s = _settings()
    assert (s.signup_enabled, s.email_enabled, s.calendar_enabled) == (False, False, False)
    assert s.jwt_jitsi_secret_key == ""
    assert s.object_store.configured is False
    assert s.agentar.worker_events_base_url == ""


def test_quota_defaults_follow_the_contract() -> None:
    a = _settings().agentar
    assert a.file_max_bytes == 10 * 1024 * 1024
    assert a.user_max_files == 20
    assert a.user_max_bytes == 100 * 1024 * 1024
    assert a.picture_max_bytes == 5 * 1024 * 1024
    assert a.meetings_per_day == 10
    assert (a.signup_rate_per_ip, a.signup_rate_per_email) == (5, 3)
    assert a.grace_minutes_default == 15
    assert a.tick_seconds == 30
    assert a.default_voice == "af_heart"


def test_object_store_needs_bucket_and_both_keys() -> None:
    partial = _settings(s3_bucket="b", s3_access_key_id="k")
    assert partial.object_store.configured is False
    full = _settings(s3_bucket="b", s3_access_key_id="k", s3_secret_access_key="s")
    assert full.object_store.configured is True
    assert "secret_access_key" not in repr(full.object_store)
    assert "access_key_id" not in repr(full.object_store)


def test_memory_backend_is_always_configured() -> None:
    assert _settings(agentar_object_store="memory").object_store.configured is True


def test_prefix_is_trimmed_and_events_url_has_no_trailing_slash() -> None:
    s = _settings(s3_prefix="/agentar/", vp_meetween_api_url="http://fastapi:8000/")
    assert s.object_store.prefix == "agentar"
    assert s.agentar.worker_events_base_url == "http://fastapi:8000"


def test_grace_minutes_default_is_bounded() -> None:
    with pytest.raises(ValidationError):
        _settings(agentar_grace_minutes_default=61)


def test_empty_voice_list_is_refused() -> None:
    with pytest.raises(ValueError):
        _settings(agentar_voices=" , ")


def test_security_defaults_follow_the_review() -> None:
    s = _settings()
    a = s.agentar
    assert s.agentar_enabled is False
    assert a.max_live_total == 1
    assert a.room_claims_per_day == 20
    assert (a.login_fail_limit, a.login_fail_window_minutes, a.login_fail_ip_limit) == (10, 15, 50)
    assert a.trusted_proxy_ips == ()
    assert (a.event_max_skew_seconds, a.event_left_grace_seconds) == (600, 300)


def test_trusted_proxy_ips_are_a_list() -> None:
    assert _settings(trusted_proxy_ips="172.18.0.5, 172.18.0.6").agentar.trusted_proxy_ips == ("172.18.0.5", "172.18.0.6")
