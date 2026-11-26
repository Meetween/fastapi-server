"""Refuse to start with a configuration that is unsafe or cannot work.

`unsafe_settings` is pure and returns one message per problem. Messages name the setting and
never contain a value. `main.startup_event` calls it first and stops the process when the list is
not empty. `settings_warnings` lists what is suspicious but harmless; startup logs it and goes on.

Extend the list here when a feature adds a rule; do not add a second check elsewhere.
"""
import ipaddress

from config import Settings

DEFAULT_JWT_SECRET = "change-me-in-production"
MIN_SECRET_BYTES = 32


def _short(value: str) -> bool:
    return len(value.encode("utf-8")) < MIN_SECRET_BYTES


def unsafe_settings(s: Settings) -> list[str]:
    problems: list[str] = []

    # Checked whatever ENABLE_AUTH says: the Agentar routes, `get_current_user` and the Jitsi
    # issue route verify tokens signed with this key even with auth off (A3 H6).
    if s.jwt_secret_key == DEFAULT_JWT_SECRET:
        problems.append("JWT_SECRET_KEY is the built in default; set a random value (openssl rand -hex 32)")
    elif _short(s.jwt_secret_key):
        problems.append(f"JWT_SECRET_KEY is shorter than {MIN_SECRET_BYTES} bytes")

    # An empty Jitsi key is allowed: the token routes then answer 503. A set one must be strong
    # and must not be the key that signs API tokens, which never leaves fastapi.
    if s.jwt_jitsi_secret_key:
        if _short(s.jwt_jitsi_secret_key):
            problems.append(f"JWT_JITSI_SECRET_KEY is shorter than {MIN_SECRET_BYTES} bytes")
        if s.jwt_jitsi_secret_key == s.jwt_secret_key:
            problems.append("JWT_JITSI_SECRET_KEY equals JWT_SECRET_KEY; the Jitsi key is shared with Prosody, use a separate value")
        # Prosody holds this key; anyone who reads it mints owner tokens for any room. Sharing it
        # with a service bearer would also let that key's holder do so.
        shared = [
            name
            for name, value in (
                ("JIGASI_MEETWEEN_API_KEY", s.jigasi_meetween_api_key),
                ("VP_MEETWEEN_API_KEY", s.vp_meetween_api_key),
                ("VP_WORKER_SECRET", s.vp_worker_secret),
            )
            if value and value == s.jwt_jitsi_secret_key
        ]
        for name in shared:
            problems.append(f"JWT_JITSI_SECRET_KEY equals {name}; every secret needs its own value")

    if s.signup_enabled and not s.enable_auth:
        problems.append("SIGNUP_ENABLED=true needs ENABLE_AUTH=true")
    elif s.signup_enabled and not s.agentar.trusted_proxy_ips:
        # Behind nginx the peer is the proxy, so every client would share one sign up and login
        # throttle key: five sign ups per hour for everyone, and one client locks all out (A3 M2).
        problems.append(
            "SIGNUP_ENABLED=true needs TRUSTED_PROXY_IPS; set the address or network nginx connects from"
        )

    for entry in s.agentar.trusted_proxy_ips:
        try:
            ipaddress.ip_network(entry, strict=False)
        except ValueError:
            problems.append("TRUSTED_PROXY_IPS holds an entry that is not an address or a network")
            break

    if s.vp_meetween_api_key and s.vp_meetween_api_key == s.vp_worker_secret:
        problems.append("VP_MEETWEEN_API_KEY equals VP_WORKER_SECRET; the two directions need different secrets")

    if s.agentar_enabled and not s.vp_meetween_api_key:
        problems.append("AGENTAR_ENABLED=true needs VP_MEETWEEN_API_KEY; the worker could not post events")
    if s.agentar_enabled and not s.jwt_jitsi_secret_key:
        problems.append("AGENTAR_ENABLED=true needs JWT_JITSI_SECRET_KEY; no bot or owner could get a room token")

    return problems


def settings_warnings(s: Settings) -> list[str]:
    warnings: list[str] = []

    # Equal to the key it changes nothing (legacy digests are verified with JWT_SECRET_KEY anyway)
    # and it signs nothing, so it is not refused. It usually means JWT_SECRET_KEY was rotated
    # into both settings, which locks out every user still on a legacy digest.
    if s.legacy_password_pepper and s.legacy_password_pepper == s.jwt_secret_key:
        warnings.append(
            "LEGACY_PASSWORD_PEPPER equals JWT_SECRET_KEY; it has no effect. After a rotation it "
            "must hold the previous key"
        )
    return warnings
