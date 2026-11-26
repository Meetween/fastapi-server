"""Refuse to start with a configuration that is unsafe or cannot work.

`unsafe_settings` is pure and returns one message per problem. Messages name the setting and
never contain a value. `main.startup_event` calls it first and stops the process when the list is
not empty.

Extend the list here when a feature adds a rule; do not add a second check elsewhere.
"""
from config import Settings

DEFAULT_JWT_SECRET = "change-me-in-production"
MIN_SECRET_BYTES = 32


def _short(value: str) -> bool:
    return len(value.encode("utf-8")) < MIN_SECRET_BYTES


def unsafe_settings(s: Settings) -> list[str]:
    problems: list[str] = []

    if s.enable_auth:
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

    if s.signup_enabled and not s.enable_auth:
        problems.append("SIGNUP_ENABLED=true needs ENABLE_AUTH=true")

    if s.vp_meetween_api_key and s.vp_meetween_api_key == s.vp_worker_secret:
        problems.append("VP_MEETWEEN_API_KEY equals VP_WORKER_SECRET; the two directions need different secrets")

    if s.agentar_enabled and not s.vp_meetween_api_key:
        problems.append("AGENTAR_ENABLED=true needs VP_MEETWEEN_API_KEY; the worker could not post events")

    return problems
