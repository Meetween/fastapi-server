"""Fakes for functions of other tasks that these tests reach through their documented seams.

`install` puts a fake attribute on a module, creating the module while it does not exist yet. The
fake wins over a real function of the same name, so the tests stay the same before and after the
other task is merged; swap them for the real ones in the integration tests.
"""
import sys
import types
from datetime import timedelta

from jose import jwt

from api.errors import AgentarError
from config import settings
from utils import clock


def install(monkeypatch, module_name: str, **attributes) -> None:
    try:
        module = __import__(module_name, fromlist=["_"])
    except ModuleNotFoundError:
        module = types.ModuleType(module_name)
        monkeypatch.setitem(sys.modules, module_name, module)
    for key, value in attributes.items():
        monkeypatch.setattr(module, key, value, raising=False)


def create_jitsi_bot_token(room: str) -> str:
    """B2's `utils.auth.create_jitsi_bot_token(room) -> str`: signed with the Jitsi key, room
    bound, no moderator claim, no name claim; a 503 AgentarError while the key is empty."""
    key = settings.jwt_jitsi_secret_key
    if not key:
        raise AgentarError(503, "jitsi_key_missing", "The Jitsi token key is not configured")
    now = clock.utcnow()
    claims = {"iss": "meetween", "aud": "jitsi", "sub": "vp-bot", "room": room, "iat": now, "exp": now + timedelta(hours=2)}
    return jwt.encode(claims, key, algorithm="HS256")


def install_bot_token(monkeypatch) -> None:
    install(monkeypatch, "utils.auth", create_jitsi_bot_token=create_jitsi_bot_token)
