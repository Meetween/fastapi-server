"""Shared fixtures of the Agentar tests, loaded by tests/conftest.py as a pytest plugin.

- `user_a`, `user_b`: two signed in users, for the 404 across owners checks.
- `object_store`: an in-memory store wired into the routes.
- `frozen_clock`: a settable clock for the lifecycle tick and the per day quotas.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api.deps.agentar import get_object_store
from main import app
from src.database.models.user import User
from tests.utils.helpers import obtain_bearer_headers
from utils import clock
from utils.object_store import MemoryObjectStore


@dataclass(frozen=True)
class AgentarUser:
    email: str
    id: str
    headers: dict[str, str]


def _signed_in(test_client: TestClient, email: str, first_name: str) -> AgentarUser:
    headers = obtain_bearer_headers(
        test_client,
        email=email,
        password="pytest-secret-123",
        first_name=first_name,
        last_name="Tester",
    )
    import asyncio

    user = asyncio.run(User.find_by_email(email))
    return AgentarUser(email=email, id=user.id, headers=headers)


@pytest.fixture
def user_a(test_client: TestClient) -> AgentarUser:
    return _signed_in(test_client, "agentar-a@meetween.test", "Alice")


@pytest.fixture
def user_b(test_client: TestClient) -> AgentarUser:
    return _signed_in(test_client, "agentar-b@meetween.test", "Bruno")


@pytest.fixture
def object_store():
    """Fresh in-memory store for the test. Routes get it through `get_object_store`."""
    store = MemoryObjectStore()
    app.dependency_overrides[get_object_store] = lambda: store
    yield store
    app.dependency_overrides.pop(get_object_store, None)


class FrozenClock:
    """Callable that replaces `utils.clock.utcnow`. Code must read the time as `clock.utcnow()`."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> datetime:
        self.now += timedelta(**delta)
        return self.now


@pytest.fixture
def frozen_clock(monkeypatch) -> FrozenClock:
    frozen = FrozenClock(datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(clock, "utcnow", frozen)
    return frozen
