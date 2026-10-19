"""
Async Valkey client lifecycle.

Valkey holds the virtual-participant routing keyspace (`bot:*`, `worker:*`).
The client is created once at app startup via `connect_valkey()` and closed in
`close_valkey()` during shutdown. Callers read it through `get_valkey()`.
"""
import logging
from typing import Optional

from valkey.asyncio import Valkey

from config import settings

logger = logging.getLogger(__name__)

_client: Optional[Valkey] = None


async def connect_valkey(url: Optional[str] = None) -> Valkey:
    global _client
    if _client is None:
        _client = Valkey.from_url(url or settings.valkey_url, decode_responses=True)
        await _client.ping()
        logger.info("Connected to Valkey")
    return _client


def get_valkey() -> Valkey:
    if _client is None:
        raise RuntimeError("Valkey not connected. Call connect_valkey() first.")
    return _client


async def close_valkey() -> None:
    global _client
    if _client is None:
        return
    try:
        await _client.aclose()
    finally:
        _client = None
        logger.info("Valkey connection closed")
