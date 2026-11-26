"""Throttles for sign up and login: an in-process sliding window behind a small interface.

One uvicorn process serves the API, so the counters live in memory (A3 ruling 2). A Redis
backend is backlog for the day a second process serves users; it implements `RateLimiter`.

`client_ip()` is the throttle key. `X-Real-IP` is the client address only when the TCP peer is a
trusted proxy (`TRUSTED_PROXY_IPS`, addresses or networks); otherwise the peer address is used.
The leftmost `X-Forwarded-For` is never read: any client can write it.
"""
import functools
import ipaddress
import math
import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Iterable, Protocol

from fastapi import Request

from api.errors import AgentarError
from config import settings

IpNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

# An IPv6 client usually holds a whole /64; one key per address would let it walk the prefix.
IPV6_KEY_PREFIX = 64
UNKNOWN_CLIENT = "unknown"


class RateLimiter(Protocol):
    def retry_after(self, key: str, limit: int, window_s: float) -> float | None:
        """Seconds until `key` may be hit again, or None when it is under `limit`."""

    def hit(self, key: str, window_s: float) -> None:
        """Record one event for `key`."""

    def reset(self, key: str) -> None:
        """Forget every event of `key`."""


class SlidingWindowLimiter:
    """Event log per key, pruned on access. Bounded: at most `max_keys` keys, the least recently
    used are dropped first. Callers check every key before they record any, so a refused request
    consumes no budget."""

    def __init__(self, *, max_keys: int = 100_000, now: Callable[[], float] = time.monotonic) -> None:
        self._events: OrderedDict[str, deque[float]] = OrderedDict()
        self._windows: dict[str, float] = {}
        self._max_keys = max_keys
        self._now = now
        self._lock = threading.Lock()

    def _prune(self, key: str, window_s: float, now: float) -> deque[float] | None:
        events = self._events.get(key)
        if events is None:
            return None
        while events and events[0] <= now - window_s:
            events.popleft()
        if not events:
            del self._events[key]
            self._windows.pop(key, None)
            return None
        return events

    def retry_after(self, key: str, limit: int, window_s: float) -> float | None:
        with self._lock:
            now = self._now()
            events = self._prune(key, window_s, now)
            if events is None or len(events) < limit:
                return None
            # The event that must expire before one more is allowed.
            return events[len(events) - limit] + window_s - now

    def hit(self, key: str, window_s: float) -> None:
        with self._lock:
            now = self._now()
            events = self._prune(key, window_s, now)
            if events is None:
                events = self._events[key] = deque()
            events.append(now)
            self._windows[key] = window_s
            self._events.move_to_end(key)
            if len(self._events) > self._max_keys:
                self._evict(now)

    def _evict(self, now: float) -> None:
        for key in list(self._events):
            self._prune(key, self._windows.get(key, 0.0), now)
        while len(self._events) > self._max_keys:
            oldest, _ = self._events.popitem(last=False)
            self._windows.pop(oldest, None)

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)
            self._windows.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._windows.clear()

    def __len__(self) -> int:
        return len(self._events)


limiter = SlidingWindowLimiter()


def get_rate_limiter() -> RateLimiter:
    """Dependency, so a test or a later Redis backend can replace the process wide instance."""
    return limiter


@functools.lru_cache(maxsize=8)
def parse_networks(entries: tuple[str, ...]) -> tuple[IpNetwork, ...]:
    """`TRUSTED_PROXY_IPS` entries as networks: a bare address is a /32 or /128. Raises
    ValueError on a malformed entry; startup checks refuse such a value."""
    return tuple(ipaddress.ip_network(entry, strict=False) for entry in entries)


def _normalise(raw: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if not raw:
        return None
    try:
        addr = ipaddress.ip_address(raw.strip())
    except ValueError:
        return None
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def _key(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    if isinstance(addr, ipaddress.IPv6Address):
        return str(ipaddress.IPv6Network((addr, IPV6_KEY_PREFIX), strict=False))
    return str(addr)


def client_ip(request: Request, trusted: tuple[IpNetwork, ...] | None = None) -> str:
    """Throttle key of the caller: an IPv4 address, an IPv6 /64, or `unknown`."""
    if trusted is None:
        trusted = parse_networks(tuple(settings.agentar.trusted_proxy_ips))
    peer = _normalise(request.client.host if request.client else None)
    if peer is None:
        return UNKNOWN_CLIENT
    if any(peer in net for net in trusted if net.version == peer.version):
        forwarded = _normalise(request.headers.get("x-real-ip"))
        if forwarded is not None:
            return _key(forwarded)
    return _key(peer)


def too_many_requests(retry_after_s: float) -> AgentarError:
    """429 `rate_limited` with a whole number of seconds in Retry-After, at least 1."""
    seconds = max(1, math.ceil(retry_after_s))
    return AgentarError(
        429,
        "rate_limited",
        "Too many attempts. Try again later.",
        headers={"Retry-After": str(seconds)},
    )


def longest_wait(limiter_: RateLimiter, checks: Iterable[tuple[str, int, float]]) -> float | None:
    """Longest wait among `(key, limit, window_s)` checks, or None when all are under limit."""
    waits = [w for key, limit, window in checks if (w := limiter_.retry_after(key, limit, window)) is not None]
    return max(waits) if waits else None
