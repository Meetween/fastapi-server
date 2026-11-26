"""Wall clock access for the Agentar lifecycle.

Code that decides on time (the tick, quotas per UTC day, grace windows) calls
``clock.utcnow()`` through this module, never ``datetime.now`` and never
``from utils.clock import utcnow``, so tests can freeze it with one patch.

MongoDB returns naive datetimes unless the client is tz aware. ``as_utc`` makes every
value timezone aware, so comparisons never mix naive and aware values.
"""
from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    """Return value as an aware UTC datetime. A naive value is taken as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
