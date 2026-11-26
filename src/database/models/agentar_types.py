"""Shared field types of the Agentar collections."""
from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator

from utils import clock

# Aware UTC whether Mongo handed back a naive value or not.
UtcDatetime = Annotated[datetime, AfterValidator(clock.as_utc)]


def now_field() -> datetime:
    """default_factory for created_at style fields. Looks the clock up at call time."""
    return clock.utcnow()
