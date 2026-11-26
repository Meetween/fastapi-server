"""Small dedicated thread pools for heavy upload work, bounded in threads and in waiting jobs.

Picture decodes and document extraction used `asyncio.to_thread`, which shares the default
executor with the live answer path (`asyncio.to_thread(build_agentar_prompt)`). A burst of
uploads could take every default thread: an extraction waiter holds one for up to a minute.
Each `BoundedExecutor` has its own threads, so it can only slow its own kind of work, and it
refuses a job with 429 `rate_limited` when `max_workers + max_waiting` jobs are already in flight
instead of queueing without limit.

The in flight count is changed only on the event loop thread, between awaits, so it needs no lock.
"""
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from api.errors import AgentarError

T = TypeVar("T")

RETRY_AFTER_S = 5


class ExecutorBusy(AgentarError):
    """429 `rate_limited`: the executor holds as many jobs as it accepts."""

    def __init__(self) -> None:
        super().__init__(
            429,
            "rate_limited",
            "Too many uploads are being processed. Try again shortly.",
            headers={"Retry-After": str(RETRY_AFTER_S)},
        )


class BoundedExecutor:
    def __init__(self, name: str, *, max_workers: int, max_waiting: int) -> None:
        self.name = name
        self.limit = max_workers + max_waiting
        self.in_flight = 0
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=name)

    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if self.in_flight >= self.limit:
            raise ExecutorBusy()
        self.in_flight += 1
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._pool, functools.partial(fn, *args, **kwargs))
        finally:
            self.in_flight -= 1
