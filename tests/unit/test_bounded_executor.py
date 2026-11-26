"""Picture decodes and document extraction run on their own bounded executors, so a burst of
uploads cannot take the default executor threads the live answer path uses
(`asyncio.to_thread(build_agentar_prompt)`), and a full executor answers 429 instead of queueing."""
import asyncio
import threading

import pytest

from api.errors import AgentarError
from utils import picture, text_extract
from utils.bounded_executor import BoundedExecutor, ExecutorBusy


def test_jobs_run_on_the_executors_own_threads() -> None:
    pool = BoundedExecutor("probe", max_workers=1, max_waiting=0)

    async def main() -> str:
        return await pool.run(lambda: threading.current_thread().name)

    assert asyncio.run(main()).startswith("probe")


def test_a_full_executor_refuses_and_frees_its_slots_afterwards() -> None:
    pool = BoundedExecutor("probe", max_workers=1, max_waiting=1)
    gate = threading.Event()

    async def main() -> None:
        first = asyncio.ensure_future(pool.run(gate.wait, 5))
        second = asyncio.ensure_future(pool.run(gate.wait, 5))
        await asyncio.sleep(0)
        with pytest.raises(ExecutorBusy):
            await pool.run(gate.wait, 5)
        gate.set()
        await asyncio.gather(first, second)
        assert pool.in_flight == 0

        def boom() -> None:
            raise ValueError("bad file")

        with pytest.raises(ValueError):
            await pool.run(boom)
        assert pool.in_flight == 0

    asyncio.run(main())


def test_busy_picture_and_extraction_executors_leave_the_default_pool_free(monkeypatch) -> None:
    """Fill both executors with blocked jobs; a default executor job still runs at once."""
    gate = threading.Event()
    monkeypatch.setattr(picture, "normalise_picture_sync", lambda data: gate.wait(5))
    monkeypatch.setattr(text_extract, "run_isolated", lambda kind, data, max_chars: gate.wait(5))

    async def main() -> None:
        held = [asyncio.ensure_future(picture.normalise_picture(b"x")) for _ in range(picture.POOL.limit)]
        held += [
            asyncio.ensure_future(text_extract.extract_text(text_extract.PDF, b"x", max_chars=10))
            for _ in range(text_extract.pool().limit)
        ]
        await asyncio.sleep(0.05)
        assert await asyncio.wait_for(asyncio.to_thread(lambda: "live answer"), timeout=1) == "live answer"

        with pytest.raises(AgentarError) as pic:
            await picture.normalise_picture(b"x")
        with pytest.raises(AgentarError) as doc:
            await text_extract.extract_text(text_extract.PDF, b"x", max_chars=10)
        for exc in (pic.value, doc.value):
            assert (exc.status_code, exc.code) == (429, "rate_limited")
            assert int(exc.headers["Retry-After"]) >= 1
        gate.set()
        await asyncio.gather(*held)

    asyncio.run(main())
