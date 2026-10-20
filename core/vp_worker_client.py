"""HTTP calls from the control route to a vp-worker bot API"""
import logging

import httpx

from config import settings
from core.vp_registry import WorkerInfo

logger = logging.getLogger(__name__)


class WorkerDispatchError(Exception):
    """The worker refused or could not be reached. The caller may try another worker."""


def _headers(worker: WorkerInfo) -> dict[str, str]:
    if not settings.vp_worker_shared_secret:
        # Fail closed: never send an empty bearer to a worker.
        raise WorkerDispatchError("VP_WORKER_SHARED_SECRET is not configured")
    return {
        "Authorization": f"Bearer {settings.vp_worker_shared_secret}",
        "X-Worker-Id": worker.worker_id,
    }


async def create_bot(worker: WorkerInfo, payload: dict) -> None:
    """POST /bots. Any 2xx counts as accepted; anything else raises WorkerDispatchError."""
    try:
        async with httpx.AsyncClient(
            timeout=settings.vp_worker_dispatch_timeout_seconds
        ) as client:
            response = await client.post(
                f"{worker.url}/bots", json=payload, headers=_headers(worker)
            )
    except httpx.HTTPError as exc:
        raise WorkerDispatchError(f"worker {worker.worker_id} unreachable: {exc}") from exc
    if not response.is_success:
        raise WorkerDispatchError(
            f"worker {worker.worker_id} rejected bot: {response.status_code}"
        )


async def delete_bot(worker: WorkerInfo, bot_id: str) -> None:
    """DELETE /bots/{id}. A 404 means the worker already dropped the bot."""
    try:
        async with httpx.AsyncClient(
            timeout=settings.vp_worker_dispatch_timeout_seconds
        ) as client:
            response = await client.delete(
                f"{worker.url}/bots/{bot_id}", headers=_headers(worker)
            )
    except httpx.HTTPError as exc:
        raise WorkerDispatchError(f"worker {worker.worker_id} unreachable: {exc}") from exc
    if response.status_code != 404 and not response.is_success:
        raise WorkerDispatchError(
            f"worker {worker.worker_id} failed delete: {response.status_code}"
        )
