"""
Valkey keyspace for virtual participants.

`bot:{id}` is written only by the control route. `worker:{id}` and
`bot:{id}/alive` are written by vp-worker to signify bot is alive; this module only reads them.
A worker spawns bots
"""
import json
from dataclasses import asdict, dataclass

from valkey.asyncio import Valkey

BOT_PREFIX = "bot:"
ALIVE_SUFFIX = "/alive"
WORKER_PREFIX = "worker:"

# A worker whose heartbeat is older than this is not offered new bots.
WORKER_HEARTBEAT_STALE_SECONDS = 30.0

STATE_PENDING = "pending"
STATE_ACTIVE = "active"


@dataclass(frozen=True)
class BotRecord:
    bot_id: str
    worker_id: str
    room_name: str
    state: str
    created_at: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "BotRecord":
        return cls(**json.loads(raw))


@dataclass(frozen=True)
class WorkerInfo:
    worker_id: str
    url: str
    load: int
    capacity: int
    heartbeat: float

    @property
    def utilisation(self) -> float:
        return self.load / self.capacity


def _bot_key(bot_id: str) -> str:
    return f"{BOT_PREFIX}{bot_id}"


def _alive_key(bot_id: str) -> str:
    return f"{BOT_PREFIX}{bot_id}{ALIVE_SUFFIX}"


def _worker_from_hash(worker_id: str, fields: dict[str, str]) -> WorkerInfo | None:
    try:
        return WorkerInfo(
            worker_id=worker_id,
            url=fields["url"],
            load=int(fields["load"]),
            capacity=int(fields["capacity"]),
            heartbeat=float(fields["heartbeat"]),
        )
    except (KeyError, ValueError):
        return None


async def claim_bot(valkey: Valkey, record: BotRecord) -> bool:
    """Atomically write the routing key. False if the bot already exists (`SET NX`)."""
    return bool(await valkey.set(_bot_key(record.bot_id), record.to_json(), nx=True))


async def put_bot(valkey: Valkey, record: BotRecord) -> None:
    await valkey.set(_bot_key(record.bot_id), record.to_json())


async def get_bot(valkey: Valkey, bot_id: str) -> BotRecord | None:
    raw = await valkey.get(_bot_key(bot_id))
    return BotRecord.from_json(raw) if raw else None


async def delete_bot(valkey: Valkey, bot_id: str) -> None:
    await valkey.delete(_bot_key(bot_id))


async def bot_is_alive(valkey: Valkey, bot_id: str) -> bool:
    return bool(await valkey.exists(_alive_key(bot_id)))


async def list_bot_ids(valkey: Valkey) -> list[str]:
    ids = []
    async for key in valkey.scan_iter(match=f"{BOT_PREFIX}*"):
        if key.endswith(ALIVE_SUFFIX):
            continue
        ids.append(key[len(BOT_PREFIX):])
    return ids


async def get_worker(valkey: Valkey, worker_id: str) -> WorkerInfo | None:
    fields = await valkey.hgetall(f"{WORKER_PREFIX}{worker_id}")
    return _worker_from_hash(worker_id, fields) if fields else None


async def list_placeable_workers(valkey: Valkey, now: float) -> list[WorkerInfo]:
    """Workers with a fresh heartbeat and spare capacity, least utilised first."""
    workers = []
    async for key in valkey.scan_iter(match=f"{WORKER_PREFIX}*"):
        worker_id = key[len(WORKER_PREFIX):]
        worker = await get_worker(valkey, worker_id)
        if worker is None:
            continue
        if now - worker.heartbeat > WORKER_HEARTBEAT_STALE_SECONDS:
            continue
        if worker.load >= worker.capacity:
            continue
        workers.append(worker)
    return sorted(workers, key=lambda w: w.utilisation)
