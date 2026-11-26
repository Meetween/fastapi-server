"""Object store for Agentar files, pictures and report exports.

A small async interface over an S3 compatible bucket. Only put, get, head, delete and list
by prefix are used, so the demo RustFS container and a real bucket are interchangeable by
configuration. Keys passed in are logical (`u/{user_id}/kb/{file_id}`); the configured
`S3_PREFIX` is added and removed inside the S3 implementation.

`S3ObjectStore` runs boto3 in a worker thread. `MemoryObjectStore` is the fake for tests and
is selected with `AGENTAR_OBJECT_STORE=memory`.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Optional, Protocol, runtime_checkable

from config import ObjectStoreSettings

DEFAULT_CHUNK_SIZE = 64 * 1024


class ObjectStoreError(Exception):
    """The store failed or is unreachable. Routes map it to 503 `storage_unavailable`."""


class ObjectNotFound(ObjectStoreError):
    pass


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    content_type: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[datetime] = None


class ObjectStream:
    """An open object: metadata plus the body as an async iterator. Close it when done."""

    def __init__(self, info: ObjectInfo, chunks: AsyncIterator[bytes]) -> None:
        self.info = info
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._chunks

    async def aclose(self) -> None:
        close = getattr(self._chunks, "aclose", None)
        if close is not None:
            await close()


@runtime_checkable
class ObjectStore(Protocol):
    async def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> ObjectInfo:
        """Write the object, replacing any existing one."""

    async def get(self, key: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> ObjectStream:
        """Open the object for streaming. Raises ObjectNotFound before any byte is read."""

    async def head(self, key: str) -> Optional[ObjectInfo]:
        """Metadata, or None when the object does not exist."""

    async def delete(self, key: str) -> None:
        """Delete the object. A missing object is not an error."""

    async def list(self, prefix: str) -> list[ObjectInfo]:
        """Every object whose key starts with prefix, ordered by key."""


class MemoryObjectStore:
    """In memory fake with the same behaviour as the S3 store, for tests."""

    def __init__(self) -> None:
        self._objects: dict[str, tuple[bytes, ObjectInfo]] = {}

    async def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> ObjectInfo:
        info = ObjectInfo(
            key=key,
            size=len(data),
            content_type=content_type,
            etag=hashlib.md5(data, usedforsecurity=False).hexdigest(),
        )
        self._objects[key] = (bytes(data), info)
        return info

    async def get(self, key: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> ObjectStream:
        try:
            data, info = self._objects[key]
        except KeyError:
            raise ObjectNotFound(key) from None

        async def chunks() -> AsyncIterator[bytes]:
            for start in range(0, len(data), chunk_size):
                yield data[start : start + chunk_size]

        return ObjectStream(info, chunks())

    async def head(self, key: str) -> Optional[ObjectInfo]:
        entry = self._objects.get(key)
        return entry[1] if entry else None

    async def delete(self, key: str) -> None:
        self._objects.pop(key, None)

    async def list(self, prefix: str) -> list[ObjectInfo]:
        return [self._objects[key][1] for key in sorted(self._objects) if key.startswith(prefix)]

    # Test helpers, not part of the Protocol.
    def keys(self) -> list[str]:
        return sorted(self._objects)

    def read(self, key: str) -> bytes:
        return self._objects[key][0]


class S3ObjectStore:
    """boto3 backed store. The client is created once; calls run in a worker thread."""

    def __init__(self, cfg: ObjectStoreSettings, client=None) -> None:
        self._bucket = cfg.bucket
        self._prefix = f"{cfg.prefix}/" if cfg.prefix else ""
        self._client = client if client is not None else self._build_client(cfg)

    @staticmethod
    def _build_client(cfg: ObjectStoreSettings):
        import boto3
        from botocore.config import Config

        return boto3.client(
            "s3",
            endpoint_url=cfg.endpoint_url or None,
            region_name=cfg.region,
            aws_access_key_id=cfg.access_key_id,
            aws_secret_access_key=cfg.secret_access_key,
            config=Config(
                s3={"addressing_style": cfg.addressing_style},
                # S3 compatible servers often reject the default CRC checksum headers of recent boto3.
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=3,
                read_timeout=15,
            ),
        )

    def _full(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _logical(self, full_key: str) -> str:
        return full_key[len(self._prefix) :] if self._prefix and full_key.startswith(self._prefix) else full_key

    async def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> ObjectInfo:
        def call():
            return self._client.put_object(
                Bucket=self._bucket, Key=self._full(key), Body=data, ContentType=content_type
            )

        response = await self._run(call)
        return ObjectInfo(key=key, size=len(data), content_type=content_type, etag=_etag(response))

    async def get(self, key: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> ObjectStream:
        def call():
            return self._client.get_object(Bucket=self._bucket, Key=self._full(key))

        response = await self._run(call)
        body = response["Body"]
        info = ObjectInfo(
            key=key,
            size=int(response.get("ContentLength", 0)),
            content_type=response.get("ContentType"),
            etag=_etag(response),
            last_modified=response.get("LastModified"),
        )

        async def chunks() -> AsyncIterator[bytes]:
            try:
                while True:
                    chunk = await asyncio.to_thread(body.read, chunk_size)
                    if not chunk:
                        return
                    yield chunk
            finally:
                await asyncio.to_thread(body.close)

        return ObjectStream(info, chunks())

    async def head(self, key: str) -> Optional[ObjectInfo]:
        def call():
            return self._client.head_object(Bucket=self._bucket, Key=self._full(key))

        try:
            response = await self._run(call)
        except ObjectNotFound:
            return None
        return ObjectInfo(
            key=key,
            size=int(response.get("ContentLength", 0)),
            content_type=response.get("ContentType"),
            etag=_etag(response),
            last_modified=response.get("LastModified"),
        )

    async def delete(self, key: str) -> None:
        def call():
            return self._client.delete_object(Bucket=self._bucket, Key=self._full(key))

        await self._run(call)

    async def list(self, prefix: str) -> list[ObjectInfo]:
        def call() -> list[ObjectInfo]:
            found: list[ObjectInfo] = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=self._full(prefix)):
                for item in page.get("Contents", []):
                    found.append(
                        ObjectInfo(
                            key=self._logical(item["Key"]),
                            size=int(item["Size"]),
                            etag=_etag(item),
                            last_modified=item.get("LastModified"),
                        )
                    )
            return sorted(found, key=lambda info: info.key)

        return await self._run(call)

    @staticmethod
    async def _run(call):
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            return await asyncio.to_thread(call)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in {"NoSuchKey", "404", "NotFound"} or status == 404:
                raise ObjectNotFound(code) from exc
            raise ObjectStoreError(code or "client error") from exc
        except BotoCoreError as exc:
            raise ObjectStoreError(type(exc).__name__) from exc


def _etag(response: dict) -> Optional[str]:
    value = response.get("ETag")
    return value.strip('"') if isinstance(value, str) else None


_store: Optional[ObjectStore] = None
_store_for: Optional[ObjectStoreSettings] = None


def get_configured_object_store(cfg: ObjectStoreSettings) -> Optional[ObjectStore]:
    """The store for these settings, built once. None while the settings leave it unconfigured."""
    global _store, _store_for
    if not cfg.configured:
        return None
    if _store is None or _store_for != cfg:
        _store = MemoryObjectStore() if cfg.backend == "memory" else S3ObjectStore(cfg)
        _store_for = cfg
    return _store
