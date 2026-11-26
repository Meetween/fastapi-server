"""Object store: the in-memory fake and the S3 implementation behave the same."""
import asyncio
import io

import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

import utils.object_store as os_mod
from config import Settings
from utils.object_store import (
    MemoryObjectStore,
    ObjectNotFound,
    ObjectStore,
    ObjectStoreError,
    S3ObjectStore,
    get_configured_object_store,
)


def run(coro):
    return asyncio.run(coro)


async def _read_all(stream) -> bytes:
    return b"".join([chunk async for chunk in stream])


def _cfg(**kw):
    base = dict(s3_bucket="bkt", s3_access_key_id="k", s3_secret_access_key="s", s3_endpoint_url="http://127.0.0.1:9000")
    return Settings(_env_file=None, **{**base, **kw}).object_store


class TestMemoryStore:
    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(MemoryObjectStore(), ObjectStore)

    def test_put_get_head_delete_list(self) -> None:
        async def go():
            store = MemoryObjectStore()
            info = await store.put("u/1/kb/a", b"hello world", content_type="text/plain")
            assert (info.size, info.content_type) == (11, "text/plain")
            await store.put("u/1/kb/b", b"x")
            await store.put("u/2/kb/c", b"y")
            stream = await store.get("u/1/kb/a", chunk_size=4)
            assert [c async for c in stream] == [b"hell", b"o wo", b"rld"]
            assert (await store.head("u/1/kb/a")).size == 11
            assert await store.head("nope") is None
            assert [i.key for i in await store.list("u/1/")] == ["u/1/kb/a", "u/1/kb/b"]
            await store.delete("u/1/kb/a")
            await store.delete("u/1/kb/a")  # missing is not an error
            assert await store.head("u/1/kb/a") is None

        run(go())

    def test_get_of_a_missing_key_raises_before_streaming(self) -> None:
        with pytest.raises(ObjectNotFound):
            run(MemoryObjectStore().get("missing"))


class TestSelection:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        monkeypatch.setattr(os_mod, "_store", None)
        monkeypatch.setattr(os_mod, "_store_for", None)

    def test_unconfigured_settings_give_no_store(self) -> None:
        assert get_configured_object_store(Settings(_env_file=None).object_store) is None

    def test_memory_backend_gives_one_shared_store(self) -> None:
        cfg = Settings(_env_file=None, agentar_object_store="memory").object_store
        assert isinstance(get_configured_object_store(cfg), MemoryObjectStore)
        assert get_configured_object_store(cfg) is get_configured_object_store(cfg)

    def test_s3_backend_builds_a_boto3_store(self) -> None:
        assert isinstance(get_configured_object_store(_cfg()), S3ObjectStore)


class TestS3Store:
    @pytest.fixture
    def stubbed(self):
        store = S3ObjectStore(_cfg(s3_prefix="agentar"))
        stubber = Stubber(store._client)
        stubber.activate()
        yield store, stubber
        stubber.deactivate()

    def test_client_uses_the_configured_endpoint_and_addressing(self) -> None:
        store = S3ObjectStore(_cfg())
        assert store._client.meta.endpoint_url == "http://127.0.0.1:9000"
        assert store._client.meta.config.s3["addressing_style"] == "path"
        assert store._client.meta.config.request_checksum_calculation == "when_required"

    def test_put_adds_the_prefix(self, stubbed) -> None:
        store, stubber = stubbed
        stubber.add_response(
            "put_object",
            {"ETag": '"abc"'},
            {"Bucket": "bkt", "Key": "agentar/u/1/kb/a", "Body": b"hi", "ContentType": "text/plain"},
        )
        info = run(store.put("u/1/kb/a", b"hi", content_type="text/plain"))
        assert (info.key, info.size, info.etag) == ("u/1/kb/a", 2, "abc")
        stubber.assert_no_pending_responses()

    def test_get_streams_in_chunks(self, stubbed) -> None:
        store, stubber = stubbed
        body = b"0123456789"
        stubber.add_response(
            "get_object",
            {"Body": StreamingBody(io.BytesIO(body), len(body)), "ContentLength": 10, "ContentType": "text/plain"},
            {"Bucket": "bkt", "Key": "agentar/u/1/kb/a"},
        )
        stream = run(store.get("u/1/kb/a", chunk_size=4))
        assert stream.info.size == 10
        assert run(_read_all(stream)) == body

    def test_missing_object_maps_to_not_found(self, stubbed) -> None:
        store, stubber = stubbed
        stubber.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)
        with pytest.raises(ObjectNotFound):
            run(store.get("u/1/kb/a"))

    def test_head_of_a_missing_object_is_none(self, stubbed) -> None:
        store, stubber = stubbed
        stubber.add_client_error("head_object", service_error_code="404", http_status_code=404)
        assert run(store.head("u/1/kb/a")) is None

    def test_other_failures_map_to_store_error(self, stubbed) -> None:
        store, stubber = stubbed
        stubber.add_client_error("delete_object", service_error_code="InternalError", http_status_code=500)
        with pytest.raises(ObjectStoreError) as exc:
            run(store.delete("u/1/kb/a"))
        assert not isinstance(exc.value, ObjectNotFound)

    def test_list_strips_the_prefix_and_follows_pages(self, stubbed) -> None:
        store, stubber = stubbed
        stubber.add_response(
            "list_objects_v2",
            {"Contents": [{"Key": "agentar/u/1/kb/b", "Size": 2}], "IsTruncated": True, "NextContinuationToken": "t"},
            {"Bucket": "bkt", "Prefix": "agentar/u/1/"},
        )
        stubber.add_response(
            "list_objects_v2",
            {"Contents": [{"Key": "agentar/u/1/kb/a", "Size": 1}], "IsTruncated": False},
            {"Bucket": "bkt", "Prefix": "agentar/u/1/", "ContinuationToken": "t"},
        )
        assert [(i.key, i.size) for i in run(store.list("u/1/"))] == [("u/1/kb/a", 1), ("u/1/kb/b", 2)]
