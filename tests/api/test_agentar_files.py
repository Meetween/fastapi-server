"""Knowledge base files: upload, list, download, delete, quotas, owners, and RustFS.

Routes run against `MemoryObjectStore` (the `object_store` fixture). One test at the end runs
the S3 class and the routes against the local RustFS and is skipped when its environment is absent.
"""
import asyncio
import hashlib
import io
import os
import struct
import uuid
import zipfile
from pathlib import Path

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from config import ObjectStoreSettings, settings
from src.database.models.agentar_file import AgentarFile as FileRecord
from src.database.models.agentar_meeting import BOT_PRESENT_STATUSES, AgentarMeeting
from utils import agentar_files
from utils.object_store import MemoryObjectStore, ObjectStoreError, S3ObjectStore
from utils.uploads import OVERHEAD_BYTES

KB = Path(__file__).resolve().parents[1] / "data" / "agentar_kb"
URL = "/api/agentar/files"
DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MISSING = "0123456789abcdef01234567"


def upload(client: TestClient, user, name="a.txt", data=b"hello world", *, scope="general", meeting_id=None, ctype="text/plain", extra=None):
    form = {"scope": scope, **({"meeting_id": meeting_id} if meeting_id else {}), **(extra or {})}
    return client.post(URL, files={"file": (name, data, ctype)}, data=form, headers=user.headers)


def read_kb(name: str) -> bytes:
    return (KB / name).read_bytes()


def run(coro):
    return asyncio.run(coro)


def make_meeting(owner_email: str, status: str = "draft", **extra) -> str:
    meeting = run(AgentarMeeting(owner=owner_email, room_name="atlas-steering", status=status, **extra).save())
    return meeting.id


def stored_record(file_id: str) -> dict:
    return run(FileRecord.get_collection().find_one({"_id": ObjectId(file_id)}))


def limits(monkeypatch, **update) -> None:
    monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update=update))


class TestUploadAndList:
    @pytest.mark.parametrize(
        ("name", "ctype", "status", "chunks", "chars"),
        [
            ("atlas-status.md", "text/markdown", "ok", 1, 350),
            ("atlas-faq.md", "text/markdown", "ok", 2, 671),
            ("atlas-sites.docx", DOCX_TYPE, "ok", 2, 241),
            ("atlas-timeline.pdf", "application/pdf", "ok", 2, 216),
        ],
    )
    def test_every_e0_file(self, test_client, user_a, object_store, name, ctype, status, chunks, chars) -> None:
        data = read_kb(name)
        response = upload(test_client, user_a, name, data, ctype="application/octet-stream")
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["content_type"] == ctype
        assert body["name"] == name
        assert body["size"] == len(data)
        assert body["sha256"] == hashlib.sha256(data).hexdigest()
        assert (body["scope"], body["meeting_id"]) == ("general", None)
        assert (body["text_status"], body["chunk_count"], body["text_chars"], body["error"]) == (status, chunks, chars, None)
        assert set(body) == {
            "id", "scope", "meeting_id", "name", "size", "content_type", "sha256",
            "text_status", "text_chars", "chunk_count", "error", "created_at",
        }
        key = f"u/{user_a.id}/kb/{body['id']}"
        assert object_store.keys() == [key]
        assert object_store.read(key) == data

    def test_chunks_are_stored_with_their_pages_and_rows(self, test_client, user_a, object_store) -> None:
        pdf = upload(test_client, user_a, "t.pdf", read_kb("atlas-timeline.pdf")).json()["id"]
        sheet = upload(test_client, user_a, "s.docx", read_kb("atlas-sites.docx")).json()["id"]
        assert [(c["n"], c["page"]) for c in stored_record(pdf)["chunks"]] == [(0, 1), (1, 2)]
        assert "Trento | 60 | 1 December 2026" in stored_record(sheet)["chunks"][0]["text"].split("\n")
        record = stored_record(sheet)
        assert record["owner"] == user_a.email and record["kind"] == "document"
        assert record["storage_key"] == f"u/{user_a.id}/kb/{sheet}"

    def test_the_client_content_type_is_ignored(self, test_client, user_a, object_store) -> None:
        as_text = upload(test_client, user_a, "report.txt", read_kb("atlas-timeline.pdf"), ctype="text/plain")
        assert as_text.json()["content_type"] == "application/pdf"
        as_pdf = upload(test_client, user_a, "report.pdf", b"just text", ctype="application/pdf")
        assert as_pdf.json()["content_type"] == "text/plain"

    def test_the_filename_is_display_metadata_only(self, test_client, user_a, object_store) -> None:
        response = upload(test_client, user_a, "../../etc/pass‮wd.txt", b"x")
        assert response.json()["name"] == "passwd.txt"
        assert all(".." not in key and "passwd" not in key for key in object_store.keys())

    def test_list_is_newest_first_with_usage_and_no_internals(self, test_client, user_a, object_store, frozen_clock) -> None:
        first = upload(test_client, user_a, "one.txt", b"1").json()
        frozen_clock.advance(minutes=1)
        second = upload(test_client, user_a, "two.md", b"# two\nbody", ctype="text/markdown").json()
        response = test_client.get(URL, headers=user_a.headers)
        assert response.status_code == 200
        body = response.json()
        assert [item["id"] for item in body["items"]] == [second["id"], first["id"]]
        assert body["items"][0]["content_type"] == "text/markdown"
        assert body["items"][0]["chunk_count"] == 1
        assert body["usage"] == {
            "files": 2, "files_max": 20, "bytes": 11, "bytes_max": 100 * 1024 * 1024, "file_max_bytes": 10 * 1024 * 1024,
        }
        flat = response.text
        assert "chunks" not in flat and "storage_key" not in flat and "owner" not in flat

    def test_empty_list_still_reports_usage(self, test_client, user_a, object_store) -> None:
        body = test_client.get(URL, headers=user_a.headers).json()
        assert body["items"] == [] and body["usage"]["files"] == 0

    def test_extraction_failure_still_stores_the_file(self, test_client, user_a, object_store) -> None:
        response = upload(test_client, user_a, "broken.pdf", b"%PDF-1.4\nnot really a pdf\n%%EOF")
        assert response.status_code == 201
        body = response.json()
        assert (body["text_status"], body["chunk_count"], body["text_chars"]) == ("failed", 0, 0)
        assert body["error"] == "The pdf could not be read"
        assert len(object_store.keys()) == 1

    def test_a_file_without_text_is_empty(self, test_client, user_a, object_store) -> None:
        body = upload(test_client, user_a, "blank.md", b"   \n\n  ").json()
        assert (body["text_status"], body["chunk_count"]) == ("empty", 0)

    def test_text_over_the_cap_is_truncated(self, test_client, user_a, object_store, monkeypatch) -> None:
        limits(monkeypatch, text_max_chars=1000)
        body = upload(test_client, user_a, "long.txt", (b"some words here. " * 200)).json()
        assert body["text_status"] == "truncated"
        assert body["text_chars"] <= 1000

    def test_created_at_follows_the_clock(self, test_client, user_a, object_store, frozen_clock) -> None:
        assert upload(test_client, user_a).json()["created_at"].startswith("2026-11-10T09:00:00")

    def test_the_stored_file_is_not_modified_by_extraction(self, test_client, user_a, object_store) -> None:
        data = b"\xef\xbb\xbfcaf\xc3\xa9 \r\n\x01 odd bytes"
        file_id = upload(test_client, user_a, "odd.txt", data).json()["id"]
        assert object_store.read(f"u/{user_a.id}/kb/{file_id}") == data


class TestRefusals:
    def test_a_chunked_body_is_411(self, test_client, user_a, object_store) -> None:
        def body():
            yield b"--x\r\n"
            yield b"--x--\r\n"

        headers = {**user_a.headers, "Content-Type": "multipart/form-data; boundary=x"}
        response = test_client.post(URL, content=body(), headers=headers)
        assert response.status_code == 411 and response.json()["code"] == "length_required"
        assert object_store.keys() == []

    def test_the_cap_is_ten_megabytes_by_default(self, test_client, user_a, object_store) -> None:
        assert settings.agentar.file_max_bytes == 10 * 1024 * 1024

    def test_exactly_the_cap_is_accepted_and_one_more_byte_is_413(self, test_client, user_a, object_store, monkeypatch) -> None:
        limits(monkeypatch, file_max_bytes=2000)
        assert upload(test_client, user_a, "ok.txt", b"a" * 2000).status_code == 201
        response = upload(test_client, user_a, "big.txt", b"a" * 2001)
        assert response.status_code == 413 and response.json()["code"] == "file_too_large"
        assert len(object_store.keys()) == 1

    def test_a_declared_length_far_over_the_cap_is_refused_before_parsing(self, test_client, user_a, object_store, monkeypatch) -> None:
        limits(monkeypatch, file_max_bytes=2000)
        headers = {**user_a.headers, "Content-Type": "multipart/form-data; boundary=zz"}
        response = test_client.post(URL, content=b"x" * (2000 + OVERHEAD_BYTES + 1), headers=headers)
        assert response.status_code == 413

    def test_two_file_parts_are_422_in_the_validation_shape(self, test_client, user_a, object_store) -> None:
        response = test_client.post(
            URL, files=[("file", ("a.txt", b"1")), ("file", ("b.txt", b"2"))], data={"scope": "general"}, headers=user_a.headers
        )
        assert response.status_code == 422
        assert isinstance(response.json()["detail"], list)
        assert object_store.keys() == []

    def test_an_unknown_form_field_is_422(self, test_client, user_a, object_store) -> None:
        response = upload(test_client, user_a, extra={"role": "admin"})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "role"]
        assert object_store.keys() == []

    def test_more_than_three_fields_are_422(self, test_client, user_a, object_store) -> None:
        response = upload(test_client, user_a, extra={"a": "1", "b": "2", "c": "3"})
        assert response.status_code == 422

    def test_a_missing_file_part_is_422(self, test_client, user_a, object_store) -> None:
        assert test_client.post(URL, data={"scope": "general"}, headers=user_a.headers).status_code == 422
        assert test_client.post(URL, files={"upload": ("a.txt", b"1")}, data={"scope": "general"}, headers=user_a.headers).status_code == 422

    def test_an_empty_file_is_422(self, test_client, user_a, object_store) -> None:
        response = upload(test_client, user_a, "empty.txt", b"")
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "file"]

    @pytest.mark.parametrize(
        ("form", "field"),
        [
            ({}, "scope"),
            ({"scope": "everyone"}, "scope"),
            ({"scope": "meeting"}, "meeting_id"),
            ({"scope": "meeting", "meeting_id": "not-an-id"}, "meeting_id"),
            ({"scope": "meeting", "meeting_id": "0123456789ABCDEF01234567"}, "meeting_id"),
            ({"scope": "general", "meeting_id": MISSING}, "meeting_id"),
        ],
    )
    def test_bad_scope_and_meeting_id_are_422(self, test_client, user_a, object_store, form, field) -> None:
        response = test_client.post(URL, files={"file": ("a.txt", b"hello")}, data=form, headers=user_a.headers)
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", field]
        assert object_store.keys() == []

    @pytest.mark.parametrize(
        ("name", "data"),
        [
            ("pic.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16),
            ("run.exe", b"MZ\x90\x00\x03\x00\x00\x00"),
            ("utf16.txt", "text".encode("utf-16")),
            ("nul.txt", b"text\x00more text"),
            ("latin1.txt", "café".encode("latin-1")),
            ("sheet.xlsx", None),
            ("jar.docx", None),
        ],
    )
    def test_unsupported_content_is_415(self, test_client, user_a, object_store, name, data) -> None:
        if name == "sheet.xlsx":
            data = _zip({"[Content_Types].xml": b"<Types/>", "xl/workbook.xml": b"<w/>"})
        if name == "jar.docx":
            data = _zip({"META-INF/MANIFEST.MF": b"Manifest-Version: 1.0"})
        response = upload(test_client, user_a, name, data, ctype=DOCX_TYPE)
        assert response.status_code == 415, response.text
        assert response.json()["code"] == "unsupported_type"
        assert object_store.keys() == []

    def test_zip_bombs_are_415_and_never_reach_the_parser(self, test_client, user_a, object_store) -> None:
        base = {"[Content_Types].xml": b"<Types/>", "word/document.xml": b"<w:document/>"}
        bombs = {
            "one big entry": {**base, "word/media/bomb.bin": b"\x00" * (60 * 1024 * 1024)},
            "ratio": {**base, "word/media/bomb.bin": b"\x00" * (20 * 1024 * 1024)},
            "many entries": {**base, **{f"e/{n}": b"" for n in range(600)}},
        }
        for entries in bombs.values():
            data = _zip(entries)
            assert len(data) < 200_000
            response = upload(test_client, user_a, "bomb.docx", data, ctype=DOCX_TYPE)
            assert response.status_code == 415
        assert object_store.keys() == []

    def test_a_directory_that_lies_about_sizes_is_stored_as_failed(self, test_client, user_a, object_store) -> None:
        archive = bytearray(
            _zip({"[Content_Types].xml": b"<Types/>", "word/document.xml": b"<w:document>" + b"A" * (30 * 1024 * 1024)})
        )
        central = archive.rfind(b"PK\x01\x02", 0, archive.rfind(b"PK\x05\x06"))
        struct.pack_into("<I", archive, central + 24, 100)
        response = upload(test_client, user_a, "liar.docx", bytes(archive))
        assert response.status_code == 201
        assert response.json()["text_status"] == "failed"

    def test_a_text_file_that_starts_with_pk_is_accepted(self, test_client, user_a, object_store) -> None:
        assert upload(test_client, user_a, "pki.txt", b"PKI roll out plan").status_code == 201

    def test_a_token_is_required(self, test_client, object_store) -> None:
        response = test_client.post(URL, files={"file": ("a.txt", b"1")}, data={"scope": "general"})
        assert response.status_code == 401

    def test_storage_must_be_configured(self, test_client, user_a) -> None:
        response = upload(test_client, user_a)
        assert response.status_code == 503 and response.json()["code"] == "storage_unavailable"


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


class TestQuotas:
    def test_defaults_follow_the_contract(self) -> None:
        cfg = settings.agentar
        assert (cfg.user_max_files, cfg.user_max_bytes) == (20, 100 * 1024 * 1024)

    def test_file_count(self, test_client, user_a, user_b, object_store, monkeypatch) -> None:
        limits(monkeypatch, user_max_files=2)
        assert upload(test_client, user_a, "1.txt", b"1").status_code == 201
        assert upload(test_client, user_a, "2.txt", b"2").status_code == 201
        response = upload(test_client, user_a, "3.txt", b"3")
        assert response.status_code == 422 and response.json()["code"] == "quota_files"
        assert len(object_store.keys()) == 2
        # the quota is per user
        assert upload(test_client, user_b, "1.txt", b"1").status_code == 201

    def test_total_bytes(self, test_client, user_a, object_store, monkeypatch) -> None:
        limits(monkeypatch, user_max_bytes=100)
        assert upload(test_client, user_a, "1.txt", b"a" * 60).status_code == 201
        response = upload(test_client, user_a, "2.txt", b"b" * 41)
        assert response.status_code == 422 and response.json()["code"] == "quota_bytes"
        assert upload(test_client, user_a, "3.txt", b"c" * 40).status_code == 201

    def test_deleting_frees_the_quota(self, test_client, user_a, object_store, monkeypatch) -> None:
        limits(monkeypatch, user_max_files=1)
        file_id = upload(test_client, user_a).json()["id"]
        assert upload(test_client, user_a).status_code == 422
        assert test_client.delete(f"{URL}/{file_id}", headers=user_a.headers).status_code == 204
        assert upload(test_client, user_a).status_code == 201

    def test_a_parallel_upload_that_slips_past_the_first_check_is_undone(self, test_client, user_a, object_store, monkeypatch) -> None:
        """The check before the insert passes for two requests that overlap; the check after it
        removes the record and the object of the one that put the user over."""
        limits(monkeypatch, user_max_files=1)
        assert upload(test_client, user_a, "1.txt", b"1").status_code == 201
        monkeypatch.setattr(agentar_files, "_check_quota", lambda usage, adding: None)
        response = upload(test_client, user_a, "2.txt", b"2")
        assert response.status_code == 422 and response.json()["code"] == "quota_files"
        assert len(object_store.keys()) == 1
        assert len(test_client.get(URL, headers=user_a.headers).json()["items"]) == 1

    def test_pictures_do_not_count(self, test_client, user_a, object_store, monkeypatch) -> None:
        limits(monkeypatch, user_max_files=1, user_max_bytes=10)
        add_picture(user_a, size=5_000_000)
        assert upload(test_client, user_a, "1.txt", b"1").status_code == 201
        assert test_client.get(URL, headers=user_a.headers).json()["usage"]["bytes"] == 1


def add_picture(user, size=1234, scope="general", meeting_id=None) -> str:
    record = FileRecord(
        owner=user.email,
        scope=scope,
        meeting_id=meeting_id,
        kind="picture",
        name="me.jpg",
        size=size,
        content_type="image/jpeg",
        sha256="0" * 64,
        storage_key=f"u/{user.id}/avatar/x.jpg",
        text_status="empty",
    )
    return run(record.save()).id


class TestPicturesAreNotKnowledgeBaseFiles:
    """B4 stores profile pictures as `kind: picture` records in the same collection."""

    def test_a_picture_is_not_listed_counted_downloaded_or_deleted_here(self, test_client, user_a, object_store) -> None:
        picture = add_picture(user_a)
        asyncio.run(object_store.put(f"u/{user_a.id}/avatar/x.jpg", b"\xff\xd8jpeg"))
        body = test_client.get(URL, headers=user_a.headers).json()
        assert body["items"] == [] and body["usage"]["files"] == 0 and body["usage"]["bytes"] == 0
        assert test_client.get(f"{URL}/{picture}/download", headers=user_a.headers).status_code == 404
        assert test_client.delete(f"{URL}/{picture}", headers=user_a.headers).status_code == 404
        assert object_store.keys() == [f"u/{user_a.id}/avatar/x.jpg"]
        assert stored_record(picture) is not None

    def test_a_document_next_to_a_picture_is_listed_alone(self, test_client, user_a, object_store) -> None:
        add_picture(user_a)
        doc = upload(test_client, user_a).json()
        body = test_client.get(URL, headers=user_a.headers).json()
        assert [item["id"] for item in body["items"]] == [doc["id"]]
        assert body["usage"]["files"] == 1


class TestDownload:
    def test_bytes_and_headers(self, test_client, user_a, object_store) -> None:
        data = read_kb("atlas-sites.docx")
        file_id = upload(test_client, user_a, "atlas-sites.docx", data).json()["id"]
        response = test_client.get(f"{URL}/{file_id}/download", headers=user_a.headers)
        assert response.status_code == 200
        assert response.content == data
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "private, no-store"
        assert response.headers["content-length"] == str(len(data))
        assert response.headers["content-disposition"].startswith('attachment; filename="atlas-sites.docx"')

    def test_a_large_file_comes_back_whole(self, test_client, user_a, object_store) -> None:
        data = (b"0123456789abcdef" * 4096 + b"\n") * 4
        file_id = upload(test_client, user_a, "big.txt", data).json()["id"]
        response = test_client.get(f"{URL}/{file_id}/download", headers=user_a.headers)
        assert response.content == data

    async def test_the_body_is_streamed_chunk_by_chunk_and_the_stream_is_closed(self) -> None:
        """The in process test client buffers a response, so the generator is checked directly."""
        from api.routes.agentar_files import _body
        from utils.object_store import ObjectInfo, ObjectStream

        closed = []

        async def pieces():
            try:
                for _ in range(5):
                    yield b"x" * 65536
            finally:
                closed.append(True)

        stream = ObjectStream(ObjectInfo(key="k", size=5 * 65536), pieces())
        assert [len(chunk) async for chunk in _body(stream)] == [65536] * 5
        assert closed == [True]

        closed.clear()
        abandoned = _body(ObjectStream(ObjectInfo(key="k", size=1), pieces()))
        await anext(abandoned)
        await abandoned.aclose()  # the client went away mid download
        assert closed == [True]

    def test_a_hostile_filename_cannot_split_the_header(self, test_client, user_a, object_store) -> None:
        file_id = upload(test_client, user_a, 'a"\r\nSet-Cookie: x=1\r\n.txt', b"hi").json()["id"]
        response = test_client.get(f"{URL}/{file_id}/download", headers=user_a.headers)
        assert "set-cookie" not in response.headers
        assert "\r" not in response.headers["content-disposition"]

    def test_an_html_upload_is_never_served_as_html(self, test_client, user_a, object_store) -> None:
        file_id = upload(test_client, user_a, "page.html", b"<script>alert(1)</script>", ctype="text/html").json()["id"]
        response = test_client.get(f"{URL}/{file_id}/download", headers=user_a.headers)
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.headers["content-disposition"].startswith("attachment;")

    def test_a_missing_object_is_404_and_a_failing_store_is_503(self, test_client, user_a, object_store) -> None:
        file_id = upload(test_client, user_a).json()["id"]
        object_store._objects.clear()
        assert test_client.get(f"{URL}/{file_id}/download", headers=user_a.headers).status_code == 404

        class Down:
            async def get(self, key, **kwargs):
                raise ObjectStoreError("down")

        from api.deps.agentar import get_object_store
        from main import app

        app.dependency_overrides[get_object_store] = lambda: Down()
        response = test_client.get(f"{URL}/{file_id}/download", headers=user_a.headers)
        assert response.status_code == 503 and response.json()["code"] == "storage_unavailable"


class TestOtherOwners:
    def test_every_route_answers_404_for_another_users_file(self, test_client, user_a, user_b, object_store) -> None:
        file_id = upload(test_client, user_a).json()["id"]
        before = object_store.keys()
        for method, path in [("get", f"{URL}/{file_id}/download"), ("delete", f"{URL}/{file_id}")]:
            response = test_client.request(method, path, headers=user_b.headers)
            assert response.status_code == 404, (method, path)
            assert response.json() == {"detail": "Not found", "code": "not_found"}
        assert test_client.get(URL, headers=user_b.headers).json()["items"] == []
        assert object_store.keys() == before
        assert test_client.get(f"{URL}/{file_id}/download", headers=user_a.headers).status_code == 200

    def test_unknown_and_malformed_ids_look_the_same(self, test_client, user_a, object_store) -> None:
        answers = {
            test_client.get(f"{URL}/{value}/download", headers=user_a.headers).text
            for value in (MISSING, "nope", "0123456789ABCDEF01234567", "1" * 25)
        }
        assert answers == {'{"detail":"Not found","code":"not_found"}'}
        assert test_client.delete(f"{URL}/nope", headers=user_a.headers).status_code == 404

    def test_other_users_files_are_not_in_the_list_filters_either(self, test_client, user_a, user_b, object_store) -> None:
        meeting = make_meeting(user_a.email)
        upload(test_client, user_a, scope="meeting", meeting_id=meeting)
        upload(test_client, user_a)
        for params in ({}, {"scope": "general"}, {"scope": "meeting", "meeting_id": meeting}, {"meeting_id": meeting}):
            body = test_client.get(URL, params=params, headers=user_b.headers).json()
            assert body["items"] == [] and body["usage"]["files"] == 0

    def test_a_meeting_of_another_user_is_404_for_upload(self, test_client, user_a, user_b, object_store) -> None:
        meeting = make_meeting(user_a.email)
        response = upload(test_client, user_b, scope="meeting", meeting_id=meeting)
        assert response.status_code == 404 and response.json()["code"] == "not_found"
        assert object_store.keys() == []

    def test_all_routes_need_a_token(self, test_client, object_store) -> None:
        for method, path in [("get", URL), ("get", f"{URL}/{MISSING}/download"), ("delete", f"{URL}/{MISSING}")]:
            assert test_client.request(method, path).status_code == 401


class TestMeetingScope:
    def test_upload_to_an_editable_meeting(self, test_client, user_a, object_store) -> None:
        for status in ("draft", "scheduled"):
            meeting = make_meeting(user_a.email, status)
            response = upload(test_client, user_a, "m.txt", b"meeting file", scope="meeting", meeting_id=meeting)
            assert response.status_code == 201, response.text
            body = response.json()
            assert (body["scope"], body["meeting_id"]) == ("meeting", meeting)
            assert f"u/{user_a.id}/m/{meeting}/kb/{body['id']}" in object_store.keys()

    def test_list_filters(self, test_client, user_a, object_store) -> None:
        meeting_one, meeting_two = make_meeting(user_a.email), make_meeting(user_a.email)
        general = upload(test_client, user_a, "g.txt").json()["id"]
        one = upload(test_client, user_a, "1.txt", scope="meeting", meeting_id=meeting_one).json()["id"]
        two = upload(test_client, user_a, "2.txt", scope="meeting", meeting_id=meeting_two).json()["id"]

        def ids(**params):
            return {item["id"] for item in test_client.get(URL, params=params, headers=user_a.headers).json()["items"]}

        assert ids() == {general, one, two}
        assert ids(scope="general") == {general}
        assert ids(scope="meeting") == {one, two}
        assert ids(scope="meeting", meeting_id=meeting_one) == {one}
        assert ids(meeting_id=meeting_two) == {two}
        usage = test_client.get(URL, params={"scope": "general"}, headers=user_a.headers).json()["usage"]
        assert usage["files"] == 3  # usage is the whole account, whatever the filter

    def test_list_rejects_bad_filters(self, test_client, user_a, object_store) -> None:
        assert test_client.get(URL, params={"scope": "x"}, headers=user_a.headers).status_code == 422
        assert test_client.get(URL, params={"meeting_id": "zz"}, headers=user_a.headers).status_code == 422

    @pytest.mark.parametrize("status", sorted(set(BOT_PRESENT_STATUSES) | {"ended", "missed", "failed"}))
    def test_a_meeting_that_is_not_editable_is_409(self, test_client, user_a, object_store, status) -> None:
        meeting = make_meeting(user_a.email, status)
        response = upload(test_client, user_a, scope="meeting", meeting_id=meeting)
        assert response.status_code == 409 and response.json()["code"] == "invalid_state"
        assert object_store.keys() == []

    def test_an_unknown_meeting_is_404(self, test_client, user_a, object_store) -> None:
        assert upload(test_client, user_a, scope="meeting", meeting_id=MISSING).status_code == 404


class TestDelete:
    def test_removes_the_record_and_the_object(self, test_client, user_a, object_store) -> None:
        file_id = upload(test_client, user_a).json()["id"]
        response = test_client.delete(f"{URL}/{file_id}", headers=user_a.headers)
        assert response.status_code == 204 and response.content == b""
        assert object_store.keys() == []
        assert stored_record(file_id) is None
        assert test_client.get(f"{URL}/{file_id}/download", headers=user_a.headers).status_code == 404
        assert test_client.delete(f"{URL}/{file_id}", headers=user_a.headers).status_code == 404

    def test_the_id_is_pulled_from_the_owners_other_meetings_only(self, test_client, user_a, user_b, object_store) -> None:
        file_id = upload(test_client, user_a).json()["id"]
        other = "f" * 24
        keep = "e" * 24
        ids = {}
        for status in ("draft", "scheduled", "missed", "failed", "ended"):
            ids[status] = make_meeting(user_a.email, status, file_ids=[keep, file_id])
        foreign = make_meeting(user_b.email, "draft", file_ids=[file_id, other])
        assert test_client.delete(f"{URL}/{file_id}", headers=user_a.headers).status_code == 204

        def file_ids(meeting_id):
            return run(AgentarMeeting.get_collection().find_one({"_id": ObjectId(meeting_id)}))["file_ids"]

        for status, meeting_id in ids.items():
            assert file_ids(meeting_id) == [keep], status
        assert file_ids(foreign) == [file_id, other]  # not ours to touch

    @pytest.mark.parametrize("status", BOT_PRESENT_STATUSES)
    def test_409_while_a_meeting_that_lists_it_has_a_bot(self, test_client, user_a, object_store, status) -> None:
        file_id = upload(test_client, user_a).json()["id"]
        draft = make_meeting(user_a.email, "draft", file_ids=[file_id])
        live = make_meeting(user_a.email, status, file_ids=[file_id])
        response = test_client.delete(f"{URL}/{file_id}", headers=user_a.headers)
        assert response.status_code == 409 and response.json()["code"] == "file_in_use"
        assert len(object_store.keys()) == 1 and stored_record(file_id) is not None
        listed = run(AgentarMeeting.get_collection().find_one({"_id": ObjectId(draft)}))["file_ids"]
        assert listed == [file_id]  # nothing was pulled by a refused delete
        # the meeting ends, the file can go
        run(AgentarMeeting.get_collection().update_one({"_id": ObjectId(live)}, {"$set": {"status": "ended", "live_lock": None}}))
        assert test_client.delete(f"{URL}/{file_id}", headers=user_a.headers).status_code == 204

    def test_409_for_a_meeting_file_of_a_live_meeting_even_if_not_listed(self, test_client, user_a, object_store) -> None:
        meeting = make_meeting(user_a.email, "draft")
        file_id = upload(test_client, user_a, scope="meeting", meeting_id=meeting).json()["id"]
        run(AgentarMeeting.get_collection().update_one({"_id": ObjectId(meeting)}, {"$set": {"status": "live"}}))
        assert test_client.delete(f"{URL}/{file_id}", headers=user_a.headers).status_code == 409

    def test_409_for_a_general_file_while_a_live_meeting_uses_the_general_knowledge_base(self, test_client, user_a, object_store) -> None:
        general = upload(test_client, user_a).json()["id"]
        make_meeting(user_a.email, "live", use_general_kb=True)
        assert test_client.delete(f"{URL}/{general}", headers=user_a.headers).status_code == 409

    def test_a_live_meeting_of_someone_else_does_not_block(self, test_client, user_a, user_b, object_store) -> None:
        file_id = upload(test_client, user_a).json()["id"]
        make_meeting(user_b.email, "live", file_ids=[file_id], use_general_kb=True)
        assert test_client.delete(f"{URL}/{file_id}", headers=user_a.headers).status_code == 204

    def test_a_failing_object_delete_does_not_fail_the_request(self, test_client, user_a, object_store, monkeypatch) -> None:
        file_id = upload(test_client, user_a).json()["id"]

        async def fail(key):
            raise ObjectStoreError("down")

        monkeypatch.setattr(object_store, "delete", fail)
        assert test_client.delete(f"{URL}/{file_id}", headers=user_a.headers).status_code == 204
        assert stored_record(file_id) is None


class TestMeetingCascade:
    """`delete_meeting_documents` is what the meeting delete calls."""

    def test_only_the_documents_of_that_meeting_go(self, test_client, user_a, user_b, object_store) -> None:
        mine, other = make_meeting(user_a.email), make_meeting(user_a.email)
        gone = [upload(test_client, user_a, f"{n}.txt", scope="meeting", meeting_id=mine).json()["id"] for n in range(2)]
        kept_other = upload(test_client, user_a, "o.txt", scope="meeting", meeting_id=other).json()["id"]
        kept_general = upload(test_client, user_a, "g.txt").json()["id"]
        picture = add_picture(user_a, scope="meeting", meeting_id=mine)
        theirs = upload(test_client, user_b, "b.txt").json()["id"]

        assert run(agentar_files.delete_meeting_documents(user_a.email, object_store, mine)) == 2

        assert all(stored_record(file_id) is None for file_id in gone)
        assert all(stored_record(file_id) is not None for file_id in (kept_other, kept_general, picture, theirs))
        assert len(object_store.keys()) == 3
        assert run(agentar_files.delete_meeting_documents(user_a.email, object_store, mine)) == 0

    def test_another_owner_cannot_use_it_on_a_meeting_id(self, test_client, user_a, user_b, object_store) -> None:
        mine = make_meeting(user_a.email)
        file_id = upload(test_client, user_a, scope="meeting", meeting_id=mine).json()["id"]
        assert run(agentar_files.delete_meeting_documents(user_b.email, object_store, mine)) == 0
        assert stored_record(file_id) is not None


class TestOrdering:
    """put, then insert; the object goes if the insert fails."""

    def test_the_object_is_deleted_when_the_insert_fails(self, test_client, user_a, object_store, monkeypatch) -> None:
        real = FileRecord.get_collection()

        class Failing:
            def __getattr__(self, name):
                return getattr(real, name)

            async def insert_one(self, doc):
                raise RuntimeError("mongo went away")

        monkeypatch.setattr(FileRecord, "get_collection", classmethod(lambda cls: Failing()))
        client = TestClient(test_client.app, raise_server_exceptions=False)
        response = client.post(URL, files={"file": ("a.txt", b"hello")}, data={"scope": "general"}, headers=user_a.headers)
        assert response.status_code == 500
        assert object_store.keys() == []

    def test_a_failing_put_is_503_and_leaves_no_record(self, test_client, user_a, object_store, monkeypatch) -> None:
        async def fail(key, data, **kwargs):
            raise ObjectStoreError("down")

        monkeypatch.setattr(object_store, "put", fail)
        response = upload(test_client, user_a)
        assert response.status_code == 503 and response.json()["code"] == "storage_unavailable"
        assert run(FileRecord.get_collection().count_documents({})) == 0

    def test_the_record_is_inserted_after_the_object_exists(self, test_client, user_a, object_store, monkeypatch) -> None:
        seen = {}
        real = FileRecord.get_collection()

        class Spy:
            def __getattr__(self, name):
                return getattr(real, name)

            async def insert_one(self, doc):
                seen["keys_at_insert"] = object_store.keys()
                seen["storage_key"] = doc["storage_key"]
                return await real.insert_one(doc)

        monkeypatch.setattr(FileRecord, "get_collection", classmethod(lambda cls: Spy()))
        upload(test_client, user_a)
        assert seen["keys_at_insert"] == [seen["storage_key"]]


# RustFS --------------------------------------------------------------------------------------------

S3_VARS = ("S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY")
rustfs_env = pytest.mark.skipif(
    not all(os.environ.get(name) for name in S3_VARS),
    reason="S3_BUCKET, S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY are not set; see local/README.md for RustFS",
)


@rustfs_env
def test_rustfs_round_trip_through_the_s3_class_and_the_routes(test_client, user_a) -> None:
    """Run with the scoped key of local/storage-fastapi.env and S3_ENDPOINT_URL=http://127.0.0.1:19000.
    Every object lives under a random prefix that the test removes, and the keys are never printed."""
    prefix = f"it-b5-{uuid.uuid4().hex[:12]}"
    cfg = ObjectStoreSettings(
        backend="s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL", "http://127.0.0.1:19000"),
        region=os.environ.get("S3_REGION", "us-east-1"),
        bucket=os.environ["S3_BUCKET"],
        prefix=prefix,
        addressing_style=os.environ.get("S3_ADDRESSING_STYLE", "path"),
        access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
    )
    store = S3ObjectStore(cfg)
    from api.deps.agentar import get_object_store
    from main import app

    async def store_semantics() -> None:
        body = os.urandom(300_000)
        info = await store.put("t/one.bin", body, content_type="application/octet-stream")
        assert info.size == len(body) and info.etag
        head = await store.head("t/one.bin")
        assert head is not None and head.size == len(body)
        stream = await store.get("t/one.bin", chunk_size=64 * 1024)
        chunks = [chunk async for chunk in stream]
        assert len(chunks) >= 4 and b"".join(chunks) == body
        assert await store.head("t/none.bin") is None
        await store.put("t/two.bin", b"2")
        assert [item.key for item in await store.list("t/")] == ["t/one.bin", "t/two.bin"]
        await store.delete("t/one.bin")
        await store.delete("t/one.bin")  # a missing object is not an error
        assert [item.key for item in await store.list("t/")] == ["t/two.bin"]
        with pytest.raises(Exception) as missing:
            await store.get("t/one.bin")
        assert type(missing.value).__name__ == "ObjectNotFound"
        await store.delete("t/two.bin")

    async def cleanup() -> None:
        for item in await store.list(""):
            await store.delete(item.key)

    app.dependency_overrides[get_object_store] = lambda: store
    try:
        run(store_semantics())
        data = read_kb("atlas-sites.docx")
        created = upload(test_client, user_a, "atlas-sites.docx", data)
        assert created.status_code == 201, created.text
        file_id = created.json()["id"]
        assert created.json()["text_status"] == "ok"
        download = test_client.get(f"{URL}/{file_id}/download", headers=user_a.headers)
        assert download.status_code == 200 and download.content == data
        assert [item.key for item in run(store.list(""))] == [f"u/{user_a.id}/kb/{file_id}"]
        assert test_client.delete(f"{URL}/{file_id}", headers=user_a.headers).status_code == 204
        assert run(store.list("")) == []
    finally:
        app.dependency_overrides.pop(get_object_store, None)
        run(cleanup())
