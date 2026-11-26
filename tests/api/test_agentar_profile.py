"""Profile, onboarding, picture and voice routes."""
import asyncio
import hashlib
import io
import struct
import zlib

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from PIL import Image

from config import settings
from src.database.models.agentar_file import AgentarFile
from src.database.models.user import User
from utils import agentar_profile as profiles
from utils.object_store import ObjectStoreError

ME = "/api/me"
PICTURE = "/api/me/picture"


def _image(size=(1280, 720), fmt="JPEG", color=(200, 30, 30), **save) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format=fmt, **save)
    return out.getvalue()


def _png_declaring(width: int, height: int) -> bytes:
    """A tiny PNG whose header declares a huge frame."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _put(client: TestClient, user, data: bytes, name="photo.jpg", ctype="image/jpeg"):
    return client.put(PICTURE, files={"file": (name, data, ctype)}, headers=user.headers)


def _stored_user(user) -> User:
    return asyncio.run(User.find_by_email(user.email))


def _picture_records(user) -> list[dict]:
    async def run():
        return await AgentarFile.get_collection().find({"owner": user.email, "kind": "picture"}).to_list(None)

    return asyncio.run(run())


class TestAuth:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", ME),
            ("patch", ME),
            ("post", "/api/me/onboarding/skip"),
            ("put", PICTURE),
            ("get", PICTURE),
            ("delete", PICTURE),
            ("get", "/api/agentar/voices"),
        ],
    )
    def test_needs_a_token(self, test_client: TestClient, method, path) -> None:
        response = test_client.request(method, path, headers={})
        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Bearer")


class TestMe:
    def test_fresh_account(self, test_client: TestClient, user_a) -> None:
        body = test_client.get(ME, headers=user_a.headers).json()
        assert body == {
            "email": user_a.email,
            "first_name": "Alice",
            "last_name": "Tester",
            "is_admin": False,
            "profile": {
                "agent_name": None,
                "voice": None,
                "language": "en",
                "onboarded_at": None,
                "onboarding_skipped_at": None,
                "has_picture": False,
            },
            "onboarding_state": "pending",
        }

    def test_no_secret_field_is_returned(self, test_client: TestClient, user_a) -> None:
        text = test_client.get(ME, headers=user_a.headers).text
        assert "password" not in text and "picture_id" not in text and "storage" not in text

    def test_patch_names_and_profile(self, test_client: TestClient, user_a) -> None:
        response = test_client.patch(
            ME,
            json={
                "first_name": "Alicia",
                "last_name": "Marsh",
                "profile": {"agent_name": "Lia", "voice": "am_michael", "language": "en"},
            },
            headers=user_a.headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert (body["first_name"], body["last_name"]) == ("Alicia", "Marsh")
        assert body["profile"]["agent_name"] == "Lia"
        assert body["profile"]["voice"] == "am_michael"
        assert test_client.get(ME, headers=user_a.headers).json() == body

    def test_patch_writes_only_the_sent_fields(self, test_client: TestClient, user_a) -> None:
        test_client.patch(ME, json={"profile": {"agent_name": "Lia", "voice": "af_bella"}}, headers=user_a.headers)
        before = _stored_user(user_a)
        test_client.patch(ME, json={"profile": {"voice": "am_fenrir"}}, headers=user_a.headers)
        after = _stored_user(user_a)
        assert after.profile.agent_name == "Lia"
        assert after.profile.voice == "am_fenrir"
        assert after.first_name == "Alice"
        assert after.password_hash == before.password_hash
        assert after.email == user_a.email and after.is_admin is False

    def test_a_patch_does_not_undo_a_concurrent_write(self, test_client: TestClient, user_a) -> None:
        """`$set` on declared fields: a field changed elsewhere after the user was read survives."""
        asyncio_user = _stored_user(user_a)

        async def other_writer():
            await User.get_collection().update_one(
                {"_id": ObjectId(asyncio_user.id)}, {"$set": {"password_hash": "rehashed-elsewhere"}}
            )

        asyncio.run(other_writer())
        test_client.patch(ME, json={"first_name": "Zed"}, headers=user_a.headers)
        assert _stored_user(user_a).password_hash == "rehashed-elsewhere"

    def test_agent_name_null_clears_it(self, test_client: TestClient, user_a) -> None:
        test_client.patch(ME, json={"profile": {"agent_name": "Lia"}}, headers=user_a.headers)
        body = test_client.patch(ME, json={"profile": {"agent_name": None}}, headers=user_a.headers).json()
        assert body["profile"]["agent_name"] is None

    def test_unicode_agent_name(self, test_client: TestClient, user_a) -> None:
        response = test_client.patch(ME, json={"profile": {"agent_name": "Zoë Łukasz-2"}}, headers=user_a.headers)
        assert response.status_code == 200
        assert response.json()["profile"]["agent_name"] == "Zoë Łukasz-2"

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"profile": {}},
            {"unknown": 1},
            {"profile": {"agent_name": "Lia", "extra": 1}},
            {"first_name": ""},
            {"first_name": "x" * 65},
            {"first_name": "Mallory (AI)"},
            {"first_name": "Agentar"},
            {"last_name": "line\nbreak"},
            {"profile": {"agent_name": "Bot (AI)"}},
            {"profile": {"agent_name": "my AGENTAR"}},
            {"profile": {"agent_name": "x" * 33}},
            {"profile": {"agent_name": " leading"}},
            {"profile": {"agent_name": "<b>x</b>"}},
            {"profile": {"voice": "zz_nobody"}},
            {"profile": {"language": "it"}},
            {"complete_onboarding": False},
            {"first_name": None},
            {"last_name": None},
            {"profile": None},
            {"profile": {"voice": None}},
            {"profile": {"language": None}},
            {"complete_onboarding": None},
        ],
    )
    def test_invalid_patch_is_422_and_changes_nothing(self, test_client: TestClient, user_a, body) -> None:
        before = test_client.get(ME, headers=user_a.headers).json()
        response = test_client.patch(ME, json=body, headers=user_a.headers)
        assert response.status_code == 422
        assert isinstance(response.json()["detail"], list)
        assert test_client.get(ME, headers=user_a.headers).json() == before

    def test_voice_must_be_offered_by_this_server(self, test_client: TestClient, user_a, monkeypatch) -> None:
        monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update={"voices": ("af_heart",)}))
        assert test_client.patch(ME, json={"profile": {"voice": "am_michael"}}, headers=user_a.headers).status_code == 422
        assert test_client.patch(ME, json={"profile": {"voice": "af_heart"}}, headers=user_a.headers).status_code == 200

    def test_users_do_not_see_each_other(self, test_client: TestClient, user_a, user_b) -> None:
        test_client.patch(ME, json={"profile": {"agent_name": "Lia"}}, headers=user_a.headers)
        assert test_client.get(ME, headers=user_b.headers).json()["profile"]["agent_name"] is None
        assert test_client.get(ME, headers=user_b.headers).json()["email"] == user_b.email


class TestOnboarding:
    def test_complete_onboarding_sets_done_once(self, test_client: TestClient, user_a, frozen_clock) -> None:
        first = test_client.patch(ME, json={"complete_onboarding": True}, headers=user_a.headers).json()
        assert first["onboarding_state"] == "done"
        stamp = first["profile"]["onboarded_at"]
        assert stamp.startswith("2026-11-10T09:00:00")
        frozen_clock.advance(hours=1)
        again = test_client.patch(
            ME, json={"complete_onboarding": True, "profile": {"agent_name": "Lia"}}, headers=user_a.headers
        ).json()
        assert again["profile"]["onboarded_at"] == stamp
        assert again["profile"]["agent_name"] == "Lia"

    def test_save_and_complete_in_one_call(self, test_client: TestClient, user_a) -> None:
        body = test_client.patch(
            ME,
            json={"profile": {"agent_name": "Lia", "voice": "af_heart"}, "complete_onboarding": True},
            headers=user_a.headers,
        ).json()
        assert body["onboarding_state"] == "done"
        assert body["profile"]["voice"] == "af_heart"

    def test_editing_without_the_flag_stays_pending(self, test_client: TestClient, user_a) -> None:
        body = test_client.patch(ME, json={"profile": {"agent_name": "Lia"}}, headers=user_a.headers).json()
        assert body["onboarding_state"] == "pending"

    def test_skip(self, test_client: TestClient, user_a, frozen_clock) -> None:
        body = test_client.post("/api/me/onboarding/skip", headers=user_a.headers).json()
        assert body["onboarding_state"] == "skipped"
        assert body["profile"]["onboarding_skipped_at"].startswith("2026-11-10T09:00:00")
        assert body["profile"]["onboarded_at"] is None

    def test_skip_is_idempotent_and_keeps_the_first_time(self, test_client: TestClient, user_a, frozen_clock) -> None:
        first = test_client.post("/api/me/onboarding/skip", headers=user_a.headers).json()
        frozen_clock.advance(minutes=5)
        second = test_client.post("/api/me/onboarding/skip", headers=user_a.headers).json()
        assert second == first

    def test_skip_after_done_leaves_done(self, test_client: TestClient, user_a) -> None:
        test_client.patch(ME, json={"complete_onboarding": True}, headers=user_a.headers)
        body = test_client.post("/api/me/onboarding/skip", headers=user_a.headers).json()
        assert body["onboarding_state"] == "done"
        assert body["profile"]["onboarding_skipped_at"] is None

    def test_completing_after_a_skip_is_done(self, test_client: TestClient, user_a) -> None:
        test_client.post("/api/me/onboarding/skip", headers=user_a.headers)
        body = test_client.patch(ME, json={"complete_onboarding": True}, headers=user_a.headers).json()
        assert body["onboarding_state"] == "done"

    def test_skip_does_not_touch_the_other_user(self, test_client: TestClient, user_a, user_b) -> None:
        test_client.post("/api/me/onboarding/skip", headers=user_a.headers)
        assert test_client.get(ME, headers=user_b.headers).json()["onboarding_state"] == "pending"


class TestFallbacks:
    @staticmethod
    def _user(first=None, last=None, agent_name=None, voice=None) -> User:
        profile = profiles.UserProfile(agent_name=agent_name, voice=voice)
        return User(email="x@y.test", password_hash="h", first_name=first, last_name=last, profile=profile)

    def test_stored_name_wins(self) -> None:
        assert profiles.effective_agent_name(self._user("Ann", "Lee", agent_name="Lia")) == "Lia"

    def test_first_and_last_name(self) -> None:
        assert profiles.effective_agent_name(self._user("Ann", "Lee")) == "Ann Lee"

    def test_first_name_only(self) -> None:
        assert profiles.effective_agent_name(self._user("Ann", None)) == "Ann"

    def test_unusable_characters_are_dropped_and_long_names_cut(self) -> None:
        name = profiles.effective_agent_name(self._user("Zoe (the) <b>", "Q" * 60))
        assert "(" not in name and "<" not in name
        assert 1 <= len(name) <= 32

    def test_the_word_agentar_falls_through(self) -> None:
        assert profiles.effective_agent_name(self._user("Agentar", "Smith")) == "Meetween"
        assert profiles.effective_agent_name(self._user("Ann", "Agentar")) == "Ann"

    def test_nothing_usable(self) -> None:
        assert profiles.effective_agent_name(self._user(None, None)) == "Meetween"
        assert profiles.effective_agent_name(self._user("()", "!!")) == "Meetween"

    def test_profile_absent(self) -> None:
        user = User(email="x@y.test", password_hash="h", first_name="Ann")
        assert profiles.effective_agent_name(user) == "Ann"
        assert profiles.effective_voice(user) == "af_heart"

    def test_voice(self, monkeypatch) -> None:
        assert profiles.effective_voice(self._user(voice="am_michael")) == "am_michael"
        assert profiles.effective_voice(self._user()) == "af_heart"
        monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update={"voices": ("af_bella", "af_heart"), "default_voice": "af_bella"}))
        assert profiles.effective_voice(self._user(voice="am_michael")) == "af_bella"


class TestVoices:
    def test_list(self, test_client: TestClient, user_a) -> None:
        body = test_client.get("/api/agentar/voices", headers=user_a.headers).json()
        assert [v["id"] for v in body] == ["af_heart", "af_bella", "am_fenrir", "am_michael"]
        assert [v["default"] for v in body] == [True, False, False, False]
        by_id = {v["id"]: v for v in body}
        assert by_id["af_heart"] == {
            "id": "af_heart",
            "label": "Heart",
            "gender": "female",
            "sample_url": "/api/agentar/voices/af_heart/sample",
            "default": True,
        }
        assert by_id["am_michael"]["gender"] == "male"
        assert {v["gender"] for v in body} == {"female", "male"}

    def test_list_follows_the_setting(self, test_client: TestClient, user_a, monkeypatch) -> None:
        monkeypatch.setattr(
            settings,
            "_agentar",
            settings.agentar.model_copy(update={"voices": ("am_michael", "af_heart"), "default_voice": "am_michael"}),
        )
        body = test_client.get("/api/agentar/voices", headers=user_a.headers).json()
        assert [(v["id"], v["default"]) for v in body] == [("am_michael", True), ("af_heart", False)]

    def test_unknown_configured_voice_is_skipped(self, test_client: TestClient, user_a, monkeypatch) -> None:
        monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update={"voices": ("af_heart", "xx_new")}))
        assert [v["id"] for v in test_client.get("/api/agentar/voices", headers=user_a.headers).json()] == ["af_heart"]

    def test_sample_is_404_while_the_file_is_missing(self, test_client: TestClient, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(profiles, "VOICE_SAMPLE_DIR", tmp_path)
        response = test_client.get("/api/agentar/voices/af_heart/sample", headers={})
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_sample_is_public_and_cached(self, test_client: TestClient, tmp_path, monkeypatch) -> None:
        wav = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
        (tmp_path / "af_heart.wav").write_bytes(wav)
        monkeypatch.setattr(profiles, "VOICE_SAMPLE_DIR", tmp_path)
        response = test_client.get("/api/agentar/voices/af_heart/sample", headers={})
        assert response.status_code == 200
        assert response.content == wav
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["cache-control"] == "public, max-age=86400"
        assert response.headers["x-content-type-options"] == "nosniff"

    def test_sample_of_a_voice_the_server_does_not_offer_is_404(self, test_client: TestClient, tmp_path, monkeypatch) -> None:
        (tmp_path / "am_michael.wav").write_bytes(b"RIFF")
        monkeypatch.setattr(profiles, "VOICE_SAMPLE_DIR", tmp_path)
        monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update={"voices": ("af_heart",)}))
        assert test_client.get("/api/agentar/voices/am_michael/sample").status_code == 404

    @pytest.mark.parametrize("voice", ["nobody", "..%2F..%2Fetc%2Fpasswd", "af_heart.wav"])
    def test_other_ids_never_reach_the_disk(self, test_client: TestClient, voice) -> None:
        assert test_client.get(f"/api/agentar/voices/{voice}/sample").status_code in {404, 422}


class TestPicture:
    def test_upload_normalises_and_stores(self, test_client: TestClient, user_a, object_store) -> None:
        response = _put(test_client, user_a, _image((1280, 720)))
        assert response.status_code == 200
        body = response.json()
        assert body["width"] == body["height"] == 512
        assert 0 < body["bytes"] <= 400 * 1024
        assert len(body["sha256"]) == 64

        keys = object_store.keys()
        assert len(keys) == 1
        prefix = f"u/{user_a.id}/avatar/"
        assert keys[0].startswith(prefix) and keys[0].endswith(".jpg")
        stored = object_store.read(keys[0])
        assert hashlib.sha256(stored).hexdigest() == body["sha256"]
        assert len(stored) == body["bytes"]
        out = Image.open(io.BytesIO(stored))
        assert (out.format, out.size) == ("JPEG", (512, 512))

        record = _picture_records(user_a)[0]
        assert record["storage_key"] == keys[0]
        assert record["owner"] == user_a.email and record["kind"] == "picture"
        assert keys[0] == f"{prefix}{record['_id']}.jpg"

        me = test_client.get(ME, headers=user_a.headers).json()
        assert me["profile"]["has_picture"] is True

    def test_png_is_accepted_and_the_declared_type_and_name_are_ignored(self, test_client: TestClient, user_a, object_store) -> None:
        response = _put(test_client, user_a, _image((600, 600), "PNG"), name="../../etc/x.exe", ctype="application/octet-stream")
        assert response.status_code == 200
        key = object_store.keys()[0]
        assert "etc" not in key and "exe" not in key and ".." not in key

    def test_get_returns_the_stored_jpeg_with_safe_headers(self, test_client: TestClient, user_a, object_store) -> None:
        put = _put(test_client, user_a, _image()).json()
        response = test_client.get(PICTURE, headers=user_a.headers)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["cache-control"] == "private, no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["etag"] == f'"{put["sha256"]}"'
        assert hashlib.sha256(response.content).hexdigest() == put["sha256"]

    def test_get_without_a_picture_is_404(self, test_client: TestClient, user_a, object_store) -> None:
        response = test_client.get(PICTURE, headers=user_a.headers)
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_replace_deletes_the_previous_object_and_record(self, test_client: TestClient, user_a, object_store) -> None:
        _put(test_client, user_a, _image(color=(255, 0, 0)))
        old_key = object_store.keys()[0]
        second = _put(test_client, user_a, _image(color=(0, 0, 255))).json()
        keys = object_store.keys()
        assert len(keys) == 1 and keys[0] != old_key
        assert hashlib.sha256(object_store.read(keys[0])).hexdigest() == second["sha256"]
        records = _picture_records(user_a)
        assert len(records) == 1 and records[0]["storage_key"] == keys[0]
        assert test_client.get(PICTURE, headers=user_a.headers).content == object_store.read(keys[0])

    def test_delete_removes_object_record_and_flag(self, test_client: TestClient, user_a, object_store) -> None:
        _put(test_client, user_a, _image())
        response = test_client.delete(PICTURE, headers=user_a.headers)
        assert response.status_code == 204 and response.content == b""
        assert object_store.keys() == []
        assert _picture_records(user_a) == []
        assert test_client.get(PICTURE, headers=user_a.headers).status_code == 404
        assert test_client.get(ME, headers=user_a.headers).json()["profile"]["has_picture"] is False

    def test_delete_without_a_picture_is_204(self, test_client: TestClient, user_a, object_store) -> None:
        assert test_client.delete(PICTURE, headers=user_a.headers).status_code == 204

    def test_upload_after_delete(self, test_client: TestClient, user_a, object_store) -> None:
        _put(test_client, user_a, _image())
        test_client.delete(PICTURE, headers=user_a.headers)
        assert _put(test_client, user_a, _image()).status_code == 200
        assert len(object_store.keys()) == 1

    def test_picture_survives_a_profile_patch(self, test_client: TestClient, user_a, object_store) -> None:
        _put(test_client, user_a, _image())
        test_client.patch(ME, json={"profile": {"agent_name": "Lia"}}, headers=user_a.headers)
        assert test_client.get(PICTURE, headers=user_a.headers).status_code == 200

    # Rejections: nothing is stored and an existing picture is untouched.

    @pytest.mark.parametrize(
        ("data", "status", "code"),
        [
            (b"GIF89a" + b"\x00" * 64, 415, "unsupported_type"),
            (b"<svg xmlns='http://www.w3.org/2000/svg'/>", 415, "unsupported_type"),
            (b"%PDF-1.7 not a picture", 415, "unsupported_type"),
            (b"", 415, "unsupported_type"),
            (_image((200, 200)), 422, "picture_invalid"),
            (_image((255, 900), "PNG"), 422, "picture_invalid"),
            (_png_declaring(30000, 30000), 422, "picture_invalid"),
            (_png_declaring(5000, 5000), 422, "picture_invalid"),
            (_png_declaring(4097, 512), 422, "picture_invalid"),
            (_image((600, 600))[:300], 422, "picture_invalid"),
        ],
        ids=["gif", "svg", "pdf", "empty", "small", "narrow", "bomb30000", "bomb5000", "wide", "truncated"],
    )
    def test_rejected_pictures(self, test_client: TestClient, user_a, object_store, data, status, code) -> None:
        keep = _put(test_client, user_a, _image()).json()
        keys = object_store.keys()
        response = _put(test_client, user_a, data, ctype="image/png")
        assert response.status_code == status
        assert response.json()["code"] == code
        assert object_store.keys() == keys
        assert test_client.get(PICTURE, headers=user_a.headers).headers["etag"] == f'"{keep["sha256"]}"'

    def test_over_the_byte_cap_is_413(self, test_client: TestClient, user_a, object_store, monkeypatch) -> None:
        monkeypatch.setattr(settings, "_agentar", settings.agentar.model_copy(update={"picture_max_bytes": 1000}))
        data = _image((600, 600), "PNG") + b"\x00" * 2000
        response = _put(test_client, user_a, data)
        assert response.status_code == 413
        assert response.json()["code"] == "file_too_large"
        assert object_store.keys() == []

    def test_chunked_body_is_411(self, test_client: TestClient, user_a, object_store) -> None:
        def body():
            yield b"--x\r\n"
            yield b"--x--\r\n"

        headers = {**user_a.headers, "Content-Type": "multipart/form-data; boundary=x"}
        response = test_client.put(PICTURE, content=body(), headers=headers)
        assert response.status_code == 411

    def test_missing_file_part_is_422(self, test_client: TestClient, user_a, object_store) -> None:
        response = test_client.put(PICTURE, files={"other": ("a.png", b"x", "image/png")}, headers=user_a.headers)
        assert response.status_code == 422

    def test_storage_down_on_put_is_503_and_nothing_changes(self, test_client: TestClient, user_a, object_store, monkeypatch) -> None:
        async def broken(*args, **kwargs):
            raise ObjectStoreError("down")

        monkeypatch.setattr(object_store, "put", broken)
        response = _put(test_client, user_a, _image())
        assert response.status_code == 503
        assert response.json()["code"] == "storage_unavailable"
        assert _picture_records(user_a) == []
        assert test_client.get(ME, headers=user_a.headers).json()["profile"]["has_picture"] is False

    def test_storage_down_on_get_is_503(self, test_client: TestClient, user_a, object_store, monkeypatch) -> None:
        _put(test_client, user_a, _image())

        async def broken(*args, **kwargs):
            raise ObjectStoreError("down")

        monkeypatch.setattr(object_store, "get", broken)
        response = test_client.get(PICTURE, headers=user_a.headers)
        assert response.status_code == 503 and response.json()["code"] == "storage_unavailable"

    def test_object_lost_from_the_store_is_404(self, test_client: TestClient, user_a, object_store) -> None:
        _put(test_client, user_a, _image())
        asyncio.run(object_store.delete(object_store.keys()[0]))
        assert test_client.get(PICTURE, headers=user_a.headers).status_code == 404

    def test_a_failed_profile_write_removes_the_new_object(self, test_client: TestClient, user_a, object_store, monkeypatch) -> None:
        _put(test_client, user_a, _image(color=(255, 0, 0)))
        keys = object_store.keys()

        async def broken(*args, **kwargs):
            raise RuntimeError("mongo down")

        monkeypatch.setattr(profiles, "_users", lambda: type("C", (), {"find_one_and_update": broken})())
        with pytest.raises(RuntimeError):
            test_client.put(PICTURE, files={"file": ("a.jpg", _image(color=(0, 0, 255)), "image/jpeg")}, headers=user_a.headers)
        assert object_store.keys() == keys
        assert len(_picture_records(user_a)) == 1

    def test_a_failed_delete_of_the_old_object_does_not_fail_the_replace(self, test_client: TestClient, user_a, object_store, monkeypatch) -> None:
        _put(test_client, user_a, _image(color=(255, 0, 0)))
        real_delete = object_store.delete

        async def broken(key):
            raise ObjectStoreError("down")

        monkeypatch.setattr(object_store, "delete", broken)
        assert _put(test_client, user_a, _image(color=(0, 0, 255))).status_code == 200
        monkeypatch.setattr(object_store, "delete", real_delete)
        assert len(object_store.keys()) == 2  # an orphan the purge finds by prefix, not a failed request


class TestPictureAcrossOwners:
    def test_b_cannot_read_a_picture_of_a(self, test_client: TestClient, user_a, user_b, object_store) -> None:
        _put(test_client, user_a, _image())
        assert test_client.get(PICTURE, headers=user_a.headers).status_code == 200
        response = test_client.get(PICTURE, headers=user_b.headers)
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_b_delete_leaves_the_picture_of_a(self, test_client: TestClient, user_a, user_b, object_store) -> None:
        _put(test_client, user_a, _image())
        keys = object_store.keys()
        assert test_client.delete(PICTURE, headers=user_b.headers).status_code == 204
        assert object_store.keys() == keys
        assert test_client.get(PICTURE, headers=user_a.headers).status_code == 200

    def test_b_upload_and_replace_leave_the_picture_of_a(self, test_client: TestClient, user_a, user_b, object_store) -> None:
        _put(test_client, user_a, _image(color=(255, 0, 0)))
        a_key = object_store.keys()[0]
        a_bytes = object_store.read(a_key)
        _put(test_client, user_b, _image(color=(0, 0, 255)))
        _put(test_client, user_b, _image(color=(0, 255, 0)))
        assert object_store.read(a_key) == a_bytes
        assert len(object_store.keys()) == 2
        assert all(k.startswith(f"u/{user_a.id}/") or k.startswith(f"u/{user_b.id}/") for k in object_store.keys())
        assert sorted(k.split("/")[1] for k in object_store.keys()) == sorted([user_a.id, user_b.id])

    def test_a_profile_pointing_at_a_foreign_record_reads_nothing(self, test_client: TestClient, user_a, user_b, object_store) -> None:
        """Defence in depth: even a poisoned picture_id cannot expose another user's object."""
        _put(test_client, user_a, _image())
        a_picture_id = _stored_user(user_a).profile.picture_id

        async def poison():
            await User.get_collection().update_one(
                {"_id": ObjectId(user_b.id)}, {"$set": {"profile.picture_id": a_picture_id}}
            )

        asyncio.run(poison())
        assert test_client.get(PICTURE, headers=user_b.headers).status_code == 404
        assert test_client.delete(PICTURE, headers=user_b.headers).status_code == 204
        assert len(object_store.keys()) == 1  # the object of A is still there
        assert test_client.get(PICTURE, headers=user_a.headers).status_code == 200

    def test_the_path_takes_no_id_from_the_client(self, test_client: TestClient, user_a, object_store) -> None:
        _put(test_client, user_a, _image())
        assert test_client.get(f"{PICTURE}/{ObjectId()}", headers=user_a.headers).status_code in {404, 405}


def test_openapi_documents_the_routes() -> None:
    from main import app

    paths = app.openapi()["paths"]
    for path in (ME, "/api/me/onboarding/skip", PICTURE, "/api/agentar/voices", "/api/agentar/voices/{voice_id}/sample"):
        assert path in paths
