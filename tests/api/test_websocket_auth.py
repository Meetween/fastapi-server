"""/ws/{meeting_id}: the Jigasi key on the handshake, strict ids, recordings kept under the base."""

import base64
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect

from api.routes import websocket as ws_route
from butler import ResultType
from butler.jitsi_api_functions import TranscriptionResult
from config import settings
from core import SessionManager
from utils.storage import StorageManager, UnsafePathError

JIGASI_KEY = "jigasi-test-key"
JIGASI_AUTH = {"Authorization": f"Bearer {JIGASI_KEY}"}
MEETING_ID = "0b7f3f0e-5d6a-4d0b-8c39-6a1f2c3d4e5f"
ROOM = "room-1"
CLOSE_UNAUTHORIZED = 4401
CLOSE_POLICY = 1008


def _caption() -> TranscriptionResult:
    return TranscriptionResult(
        participant_id="p1",
        message_id=1,
        timestamp=1_700_000_000_000.0,
        is_interim=False,
        language="en-US",
        stability=1.0,
        text="hello there",
        confidence=1.0,
        result_type=ResultType.FINAL_RESULT,
    )


def _audio(room_name: str = ROOM, meeting_id: str = MEETING_ID) -> dict:
    return {
        "type": "audio",
        "language": "en-US",
        "participant_id": "p1",
        "room_name": room_name,
        "meeting_id": meeting_id,
        "audio": base64.b64encode(b"\x00\x01" * 1600).decode(),
    }


def _refused(test_client: TestClient, path: str, headers: dict | None = None) -> WebSocketDisconnect:
    with pytest.raises(WebSocketDisconnect) as exc:
        with test_client.websocket_connect(path, headers=headers or {}):
            pass
    return exc.value


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


@pytest.fixture
def base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Real recordings on disk, under tmp_path/rec; tmp_path itself is the outside to watch."""
    rec = tmp_path / "rec"
    rec.mkdir()
    monkeypatch.setattr(settings, "recordings_base_path", rec)
    return rec


@pytest.fixture(autouse=True)
def _pipeline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "enable_auth", True)
    monkeypatch.setattr(settings, "jigasi_meetween_api_key", JIGASI_KEY)
    monkeypatch.setattr(settings, "asr_utterance_mode", False)
    monkeypatch.setattr(SessionManager, "_sessions", {})
    monkeypatch.setattr(SessionManager, "_connections", {})
    monkeypatch.setattr(SessionManager, "_ended_meetings", {})
    monkeypatch.setattr(ws_route.EngineManager, "get_lmm", MagicMock(return_value=object()))
    monkeypatch.setattr(SessionManager, "get_or_create_caption_handler", MagicMock(return_value=object()))
    monkeypatch.setattr(ws_route, "_measure_speech", AsyncMock(return_value=(True, 1000, 0, 0)))
    monkeypatch.setattr(ws_route, "caption_audio", AsyncMock(return_value=_caption()))
    monkeypatch.setattr(ws_route, "_get_translations", AsyncMock(return_value=None))


@pytest.mark.integration
class TestHandshakeKey:
    def test_no_key_is_closed_4401_before_accept(self, test_client: TestClient):
        assert _refused(test_client, f"/ws/{MEETING_ID}").code == CLOSE_UNAUTHORIZED
        assert MEETING_ID not in SessionManager._connections

    @pytest.mark.parametrize(
        "header",
        [
            f"Bearer {JIGASI_KEY}x",
            f"Bearer {JIGASI_KEY[:-1]}",
            "Bearer wrong-key",
            f"Basic {JIGASI_KEY}",
            JIGASI_KEY,
            "Bearer",
            "Bearer ",
        ],
    )
    def test_wrong_key_is_closed_4401(self, test_client: TestClient, header: str):
        assert _refused(test_client, f"/ws/{MEETING_ID}", {"Authorization": header}).code == CLOSE_UNAUTHORIZED

    def test_non_ascii_key_is_closed_4401_not_an_error(self, test_client: TestClient):
        headers = {"Authorization": "Bearer jigasi-test-kéy".encode("utf-8")}
        assert _refused(test_client, f"/ws/{MEETING_ID}", headers).code == CLOSE_UNAUTHORIZED

    def test_unset_key_refuses_every_bearer(self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings, "jigasi_meetween_api_key", "")
        assert _refused(test_client, f"/ws/{MEETING_ID}", {"Authorization": "Bearer anything"}).code == CLOSE_UNAUTHORIZED

    def test_user_access_token_is_not_the_service_key(self, test_client: TestClient, user_a):
        assert _refused(test_client, f"/ws/{MEETING_ID}", user_a.headers).code == CLOSE_UNAUTHORIZED

    def test_key_is_checked_before_the_meeting_id(self, test_client: TestClient):
        assert _refused(test_client, "/ws/%2E%2E").code == CLOSE_UNAUTHORIZED

    def test_good_key_is_accepted_and_captions(self, test_client: TestClient, base: Path):
        with test_client.websocket_connect(f"/ws/{MEETING_ID}", headers=JIGASI_AUTH) as ws:
            ws.send_json(_audio())
            reply = ws.receive_json()
        assert reply["transcript"] == "hello there"
        assert reply["meeting_id"] == MEETING_ID

    def test_auth_off_keeps_the_existing_switch(self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch, base: Path):
        monkeypatch.setattr(settings, "enable_auth", False)
        with test_client.websocket_connect(f"/ws/{MEETING_ID}") as ws:
            ws.send_json(_audio())
            assert ws.receive_json()["transcript"] == "hello there"

    def test_ended_meeting_still_answers_409_with_a_good_key(self, test_client: TestClient):
        assert test_client.post(f"/meetings/{MEETING_ID}/ended", headers=JIGASI_AUTH).status_code == 200
        with pytest.raises(WebSocketDenialResponse) as exc:
            with test_client.websocket_connect(f"/ws/{MEETING_ID}", headers=JIGASI_AUTH):
                pass
        assert exc.value.status_code == 409


@pytest.mark.integration
class TestMeetingIdPattern:
    @pytest.mark.parametrize(
        "segment",
        [
            "%2E%2E",  # ..
            "%2E",  # .
            "abc.def",
            "%EF%BC%8E%EF%BC%8E",  # fullwidth dots, NFKC folds them to ..
            "abc%5C..%5Cx",  # backslashes
            "abc%00",  # NUL
            "abc%0A",  # trailing newline, which `$` would accept
            "abc%E2%80%8B",  # zero width space
            "abc%E2%80%AEgpj",  # right to left override
            "%D0%B0bc",  # Cyrillic a
            "-abc",
            "_abc",
            "a" * 129,
        ],
    )
    def test_invalid_meeting_id_is_closed_1008(self, test_client: TestClient, base: Path, segment: str):
        assert _refused(test_client, f"/ws/{segment}", JIGASI_AUTH).code == CLOSE_POLICY
        assert _tree(base.parent) == ["rec"]

    def test_encoded_separators_never_reach_the_handler(self, test_client: TestClient, base: Path):
        with pytest.raises((WebSocketDisconnect, WebSocketDenialResponse)):
            with test_client.websocket_connect("/ws/..%2F..%2Fescape", headers=JIGASI_AUTH):
                pass
        assert _tree(base.parent) == ["rec"]

    @pytest.mark.parametrize("value", [MEETING_ID, MEETING_ID.upper(), "ws_test_1769430565", "a", "a" * 128])
    def test_valid_meeting_ids(self, value: str):
        assert ws_route.valid_meeting_id(value)

    @pytest.mark.parametrize("value", ["", "..", "a/b", "abc\n", "abc\x00", "．．", "a" * 129, None, 7])
    def test_invalid_meeting_ids(self, value):
        assert not ws_route.valid_meeting_id(value)


@pytest.mark.integration
class TestRoomNameAndRecordingPath:
    @pytest.mark.parametrize(
        "room_name",
        [
            "..",
            ".",
            "../escape",
            "../../escape",
            "room/../../escape",
            "room\\..\\escape",
            "ABSOLUTE",  # replaced by an absolute path under tmp_path in the test
            "Room-1",
            "room-1\n",
            "room\x00",
            "rоom",  # Cyrillic o
            "room​",
            "．．",
            "room.1",
            "-room",
            "",
            "a" * 64,
        ],
    )
    def test_invalid_room_name_is_dropped_before_any_path(
        self, test_client: TestClient, base: Path, room_name: str
    ):
        if room_name == "ABSOLUTE":
            room_name = str(base.parent / "absolute-escape")
        with test_client.websocket_connect(f"/ws/{MEETING_ID}", headers=JIGASI_AUTH) as ws:
            ws.send_json(_audio(room_name=room_name))
            reply = ws.receive_json()
        assert reply == {"type": "error", "message": "Invalid room_name", "code": "INVALID_REQUEST"}
        assert _tree(base.parent) == ["rec"]
        ws_route.caption_audio.assert_not_awaited()

    def test_valid_recording_path_still_records(self, test_client: TestClient, base: Path):
        with test_client.websocket_connect(f"/ws/{MEETING_ID}", headers=JIGASI_AUTH) as ws:
            ws.send_json(_audio())
            assert ws.receive_json()["transcript"] == "hello there"
            ws.send_json(_audio())
            assert ws.receive_json()["transcript"] == "hello there"

        meeting_dir = base / ROOM / MEETING_ID
        assert sorted(p.name for p in meeting_dir.iterdir()) == ["chunk_0001.wav", "chunk_0002.wav", "metadata.json"]
        metadata = json.loads((meeting_dir / "metadata.json").read_text())
        assert metadata["room_name"] == ROOM
        assert metadata["chunk_count"] == 2
        assert "end_time" in metadata  # stop_recording ran on close
        assert _tree(base.parent) == sorted(
            ["rec", f"rec/{ROOM}", f"rec/{ROOM}/{MEETING_ID}"]
            + [f"rec/{ROOM}/{MEETING_ID}/{n}" for n in ("chunk_0001.wav", "chunk_0002.wav", "metadata.json")]
        )

    def test_symlinked_room_is_refused_and_nothing_lands_outside(self, test_client: TestClient, base: Path):
        outside = base.parent / "outside"
        outside.mkdir()
        (base / ROOM).symlink_to(outside, target_is_directory=True)

        with test_client.websocket_connect(f"/ws/{MEETING_ID}", headers=JIGASI_AUTH) as ws:
            ws.send_json(_audio())
            reply = ws.receive_json()

        assert reply == {"type": "error", "message": "Recording path refused", "code": "INVALID_REQUEST"}
        assert list(outside.iterdir()) == []
        ws_route.caption_audio.assert_not_awaited()


class TestStorageMeetingPath:
    def test_valid_path_is_under_the_resolved_base(self, base: Path):
        assert StorageManager.meeting_path(base, ROOM, MEETING_ID) == base.resolve() / ROOM / MEETING_ID

    @pytest.mark.parametrize(
        "room_name, meeting_id",
        [
            ("..", MEETING_ID),
            (".", MEETING_ID),
            (ROOM, ".."),
            ("../escape", MEETING_ID),
            (ROOM, "../../escape"),
            ("/tmp/escape", MEETING_ID),
            (ROOM, "/tmp/escape"),
            ("a/b", MEETING_ID),
            ("a\\b", MEETING_ID),
            ("room\x00", MEETING_ID),
            ("room\n", MEETING_ID),
            ("", MEETING_ID),
            (ROOM, ""),
            (None, MEETING_ID),
            ("a" * 256, MEETING_ID),
        ],
    )
    def test_traversal_names_raise(self, base: Path, room_name, meeting_id):
        with pytest.raises(UnsafePathError):
            StorageManager.meeting_path(base, room_name, meeting_id)
        with pytest.raises(UnsafePathError):
            StorageManager.create_meeting_directory(base, room_name, meeting_id)
        assert StorageManager.find_meeting_directory(base, room_name, meeting_id) is None
        assert _tree(base.parent) == ["rec"]

    def test_symlinked_room_outside_raises(self, base: Path):
        outside = base.parent / "outside"
        outside.mkdir()
        (base / ROOM).symlink_to(outside, target_is_directory=True)
        with pytest.raises(UnsafePathError):
            StorageManager.create_meeting_directory(base, ROOM, MEETING_ID)
        assert list(outside.iterdir()) == []

    def test_symlinked_meeting_outside_raises(self, base: Path):
        outside = base.parent / "outside"
        outside.mkdir()
        (base / ROOM).mkdir()
        (base / ROOM / MEETING_ID).symlink_to(outside, target_is_directory=True)
        with pytest.raises(UnsafePathError):
            StorageManager.meeting_path(base, ROOM, MEETING_ID)
        assert StorageManager.find_meeting_directory(base, ROOM, MEETING_ID) is None

    def test_symlink_inside_the_base_is_refused_too(self, base: Path):
        (base / "other").mkdir()
        (base / ROOM).symlink_to(base / "other", target_is_directory=True)
        with pytest.raises(UnsafePathError):
            StorageManager.meeting_path(base, ROOM, MEETING_ID)

    def test_symlinked_base_is_followed_once(self, tmp_path: Path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        path = StorageManager.create_meeting_directory(link, ROOM, MEETING_ID)
        assert path == real.resolve() / ROOM / MEETING_ID
        assert path.is_dir()

    def test_find_returns_existing_and_none_for_missing(self, base: Path):
        assert StorageManager.find_meeting_directory(base, ROOM, MEETING_ID) is None
        created = StorageManager.create_meeting_directory(base, ROOM, MEETING_ID)
        assert StorageManager.find_meeting_directory(base, ROOM, MEETING_ID) == created

    def test_cleanup_never_follows_a_symlinked_room(self, base: Path):
        outside = base.parent / "outside"
        old = outside / "old-meeting"
        old.mkdir(parents=True)
        start = (datetime.utcnow() - timedelta(days=settings.recording_retention_days + 3)).isoformat()
        (old / "metadata.json").write_text(json.dumps({"start_time": start}))
        (base / "linked").symlink_to(outside, target_is_directory=True)

        result = StorageManager.cleanup_old_recordings()

        assert result["deleted_count"] == 0
        assert (old / "metadata.json").exists()
        assert os.path.islink(base / "linked")
