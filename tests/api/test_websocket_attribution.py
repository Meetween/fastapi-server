"""Transcript lines stored from the websocket carry the sender and a timestamp."""

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.routes import websocket as ws_route
from butler import ResultType
from butler.jitsi_api_functions import TranscriptionResult
from config import settings
from core import SessionManager

MEETING_ID = "6f0c1b52-3c1a-4d0e-9a55-2f4f1d1a9b11"
ROOM = "attribution-room"
TIMESTAMP = 1_700_000_000_123.0


def _result(is_interim: bool = False) -> TranscriptionResult:
    return TranscriptionResult(
        participant_id=None,
        message_id=1,
        timestamp=TIMESTAMP,
        is_interim=is_interim,
        language="en-US",
        stability=1.0,
        text="hello there",
        confidence=1.0,
        result_type=ResultType.FINAL_RESULT,
    )


def _audio_message(participant_id):
    message = {
        "type": "audio",
        "language": "en-US",
        "room_name": ROOM,
        "meeting_id": MEETING_ID,
        "audio": base64.b64encode(b"\x00\x01" * 1600).decode(),
    }
    if participant_id is not None:
        message["participant_id"] = participant_id
    return message


@pytest.fixture(autouse=True)
def _stub_pipeline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(SessionManager, "_sessions", {})
    monkeypatch.setattr(SessionManager, "_ended_meetings", {})
    monkeypatch.setattr(ws_route.EngineManager, "get_lmm", MagicMock(return_value=object()))
    monkeypatch.setattr(
        SessionManager, "get_or_create_caption_handler", MagicMock(return_value=object())
    )
    monkeypatch.setattr(ws_route.AudioRecorder, "init_recording", MagicMock())
    monkeypatch.setattr(ws_route.AudioRecorder, "save_chunk", AsyncMock())
    monkeypatch.setattr(ws_route, "caption_audio", AsyncMock(return_value=_result()))
    monkeypatch.setattr(ws_route, "_get_translations", AsyncMock(return_value=None))


@pytest.mark.integration
class TestWebsocketAttribution:
    def test_audio_message_stores_line_with_participant_and_timestamp(
        self, test_client: TestClient
    ):
        SessionManager.add_participants(
            meeting_id=MEETING_ID,
            room_name=ROOM,
            participants=[{"participant_id": "p1", "username": "Ada"}],
        )

        with test_client.websocket_connect(f"/ws/{MEETING_ID}") as ws:
            ws.send_json(_audio_message("p1"))
            assert ws.receive_json()["participant_id"] == "p1"

        entries = SessionManager._sessions[MEETING_ID].transcript
        assert len(entries) == 1
        assert entries[0].participant_id == "p1"
        assert entries[0].username == "Ada"
        assert entries[0].text == "hello there"
        assert entries[0].timestamp == TIMESTAMP
        assert SessionManager.get_transcript(MEETING_ID, ROOM) == "Ada: hello there"

    def test_interim_result_is_not_stored(self, test_client: TestClient, monkeypatch):
        monkeypatch.setattr(
            ws_route, "caption_audio", AsyncMock(return_value=_result(is_interim=True))
        )

        with test_client.websocket_connect(f"/ws/{MEETING_ID}") as ws:
            ws.send_json(_audio_message("p1"))
            ws.receive_json()

        session = SessionManager._sessions.get(MEETING_ID)
        assert session is None or session.transcript == []


@pytest.mark.integration
class TestWebsocketVirtualParticipant:
    @pytest.fixture(autouse=True)
    def _participants(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings, "vp_bot_name_suffix", "(bot)")
        SessionManager.add_participants(
            meeting_id=MEETING_ID,
            room_name=ROOM,
            participants=[
                {"participant_id": "p1", "username": "Ada"},
                {"participant_id": "bot1", "username": "Agentar (bot)"},
            ],
        )

    def test_bot_audio_is_neither_transcribed_nor_stored(self, test_client: TestClient):
        with test_client.websocket_connect(f"/ws/{MEETING_ID}") as ws:
            ws.send_json(_audio_message("bot1"))
            ws.send_json(_audio_message("p1"))
            # Only the human chunk is answered: the bot chunk produced no message before it.
            assert ws.receive_json()["participant_id"] == "p1"

        assert ws_route.caption_audio.await_count == 1
        assert ws_route.AudioRecorder.save_chunk.await_count == 2
        assert SessionManager.get_transcript(MEETING_ID, ROOM) == "Ada: hello there"

    def test_unset_suffix_transcribes_the_bot(self, test_client: TestClient, monkeypatch):
        monkeypatch.setattr(settings, "vp_bot_name_suffix", "")

        with test_client.websocket_connect(f"/ws/{MEETING_ID}") as ws:
            ws.send_json(_audio_message("bot1"))
            assert ws.receive_json()["participant_id"] == "bot1"

        assert SessionManager.get_transcript(MEETING_ID, ROOM) == "Agentar (bot): hello there"

    def test_participant_missing_from_the_list_is_transcribed(self, test_client: TestClient):
        with test_client.websocket_connect(f"/ws/{MEETING_ID}") as ws:
            ws.send_json(_audio_message("late"))
            assert ws.receive_json()["participant_id"] == "late"

        assert SessionManager.get_transcript(MEETING_ID, ROOM) == "Unknown: hello there"
