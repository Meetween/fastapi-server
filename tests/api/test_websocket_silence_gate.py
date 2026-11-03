"""Caption websocket: chunks without speech never reach the model."""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from butler import ResultType, TranscriptionResult
from config import settings
from core import SessionManager
from utils import new_meeting_id

SR = 16000


def _pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def _silence() -> bytes:
    return _pcm(np.zeros(SR * 2))


def _speech() -> bytes:
    t = np.arange(SR * 2) / SR
    return _pcm(2500 * np.sin(2 * np.pi * 200 * t))


def _message(meeting_id: str, audio: bytes, **extra) -> str:
    return json.dumps(
        {
            "type": "audio",
            "meeting_id": meeting_id,
            "room_name": "room-1",
            "participant_id": "p1",
            "audio": base64.b64encode(audio).decode(),
            **extra,
        }
    )


@pytest.fixture
def caption_mock():
    async def fake_caption(*args, **kwargs):
        handler = kwargs["caption_handler"]
        _, _, message_id = handler.process_chunk("hello there")
        return TranscriptionResult(
            participant_id="p1",
            message_id=message_id,
            timestamp=1.0,
            is_interim=False,
            language="en-US",
            stability=1.0,
            text="hello there",
            confidence=1.0,
            result_type=ResultType.FINAL_RESULT,
        )

    recorder = MagicMock()
    recorder.save_chunk = AsyncMock()
    translate = AsyncMock(return_value="hallo")
    caption = AsyncMock(side_effect=fake_caption)
    with (
        patch("api.routes.websocket.caption_audio", caption),
        patch("api.routes.websocket.translate_audio", translate),
        patch("api.routes.websocket.AudioRecorder", recorder),
    ):
        caption.translate = translate
        yield caption


@pytest.fixture
def meeting_id():
    mid = new_meeting_id()
    yield mid
    SessionManager._sessions.pop(mid, None)


class TestWebsocketSilenceGate:
    def test_silent_chunk_is_not_sent_to_the_model(self, test_client: TestClient, caption_mock, meeting_id):
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _silence()))
            # The socket is handled in order: the answer to the speech chunk is the first frame back.
            ws.send_text(_message(meeting_id, _speech()))
            reply = ws.receive_json()

        assert reply["transcript"] == "hello there"
        assert caption_mock.await_count == 1

    def test_silent_chunk_adds_nothing_to_the_transcript(self, test_client: TestClient, caption_mock, meeting_id):
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _silence()))
            ws.send_text(_message(meeting_id, _silence()))
            ws.send_text("not json")  # synchronisation point: its error reply proves both chunks were handled
            assert ws.receive_json()["code"] == "INVALID_JSON"

        assert caption_mock.await_count == 0
        assert SessionManager.get_transcript(meeting_id=meeting_id, room_name="room-1") in (None, "")

    def test_speech_chunk_is_sent_to_the_model(self, test_client: TestClient, caption_mock, meeting_id):
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech()))
            reply = ws.receive_json()

        assert reply["transcript"] == "hello there"
        assert caption_mock.await_count == 1

    def test_silent_chunk_skips_translation_too(self, test_client: TestClient, caption_mock, meeting_id):
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _silence(), target_languages=["de"]))
            ws.send_text("not json")
            assert ws.receive_json()["code"] == "INVALID_JSON"

        assert caption_mock.await_count == 0
        assert caption_mock.translate.await_count == 0

    def test_chunk_at_the_threshold_is_sent(self, test_client: TestClient, caption_mock, meeting_id, monkeypatch):
        monkeypatch.setattr(settings, "asr_silence_rms", 150.0)
        at_threshold = _pcm(np.full(SR * 2, 150.0))
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, at_threshold))
            ws.receive_json()

        assert caption_mock.await_count == 1

    def test_chunk_just_below_the_threshold_is_dropped(self, test_client: TestClient, caption_mock, meeting_id, monkeypatch):
        monkeypatch.setattr(settings, "asr_silence_rms", 150.0)
        below = _pcm(np.full(SR * 2, 149.0))
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, below))
            ws.send_text("not json")
            assert ws.receive_json()["code"] == "INVALID_JSON"

        assert caption_mock.await_count == 0

    def test_gate_off_keeps_sending_silence(self, test_client: TestClient, caption_mock, meeting_id, monkeypatch):
        monkeypatch.setattr(settings, "asr_silence_rms", 0.0)
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _silence()))
            ws.receive_json()

        assert caption_mock.await_count == 1


def test_gate_defaults():
    from config import Settings

    fresh = Settings()
    assert fresh.asr_silence_rms == 150.0
    assert fresh.asr_silence_min_active_ms == 100
