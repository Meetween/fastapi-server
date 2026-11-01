"""Caption websocket in utterance mode: speech chunks are joined and transcribed once per utterance."""

import asyncio
import base64
import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from butler import ResultType, TranscriptionResult
from config import settings
from core import SessionManager
from utils import new_meeting_id

SR = 16000
CHUNK_BYTES = SR * 2 * 2  # one 2 s chunk of 16 kHz PCM16
JIGASI_KEY = "jigasi-test-key"


def _pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def _silence() -> bytes:
    return _pcm(np.zeros(SR * 2))


def _speech() -> bytes:
    t = np.arange(SR * 2) / SR
    return _pcm(2500 * np.sin(2 * np.pi * 200 * t))


def _tail(speech_ms: int = 300) -> bytes:
    """A 2 s chunk holding `speech_ms` of tone and silence after it, the end of a line."""
    t = np.arange(SR * 2) / SR
    tone = 2500 * np.sin(2 * np.pi * 200 * t)
    tone[SR * speech_ms // 1000:] = 0
    return _pcm(tone)


def _message(meeting_id: str, audio: bytes, participant_id: str = "p1", **extra) -> str:
    return json.dumps(
        {
            "type": "audio",
            "meeting_id": meeting_id,
            "room_name": "room-1",
            "participant_id": participant_id,
            "audio": base64.b64encode(audio).decode(),
            **extra,
        }
    )


def _result(text: str, message_id: int = 0, timestamp: float = 1.0) -> TranscriptionResult:
    return TranscriptionResult(
        participant_id="p1",
        message_id=message_id,
        timestamp=timestamp,
        is_interim=False,
        language="en-US",
        stability=1.0,
        text=text,
        confidence=1.0,
        result_type=ResultType.FINAL_RESULT,
    )


def _sync(ws) -> None:
    """Every message sent before this one has been handled once its error reply is back."""
    ws.send_text("not json")
    assert ws.receive_json()["code"] == "INVALID_JSON"


@pytest.fixture
def pipeline():
    async def fake_caption(*args, **kwargs):
        _, _, message_id = kwargs["caption_handler"].process_chunk("chunk text")
        return _result("chunk text", message_id=message_id, timestamp=kwargs["timestamp"])

    async def fake_transcribe(*args, **kwargs):
        seconds = len(kwargs["audio_bytes"]) // (SR * 2)
        return _result(f"utterance of {seconds} s")

    recorder = MagicMock()
    recorder.save_chunk = AsyncMock()
    caption = AsyncMock(side_effect=fake_caption)
    transcribe = AsyncMock(side_effect=fake_transcribe)
    with (
        patch("api.routes.websocket.caption_audio", caption),
        patch("core.utterance_collector.transcribe_audio", transcribe),
        patch("api.routes.websocket.translate_audio", AsyncMock(return_value="hallo")),
        patch("api.routes.websocket.AudioRecorder", recorder),
    ):
        yield caption, transcribe


@pytest.fixture
def meeting_id():
    mid = new_meeting_id()
    yield mid
    SessionManager._sessions.pop(mid, None)
    SessionManager._ended_meetings.pop(mid, None)


@pytest.fixture
def utterance_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "asr_utterance_mode", True)
    monkeypatch.setattr(settings, "asr_utterance_max_s", 10.0)
    monkeypatch.setattr(settings, "asr_utterance_idle_s", 60.0)
    monkeypatch.setattr(settings, "asr_utterance_min_speech_ms", 500)
    monkeypatch.setattr(settings, "asr_silence_rms", 150.0)
    # Tones stand in for speech here, which the speech detector rightly rejects, so these tests
    # drive the RMS gate. TestCleanAsr covers the detector.
    monkeypatch.setattr(settings, "asr_vad", False)
    # The transcript tests below keep every 2 s caption final; TestFinalCaption turns this on.
    monkeypatch.setattr(settings, "asr_utterance_final_caption", False)


def _lines(meeting_id: str) -> list:
    session = SessionManager._sessions.get(meeting_id)
    return [] if session is None or session.transcript is None else list(session.transcript)


def _seconds(call) -> float:
    return len(call.kwargs["audio_bytes"]) / (SR * 2)


@pytest.mark.usefixtures("utterance_mode")
class TestUtteranceMode:
    def test_utterance_closes_on_the_first_silent_chunk(self, test_client: TestClient, pipeline, meeting_id):
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            for _ in range(3):
                ws.send_text(_message(meeting_id, _speech()))
                ws.receive_json()
            ws.send_text(_message(meeting_id, _silence()))
            _sync(ws)

        assert transcribe.await_count == 1
        assert _seconds(transcribe.await_args_list[0]) == 6.0
        assert [e.text for e in _lines(meeting_id)] == ["utterance of 6 s"]

    def test_utterance_closes_at_the_cap(self, test_client: TestClient, pipeline, meeting_id, monkeypatch):
        monkeypatch.setattr(settings, "asr_utterance_max_s", 4.0)
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            for _ in range(5):
                ws.send_text(_message(meeting_id, _speech()))
                ws.receive_json()

        # 4 s, 4 s at the cap, then the last 2 s on close.
        assert [_seconds(c) for c in transcribe.await_args_list] == [4.0, 4.0, 2.0]
        assert len(_lines(meeting_id)) == 3

    def test_open_utterance_is_flushed_on_close(self, test_client: TestClient, pipeline, meeting_id):
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech()))
            ws.receive_json()
            _sync(ws)
            assert transcribe.await_count == 0

        assert transcribe.await_count == 1
        assert [e.text for e in _lines(meeting_id)] == ["utterance of 2 s"]

    def test_utterance_closes_when_the_speaker_goes_quiet(
        self, test_client: TestClient, pipeline, meeting_id, monkeypatch
    ):
        # Jigasi's VAD drops silence, so a speaker who stops talking sends nothing at all.
        monkeypatch.setattr(settings, "asr_utterance_idle_s", 0.05)
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech()))
            ws.receive_json()
            ws.send_text(_message(meeting_id, _speech()))
            ws.receive_json()
            deadline = time.monotonic() + 2.0
            while not _lines(meeting_id) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert [e.text for e in _lines(meeting_id)] == ["utterance of 4 s"]

        assert transcribe.await_count == 1

    def test_lone_short_tail_is_dropped(self, test_client: TestClient, pipeline, meeting_id):
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _tail(300)))
            # Past the silence gate, so it is still captioned.
            assert ws.receive_json()["transcript"] == "chunk text"
            ws.send_text(_message(meeting_id, _silence()))
            _sync(ws)

        assert transcribe.await_count == 0
        assert _lines(meeting_id) == []

    def test_short_tail_joins_the_speech_before_it(self, test_client: TestClient, pipeline, meeting_id):
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            for audio in (_speech(), _tail(300)):
                ws.send_text(_message(meeting_id, audio))
                ws.receive_json()
            ws.send_text(_message(meeting_id, _silence()))
            _sync(ws)

        assert [_seconds(c) for c in transcribe.await_args_list] == [4.0]
        assert [e.text for e in _lines(meeting_id)] == ["utterance of 4 s"]

    def test_tail_with_enough_speech_is_kept(self, test_client: TestClient, pipeline, meeting_id):
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _tail(600)))
            ws.receive_json()

        assert transcribe.await_count == 1

    def test_captions_are_unchanged(self, test_client: TestClient, pipeline, meeting_id):
        caption, transcribe = pipeline
        replies = []
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            for audio in (_speech(), _speech(), _silence(), _speech()):
                ws.send_text(_message(meeting_id, audio))
                if audio != _silence():
                    replies.append(ws.receive_json())
            _sync(ws)

        assert caption.await_count == 3
        assert all(caption.await_args_list[i].kwargs["audio_bytes"] == _speech() for i in range(3))
        assert [r["transcript"] for r in replies] == ["chunk text"] * 3
        assert len({r["message_id"] for r in replies}) == 3
        assert not any(r["is_interim"] for r in replies)

    def test_transcript_gets_one_line_per_utterance(self, test_client: TestClient, pipeline, meeting_id):
        SessionManager.add_participants(
            meeting_id=meeting_id,
            room_name="room-1",
            participants=[{"participant_id": "p1", "username": "Ada"}],
        )
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            for audio in (_speech(), _speech(), _silence(), _speech(), _silence()):
                ws.send_text(_message(meeting_id, audio))
                if audio != _silence():
                    ws.receive_json()
            _sync(ws)

        lines = _lines(meeting_id)
        assert [e.text for e in lines] == ["utterance of 4 s", "utterance of 2 s"]
        assert all(e.participant_id == "p1" and e.username == "Ada" for e in lines)
        assert lines[0].timestamp < lines[1].timestamp
        # The line carries the arrival time of its first chunk, taken just before that chunk's caption call.
        assert 0 <= caption.await_args_list[0].kwargs["timestamp"] - lines[0].timestamp < 1000

    def test_participants_have_separate_utterances(self, test_client: TestClient, pipeline, meeting_id):
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            for pid in ("p1", "p2", "p1"):
                ws.send_text(_message(meeting_id, _speech(), participant_id=pid))
                ws.receive_json()
            ws.send_text(_message(meeting_id, _silence(), participant_id="p2"))
            _sync(ws)

        lines = _lines(meeting_id)
        assert [(e.participant_id, e.text) for e in lines] == [("p2", "utterance of 2 s"), ("p1", "utterance of 4 s")]

    def test_silence_without_open_utterance_calls_nothing(self, test_client: TestClient, pipeline, meeting_id):
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _silence()))
            _sync(ws)

        assert caption.await_count == 0
        assert transcribe.await_count == 0
        assert _lines(meeting_id) == []

    def test_virtual_participant_is_not_buffered(
        self, test_client: TestClient, pipeline, meeting_id, monkeypatch
    ):
        monkeypatch.setattr(settings, "vp_bot_name_suffix", "(Agentar)")
        SessionManager.add_participants(
            meeting_id=meeting_id,
            room_name="room-1",
            participants=[{"participant_id": "bot", "username": "Ada (Agentar)"}],
        )
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech(), participant_id="bot"))
            _sync(ws)

        assert caption.await_count == 0
        assert transcribe.await_count == 0

    def test_end_signal_drops_open_utterances(
        self, test_client: TestClient, pipeline, meeting_id, monkeypatch
    ):
        monkeypatch.setattr(settings, "enable_auth", True)
        monkeypatch.setattr(settings, "jigasi_meetween_api_key", JIGASI_KEY)
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech()))
            ws.receive_json()
            test_client.post(f"/meetings/{meeting_id}/ended", headers={"Authorization": f"Bearer {JIGASI_KEY}"})

        assert transcribe.await_count == 0
        assert meeting_id not in SessionManager._sessions

    def test_failed_transcription_stores_nothing_and_keeps_captions(
        self, test_client: TestClient, pipeline, meeting_id
    ):
        caption, transcribe = pipeline
        transcribe.side_effect = RuntimeError("triton down")
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech()))
            assert ws.receive_json()["transcript"] == "chunk text"
            ws.send_text(_message(meeting_id, _silence()))
            ws.send_text(_message(meeting_id, _speech()))
            assert ws.receive_json()["transcript"] == "chunk text"

        assert transcribe.await_count == 2
        assert _lines(meeting_id) == []


class TestFinalCaption:
    @pytest.fixture(autouse=True)
    def final_caption(self, utterance_mode, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings, "asr_utterance_final_caption", True)

    def test_chunks_are_interim_and_the_utterance_is_final(self, test_client: TestClient, pipeline, meeting_id):
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech()))
            first = ws.receive_json()
            ws.send_text(_message(meeting_id, _speech()))
            second = ws.receive_json()
            ws.send_text(_message(meeting_id, _silence()))
            final = ws.receive_json()
            ws.send_text(_message(meeting_id, _speech()))
            next_one = ws.receive_json()
            _sync(ws)

        assert (first["is_interim"], first["transcript"]) == (True, "chunk text")
        assert (second["is_interim"], second["transcript"]) == (True, "chunk text chunk text")
        assert final["is_interim"] is False
        assert final["transcript"] == "utterance of 4 s"
        assert (final["participant_id"], final["language"], final["meeting_id"]) == ("p1", "en-US", meeting_id)
        # One message id and one timestamp, the first chunk's, for the whole utterance.
        assert first["message_id"] == second["message_id"] == final["message_id"]
        assert first["timestamp"] == second["timestamp"] == final["timestamp"] == _lines(meeting_id)[0].timestamp
        assert next_one["is_interim"] is True
        assert next_one["message_id"] != final["message_id"]

    def test_final_comes_when_the_speaker_goes_quiet(self, test_client: TestClient, pipeline, meeting_id, monkeypatch):
        monkeypatch.setattr(settings, "asr_utterance_idle_s", 0.05)
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech()))
            interim = ws.receive_json()
            final = ws.receive_json()

        assert interim["is_interim"] is True
        assert (final["is_interim"], final["transcript"]) == (False, "utterance of 2 s")
        assert final["message_id"] == interim["message_id"]

    def test_final_waits_for_the_interim_of_the_chunk_that_hit_the_cap(
        self, test_client: TestClient, pipeline, meeting_id, monkeypatch
    ):
        monkeypatch.setattr(settings, "asr_utterance_max_s", 4.0)
        caption, transcribe = pipeline
        fake_caption = caption.side_effect

        async def slow_caption(*args, **kwargs):
            # The utterance closes before this chunk's caption call; its transcription is faster.
            await asyncio.sleep(0.05)
            return await fake_caption(*args, **kwargs)

        caption.side_effect = slow_caption
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech()))
            ws.send_text(_message(meeting_id, _speech()))
            replies = [ws.receive_json() for _ in range(3)]

        assert [(r["is_interim"], r["transcript"]) for r in replies] == [
            (True, "chunk text"),
            (True, "chunk text chunk text"),
            (False, "utterance of 4 s"),
        ]
        assert len({r["message_id"] for r in replies}) == 1

    def test_final_carries_the_joined_translations(self, test_client: TestClient, pipeline, meeting_id):
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            for _ in range(2):
                ws.send_text(_message(meeting_id, _speech(), language="en-US", target_languages=["de"]))
                interim = ws.receive_json()
            ws.send_text(_message(meeting_id, _silence(), language="en-US", target_languages=["de"]))
            final = ws.receive_json()

        # Each interim keeps its own 2 s translation; Jigasi relays translations of finals only.
        assert interim["translations"] == [{"target_language": "de", "text": "hallo"}]
        assert final["translations"] == [{"target_language": "de", "text": "hallo hallo"}]

    def test_dropped_utterance_ends_with_its_captions(self, test_client: TestClient, pipeline, meeting_id):
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _tail(300)))
            interim = ws.receive_json()
            ws.send_text(_message(meeting_id, _silence()))
            final = ws.receive_json()
            _sync(ws)

        assert transcribe.await_count == 0
        assert _lines(meeting_id) == []
        assert (final["is_interim"], final["transcript"]) == (False, "chunk text")
        assert final["message_id"] == interim["message_id"]

    def test_failed_transcription_ends_with_its_captions(self, test_client: TestClient, pipeline, meeting_id):
        caption, transcribe = pipeline
        transcribe.side_effect = RuntimeError("triton down")
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech()))
            interim = ws.receive_json()
            ws.send_text(_message(meeting_id, _silence()))
            final = ws.receive_json()

        assert _lines(meeting_id) == []
        assert (final["is_interim"], final["transcript"], final["message_id"]) == (
            False, "chunk text", interim["message_id"],
        )

    def test_final_is_skipped_once_the_socket_is_closed(
        self, test_client: TestClient, pipeline, meeting_id, caplog
    ):
        caption, transcribe = pipeline
        with caplog.at_level(logging.INFO, logger="api.routes.websocket"):
            with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
                ws.send_text(_message(meeting_id, _speech()))
                ws.receive_json()

        # The open utterance is still transcribed and stored on close; only its caption has nowhere to go.
        assert [e.text for e in _lines(meeting_id)] == ["utterance of 2 s"]
        assert "Final caption not sent, socket closed" in caplog.text
        assert "Final caption not sent: " not in caplog.text

    def test_end_signal_sends_no_final(self, test_client: TestClient, pipeline, meeting_id, monkeypatch):
        monkeypatch.setattr(settings, "enable_auth", True)
        monkeypatch.setattr(settings, "jigasi_meetween_api_key", JIGASI_KEY)
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech()))
            ws.receive_json()
            test_client.post(f"/meetings/{meeting_id}/ended", headers={"Authorization": f"Bearer {JIGASI_KEY}"})

        assert transcribe.await_count == 0


class TestCleanAsr:
    """Speech detector in front of the model."""

    @pytest.fixture(autouse=True)
    def clean_asr(self, utterance_mode, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings, "asr_vad", None)
        monkeypatch.setattr(settings, "asr_vad_threshold", 0.5)
        monkeypatch.setattr(settings, "asr_vad_min_speech_ms", 96)
        monkeypatch.setattr(settings, "asr_utterance_final_caption", True)

    @pytest.fixture
    def vad(self):
        """Speech ms by chunk: _speech() is all voice, _tail(ms) holds ms of voice, the rest none."""
        rated = {_speech(): 2000, _tail(300): 300, _tail(600): 600}
        detector = MagicMock(side_effect=lambda audio, threshold: rated.get(audio, 0))
        with patch("api.routes.websocket.pcm16_speech_ms", detector):
            yield detector

    def test_detector_is_on_in_utterance_mode(self, test_client: TestClient, pipeline, meeting_id, vad):
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech()))
            ws.receive_json()
            _sync(ws)

        assert vad.call_args.kwargs == {"threshold": 0.5}

    def test_loud_chunk_without_voice_never_reaches_the_model(
        self, test_client: TestClient, pipeline, meeting_id, vad
    ):
        caption, transcribe = pipeline
        # Loud enough for the RMS gate, but the detector hears no voice in it: fan, music, echo.
        noise = _pcm(np.random.default_rng(3).normal(0, 2000, SR * 2))
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _speech()))
            ws.receive_json()
            ws.send_text(_message(meeting_id, noise))
            final = ws.receive_json()
            _sync(ws)

        assert caption.await_count == 1
        # The voiceless chunk also closes the utterance, without its audio.
        assert [_seconds(c) for c in transcribe.await_args_list] == [2.0]
        assert (final["is_interim"], final["transcript"]) == (False, "utterance of 2 s")

    def test_faint_voice_under_the_rms_floor_never_reaches_the_model(
        self, test_client: TestClient, pipeline, meeting_id, vad
    ):
        caption, transcribe = pipeline
        # The bot's answer leaking back through a laptop microphone: voice, but faint.
        t = np.arange(SR * 2) / SR
        echo = _pcm(60 * np.sin(2 * np.pi * 200 * t))
        vad.side_effect = lambda audio, threshold: 2000
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, echo))
            _sync(ws)

        assert caption.await_count == 0
        assert vad.call_count == 0

    def test_utterance_minimum_counts_voice_not_loudness(
        self, test_client: TestClient, pipeline, meeting_id, vad
    ):
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            # 300 ms of voice passes the chunk minimum but not the utterance minimum of 500 ms.
            ws.send_text(_message(meeting_id, _tail(300)))
            ws.receive_json()
            ws.send_text(_message(meeting_id, _silence()))
            ws.receive_json()
            ws.send_text(_message(meeting_id, _tail(600)))
            ws.receive_json()
            ws.send_text(_message(meeting_id, _silence()))
            ws.receive_json()
            _sync(ws)

        assert [_seconds(c) for c in transcribe.await_args_list] == [2.0]

    def test_detector_off_falls_back_to_the_rms_gate(
        self, test_client: TestClient, pipeline, meeting_id, vad, monkeypatch
    ):
        monkeypatch.setattr(settings, "asr_vad", False)
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            ws.send_text(_message(meeting_id, _pcm(np.random.default_rng(3).normal(0, 2000, SR * 2))))
            ws.receive_json()
            _sync(ws)

        assert vad.call_count == 0
        assert caption.await_count == 1


class TestUtteranceModeOff:
    def test_mode_off_stores_one_line_per_caption_chunk(
        self, test_client: TestClient, pipeline, meeting_id, monkeypatch
    ):
        monkeypatch.setattr(settings, "asr_utterance_mode", False)
        caption, transcribe = pipeline
        with test_client.websocket_connect(f"/ws/{meeting_id}") as ws:
            for audio in (_speech(), _speech(), _silence(), _speech()):
                ws.send_text(_message(meeting_id, audio))
                if audio != _silence():
                    ws.receive_json()
            _sync(ws)

        assert transcribe.await_count == 0
        lines = _lines(meeting_id)
        assert [e.text for e in lines] == ["chunk text"] * 3
        assert [e.timestamp for e in lines] == [c.kwargs["timestamp"] for c in caption.await_args_list]


def test_utterance_defaults():
    from config import Settings

    fresh = Settings()
    assert fresh.asr_utterance_mode is False
    assert fresh.asr_utterance_max_s == 10.0
    assert fresh.asr_utterance_idle_s == 3.0
    assert fresh.asr_utterance_min_speech_ms == 500
    assert fresh.asr_utterance_final_caption is True
    assert fresh.asr_vad is None
    assert fresh.asr_vad_threshold == 0.5
    assert fresh.asr_vad_min_speech_ms == 96


@pytest.mark.parametrize("mode, setting, enabled", [
    (False, None, False), (True, None, True), (True, False, False), (False, True, True),
])
def test_detector_follows_utterance_mode_unless_set(mode, setting, enabled):
    from config import Settings

    fresh = Settings(asr_utterance_mode=mode, asr_vad=setting)
    assert fresh.asr_vad_enabled is enabled
