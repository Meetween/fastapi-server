"""Caption websocket: a caption too long for its audio never reaches the transcript.

C2 run 2 (K/C2/logs/run2-report.txt, line 10): a 0.5 s chunk came back from the model as a
speech of about 1,500 characters that nobody said. The cap is ASR_CAPTION_MAX_CHARS_PER_S
characters per second of audio plus ASR_CAPTION_CHARS_SLACK. Jigasi sends no duration, so the
duration is the byte length of the 16 kHz mono PCM16 chunk. A dropped caption is not sent to the
room, not stored, so neither the Agentar prompt nor the report can read it, and it is counted.
"""
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from butler import ResultType, TranscriptionResult
from config import settings
from core import SessionManager
from utils import caption_plausibility as plausibility
from utils import new_meeting_id

JIGASI_KEY = "jigasi-test-key"
JIGASI_AUTH = {"Authorization": f"Bearer {JIGASI_KEY}"}
SR = 16000

# C2 run 2, transcript line 10, the start of the invented speech (the full text is 1,486 characters).
INVENTED = (
    "On the 20th of June, we will have a meeting with the Minister for Foreign Affairs and Trade to discuss "
    "our plans in relation to the European Union's Green Deal. I would like to thank the rapporteur, Mr. Duff, "
    "as well as all those who contributed to the report on behalf of the Committee on International Trade. "
)
INVENTED = (INVENTED * 6)[:1486]


@pytest.fixture(autouse=True)
def _jigasi_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "enable_auth", True)
    monkeypatch.setattr(settings, "jigasi_meetween_api_key", JIGASI_KEY)


@pytest.fixture(autouse=True)
def _fresh_counts():
    plausibility.reset_counts()
    yield
    plausibility.reset_counts()


def _pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def _speech(seconds: float) -> bytes:
    t = np.arange(int(SR * seconds)) / SR
    return _pcm(2500 * np.sin(2 * np.pi * 200 * t))


def _silence(seconds: float = 2.0) -> bytes:
    return _pcm(np.zeros(int(SR * seconds)))


def _message(meeting_id: str, audio: bytes) -> str:
    return json.dumps({
        "type": "audio", "meeting_id": meeting_id, "room_name": "room-1", "participant_id": "p1",
        "audio": base64.b64encode(audio).decode(),
    })


def _result(text: str, message_id: int = 0) -> TranscriptionResult:
    return TranscriptionResult(
        participant_id="p1", message_id=message_id, timestamp=1.0, is_interim=False, language="en-US",
        stability=1.0, text=text, confidence=1.0, result_type=ResultType.FINAL_RESULT,
    )


def _sync(ws) -> None:
    ws.send_text("not json")
    assert ws.receive_json()["code"] == "INVALID_JSON"


@pytest.fixture
def captions():
    """The caption call answers with the texts put in `texts`, in turn."""
    texts: list[str] = []

    async def fake_caption(*args, **kwargs):
        text = texts.pop(0)
        _, _, message_id = kwargs["caption_handler"].process_chunk(text)
        return _result(text, message_id)

    recorder = MagicMock()
    recorder.save_chunk = AsyncMock()
    with (
        patch("api.routes.websocket.caption_audio", AsyncMock(side_effect=fake_caption)),
        patch("api.routes.websocket.translate_audio", AsyncMock(return_value="hallo")),
        patch("api.routes.websocket.AudioRecorder", recorder),
    ):
        yield texts


@pytest.fixture
def meeting_id():
    mid = new_meeting_id()
    yield mid
    SessionManager._sessions.pop(mid, None)
    SessionManager._ended_meetings.pop(mid, None)


@pytest.fixture
def chunk_mode(monkeypatch):
    """The C2 setting: one transcript line per caption chunk, output guard unset (off)."""
    monkeypatch.setattr(settings, "asr_utterance_mode", False)
    monkeypatch.setattr(settings, "asr_output_guard", None)
    monkeypatch.setattr(settings, "asr_vad", False)
    monkeypatch.setattr(settings, "asr_silence_rms", 150.0)


def _texts(meeting_id: str) -> list[str]:
    session = SessionManager._sessions.get(meeting_id)
    return [] if session is None else [e.text for e in session.transcript]


@pytest.mark.usefixtures("chunk_mode")
class TestChunkCaptions:
    def test_the_c2_chunk_is_dropped_counted_and_never_sent(self, test_client: TestClient, captions, meeting_id):
        captions.extend([INVENTED, "Of November, 2013."])
        with test_client.websocket_connect(f"/ws/{meeting_id}", headers=JIGASI_AUTH) as ws:
            ws.send_text(_message(meeting_id, _speech(0.5)))
            ws.send_text(_message(meeting_id, _speech(0.5)))
            reply = ws.receive_json()
            _sync(ws)

        assert reply["transcript"] == "Of November, 2013."
        assert _texts(meeting_id) == ["Of November, 2013."]
        assert plausibility.dropped_counts() == {"implausible_length": 1}

    def test_the_cap_is_chars_per_second_plus_slack(self, test_client: TestClient, captions, meeting_id, monkeypatch):
        monkeypatch.setattr(settings, "asr_caption_max_chars_per_s", 25.0)
        monkeypatch.setattr(settings, "asr_caption_chars_slack", 20)
        at_cap, over_cap = "a" * 70, "b" * 71  # 2 s: 25 * 2 + 20 = 70
        captions.extend([at_cap, over_cap])
        with test_client.websocket_connect(f"/ws/{meeting_id}", headers=JIGASI_AUTH) as ws:
            ws.send_text(_message(meeting_id, _speech(2.0)))
            ws.receive_json()
            ws.send_text(_message(meeting_id, _speech(2.0)))
            _sync(ws)

        assert _texts(meeting_id) == [at_cap]
        assert plausibility.dropped_counts() == {"implausible_length": 1}

    def test_zero_turns_the_cap_off(self, test_client: TestClient, captions, meeting_id, monkeypatch):
        monkeypatch.setattr(settings, "asr_caption_max_chars_per_s", 0.0)
        captions.append(INVENTED)
        with test_client.websocket_connect(f"/ws/{meeting_id}", headers=JIGASI_AUTH) as ws:
            ws.send_text(_message(meeting_id, _speech(0.5)))
            ws.receive_json()
            _sync(ws)

        assert _texts(meeting_id) == [INVENTED]
        assert plausibility.dropped_counts() == {}

    def test_a_caption_loop_is_dropped_and_counted(self, test_client: TestClient, captions, meeting_id):
        captions.extend(["very very very", "Yes!"])
        with test_client.websocket_connect(f"/ws/{meeting_id}", headers=JIGASI_AUTH) as ws:
            ws.send_text(_message(meeting_id, _speech(2.0)))
            ws.send_text(_message(meeting_id, _speech(2.0)))
            assert ws.receive_json()["transcript"] == "Yes!"
            _sync(ws)

        assert _texts(meeting_id) == ["Yes!"]
        assert plausibility.dropped_counts() == {"repeat_loop": 1}

    def test_model_reply_drops_are_counted_too(self, test_client: TestClient, captions, meeting_id, monkeypatch):
        monkeypatch.setattr(settings, "asr_output_guard", True)
        captions.extend(["I'm sorry, could you please repeat that?", "Leon."])
        with test_client.websocket_connect(f"/ws/{meeting_id}", headers=JIGASI_AUTH) as ws:
            ws.send_text(_message(meeting_id, _speech(1.0)))
            ws.send_text(_message(meeting_id, _speech(1.0)))
            ws.receive_json()
            _sync(ws)

        assert _texts(meeting_id) == ["Leon."]
        assert plausibility.dropped_counts() == {"model_reply": 1}

    def test_health_shows_the_counts(self, test_client: TestClient, captions, meeting_id):
        captions.append(INVENTED)
        with test_client.websocket_connect(f"/ws/{meeting_id}", headers=JIGASI_AUTH) as ws:
            ws.send_text(_message(meeting_id, _speech(0.5)))
            _sync(ws)

        assert test_client.get("/health").json()["captions_dropped"] == {"implausible_length": 1}


class TestUtteranceText:
    @pytest.fixture
    def utterance_mode(self, monkeypatch):
        monkeypatch.setattr(settings, "asr_utterance_mode", True)
        monkeypatch.setattr(settings, "asr_utterance_max_s", 10.0)
        monkeypatch.setattr(settings, "asr_utterance_idle_s", 60.0)
        monkeypatch.setattr(settings, "asr_utterance_min_speech_ms", 500)
        monkeypatch.setattr(settings, "asr_vad", False)
        monkeypatch.setattr(settings, "asr_silence_rms", 150.0)
        monkeypatch.setattr(settings, "asr_utterance_final_caption", False)

    def test_an_utterance_text_too_long_for_its_audio_is_not_stored(
        self, test_client: TestClient, captions, meeting_id, utterance_mode
    ):
        captions.append("We'll sign the")
        transcribe = AsyncMock(return_value=_result(INVENTED))
        with patch("core.utterance_collector.transcribe_audio", transcribe):
            with test_client.websocket_connect(f"/ws/{meeting_id}", headers=JIGASI_AUTH) as ws:
                ws.send_text(_message(meeting_id, _speech(2.0)))
                ws.receive_json()
                ws.send_text(_message(meeting_id, _silence()))
                _sync(ws)

        assert transcribe.await_count == 1
        assert _texts(meeting_id) == []
        assert plausibility.dropped_counts() == {"implausible_length": 1}

    def test_a_plausible_utterance_is_stored(self, test_client: TestClient, captions, meeting_id, utterance_mode):
        captions.append("We'll sign the")
        transcribe = AsyncMock(return_value=_result("We'll sign the data processing agreement."))
        with patch("core.utterance_collector.transcribe_audio", transcribe):
            with test_client.websocket_connect(f"/ws/{meeting_id}", headers=JIGASI_AUTH) as ws:
                ws.send_text(_message(meeting_id, _speech(2.0)))
                ws.receive_json()
                ws.send_text(_message(meeting_id, _silence()))
                _sync(ws)

        assert _texts(meeting_id) == ["We'll sign the data processing agreement."]
        assert plausibility.dropped_counts() == {}


class TestRule:
    def test_reason(self) -> None:
        assert plausibility.drop_reason(INVENTED, 0.5) == "implausible_length"
        assert plausibility.drop_reason("Of November, 2013.", 0.5) is None
        assert plausibility.drop_reason("uh uh uh uh", 2.0) == "repeat_loop"
        assert plausibility.drop_reason("no, no", 2.0) is None
        assert plausibility.drop_reason("", 2.0) is None

    def test_duration_of_pcm16(self) -> None:
        assert plausibility.audio_seconds(b"\x00" * 16000) == 0.5

    def test_defaults(self) -> None:
        from config import Settings

        fresh = Settings()
        assert fresh.asr_caption_max_chars_per_s == 25.0
        assert fresh.asr_caption_chars_slack == 20
