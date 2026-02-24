"""
Pytest configuration and shared fixtures

This module provides shared fixtures for all tests.
"""
import json
import pytest
import base64
import os
import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from pydub import AudioSegment
from main import app
from core import EngineManager


# ============================================================================
# Test Configuration Fixtures (Session-scoped)
# ============================================================================

@pytest.fixture(scope="session")
def test_room_name() -> str:
    """Test room name for API requests (session-scoped for consistency)"""
    return os.getenv("TEST_ROOM_NAME", "test_room")


@pytest.fixture(scope="session")
def test_meeting_id() -> str:
    """Test meeting ID for API requests (session-scoped for consistency)"""
    return os.getenv("TEST_MEETING_ID", "ws_test_1769430565")


@pytest.fixture(scope="session", autouse=True)
def fake_recordings_dir(test_room_name: str, test_meeting_id: str):
    """
    Create a temporary recording directory with minimal metadata and one chunk
    so that "from recordings" API tests (transcribe/summarize/catchup) can run
    without real disk recordings. Patches config.settings.recordings_base_path.
    """
    tmp = Path(tempfile.mkdtemp(prefix="fastapi_test_recordings_"))
    meeting_path = tmp / test_room_name / test_meeting_id
    meeting_path.mkdir(parents=True, exist_ok=True)

    metadata = {"chunk_count": 1, "total_duration": 60.0}
    (meeting_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    chunk_path = meeting_path / "chunk_0001.wav"
    audio = AudioSegment.silent(duration=600, frame_rate=16000)
    audio.export(str(chunk_path), format="wav")

    from config import settings
    with patch.object(settings, "recordings_base_path", tmp), \
         patch.object(settings, "min_catchup_duration_seconds", 0):
        yield tmp


# ============================================================================
# Test Data Fixtures (Function-scoped)
# ============================================================================

@pytest.fixture
def dummy_audio_bytes() -> bytes:
    """
    Generate valid dummy audio bytes for testing (WebM format).
    
    Creates a short silent audio segment (0.5 seconds) and exports it as WebM
    to ensure it can be decoded by ffmpeg/pydub in the API routes.
    """
    # Create a silent audio segment: 0.5 seconds, mono, 16kHz
    # This matches the format expected by the API routes
    audio = AudioSegment.silent(duration=500, frame_rate=16000)
    
    # Export to WebM format (in-memory)
    buffer = io.BytesIO()
    audio.export(buffer, format="webm")
    buffer.seek(0)
    
    return buffer.read()


@pytest.fixture
def dummy_audio_base64(dummy_audio_bytes: bytes) -> str:
    """Generate base64-encoded dummy audio"""
    return base64.b64encode(dummy_audio_bytes).decode('utf-8')


@pytest.fixture
def sample_text() -> str:
    """Sample text for text-based requests"""
    return (
        "This is a test meeting transcript. We discussed the quarterly results "
        "and the upcoming product launch. The team agreed on the timeline and "
        "budget allocation."
    )

# ============================================================================
# Application & Client Fixtures
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def mock_engine_manager():
    """
    Mock EngineManager and audio processing to avoid loading actual models during tests.
    
    This fixture runs automatically before all tests and mocks:
    - EngineManager to avoid model loading
    
    The mocks persist for the entire test session.
    """
    # Create a mock engine with _generate method (used by summarize_audio and transcribe_audio)
    mock_engine = Mock()
    mock_engine._generate = Mock(return_value="Mock summary/transcription text for testing")
    mock_engine.generate = Mock(return_value="Mock summary text for testing")
    
    # Mock butler functions to return test data without processing audio
    def mock_summarize_audio(*args, **kwargs):
        return "Mock summary text for testing"
    
    def mock_transcribe_audio(*args, **kwargs):
        from butler import TranscriptionResult, ResultType
        return TranscriptionResult(
            participant_id=kwargs.get('participant_id'),
            message_id=0,
            timestamp=0.0,
            is_interim=False,
            language=kwargs.get('language', 'en-US'),
            stability=1.0,
            text="Mock transcription text for testing",
            confidence=1.0,
            result_type=ResultType.FINAL_RESULT,
        )
    
    # Mock EngineManager methods - these patches persist for the session
    get_lmm_patcher = patch.object(EngineManager, 'get_lmm', return_value=mock_engine)
    initialize_patcher = patch.object(EngineManager, 'initialize', return_value=None)
    cleanup_patcher = patch.object(EngineManager, 'cleanup', return_value=None)
    
    
    # Mock butler functions where they're used in routes
    summarize_patcher = patch('api.routes.summary.summarize_audio', side_effect=mock_summarize_audio)
    transcribe_patcher = patch('api.routes.transcription.transcribe_audio', side_effect=mock_transcribe_audio)
    
    # Start all patches
    get_lmm_patcher.start()
    initialize_patcher.start()
    cleanup_patcher.start()
    summarize_patcher.start()
    transcribe_patcher.start()
    
    yield
    
    # Stop all patches
    get_lmm_patcher.stop()
    initialize_patcher.stop()
    cleanup_patcher.stop()
    summarize_patcher.stop()
    transcribe_patcher.stop()


@pytest.fixture(scope="session")
def test_client(mock_engine_manager) -> TestClient:
    """
    FastAPI TestClient for making test requests (session-scoped).
    
    This allows testing the FastAPI application without running a server.
    The TestClient simulates HTTP requests directly to the app.
    
    Session-scoped to ensure startup/shutdown events are called once per test session.
    The mock_engine_manager fixture ensures EngineManager is mocked before creating the client.
    """
    return TestClient(app)


@pytest.fixture
def client(test_client: TestClient) -> TestClient:
    """
    FastAPI TestClient fixture (function-scoped wrapper).
    
    Returns the session-scoped test client for use in individual tests.
    """
    return test_client

