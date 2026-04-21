"""
Pytest configuration and shared fixtures

This module provides shared fixtures for all tests.
"""
import asyncio
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
from src.database import mongo as mongo_module


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
# Recording Storage Fixtures (Function-scoped)
# ============================================================================

@pytest.fixture
def temp_recordings_dir(tmp_path):
    """Create a temporary recordings directory for testing"""
    from config import settings
    
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir(parents=True)
    
    original_path = settings.recordings_base_path
    settings.recordings_base_path = recordings_dir
    
    yield recordings_dir
    
    settings.recordings_base_path = original_path


@pytest.fixture
def create_test_recording(temp_recordings_dir):
    """Helper fixture to create test recordings"""
    import json
    from datetime import datetime
    
    def _create(room_name: str, meeting_id: str, start_time: datetime, end_time: datetime = None, chunks: int = 1):
        """Create a test recording with metadata and optional chunk files"""
        meeting_path = temp_recordings_dir / room_name / meeting_id
        meeting_path.mkdir(parents=True)
        
        metadata = {
            "meeting_id": meeting_id,
            "room_name": room_name,
            "start_time": start_time.isoformat(),
            "chunk_count": chunks,
            "total_duration": chunks * 5.0
        }
        
        if end_time:
            metadata["end_time"] = end_time.isoformat()
        
        metadata_path = meeting_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        for i in range(1, chunks + 1):
            chunk_path = meeting_path / f"chunk_{i:04d}.wav"
            chunk_path.write_bytes(b"dummy audio data" * 100)
        
        return meeting_path
    
    return _create

# ============================================================================
# Application & Client Fixtures
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def mock_mongo_client():
    """
    Replace the Motor client factory with an in-memory mongomock client so
    tests run without a live MongoDB instance. The mock is connected here
    because the TestClient fixture below does not enter the FastAPI lifespan
    context, so app startup events do not fire.
    """
    from mongomock_motor import AsyncMongoMockClient

    patcher = patch.object(
        mongo_module, "_create_client", lambda uri: AsyncMongoMockClient()
    )
    patcher.start()
    asyncio.run(mongo_module.async_mongo_manager.connect())
    yield
    asyncio.run(mongo_module.async_mongo_manager.close())
    patcher.stop()


@pytest.fixture(autouse=True)
def _clear_mongo_collections(mock_mongo_client):
    """Reset Mongo state between tests to keep them independent."""
    yield
    from src.database.models import MeetingContext, User

    for model in (User, MeetingContext):
        asyncio.run(model.get_collection().delete_many({}))


@pytest.fixture(scope="session", autouse=True)
def mock_engine_manager():
    """
    Mock EngineManager and audio processing to avoid loading actual models during tests.
    
    This fixture runs automatically before all tests and mocks:
    - EngineManager to avoid model loading
    
    The mocks persist for the entire test session.
    """
    # Create a mock engine with generation methods (used by summarize_audio, transcribe_audio, chat)
    mock_engine = Mock()
    mock_engine.max_context_chars = 300_000
    mock_engine.max_history_chars = 600_000
    # Async mock for text-only generation
    async def mock_generate_text(*args, **kwargs):
        return "Mock chat response text for testing"
    mock_engine._generate_text = mock_generate_text
    
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

    async def mock_translate_text(*args, **kwargs):
        return "Mock translation text for testing"

    async def mock_translate_audio(*args, **kwargs):
        return "Mock audio translation text for testing"

    translate_text_patcher = patch('api.routes.translation.translate_text', side_effect=mock_translate_text)
    translate_audio_patcher = patch('api.routes.translation.translate_audio', side_effect=mock_translate_audio)

    # Start all patches
    get_lmm_patcher.start()
    initialize_patcher.start()
    cleanup_patcher.start()
    summarize_patcher.start()
    transcribe_patcher.start()
    translate_text_patcher.start()
    translate_audio_patcher.start()

    yield

    # Stop all patches
    get_lmm_patcher.stop()
    initialize_patcher.stop()
    cleanup_patcher.stop()
    summarize_patcher.stop()
    transcribe_patcher.stop()
    translate_text_patcher.stop()
    translate_audio_patcher.stop()


_TEST_AUTH_EMAIL = "pytest-guard@meetween.test"
_TEST_AUTH_PASSWORD = "pytest-secret-123"


@pytest.fixture
def admin_headers(test_client: TestClient) -> dict[str, str]:
    """Bearer token for the bootstrap admin user."""
    from tests.utils.helpers import obtain_admin_headers

    return obtain_admin_headers(test_client)


@pytest.fixture
def auth_headers(test_client: TestClient) -> dict[str, str]:
    """Bearer token for a dedicated test user (re-created each test after Mongo reset)."""
    from tests.utils.helpers import obtain_bearer_headers

    return obtain_bearer_headers(
        test_client,
        email=_TEST_AUTH_EMAIL,
        password=_TEST_AUTH_PASSWORD,
        first_name="Pytest",
        last_name="Guard",
    )


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
def client(test_client: TestClient, auth_headers: dict[str, str]) -> TestClient:
    """
    Test client with a valid JWT for protected API routes.
    Use ``test_client`` when calling endpoints without authentication.
    """
    return TestClient(app, headers=auth_headers)

