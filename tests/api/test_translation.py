"""
Tests for the translation endpoint (/api/translate)
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from models import TaskType
from tests.utils.assertions import assert_successful_response, assert_error_response, assert_translation_response
from tests.utils.helpers import create_mt_payload, create_st_payload


@pytest.mark.integration
@pytest.mark.regression
class TestMTEndpoint:
    """Test suite for text translation (type=MT)"""

    def test_mt_basic(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str,
    ):
        """Basic text translation returns a valid translation response"""
        payload = create_mt_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            text=sample_text,
            target_language="de",
            participant_id="participant_123"
        )
        response = client.post("/api/translate", json=payload)

        data = assert_successful_response(response)
        assert_translation_response(data, expected_type=TaskType.TEXT_TRANSLATION)
        assert data["target_language"] == "de"

    @pytest.mark.parametrize("source,target", [
        ("en", "de"),
        ("en", "fr"),
        ("fr", "de"),
        ("de", "en"),
    ])
    def test_mt_language_pairs(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str,
        source: str,
        target: str,
    ):
        """Text translation works for various language pairs"""
        payload = create_mt_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            text=sample_text,
            language=source,
            target_language=target,

        )
        response = client.post("/api/translate", json=payload)

        data = assert_successful_response(response)
        assert_translation_response(data, expected_type=TaskType.TEXT_TRANSLATION)
        assert data["target_language"] == target

    def test_mt_with_participant_id(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str,
    ):
        """Participant ID is preserved in the response"""
        payload = create_mt_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            text=sample_text,
            participant_id="participant-123",
        )
        response = client.post("/api/translate", json=payload)

        data = assert_successful_response(response)
        assert data["participant_id"] == "participant-123"

    def test_mt_missing_text(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
    ):
        """Missing text field returns 422"""
        payload = {
            "type": TaskType.TEXT_TRANSLATION,
            "room_name": test_room_name,
            "meeting_id": test_meeting_id,
            "language": "en",
            "target_language": "de",
        }
        response = client.post("/api/translate", json=payload)
        assert_error_response(response, expected_status=422)

    def test_mt_missing_target_language(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str,
    ):
        """Missing target_language field returns 422"""
        payload = {
            "type": TaskType.TEXT_TRANSLATION,
            "room_name": test_room_name,
            "meeting_id": test_meeting_id,
            "text": sample_text,
            "language": "en",
        }
        response = client.post("/api/translate", json=payload)
        assert_error_response(response, expected_status=422)

    def test_mt_model_not_loaded(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        sample_text: str,
    ):
        """Returns 503 when the model is not loaded"""
        from core import EngineManager
        with patch.object(EngineManager, "get_lmm", return_value=None):
            payload = create_mt_payload(
                room_name=test_room_name,
                meeting_id=test_meeting_id,
                text=sample_text,
            )
            response = client.post("/api/translate", json=payload)
        assert_error_response(response, expected_status=503)


@pytest.mark.integration
@pytest.mark.regression
class TestSTEndpoint:
    """Test suite for speech translation (type=ST)"""

    def test_st_basic(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        dummy_audio_bytes: bytes,
    ):
        """Basic speech translation returns a valid translation response"""
        payload = create_st_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            audio_data=dummy_audio_bytes,
            target_language="de",
        )
        response = client.post("/api/translate", json=payload)

        data = assert_successful_response(response)
        assert_translation_response(data, expected_type=TaskType.SPEECH_TRANSLATION)
        assert data["target_language"] == "de"

    @pytest.mark.parametrize("target", ["de", "fr", "es"])
    def test_st_target_languages(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        dummy_audio_bytes: bytes,
        target: str,
    ):
        """Speech translation works for various target languages"""
        payload = create_st_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            audio_data=dummy_audio_bytes,
            target_language=target,
        )
        response = client.post("/api/translate", json=payload)

        data = assert_successful_response(response)
        assert_translation_response(data, expected_type=TaskType.SPEECH_TRANSLATION)
        assert data["target_language"] == target

    def test_st_empty_audio(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
    ):
        """Empty audio data returns 422"""
        import base64
        payload = create_st_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            audio_data=b"",
        )
        response = client.post("/api/translate", json=payload)
        assert_error_response(response, expected_status=422)

    def test_st_invalid_audio(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
    ):
        """Invalid base64 audio data returns 422"""
        payload = {
            "type": TaskType.SPEECH_TRANSLATION,
            "room_name": test_room_name,
            "meeting_id": test_meeting_id,
            "audio": "not-valid-base64!!!",
            "language": "en",
            "target_language": "de",
        }
        response = client.post("/api/translate", json=payload)
        assert_error_response(response, expected_status=422)

    def test_st_missing_target_language(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        dummy_audio_bytes: bytes,
    ):
        """Missing target_language field returns 422"""
        import base64
        payload = {
            "type": TaskType.SPEECH_TRANSLATION,
            "room_name": test_room_name,
            "meeting_id": test_meeting_id,
            "audio": base64.b64encode(dummy_audio_bytes).decode("utf-8"),
            "language": "en",
        }
        response = client.post("/api/translate", json=payload)
        assert_error_response(response, expected_status=422)

    def test_st_model_not_loaded(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
        dummy_audio_bytes: bytes,
    ):
        """Returns 503 when the model is not loaded"""
        from core import EngineManager
        with patch.object(EngineManager, "get_lmm", return_value=None):
            payload = create_st_payload(
                room_name=test_room_name,
                meeting_id=test_meeting_id,
                audio_data=dummy_audio_bytes,
            )
            response = client.post("/api/translate", json=payload)
        assert_error_response(response, expected_status=503)


@pytest.mark.integration
class TestTranslateEndpointValidation:
    """Validation and error handling tests shared across both types"""

    def test_invalid_type_returns_422(
        self,
        client: TestClient,
        test_room_name: str,
        test_meeting_id: str,
    ):
        """Unknown type value returns 422"""
        payload = {
            "type": "INVALID",
            "room_name": test_room_name,
            "meeting_id": test_meeting_id,
        }
        response = client.post("/api/translate", json=payload)
        assert_error_response(response, expected_status=422)

    def test_empty_body_returns_422(self, client: TestClient):
        """Empty request body returns 422"""
        response = client.post("/api/translate", json={})
        assert_error_response(response, expected_status=422)
