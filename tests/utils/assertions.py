"""
Custom assertion helpers for API tests

Works with both FastAPI TestClient and httpx responses.
"""
from typing import Any, Dict, Optional, Union
from fastapi.testclient import TestClient
import httpx
from models import TaskType

# Type alias for response objects (TestClient and httpx are compatible)
Response = Union[httpx.Response, Any]


def assert_successful_response(response: Response, expected_status: int = 200) -> Dict[str, Any]:
    """
    Assert that a response is successful and return the JSON data.
    
    Args:
        response: HTTP response object
        expected_status: Expected HTTP status code (default: 200)
        
    Returns:
        Parsed JSON response data
        
    Raises:
        AssertionError: If response is not successful
    """
    assert response.status_code == expected_status, (
        f"Expected status {expected_status}, got {response.status_code}. "
        f"Response: {response.text[:500]}"
    )
    return response.json()


def assert_error_response(
    response: Response,
    expected_status: int,
    expected_detail: Optional[str] = None
) -> Dict[str, Any]:
    """
    Assert that a response is an error response.
    
    Args:
        response: HTTP response object
        expected_status: Expected HTTP status code
        expected_detail: Optional expected error detail message
        
    Returns:
        Parsed JSON response data
        
    Raises:
        AssertionError: If response doesn't match expected error
    """
    assert response.status_code == expected_status, (
        f"Expected error status {expected_status}, got {response.status_code}. "
        f"Response: {response.text[:500]}"
    )
    data = response.json()
    
    if expected_detail:
        assert "detail" in data or "error" in data, "Error response should contain detail or error field"
        error_msg = data.get("detail") or data.get("error", "")
        assert expected_detail.lower() in error_msg.lower(), (
            f"Expected error detail to contain '{expected_detail}', got '{error_msg}'"
        )
    
    return data


def assert_task_response(data: Dict[str, Any], expected_type: str = TaskType.TEXT_SUMMARY) -> None:
    """
    Assert that a response matches the MeetweenTaskResponse structure.
    
    Args:
        data: Response JSON data
        expected_type: Expected task type (default: TaskType.TEXT_SUMMARY)
        
    Raises:
        AssertionError: If response doesn't match expected structure
    """
    assert "type" in data or "result" in data, "Response should contain type or result field"
    
    if "type" in data:
        assert data["type"] == expected_type, f"Expected type '{expected_type}', got '{data['type']}'"
    
    if "result" in data:
        assert isinstance(data["result"], str), "Result should be a string"
        assert len(data["result"].strip()) > 0, "Result should not be empty"


def assert_transcription_response(data: Dict[str, Any]) -> None:
    """
    Assert that a response matches the MeetweenTranscriptionResponse structure.
    
    Args:
        data: Response JSON data
        
    Raises:
        AssertionError: If response doesn't match expected structure
    """
    assert "type" in data or "transcript" in data, "Response should contain type or transcript field"

    if "type" in data:
        assert data["type"] == TaskType.TRANSCRIPTION, (
            f"Expected type '{TaskType.TRANSCRIPTION}', got '{data['type']}'"
        )
    
    if "transcript" in data:
        assert isinstance(data["transcript"], str), "Transcript should be a string"


def assert_translation_response(data: Dict[str, Any], expected_type: str = TaskType.TEXT_TRANSLATION) -> None:
    """
    Assert that a response matches the MeetweenTranslationResponse structure.

    Args:
        data: Response JSON data
        expected_type: Expected task type (TaskType.TEXT_TRANSLATION or TaskType.SPEECH_TRANSLATION)

    Raises:
        AssertionError: If response doesn't match expected structure
    """
    assert data.get("type") == expected_type, f"Expected type '{expected_type}', got '{data.get('type')}'"
    assert "result" in data, "Translation response should contain 'result'"
    assert isinstance(data["result"], str), "Result should be a string"
    assert len(data["result"].strip()) > 0, "Result should not be empty"
    assert "target_language" in data, "Translation response should contain 'target_language'"


def assert_file_download_response(response: Response, expected_content_type: Optional[str] = None) -> bytes:
    """
    Assert that a response is a valid file download.
    
    Args:
        response: HTTP response object
        expected_content_type: Optional expected content type
        
    Returns:
        File content as bytes
        
    Raises:
        AssertionError: If response is not a valid file download
    """
    assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
    
    content_type = response.headers.get('Content-Type', '')
    if expected_content_type:
        assert expected_content_type in content_type, (
            f"Expected content type to contain '{expected_content_type}', got '{content_type}'"
        )
    else:
        assert 'text' in content_type.lower() or 'application' in content_type.lower(), (
            f"Expected text or application content type, got {content_type}"
        )
    
    content_disposition = response.headers.get('Content-Disposition', '')
    assert 'attachment' in content_disposition.lower() or 'filename' in content_disposition.lower(), (
        "Response should indicate file download"
    )
    
    file_content = response.content
    assert len(file_content) > 0, "Downloaded file should not be empty"
    
    try:
        text_content = file_content.decode('utf-8')
        assert len(text_content.strip()) > 0, "File content should not be empty or only whitespace"
    except UnicodeDecodeError as e:
        raise AssertionError(f"File content is not valid UTF-8: {e}")
    
    return file_content

