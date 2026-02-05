# Test Suite

This directory contains a comprehensive test suite for the FastAPI server using pytest, following best practices for test organization, maintainability, and reliability.

## Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared pytest fixtures and configuration
├── README.md                # This file
├── api/                     # API endpoint tests (organized by feature)
│   ├── __init__.py
│   ├── test_summary.py      # Summary endpoint tests
│   ├── test_transcription.py # Transcription endpoint tests
│   ├── test_catchup.py      # Catchup endpoint tests
│   ├── test_health.py       # Health check endpoint tests
│   └── test_errors.py       # Error handling and validation tests
└── utils/                   # Test utilities and helpers
    ├── __init__.py
    ├── assertions.py        # Custom assertion helpers
    └── helpers.py           # Test data generation helpers
```


## Running Tests

### Run all tests
```bash
pytest
```

### Run only unit tests
```bash
pytest -m unit
```

### Run only integration tests
```bash
pytest -m integration
```

### Run only regression tests
```bash
pytest -m regression
```

### Run specific test file
```bash
pytest tests/api/test_summary.py
```

### Run specific test class
```bash
pytest tests/api/test_summary.py::TestSummarizeEndpoint
```

### Run specific test method
```bash
pytest tests/api/test_summary.py::TestSummarizeEndpoint::test_summarize_with_audio
```

### Run with coverage report
```bash
pytest --cov=. --cov-report=html
```

### Run with verbose output
```bash
pytest -v
```

### Run tests in parallel (requires pytest-xdist)
```bash
pytest -n auto
```

### Run tests matching a pattern
```bash
pytest -k "summary"
```

## Configuration

Test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`.

### Environment Variables

- `TEST_BASE_URL` - Base URL for the API server (default: `http://localhost:8002`)
- `TEST_ROOM_NAME` - Test room name (default: hardcoded test value)
- `TEST_MEETING_ID` - Test meeting ID (default: hardcoded test value)

## Writing New Tests

### 1. Choose the Right Location
- API endpoint tests → `tests/api/test_<feature>.py`
- Unit tests → `tests/unit/test_<module>.py`
- Utility tests → `tests/utils/test_<utility>.py`

### 2. Follow the Structure
```python
import pytest
import httpx
from tests.utils.assertions import assert_successful_response
from tests.utils.helpers import create_audio_payload

@pytest.mark.integration
@pytest.mark.regression
class TestMyEndpoint:
    """Test suite for /api/my-endpoint"""

    def test_my_endpoint_success(
        self,
        base_url: str,
        test_room_name: str,
        test_meeting_id: str,
        httpx_client: httpx.Client
    ):
        """Test my endpoint with valid input"""
        # Arrange
        payload = create_audio_payload(
            room_name=test_room_name,
            meeting_id=test_meeting_id,
            audio_data=b'\x00' * 1000
        )
        
        # Act
        response = httpx_client.post(
            f"{base_url}/api/my-endpoint",
            json=payload
        )
        
        # Assert
        data = assert_successful_response(response)
        assert "expected_field" in data
```

### 3. Use Helpers and Assertions
- Use `tests.utils.helpers` for creating test payloads
- Use `tests.utils.assertions` for consistent validation
- Leverage fixtures from `conftest.py`

### 4. Add Appropriate Markers
- Mark integration tests with `@pytest.mark.integration`
- Mark critical tests with `@pytest.mark.regression`
- Mark slow tests with `@pytest.mark.slow`

### 5. Test Error Cases
Always include tests for:
- Missing required fields
- Invalid input data
- Edge cases
- Error responses

## Test Utilities

### Assertion Helpers (`tests/utils/assertions.py`)

- `assert_successful_response()` - Verify successful HTTP response
- `assert_error_response()` - Verify error HTTP response
- `assert_task_response()` - Verify task response structure
- `assert_transcription_response()` - Verify transcription response structure
- `assert_file_download_response()` - Verify file download response

### Helper Functions (`tests/utils/helpers.py`)

- `create_audio_payload()` - Create payload with audio data
- `create_text_payload()` - Create payload with text data
- `create_generic_payload()` - Create generic payload (no audio/text)
- `generate_test_meeting_id()` - Generate unique meeting ID
- `generate_test_room_name()` - Generate unique room name

## Fixtures

### Configuration Fixtures (Session-scoped)
- `base_url` - API server base URL
- `test_room_name` - Test room name
- `test_meeting_id` - Test meeting ID

### Test Data Fixtures (Function-scoped)
- `dummy_audio_bytes` - Dummy audio bytes
- `dummy_audio_base64` - Base64-encoded dummy audio
- `sample_text` - Sample text for testing
- `unique_meeting_id` - Unique meeting ID per test
- `unique_room_name` - Unique room name per test

### Client Fixtures (Function-scoped)
- `httpx_client` - HTTP client for making requests

## Continuous Integration

Tests are designed to run in CI/CD pipelines. Make sure to:
1. Set `TEST_BASE_URL` environment variable
2. Ensure the server is running before integration tests
3. Use appropriate markers to skip slow tests if needed

## Troubleshooting

### Tests failing with connection errors
- Ensure the server is running on the configured `TEST_BASE_URL`
- Check firewall/network settings

### Tests failing with validation errors
- Verify test data matches expected API schema
- Check that required fields are included in payloads

### Slow test execution
- Use `-m "not slow"` to skip slow tests during development
- Consider using `pytest-xdist` for parallel execution
