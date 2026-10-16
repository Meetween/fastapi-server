"""
Tests for health check endpoints
"""
import pytest
from fastapi.testclient import TestClient
from config import settings
from tests.utils.assertions import assert_successful_response


@pytest.mark.integration
class TestHealthEndpoints:
    """Test suite for health check endpoints"""

    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint"""
        # Arrange & Act
        response = client.get("/")
        
        # Assert
        data = assert_successful_response(response)
        assert "service" in data
        assert "version" in data
        assert data["commit"] == settings.app_commit
        assert "status" in data
        assert data["status"] == "running"
        assert "engine_loaded" in data

    def test_health_check_endpoint(self, client: TestClient):
        """Test health check endpoint"""
        # Arrange & Act
        response = client.get("/health")
        
        # Assert
        data = assert_successful_response(response)
        assert "status" in data
        assert data["status"] == "healthy"
        assert data["version"] == settings.app_version
        assert data["commit"] == settings.app_commit
        assert "engine" in data
