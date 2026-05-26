import os
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "test-secret-key-12345")

@pytest.fixture
def client():
    # Import inside fixture so the env var is set before app reads it
    from server import app
    return TestClient(app)

def test_protected_requires_bearer(client):
    response = client.get("/_test_protected")
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.json()["detail"]

def test_protected_rejects_wrong_bearer(client):
    response = client.get("/_test_protected", headers={"Authorization": "Bearer wrong-key"})
    assert response.status_code == 403
    assert "Invalid API key" in response.json()["detail"]

def test_protected_accepts_correct_bearer(client):
    response = client.get("/_test_protected", headers={"Authorization": "Bearer test-secret-key-12345"})
    assert response.status_code == 200

def test_healthz_does_not_require_bearer(client):
    response = client.get("/healthz")
    assert response.status_code == 200
