# tests/test_auth.py
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

def test_protected_returns_500_when_api_key_unset(monkeypatch):
    """When CLOUD_API_KEY is missing from the worker env, /_test_protected
    must return 500 — not 401/403 — because the worker is misconfigured,
    not the caller.
    """
    # Override the autouse set_api_key fixture by deleting after it set
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    # Need a fresh client AFTER env mutation so the import-time check sees it
    from server import app
    client = TestClient(app)
    response = client.get("/_test_protected", headers={"Authorization": "Bearer anything"})
    assert response.status_code == 500
    assert "CLOUD_API_KEY not configured" in response.json()["detail"]
