# tests/test_server.py
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUD_API_KEY", "test-key-server")
    monkeypatch.setenv("DISABLE_IDLE_WATCHER", "1")
    monkeypatch.setenv("HLS_SERVE_DIR", str(tmp_path / "hls"))
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://fake-pod-id-8080.proxy.runpod.net")


@pytest.fixture
def client():
    from server import app
    return TestClient(app)


def test_healthz_returns_200_and_status_ok(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
