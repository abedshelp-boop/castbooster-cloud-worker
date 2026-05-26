# tests/test_stop_endpoint.py
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUD_API_KEY", "stop-key")
    monkeypatch.setenv("DISABLE_IDLE_WATCHER", "1")
    monkeypatch.setenv("HLS_SERVE_DIR", str(tmp_path / "hls"))
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://x.proxy.runpod.net")


@pytest.fixture
def client():
    from server import app
    return TestClient(app)


def test_stop_requires_auth(client):
    response = client.post("/stop")
    assert response.status_code == 401


def test_stop_triggers_self_terminate(client, monkeypatch):
    called = []
    monkeypatch.setattr("server._self_terminate_pod", lambda: called.append(True))
    response = client.post("/stop", headers={"Authorization": "Bearer stop-key"})
    assert response.status_code == 200
    assert response.json() == {"status": "shutting_down"}
    assert called == [True]
