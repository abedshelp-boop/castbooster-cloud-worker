# tests/test_diag_endpoint.py
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUD_API_KEY", "diag-key")
    monkeypatch.setenv("DISABLE_IDLE_WATCHER", "1")
    monkeypatch.setenv("HLS_SERVE_DIR", str(tmp_path / "hls"))
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://x.proxy.runpod.net")


@pytest.fixture
def client():
    from server import app
    return TestClient(app)


def test_diag_requires_auth(client):
    response = client.get("/diag")
    assert response.status_code == 401


def test_diag_returns_structured_environment_info(client):
    """The /diag endpoint should not crash even when vsmlrt/vapoursynth aren't
    installed (i.e. running locally on Windows ARM64 without the deps). Should
    return a 200 with the schema we expect, populated with errors where probes
    failed."""
    response = client.get("/diag", headers={"Authorization": "Bearer diag-key"})
    assert response.status_code == 200
    body = response.json()
    # Top-level keys present
    assert "env" in body
    assert "binaries" in body
    assert "vsmlrt" in body
    assert "vapoursynth_core" in body
    assert "model_files" in body
    assert "libvstrt_candidates" in body
    # env keys populated (some empty strings are fine)
    assert "RUNPOD_POD_ID" in body["env"]
    assert "PUBLIC_BASE_URL" in body["env"]
    # vsmlrt should have import_ok set (False on dev machine without it)
    assert "import_ok" in body["vsmlrt"]
