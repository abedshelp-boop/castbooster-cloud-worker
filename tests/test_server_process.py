"""HTTP tests for the async /process + /process_status behavior.

These exercise the path through PipelineManager. The existing
test_process_endpoint.py covers pre-flight validation (auth, CRLF
headers, etc.) that hasn't changed semantically.
"""
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUD_API_KEY", "test-key-async")
    monkeypatch.setenv("DISABLE_IDLE_WATCHER", "1")
    monkeypatch.setenv("HLS_SERVE_DIR", str(tmp_path / "hls"))
    (tmp_path / "hls").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://fake-pod-8080.proxy.runpod.net")


@pytest.fixture
def client_and_manager(monkeypatch, tmp_path):
    """Returns (client, install_fake) where install_fake replaces the
    manager's run_pipeline callable with a test fake.
    """
    from server import app
    c = TestClient(app)
    # Trigger startup so app.state.pipeline_manager exists.
    with c:
        def install(fake):
            app.state.pipeline_manager._run_pipeline = fake  # type: ignore[attr-defined]
        yield c, install


def test_process_returns_200_and_hls_url_within_1s_when_pipeline_is_slow(client_and_manager):
    """Issue 1 fix: /process must not block for the encode duration."""
    c, install = client_and_manager
    block = threading.Event()

    def slow_pipeline(**kwargs):
        block.wait(timeout=5)
        from pipeline_types import PipelineResult
        return PipelineResult(returncode=0, stdout="", stderr="")

    install(slow_pipeline)
    t0 = time.monotonic()
    r = c.post(
        "/process",
        headers={"Authorization": "Bearer test-key-async"},
        json={"source_url": "https://test.m3u8"},
    )
    elapsed = time.monotonic() - t0

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hls_url"] == "https://fake-pod-8080.proxy.runpod.net/hls/playlist.m3u8"
    assert body["state"] == "running"
    assert body["source_url"] == "https://test.m3u8"
    assert body["started_at"] is not None
    assert elapsed < 1.0, f"/process blocked for {elapsed:.3f}s — must return immediately"

    block.set()


def test_process_returns_409_when_pipeline_already_running(client_and_manager):
    c, install = client_and_manager
    block = threading.Event()

    def slow_pipeline(**kwargs):
        block.wait(timeout=5)
        from pipeline_types import PipelineResult
        return PipelineResult(returncode=0, stdout="", stderr="")

    install(slow_pipeline)
    r1 = c.post(
        "/process",
        headers={"Authorization": "Bearer test-key-async"},
        json={"source_url": "https://first.m3u8"},
    )
    assert r1.status_code == 200

    r2 = c.post(
        "/process",
        headers={"Authorization": "Bearer test-key-async"},
        json={"source_url": "https://second.m3u8"},
    )
    assert r2.status_code == 409
    body = r2.json()
    assert body["state"] == "running"
    assert body["source_url"] == "https://first.m3u8"
    assert body["hls_url"].endswith("/hls/playlist.m3u8")
    assert body["started_at"] is not None

    block.set()
