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


def test_process_status_returns_idle_on_fresh_server(client_and_manager):
    c, _ = client_and_manager
    r = c.get("/process_status", headers={"Authorization": "Bearer test-key-async"})
    assert r.status_code == 200
    s = r.json()
    assert s["state"] == "idle"
    assert s["hls_url"] == "https://fake-pod-8080.proxy.runpod.net/hls/playlist.m3u8"
    assert s["source_url"] is None
    assert s["started_at"] is None
    assert s["playlist_ready"] is False
    assert s["n_segments"] == 0
    assert s["error_type"] is None


def test_process_status_running_while_thread_is_alive(client_and_manager):
    c, install = client_and_manager
    block = threading.Event()

    def slow_pipeline(**kwargs):
        block.wait(timeout=5)
        from pipeline_types import PipelineResult
        return PipelineResult(returncode=0, stdout="", stderr="")

    install(slow_pipeline)
    c.post("/process", headers={"Authorization": "Bearer test-key-async"}, json={"source_url": "https://x.m3u8"})

    s1 = c.get("/process_status", headers={"Authorization": "Bearer test-key-async"}).json()
    assert s1["state"] == "running"
    assert s1["source_url"] == "https://x.m3u8"
    assert s1["elapsed_s"] >= 0
    time.sleep(0.05)
    s2 = c.get("/process_status", headers={"Authorization": "Bearer test-key-async"}).json()
    assert s2["elapsed_s"] >= s1["elapsed_s"], "elapsed_s must be monotonic"

    block.set()


def test_process_status_failed_after_thread_raise(client_and_manager):
    from server import app
    c, install = client_and_manager

    def boom(**kwargs):
        raise RuntimeError("native segfault simulation")

    install(boom)
    c.post("/process", headers={"Authorization": "Bearer test-key-async"}, json={"source_url": "https://x.m3u8"})
    app.state.pipeline_manager._thread.join(timeout=5)  # type: ignore[attr-defined]
    s = c.get("/process_status", headers={"Authorization": "Bearer test-key-async"}).json()
    assert s["state"] == "failed"
    assert s["error_type"] == "RuntimeError"
    assert "native segfault simulation" in s["error_msg"]
    assert s["traceback"] is not None and "RuntimeError" in s["traceback"]


def test_process_status_reports_playlist_ready_once_segment_exists(client_and_manager, tmp_path):
    c, install = client_and_manager
    block = threading.Event()
    hls_dir = tmp_path / "hls"  # matches env fixture

    def slow_pipeline(**kwargs):
        # Create a non-zero segment then block, so status() observes RUNNING
        # + playlist_ready=True.
        (hls_dir / "segment_005.ts").write_bytes(b"x" * 4096)
        block.wait(timeout=5)
        from pipeline_types import PipelineResult
        return PipelineResult(returncode=0, stdout="", stderr="")

    install(slow_pipeline)
    c.post("/process", headers={"Authorization": "Bearer test-key-async"}, json={"source_url": "https://x.m3u8"})

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        s = c.get("/process_status", headers={"Authorization": "Bearer test-key-async"}).json()
        if s["playlist_ready"]:
            break
        time.sleep(0.02)
    s = c.get("/process_status", headers={"Authorization": "Bearer test-key-async"}).json()
    assert s["playlist_ready"] is True
    assert s["n_segments"] == 1

    block.set()
