# tests/test_process_endpoint.py
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUD_API_KEY", "test-key-process")
    monkeypatch.setenv("DISABLE_IDLE_WATCHER", "1")
    monkeypatch.setenv("HLS_SERVE_DIR", str(tmp_path / "hls"))
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://fake-pod-id-8080.proxy.runpod.net")


@pytest.fixture
def client():
    from server import app
    return TestClient(app)


def test_process_requires_auth(client):
    response = client.post("/process", json={"source_url": "file:///x.ts"})
    assert response.status_code == 401


def test_process_starts_pipeline_and_returns_hls_url(client, monkeypatch):
    """v0.3.0: /process returns 200 with hls_url + state=\"running\"
    immediately. The pipeline runs in a background thread; verify the
    fake was invoked by joining the thread."""
    fake_calls = []

    def fake_pipeline(**kwargs):
        fake_calls.append(kwargs)
        from pipeline_types import PipelineResult
        return PipelineResult(returncode=0, stdout="", stderr="")

    # Replace the manager's run_pipeline (set up at startup) with our fake.
    from server import app
    with client:
        app.state.pipeline_manager._run_pipeline = fake_pipeline  # type: ignore[attr-defined]
        response = client.post(
            "/process",
            headers={"Authorization": "Bearer test-key-process"},
            json={"source_url": "https://egydead.example/anime.m3u8"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["hls_url"].startswith("https://fake-pod-id-8080.proxy.runpod.net/hls/")
        assert body["hls_url"].endswith("/playlist.m3u8")
        assert body["state"] == "running"

        # Join the background thread so we know the fake ran.
        app.state.pipeline_manager._thread.join(timeout=5)  # type: ignore[attr-defined]
    assert len(fake_calls) == 1
    assert fake_calls[0]["source_url"] == "https://egydead.example/anime.m3u8"


@pytest.mark.skip(reason="needs /process_status endpoint — unskipped in Task 14")
def test_process_pipeline_failure_surfaces_via_process_status(client, monkeypatch):
    """v0.3.0: /process always returns 200 on successful start. Pipeline
    failures (rc != 0 or exceptions) surface via /process_status as
    state=\"failed\" — not via a synchronous 502 from /process."""
    def fake_fails(**kwargs):
        from pipeline_types import PipelineResult
        return PipelineResult(returncode=1, stdout="", stderr="ffmpeg error: bad source")

    from server import app
    with client:
        app.state.pipeline_manager._run_pipeline = fake_fails  # type: ignore[attr-defined]
        r = client.post(
            "/process",
            headers={"Authorization": "Bearer test-key-process"},
            json={"source_url": "file:///bad.ts"},
        )
        assert r.status_code == 200, "start succeeds even if pipeline will fail"
        app.state.pipeline_manager._thread.join(timeout=5)  # type: ignore[attr-defined]

        # Now poll status — should report failed.
        s = client.get(
            "/process_status",
            headers={"Authorization": "Bearer test-key-process"},
        ).json()
        assert s["state"] == "failed"
        assert s["pipeline_returncode"] == 1
        assert s["error_type"] == "PipelineNonZeroExit"
        assert "ffmpeg error" in s["stderr_tail"]


def test_process_returns_400_on_crlf_in_headers(client):
    """CRLF in source_headers must be rejected by the pipeline guard,
    surfaced as 400 (bad input), not 500 (server error).

    v0.1.15: response body has `error_type` + `error_msg` keys (no `detail`).
    """
    response = client.post(
        "/process",
        headers={"Authorization": "Bearer test-key-process"},
        json={
            "source_url": "https://example.com/anime.m3u8",
            "source_headers": {"X-Bad": "v\r\nX-Smuggled: pwned"},
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert "error_type" in body
    assert "error_msg" in body
    # The pipeline rejects CRLF; the message must say so.
    assert "CR/LF" in body["error_msg"]
