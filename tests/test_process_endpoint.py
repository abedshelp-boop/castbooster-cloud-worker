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
    """Posting a source URL should start ffmpeg and return the public HLS URL."""
    fake_calls = []

    def fake_run_passthrough(**kwargs):
        fake_calls.append(kwargs)
        from pipeline_types import PipelineResult
        return PipelineResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("server.run_pipeline_for_request", fake_run_passthrough)

    response = client.post(
        "/process",
        headers={"Authorization": "Bearer test-key-process"},
        json={
            "source_url": "https://egydead.example/anime.m3u8",
            "source_headers": {"Cookie": "session=abc"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["hls_url"].startswith("https://fake-pod-id-8080.proxy.runpod.net/hls/")
    assert body["hls_url"].endswith("/playlist.m3u8")
    assert len(fake_calls) == 1
    assert fake_calls[0]["source_url"] == "https://egydead.example/anime.m3u8"


def test_process_returns_502_on_pipeline_failure(client, monkeypatch):
    """v0.1.14: response body shape is `{error, returncode, stderr_tail}` (not
    `{detail}`). The shape now matches probe endpoints — no HTTPException in
    /process anymore (see server.py /process commentary for the why)."""
    def fake_run_passthrough(**kwargs):
        from pipeline_types import PipelineResult
        return PipelineResult(returncode=1, stdout="", stderr="ffmpeg error: bad source")

    monkeypatch.setattr("server.run_pipeline_for_request", fake_run_passthrough)

    response = client.post(
        "/process",
        headers={"Authorization": "Bearer test-key-process"},
        json={"source_url": "file:///bad.ts"},
    )
    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "pipeline returncode nonzero"
    assert body["returncode"] == 1
    assert "ffmpeg error" in body["stderr_tail"]


def test_process_returns_400_on_crlf_in_headers(client):
    """CRLF in source_headers must be rejected by the pipeline guard,
    surfaced as 400 (bad input), not 500 (server error).

    v0.1.14: response body has `error_type` + `error_msg` keys (no `detail`).
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
