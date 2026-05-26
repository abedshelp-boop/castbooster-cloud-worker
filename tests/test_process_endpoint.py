# tests/test_process_endpoint.py
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUD_API_KEY", "test-key-process")
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
        from pipeline import PipelineResult
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
    def fake_run_passthrough(**kwargs):
        from pipeline import PipelineResult
        return PipelineResult(returncode=1, stdout="", stderr="ffmpeg error: bad source")

    monkeypatch.setattr("server.run_pipeline_for_request", fake_run_passthrough)

    response = client.post(
        "/process",
        headers={"Authorization": "Bearer test-key-process"},
        json={"source_url": "file:///bad.ts"},
    )
    assert response.status_code == 502
    assert "ffmpeg error" in response.json()["detail"]
