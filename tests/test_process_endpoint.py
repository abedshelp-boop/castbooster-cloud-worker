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
    fake was invoked by joining the thread.

    v0.3.1 (Pillar 3.6): the pipeline receives the LOOPBACK proxy URL,
    not the client's source_url, because BestSource can't inject HTTP
    headers and we route via /source_proxy. The client-facing response
    still surfaces the original source_url (display_source_url field).
    """
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
        # Client-facing source_url is the user's original URL.
        assert body["source_url"] == "https://egydead.example/anime.m3u8"

        # Join the background thread so we know the fake ran.
        app.state.pipeline_manager._thread.join(timeout=5)  # type: ignore[attr-defined]
    assert len(fake_calls) == 1
    # Pipeline-internal source_url is the loopback proxy URL, NOT the
    # user-provided URL — this is the whole point of Pillar 3.6.
    pipeline_source = fake_calls[0]["source_url"]
    assert pipeline_source.startswith("http://127.0.0.1:8001/source_proxy?token="), (
        f"pipeline must receive loopback proxy URL, got {pipeline_source!r}"
    )
    # Headers must be stripped — the proxy already injects them upstream.
    assert fake_calls[0]["extra_input_headers"] is None


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


def test_process_accepts_source_headers_returns_200(client):
    """v0.3.1 (Pillar 3.6): non-empty source_headers no longer 400. They
    get registered with app.state.source_registry, and BestSource reads
    through /source_proxy which injects them upstream. This unblocks
    egydead-class sources (Abed's current daily-use streaming platform).

    v0.3.0 returned 400 NotImplementedError; that's gone.
    """
    fake_calls = []

    def fake_pipeline(**kwargs):
        fake_calls.append(kwargs)
        from pipeline_types import PipelineResult
        return PipelineResult(returncode=0, stdout="", stderr="")

    from server import app
    with client:
        app.state.pipeline_manager._run_pipeline = fake_pipeline  # type: ignore[attr-defined]
        response = client.post(
            "/process",
            headers={"Authorization": "Bearer test-key-process"},
            json={
                "source_url": "https://egydead.example/anime.m3u8",
                "source_headers": {
                    "Cookie": "session=abc123",
                    "Referer": "https://egydead.example/",
                },
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["state"] == "running"
        # Client-facing source_url unchanged.
        assert body["source_url"] == "https://egydead.example/anime.m3u8"

        # Verify the registry actually got the headers (the cleanup hook
        # frees them on terminal — drain the thread to confirm cleanup).
        # We can't poke the token easily, but we can check the registry
        # had at least one entry while the pipeline was prepping. Easier:
        # confirm the pipeline received a loopback URL.
        app.state.pipeline_manager._thread.join(timeout=5)  # type: ignore[attr-defined]

    assert len(fake_calls) == 1
    pipeline_source = fake_calls[0]["source_url"]
    assert pipeline_source.startswith("http://127.0.0.1:8001/source_proxy?token=")


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
