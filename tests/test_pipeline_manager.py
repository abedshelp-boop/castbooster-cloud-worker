import threading
import time
from pathlib import Path
from pipeline_manager import PipelineManager, PipelineState


def test_idle_on_init(tmp_path):
    m = PipelineManager(
        output_dir=tmp_path,
        public_base_url="https://fake-pod-8080.proxy.runpod.net",
    )
    s = m.status()
    assert s["state"] == "idle"
    assert s["hls_url"] == "https://fake-pod-8080.proxy.runpod.net/hls/playlist.m3u8"
    assert s["source_url"] is None
    assert s["started_at"] is None
    assert s["elapsed_s"] is None
    assert s["playlist_ready"] is False
    assert s["n_segments"] == 0
    assert s["pipeline_returncode"] is None
    assert s["error_type"] is None
    assert s["error_msg"] is None
    assert s["stderr_tail"] is None


def test_start_transitions_to_running_immediately(tmp_path):
    block = threading.Event()

    def fake_pipeline(**kwargs):
        block.wait(timeout=5)
        from pipeline_types import PipelineResult
        return PipelineResult(returncode=0, stdout="", stderr="")

    m = PipelineManager(
        output_dir=tmp_path,
        public_base_url="https://fake.example",
        run_pipeline=fake_pipeline,
    )
    t0 = time.monotonic()
    outcome = m.start(source_url="https://test.m3u8", headers=None)
    elapsed = time.monotonic() - t0

    assert outcome.success is True
    assert elapsed < 0.5, f"start() blocked for {elapsed:.3f}s — must return immediately"
    s = m.status()
    assert s["state"] == "running"
    assert s["source_url"] == "https://test.m3u8"
    assert s["started_at"] is not None
    assert s["elapsed_s"] is not None and s["elapsed_s"] >= 0
    block.set()  # unblock the fake worker
    m._thread.join(timeout=5)  # type: ignore[union-attr]
