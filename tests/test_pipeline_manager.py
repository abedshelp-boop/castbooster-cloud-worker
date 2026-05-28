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
