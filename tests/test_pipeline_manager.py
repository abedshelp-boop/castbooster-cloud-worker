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


def test_completion_rc_zero_transitions_to_completed(tmp_path):
    from pipeline_types import PipelineResult

    def fake_pipeline(**kwargs):
        return PipelineResult(returncode=0, stdout="", stderr="")

    m = PipelineManager(
        output_dir=tmp_path,
        public_base_url="https://fake.example",
        run_pipeline=fake_pipeline,
    )
    m.start(source_url="https://test.m3u8", headers=None)
    m._thread.join(timeout=5)  # type: ignore[union-attr]

    s = m.status()
    assert s["state"] == "completed"
    assert s["pipeline_returncode"] == 0
    assert s["error_type"] is None
    assert s["error_msg"] is None


def test_completion_rc_nonzero_transitions_to_failed(tmp_path):
    from pipeline_types import PipelineResult

    def fake_pipeline(**kwargs):
        return PipelineResult(returncode=2, stdout="", stderr="clip build failed: bs.VideoSource Couldn't open URL")

    m = PipelineManager(
        output_dir=tmp_path,
        public_base_url="https://fake.example",
        run_pipeline=fake_pipeline,
    )
    m.start(source_url="https://bad.m3u8", headers=None)
    m._thread.join(timeout=5)  # type: ignore[union-attr]

    s = m.status()
    assert s["state"] == "failed"
    assert s["pipeline_returncode"] == 2
    assert s["error_type"] == "PipelineNonZeroExit"
    assert "returncode=2" in s["error_msg"]
    assert "Couldn't open URL" in s["stderr_tail"]


def test_thread_raises_transitions_to_failed(tmp_path):
    def fake_pipeline(**kwargs):
        raise RuntimeError("boom — vapoursynth.so segfault simulated")

    m = PipelineManager(
        output_dir=tmp_path,
        public_base_url="https://fake.example",
        run_pipeline=fake_pipeline,
    )
    m.start(source_url="https://test.m3u8", headers=None)
    m._thread.join(timeout=5)  # type: ignore[union-attr]

    s = m.status()
    assert s["state"] == "failed"
    assert s["error_type"] == "RuntimeError"
    assert "boom" in s["error_msg"]
    assert s["traceback"] is not None
    assert "RuntimeError" in s["traceback"]


def test_second_start_while_running_returns_already_running(tmp_path):
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
    first = m.start(source_url="https://first.m3u8", headers=None)
    assert first.success is True

    second = m.start(source_url="https://second.m3u8", headers=None)
    assert second.success is False
    assert second.snapshot["state"] == "running"
    assert second.snapshot["source_url"] == "https://first.m3u8", (
        "second.snapshot must report the CURRENTLY running pipeline, not the rejected one"
    )

    block.set()
    m._thread.join(timeout=5)  # type: ignore[union-attr]


def test_start_after_completed_wipes_output_dir(tmp_path):
    from pipeline_types import PipelineResult

    # First run: completes successfully
    def fake_completes(**kwargs):
        return PipelineResult(returncode=0, stdout="", stderr="")

    m = PipelineManager(
        output_dir=tmp_path,
        public_base_url="https://fake.example",
        run_pipeline=fake_completes,
    )
    m.start(source_url="https://first.m3u8", headers=None)
    m._thread.join(timeout=5)  # type: ignore[union-attr]
    assert m.status()["state"] == "completed"

    # Plant a stale segment + playlist that the next start() must wipe.
    (tmp_path / "segment_007.ts").write_bytes(b"stale" * 1000)
    (tmp_path / "playlist.m3u8").write_text("#EXTM3U\nstale\n")

    # Second start: from a fresh (un-blocked) callable so we can observe RUNNING.
    block = threading.Event()

    def fake_blocks(**kwargs):
        block.wait(timeout=5)
        return PipelineResult(returncode=0, stdout="", stderr="")

    m._run_pipeline = fake_blocks  # type: ignore[attr-defined]
    m.start(source_url="https://second.m3u8", headers=None)
    s = m.status()
    assert s["state"] == "running"
    assert not (tmp_path / "segment_007.ts").exists(), "stale segment must be wiped"
    assert not (tmp_path / "playlist.m3u8").exists(), "stale playlist must be wiped"

    block.set()
    m._thread.join(timeout=5)  # type: ignore[union-attr]


def test_start_after_failed_wipes_output_dir(tmp_path):
    from pipeline_types import PipelineResult

    def fake_fails(**kwargs):
        return PipelineResult(returncode=2, stdout="", stderr="failed")

    m = PipelineManager(
        output_dir=tmp_path,
        public_base_url="https://fake.example",
        run_pipeline=fake_fails,
    )
    m.start(source_url="https://first.m3u8", headers=None)
    m._thread.join(timeout=5)  # type: ignore[union-attr]
    assert m.status()["state"] == "failed"

    (tmp_path / "segment_003.ts").write_bytes(b"stale" * 1000)

    block = threading.Event()

    def fake_blocks(**kwargs):
        block.wait(timeout=5)
        return PipelineResult(returncode=0, stdout="", stderr="")

    m._run_pipeline = fake_blocks  # type: ignore[attr-defined]
    m.start(source_url="https://second.m3u8", headers=None)
    assert m.status()["state"] == "running"
    assert not (tmp_path / "segment_003.ts").exists()
    # Error fields must reset on restart
    assert m.status()["error_type"] is None
    assert m.status()["error_msg"] is None

    block.set()
    m._thread.join(timeout=5)  # type: ignore[union-attr]


def test_playlist_ready_false_during_running_with_no_segments(tmp_path):
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
    m.start(source_url="https://test.m3u8", headers=None)
    assert m.status()["playlist_ready"] is False
    assert m.status()["n_segments"] == 0

    block.set()
    m._thread.join(timeout=5)  # type: ignore[union-attr]


def test_playlist_ready_true_once_any_segment_exists(tmp_path):
    block = threading.Event()

    def fake_pipeline(**kwargs):
        # Touch a non-zero segment file then block.
        (tmp_path / "segment_002.ts").write_bytes(b"x" * 1024)
        block.wait(timeout=5)
        from pipeline_types import PipelineResult
        return PipelineResult(returncode=0, stdout="", stderr="")

    m = PipelineManager(
        output_dir=tmp_path,
        public_base_url="https://fake.example",
        run_pipeline=fake_pipeline,
    )
    m.start(source_url="https://test.m3u8", headers=None)
    # Poll briefly: the fake creates the file then blocks; we should see ready
    # within ~100ms.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        s = m.status()
        if s["playlist_ready"]:
            break
        time.sleep(0.02)
    s = m.status()
    assert s["playlist_ready"] is True, "should flip True for segment_002.ts (no segment_000 reliance)"
    assert s["n_segments"] == 1

    block.set()
    m._thread.join(timeout=5)  # type: ignore[union-attr]


def test_tier0_race_pipeline_fails_before_start_returns(tmp_path):
    """Regression-canary for the May 20 Pillar 3.3 multi-proc state-mirror
    race (see claude-code/gotchas/decision-review-log.md).

    The fake pipeline raises IMMEDIATELY on its first call — so the worker
    thread can transition to FAILED before the test reads state. If anyone
    refactors start() to set state=RUNNING AFTER thread.start() (the bug
    shape from 2026-05-20), this test will see state=RUNNING after join
    and fail.

    Today's design holds the lock through thread.start() and only releases
    on `with self._lock:` exit, so the worker can't acquire the lock until
    after start() returns. Once we join, state must be FAILED.
    """
    def fake_pipeline(**kwargs):
        raise RuntimeError("immediate failure simulating native segfault")

    m = PipelineManager(
        output_dir=tmp_path,
        public_base_url="https://fake.example",
        run_pipeline=fake_pipeline,
    )
    outcome = m.start(source_url="https://test.m3u8", headers=None)
    assert outcome.success is True, "start() itself must succeed; failure surfaces via status"

    m._thread.join(timeout=5)  # type: ignore[union-attr]

    s = m.status()
    assert s["state"] == "failed", (
        f"Tier-0 invariant violated: state is {s['state']!r}. Did someone "
        f"add `self._state = RUNNING` AFTER thread.start() returned? The "
        f"worker already transitioned to FAILED; do not overwrite."
    )
    assert s["error_type"] == "RuntimeError"
    assert "immediate failure" in s["error_msg"]
