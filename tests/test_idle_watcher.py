# tests/test_idle_watcher.py
import time
from pathlib import Path

import pytest

from idle_watcher import IdleWatcher


def test_no_recent_files_triggers_shutdown(tmp_path, monkeypatch):
    """If watch_dir has no files newer than idle_seconds ago, trigger shutdown."""
    shutdown_called = []

    def fake_shutdown():
        shutdown_called.append(True)

    watcher = IdleWatcher(
        watch_dir=tmp_path,
        idle_seconds=1,
        check_interval_s=0.1,
        on_shutdown=fake_shutdown,
    )
    watcher.start()
    time.sleep(1.5)  # exceed idle window
    watcher.stop()
    assert shutdown_called, "Expected shutdown after idle window"


def test_recent_file_prevents_shutdown(tmp_path):
    """If a recent file exists in watch_dir, shutdown should NOT trigger."""
    shutdown_called = []

    def fake_shutdown():
        shutdown_called.append(True)

    # Create a file and keep it fresh
    f = tmp_path / "segment_001.ts"
    f.write_text("x")

    watcher = IdleWatcher(
        watch_dir=tmp_path,
        idle_seconds=2,
        check_interval_s=0.1,
        on_shutdown=fake_shutdown,
    )
    watcher.start()
    # Touch file every 100ms to keep it fresh
    for _ in range(10):
        f.touch()
        time.sleep(0.1)
    watcher.stop()
    assert not shutdown_called, "Should not shut down while files are fresh"


def test_hard_lifetime_cap_triggers_shutdown(tmp_path):
    """Even if files stay fresh, exceed hard_max_lifetime_s should shut down."""
    shutdown_called = []

    def fake_shutdown():
        shutdown_called.append(True)

    f = tmp_path / "segment_001.ts"
    f.write_text("x")

    watcher = IdleWatcher(
        watch_dir=tmp_path,
        idle_seconds=60,         # tolerant of idle
        hard_max_lifetime_s=0.5,  # but hard-kill after 0.5s
        check_interval_s=0.1,
        on_shutdown=fake_shutdown,
    )
    watcher.start()
    for _ in range(10):
        f.touch()
        time.sleep(0.1)
    watcher.stop()
    assert shutdown_called, "Hard lifetime cap should trigger shutdown"
