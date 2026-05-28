"""Owns the background pipeline thread + state machine + playlist_ready check.

Singleton instance attached to FastAPI's app.state.pipeline_manager at
startup. Encapsulates the Tier-0 invariant from the May 20 Pillar 3.3
multi-proc race: state transitions to RUNNING happen under the lock
BEFORE thread.start(); the worker thread guards every transition with
`if self._state is PipelineState.RUNNING` so a future cancel() can't
be silently overwritten.
"""
from __future__ import annotations

import enum
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pipeline_types import PipelineResult


class PipelineState(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class StartOutcome:
    success: bool
    snapshot: dict[str, Any]

    @classmethod
    def started(cls, snapshot: dict[str, Any]) -> "StartOutcome":
        return cls(success=True, snapshot=snapshot)

    @classmethod
    def already_running(cls, snapshot: dict[str, Any]) -> "StartOutcome":
        return cls(success=False, snapshot=snapshot)


class PipelineManager:
    def __init__(
        self,
        output_dir: Path,
        public_base_url: str,
        run_pipeline: Callable[..., PipelineResult] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._state = PipelineState.IDLE
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        # _source_url is what the pipeline actually feeds BestSource —
        # the loopback proxy URL in v0.3.1. _display_source_url is what
        # the client provided and what /process_status surfaces.
        self._source_url: str | None = None
        self._display_source_url: str | None = None
        self._result: PipelineResult | None = None
        self._error_type: str | None = None
        self._error_msg: str | None = None
        self._traceback: str | None = None
        self._output_dir = Path(output_dir)
        self._public_base_url = public_base_url.rstrip("/")
        # Lazy-defaulted so tests can inject without importing run_rife
        # (which transitively imports vapoursynth, only present on the pod).
        self._run_pipeline = run_pipeline
        # Per-run terminal callback — fires exactly once on COMPLETED /
        # FAILED. Used by server.py to unregister the source-proxy token
        # without coupling PipelineManager to the SourceRegistry module.
        self._on_terminal: Callable[[], None] | None = None

    def _hls_url(self) -> str:
        return f"{self._public_base_url}/hls/playlist.m3u8" if self._public_base_url else ""

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            elapsed = (now - self._started_at) if self._started_at else None
            stderr_tail = (
                self._result.stderr[-1500:]
                if self._result is not None and self._result.stderr
                else None
            )
            returncode = self._result.returncode if self._result is not None else None
            snapshot = {
                "state": self._state.value,
                "hls_url": self._hls_url() or None,
                # Surface the client-provided URL, not the internal loopback
                # proxy URL — the proxy is implementation detail.
                "source_url": self._display_source_url,
                "started_at": self._started_at,
                "elapsed_s": elapsed,
                "playlist_ready": False,  # filled below
                "n_segments": 0,           # filled below
                "pipeline_returncode": returncode,
                "error_type": self._error_type,
                "error_msg": self._error_msg,
                "stderr_tail": stderr_tail,
                "traceback": self._traceback,
            }
            current_state = self._state
        # FS check happens outside the lock (FS is its own coherence boundary).
        snapshot["n_segments"], snapshot["playlist_ready"] = self._scan_segments(current_state)
        return snapshot

    def _scan_segments(self, current_state: PipelineState) -> tuple[int, bool]:
        try:
            segments = [
                p for p in self._output_dir.glob("segment_*.ts") if p.stat().st_size > 0
            ]
        except FileNotFoundError:
            return 0, False
        n = len(segments)
        ready = current_state in (PipelineState.RUNNING, PipelineState.COMPLETED) and n > 0
        return n, ready

    def start(
        self,
        source_url: str,
        headers: dict[str, str] | None,
        on_terminal: Callable[[], None] | None = None,
        display_source_url: str | None = None,
    ) -> StartOutcome:
        """Spawn a pipeline thread. `on_terminal`, if provided, is invoked
        exactly once when the pipeline transitions to COMPLETED or FAILED
        (including the ConfigurationError + worker-thread-raised paths).
        It's best-effort: a callback exception is logged and swallowed so
        a buggy cleanup hook can't leak the manager into a stuck state."""
        with self._lock:
            if self._state is PipelineState.RUNNING:
                return StartOutcome.already_running(self._snapshot_locked())
            # From a terminal state, wipe so a stale playlist/segments from
            # the previous pipeline don't leak into the new run.
            if self._state in (PipelineState.COMPLETED, PipelineState.FAILED):
                self._wipe_output_dir_locked()
            self._reset_fields_locked()
            # Tier-0 invariant: transition BEFORE thread.start so the worker
            # never sees IDLE/COMPLETED/FAILED on its first lock acquire.
            self._state = PipelineState.RUNNING
            self._source_url = source_url
            self._display_source_url = display_source_url or source_url
            self._started_at = time.time()
            self._on_terminal = on_terminal
            self._thread = threading.Thread(
                target=self._run_target,
                args=(source_url, headers),
                name="pipeline-worker",
                daemon=True,
            )
            self._thread.start()
            return StartOutcome.started(self._snapshot_locked())

    def _snapshot_locked(self) -> dict[str, Any]:
        # Caller holds self._lock. Mirror of status() but without the FS
        # check (snapshot is used at start() time when no segments exist yet).
        return {
            "state": self._state.value,
            "hls_url": self._hls_url() or None,
            # Surface the client-provided URL, not the internal loopback
            # proxy URL — the proxy is implementation detail.
            "source_url": self._display_source_url,
            "started_at": self._started_at,
            "elapsed_s": (time.time() - self._started_at) if self._started_at else None,
            "playlist_ready": False,
            "n_segments": 0,
            "pipeline_returncode": (
                self._result.returncode if self._result is not None else None
            ),
            "error_type": self._error_type,
            "error_msg": self._error_msg,
            "stderr_tail": (
                self._result.stderr[-1500:]
                if self._result is not None and self._result.stderr
                else None
            ),
            "traceback": self._traceback,
        }

    def _reset_fields_locked(self) -> None:
        self._result = None
        self._error_type = None
        self._error_msg = None
        self._traceback = None

    def _wipe_output_dir_locked(self) -> None:
        """Remove segment_*.ts files + playlist.m3u8 from output_dir, but
        preserve the directory itself (nginx is serving it). Best-effort —
        a stray file we can't delete shouldn't block a new pipeline.
        """
        if not self._output_dir.exists():
            return
        for p in self._output_dir.iterdir():
            if p.is_file() and (p.name.startswith("segment_") or p.name == "playlist.m3u8"):
                try:
                    p.unlink()
                except OSError:
                    pass

    def _run_target(self, source_url: str, headers: dict[str, str] | None) -> None:
        run = self._run_pipeline
        if run is None:
            # No pipeline injected (test misconfiguration or production bug).
            # Treat as failure rather than leave state in RUNNING forever.
            with self._lock:
                if self._state is PipelineState.RUNNING:
                    self._state = PipelineState.FAILED
                    self._error_type = "ConfigurationError"
                    self._error_msg = "run_pipeline callable was not configured"
            self._fire_terminal_callback()
            return
        try:
            result = run(
                source_url=source_url,
                output_dir=self._output_dir,
                extra_input_headers=headers,
            )
        except BaseException as exc:
            with self._lock:
                if self._state is PipelineState.RUNNING:
                    self._state = PipelineState.FAILED
                    self._error_type = type(exc).__name__
                    self._error_msg = str(exc)[:1500]
                    self._traceback = traceback.format_exc()[-2000:]
            self._fire_terminal_callback()
            return
        with self._lock:
            if self._state is PipelineState.RUNNING:
                self._result = result
                if result.returncode == 0:
                    self._state = PipelineState.COMPLETED
                else:
                    self._state = PipelineState.FAILED
                    self._error_type = "PipelineNonZeroExit"
                    self._error_msg = f"returncode={result.returncode}"
        self._fire_terminal_callback()

    def _fire_terminal_callback(self) -> None:
        """Invoke + clear the on_terminal hook. Best-effort: a callback
        exception must not propagate out of the worker thread, since the
        thread is already past its terminal state transition and Python's
        default unhandled-exception handler would just print to stderr."""
        with self._lock:
            cb = self._on_terminal
            self._on_terminal = None
        if cb is None:
            return
        try:
            cb()
        except BaseException as e:
            # Print but never re-raise; pipeline is already terminal.
            print(
                f"[pipeline_manager] on_terminal callback failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
