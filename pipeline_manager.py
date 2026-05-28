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
        self._source_url: str | None = None
        self._result: PipelineResult | None = None
        self._error_type: str | None = None
        self._error_msg: str | None = None
        self._traceback: str | None = None
        self._output_dir = Path(output_dir)
        self._public_base_url = public_base_url.rstrip("/")
        # Lazy-defaulted so tests can inject without importing run_rife
        # (which transitively imports vapoursynth, only present on the pod).
        self._run_pipeline = run_pipeline

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
                "source_url": self._source_url,
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
