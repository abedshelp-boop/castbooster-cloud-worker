---
agent: claude-code
created: 2026-05-28
project: castbooster-cloud-worker
pillar: 3.6 cloud VFI MVP
phase: streaming fixes before laptop orchestrator (Tasks 13-26)
tags: [design, streaming, async, nvenc, gop, pipeline-manager]
status: draft — awaiting user review before writing-plans
---

# Async /process + NVENC keyframe fix — Design

## Context

[Task 12 acceptance](../../../../vault-global/shared/projects/castbooster-cloud-worker/decisions/2026-05-28-task-12-acceptance-success.md) shipped `v0.2.8`: HTTP HLS source → BestSource → RIFE v4.6 TRT FP16 2x → NVENC h264 → HLS works end-to-end on a RunPod RTX 6000 Ada pod. Two open Phase-2 issues block real TV cast:

1. **`/process` blocks for the full encode duration (3-15 min).** CloudFlare cuts client connections at 60s; the laptop orchestrator (Task 16) needs the HLS URL within seconds so it can hand it to Chromecast.
2. **NVENC produces one giant segment (~634s) instead of 4s segments.** `-hls_time 4` is honored, but the source has no keyframes within the muxer window so the HLS muxer keeps all frames in a single `.ts`.

Both are surfaced in [`gotchas/2026-05-27-cloud-worker-week1-acceptance.md`](../../../../vault-global/shared/projects/Chrome-cast-extension/gotchas/2026-05-27-cloud-worker-week1-acceptance.md), at the "Open Phase-2 issues" section in the Task 12 acceptance writeup. A third issue (TRT engine cold-start ~30-60s on a fresh pod) is acceptable for the MVP and is out of scope here — Task 20's warmup indicator will surface it to the user.

These fixes are upstream of the laptop orchestrator: fixing them first prevents the orchestrator design from being warped around a synchronous server.

## Goals

- `/process` returns within 1s with a predicted HLS URL, regardless of how long the pipeline will take.
- A new `/process_status` endpoint lets the orchestrator poll for "is the playlist safe to give to Chromecast?" — `playlist_ready=true` once segment 0 is written.
- NVENC + HLS muxer produce regular ~4s segments end-to-end.
- The new pattern preserves the v0.2.8 happy path: same source URL, same RIFE pipeline, same nginx static serving from `/var/hls`.
- Ship as `v0.3.0` (minor bump signals new streaming pattern + new endpoint, not just a fix).

## Non-goals

- Graceful cancellation of an in-flight pipeline. `/stop` continues to self-terminate the pod; the OS reaps ffmpeg/VS. A future `manager.cancel()` is possible but not needed for the MVP.
- Multi-pipeline parallelism. One pod runs one pipeline at a time. `/process` while RUNNING returns 409.
- Pre-flight source-URL reachability checks before thread spawn. Bad URLs fail fast inside the thread; `/process_status` surfaces the error within ~1s of `/process`.
- Laptop orchestrator, extension wiring, popup warmup UI — those are Tasks 13-26.

## Architecture

One new file. The existing layout stays the same; nothing in `run_rife.py` changes structurally.

```
castbooster-cloud-worker/
├── server.py              # /process, /process_status, /diag, /probe/* — minor edits
├── run_rife.py            # +1 ffmpeg flag for keyframes (Issue 2), otherwise unchanged
├── pipeline_manager.py    # NEW — PipelineManager + PipelineState enum
├── pipeline_types.py      # existing PipelineResult lives here
├── idle_watcher.py        # unchanged — still watches /var/hls mtimes
└── tests/
    ├── test_pipeline_manager.py   # NEW — state-machine unit tests
    └── test_server_process.py     # NEW — async /process via FastAPI TestClient
```

**Boundary contract:** `PipelineManager` knows nothing about FastAPI or HTTP. It owns the thread, state, file-system presence checks, and the locking invariants. `server.py` translates HTTP requests into manager calls (`start`, `status`) and serializes responses. `run_rife.py` stays synchronous; the manager is the only place that spawns a thread.

A single `PipelineManager` instance is attached to FastAPI at startup (via `app.state.pipeline_manager`). It's a singleton because a pod runs one pipeline at a time.

## State machine

Four states, one enum:

```
   ┌──────┐  start()    ┌─────────┐  run_rife OK    ┌───────────┐
   │ IDLE │ ──────────► │ RUNNING │ ──────────────► │ COMPLETED │
   └──────┘             └─────────┘                 └───────────┘
       ▲                     │                            │
       │                     │ run_rife rc!=0 or          │
       │                     │ thread raises              │
       │                     ▼                            │
       │                ┌────────┐    start() (wipes HLS) │
       │                │ FAILED │ ◄──────────────────────┘
       │                └────────┘
       │                     │ start() (wipes HLS)
       └─────────────────────┘
```

**Transitions:**

- `IDLE → RUNNING`: `/process` accepts; no prior segments to wipe.
- `RUNNING → COMPLETED`: thread exits, `PipelineResult.returncode == 0`.
- `RUNNING → FAILED`: thread exits with non-zero rc OR raises before returning a `PipelineResult`; `error_type`/`error_msg`/`traceback` captured.
- `COMPLETED → RUNNING` and `FAILED → RUNNING`: `/process` accepts, wipes `/var/hls`, resets `started_at`, `result`, `error_*`, spawns new thread.
- `RUNNING → /process`: rejected with **409**, body `{state, hls_url, started_at, source_url}`.

`playlist_ready` is **not** a state — it's a derived property computed at `status()` read time: `state in {RUNNING, COMPLETED}` AND `glob(_output_dir / "segment_*.ts")` returns at least one file with size > 0. We can't check `segment_000.ts` specifically because `run_rife.py` runs ffmpeg with `-hls_flags delete_segments -hls_list_size 6`, which deletes the oldest segment once seven have been written. A glob+size check survives the rolling window. Filesystem is its own coherence boundary, so this read doesn't need to hold the manager lock past the state-snapshot.

### Tier-0 invariant (the multi-proc race lesson)

The May 20 Pillar 3.3 incident logged in [decision-review-log.md](../../../../vault-global/claude-code/gotchas/decision-review-log.md) showed: when a method spawns a thread that can transition shared state, the caller MUST NOT unconditionally re-set state after spawn returns — the thread may have already transitioned to a terminal state.

Applied here:

1. `PipelineManager.start()` transitions to `RUNNING` **under the lock, BEFORE `thread.start()`**.
2. `_run_target` acquires the same lock to transition to `COMPLETED`/`FAILED` and **guards each transition with `if self._state is PipelineState.RUNNING`** — so any future `cancel()` that sets state to a non-RUNNING value won't be silently overwritten.
3. The HTTP handler (`server.py /process`) does NOT touch state after calling `manager.start()` — it only formats the response from what `manager` reports.

This is the inverse of the May 20 bug. There, the slot's handler set `state=WARMING` after `slot.start()` returned, overwriting `FAILED` that the side task had already set. Here, no caller of `start()` ever touches state after spawn.

## HTTP API surface

### POST /process

Request body unchanged from v0.2.8: `ProcessRequest{source_url, source_headers, output_resolution}`.

| Outcome | Status | Body |
|---|---|---|
| Started new pipeline (from IDLE, COMPLETED, or FAILED) | **200** | `{hls_url, state: "running", started_at, source_url}` |
| Pipeline already running | **409** | `{state: "running", hls_url, started_at, source_url}` — orchestrator can call `/stop` or wait |
| `PUBLIC_BASE_URL` not configured | **500** | `{error: "PUBLIC_BASE_URL not configured"}` |
| Input validation error (`ValueError`/`NotImplementedError` raised synchronously before thread spawn) | **400** | `{error_type, error_msg}` |

`/process` returns in well under a second — it spawns the thread, never waits for it. Pipeline errors (bad URL, RIFE crash, etc.) surface via `/process_status` once the thread has had a chance to fail.

### GET /process_status

Always **200**, even in IDLE.

```json
{
  "state": "idle | running | completed | failed",
  "hls_url": null | "https://<pod>-8080.proxy.runpod.net/hls/playlist.m3u8",
  "source_url": null | "https://...",
  "started_at": null | 1748470000.123,
  "elapsed_s": null | 47.2,
  "playlist_ready": false | true,
  "n_segments": 0,
  "pipeline_returncode": null | 0 | 124 | ...,
  "error_type": null | "ValueError" | "BrokenPipeError" | ...,
  "error_msg": null | "<short>",
  "stderr_tail": null | "<~1500 chars>"
}
```

- `hls_url` is non-null whenever `PUBLIC_BASE_URL` is configured (so orchestrator can stash it across polls).
- `playlist_ready` is the orchestrator's gating signal: hand the URL to Chromecast as soon as this flips `true`.
- `n_segments` is a small QoS metric — count of `segment_*.ts` files currently in `/var/hls`. Cheap (`os.listdir`). This plateaus at ~6 during steady-state RUNNING because `-hls_list_size 6 -hls_flags delete_segments` keeps a rolling window.
- `error_*` and `stderr_tail` populated only when `state == "failed"`.

### POST /stop

Unchanged. Pod self-terminates via RunPod REST API. The worker thread is reaped when the OS tears down the container.

### What /process no longer does

The current 502-with-traceback path goes away from the **synchronous** response. The thread can still produce `PipelineResult{returncode != 0}` or raise, but errors surface via `/process_status` as `state="failed"` with structured fields. Tracebacks that previously rode in an `HTTPException(detail=...)` (the Cloudflare-502 trap from the [2026-05-28 gotcha](../../../../vault-global/shared/projects/Chrome-cast-extension/gotchas/2026-05-27-cloud-worker-week1-acceptance.md), lines 1117-1136) are now in clean JSON fields that survive the Cloudflare edge.

## Locking & concurrency

Single `threading.RLock` (re-entrant) on `PipelineManager`. All state-field reads/writes go through it. Filesystem checks (`segment_000.ts` exists) happen outside the lock — FS is its own coherence boundary.

**`StartOutcome` return type:** a small dataclass `StartOutcome(success: bool, snapshot: dict)`. The `snapshot` field has the same shape as `/process_status`'s response body, computed at the moment of the start call. The HTTP handler in `server.py` uses `outcome.success` to choose 200 vs 409 and serializes `outcome.snapshot` as the body (filtered to the fields documented above for /process).

Sketch:

```python
class PipelineManager:
    def __init__(self, output_dir: Path, public_base_url: str):
        self._lock = threading.RLock()
        self._state = PipelineState.IDLE
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._source_url: str | None = None
        self._result: PipelineResult | None = None
        self._error_type: str | None = None
        self._error_msg: str | None = None
        self._traceback: str | None = None
        self._output_dir = output_dir
        self._public_base_url = public_base_url

    def start(self, source_url, headers) -> "StartOutcome":
        with self._lock:
            if self._state is PipelineState.RUNNING:
                return StartOutcome.already_running(self._snapshot_locked())
            # Auto-wipe from COMPLETED / FAILED — clean restart.
            if self._state in (PipelineState.COMPLETED, PipelineState.FAILED):
                self._wipe_output_dir_locked()
            self._reset_fields_locked()
            self._state = PipelineState.RUNNING       # transition BEFORE spawn
            self._source_url = source_url
            self._started_at = time.time()
            self._thread = threading.Thread(
                target=self._run_target,
                args=(source_url, headers),
                name="pipeline-worker",
                daemon=True,
            )
            self._thread.start()
            return StartOutcome.started(self._snapshot_locked())

    def _run_target(self, source_url, headers):
        try:
            result = run_rife(source_url=source_url,
                              output_dir=self._output_dir,
                              extra_input_headers=headers)
        except BaseException as exc:
            with self._lock:
                if self._state is PipelineState.RUNNING:   # Tier-0 guard
                    self._state = PipelineState.FAILED
                    self._error_type = type(exc).__name__
                    self._error_msg = str(exc)[:1500]
                    self._traceback = traceback.format_exc()[-2000:]
            return
        with self._lock:
            if self._state is PipelineState.RUNNING:       # Tier-0 guard
                self._result = result
                if result.returncode == 0:
                    self._state = PipelineState.COMPLETED
                else:
                    self._state = PipelineState.FAILED
                    self._error_type = "PipelineNonZeroExit"
                    self._error_msg = f"returncode={result.returncode}"
                    # stderr_tail captured from result.stderr in snapshot()
```

**Lock-scope rule:** never hold the lock across `run_rife()` — that would serialize the worker thread on the lock for the entire encode duration, defeating the whole point. The lock protects only field reads/writes; `run_rife` runs lock-free.

## NVENC keyframe fix (Issue 2)

Single change in `run_rife.py:120-137`. Add one ffmpeg flag pair to the existing command:

```python
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "yuv4mpegpipe",
    "-i", "-",
    "-c:v", "h264_nvenc",
    "-preset", "p4",
    "-tune", "ll",
    "-b:v", "8M",
    "-maxrate", "12M",
    "-bufsize", "16M",
    "-pix_fmt", "yuv420p",
    "-force_key_frames", "expr:gte(t,n_forced*4)",   # ← NEW: IDR every 4s, time-based
    "-f", "hls",
    "-hls_time", "4",
    "-hls_list_size", "6",
    "-hls_flags", "delete_segments+independent_segments",
    "-hls_segment_filename", str(segment_pattern),
    str(playlist),
]
```

Just `-force_key_frames`. **Not** `-g 240` from the original handoff sketch: that arg is frame-count-based and the comment in the handoff doc ("4s × 60fps") was off — the RIFE output is 120fps (2x interp on the 60fps test source), so `-g 240` would force IDR every 2s instead of 4s, and on a 30fps source input (60fps output) the same value would cut at 4s but only by coincidence. The expression `expr:gte(t,n_forced*4)` is **time-based**, so it produces clean 4s intervals regardless of source framerate. NVENC respects `-force_key_frames` for IDR insertion; the HLS muxer then cuts each segment at the forced keyframes.

## Test plan

Two new test files. Both use the existing `py -3.12 -m pytest` setup. Existing 25 tests must stay green; target ~42 total.

### `tests/test_pipeline_manager.py` (NEW)

Unit tests against `PipelineManager` in isolation. The fake `run_pipeline` is a callable injected via the manager's constructor (default = real `run_rife`); tests inject a fake that blocks on a `threading.Event` or returns/raises immediately.

1. **idle on init** — fresh manager has `state == IDLE`, `playlist_ready == False`, all error fields None.
2. **start transitions to RUNNING immediately** — fake `run_pipeline` blocks on Event; `start()` returns within 100ms and `status()` reports RUNNING.
3. **second start while RUNNING returns already_running** — start one, try another with different source_url; assert outcome flag, state unchanged.
4. **completion: rc=0 transitions to COMPLETED** — fake returns `PipelineResult(0, "", "")`; assert state, result, no errors.
5. **completion: rc!=0 transitions to FAILED** — fake returns `PipelineResult(2, "", "clip build failed")`; assert FAILED with `error_type="PipelineNonZeroExit"`, stderr_tail set.
6. **thread raises transitions to FAILED** — fake raises `RuntimeError("boom")`; assert FAILED with `error_type="RuntimeError"`, traceback captured.
7. **start after COMPLETED wipes output dir + starts fresh** — write a fake `segment_007.ts` (any rolling-window slot), complete a pipeline, start another; assert old segment + any playlist.m3u8 are gone, new pipeline RUNNING.
8. **start after FAILED wipes + restarts** — symmetric to (7).
9. **playlist_ready false during RUNNING with no segments** — fake blocks; assert False.
10. **playlist_ready true once any segment_*.ts with size > 0 exists** — touch e.g. `segment_002.ts` mid-run (verifying the glob-based check, not a hardcoded `segment_000` lookup); assert True.
11. **Tier-0 race: pipeline fails before start() returns** — fake raises immediately on first call (no Event wait); join thread; assert final state is FAILED, not RUNNING. This pins the lock-ordering invariant. If this regresses to RUNNING, someone has reintroduced the May 20 multi-proc bug shape.

### `tests/test_server_process.py` (NEW)

HTTP layer via FastAPI's `TestClient`. The pipeline_manager attached to `app.state` uses a fake `run_pipeline` callable so tests don't need a GPU.

12. **POST /process returns 200 + hls_url within 1s when pipeline is slow** — fake blocks; assert wall-time of `/process` call is < 1s, body has `hls_url`.
13. **POST /process returns 409 when already running** — start one, post another; assert 409 body shape (`state`, `hls_url`, `started_at`, `source_url`).
14. **GET /process_status returns idle on fresh server** — assert all fields null/zero/false except `state == "idle"`.
15. **GET /process_status returns running while thread is alive** — assert state, hls_url, started_at, elapsed_s monotonic across two reads.
16. **GET /process_status returns failed after thread raise** — fake raises; poll status; assert `state == "failed"` with error_type, error_msg, traceback.
17. **GET /process_status reports playlist_ready=true once any segment exists** — touch a fake `segment_002.ts` via the manager's output_dir; assert `playlist_ready` flips True on next status read.

## Version, release, deploy

- Bump `pyproject.toml` version → `0.3.0`.
- Bump `server.py` `FastAPI(title="castbooster-cloud-worker", version="0.3.0")`.
- Git tag `v0.3.0`, push, GHA builds + publishes `ghcr.io/abedshelp-boop/castbooster-cloud-worker:v0.3.0`.

### Pod verification

One RTX 6000 Ada Secure pod (~$0.74/hr), ~10-15min wall time, ~$0.15 expected cost.

Acceptance criteria:

1. `/process` returns 200 with `hls_url` within 1s.
2. `/process_status` flips `playlist_ready=true` within 90s (TRT cold-start + first segment).
3. `playlist.m3u8` shows multiple `#EXTINF:4.0xx,` lines, not one giant one.
4. `ls /var/hls/segment_*.ts` shows N segments where N ≈ source_duration / 4.
5. `/process_status` reports `state="completed"` when source EOS reached.
6. `n_segments` reaches the rolling-window cap (~6) and stays there during steady-state RUNNING. Verified by polling status across at least 30s of encode.

If any criterion fails, capture `/diag` + pod logs (via RunPod web UI), document in master gotchas, dispatch a fix.

## References

- Handoff: [`~/.claude/plans/next-session-prompt-pillar-3.6-streaming-fixes-and-laptop-orchestrator.md`](../../../../.claude/plans/next-session-prompt-pillar-3.6-streaming-fixes-and-laptop-orchestrator.md)
- Master gotchas: [`vault-global/shared/projects/Chrome-cast-extension/gotchas/2026-05-27-cloud-worker-week1-acceptance.md`](../../../../vault-global/shared/projects/Chrome-cast-extension/gotchas/2026-05-27-cloud-worker-week1-acceptance.md)
- Task 12 success: [`vault-global/shared/projects/castbooster-cloud-worker/decisions/2026-05-28-task-12-acceptance-success.md`](../../../../vault-global/shared/projects/castbooster-cloud-worker/decisions/2026-05-28-task-12-acceptance-success.md)
- May 20 multi-proc race (Tier-0 invariant): [`vault-global/claude-code/gotchas/decision-review-log.md`](../../../../vault-global/claude-code/gotchas/decision-review-log.md)
- Cloudflare 502 + HTTPException trap: master gotchas lines 1117-1136
- Pillar 3.6 plan: [`Chrome-cast-extension/docs/superpowers/plans/2026-05-24-pillar-3.6-cloud-vfi-impl.md`](../../../../Chrome-cast-extension/docs/superpowers/plans/2026-05-24-pillar-3.6-cloud-vfi-impl.md) — Tasks 13-26 follow this fix.

## Open follow-ups (deferred)

- Engine cold-start ~30-60s — surface via Task 20's popup warmup indicator.
- Graceful pipeline cancellation (`manager.cancel()` + `_cancel_event` checked between frames in `run_rife`).
- Multi-pipeline parallelism on a single pod (not needed; one pod = one cast).
- Pre-flight source URL HEAD check before thread spawn (orchestrator already handles FAILED gracefully).
