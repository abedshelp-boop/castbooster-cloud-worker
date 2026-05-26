# idle_watcher.py
import threading
import time
from pathlib import Path
from typing import Callable


class IdleWatcher:
    """Background thread that triggers shutdown when no recent activity in watch_dir.

    Activity = file mtimes inside watch_dir (HLS segments produced by ffmpeg).
    If no file has been modified within `idle_seconds`, calls `on_shutdown`.
    Also enforces a hard maximum lifetime regardless of activity, as a safety
    net against runaway cloud cost.
    """

    def __init__(
        self,
        watch_dir: Path,
        idle_seconds: float,
        on_shutdown: Callable[[], None],
        check_interval_s: float = 30.0,
        hard_max_lifetime_s: float = 6 * 60 * 60,
    ):
        self.watch_dir = Path(watch_dir)
        self.idle_seconds = idle_seconds
        self.on_shutdown = on_shutdown
        self.check_interval_s = check_interval_s
        self.hard_max_lifetime_s = hard_max_lifetime_s
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started_at: float | None = None

    def start(self):
        if self._thread is not None:
            return
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self):
        triggered = False
        while not self._stop_event.is_set() and not triggered:
            should_trigger = False
            try:
                now = time.time()

                # Hard lifetime check
                if self._started_at and (now - self._started_at) >= self.hard_max_lifetime_s:
                    should_trigger = True
                else:
                    # Idle check — find newest file mtime in watch_dir
                    newest_mtime = 0.0
                    if self.watch_dir.exists():
                        for path in self.watch_dir.iterdir():
                            if path.is_file():
                                mtime = path.stat().st_mtime
                                if mtime > newest_mtime:
                                    newest_mtime = mtime

                    if newest_mtime == 0.0:
                        # No files yet — count from watcher start time
                        age = now - (self._started_at or now)
                    else:
                        age = now - newest_mtime

                    if age >= self.idle_seconds:
                        should_trigger = True
            except Exception as e:
                # Detection-side errors (FS races, perms) are transient — log + continue
                print(f"[idle_watcher] detection error: {e}")

            if should_trigger:
                try:
                    self.on_shutdown()
                except Exception as e:
                    # Cost kill switch failed — visible, but watcher's done its job;
                    # caller can re-create watcher if they want to retry
                    print(f"[idle_watcher] on_shutdown failed: {e}")
                triggered = True
                break

            self._stop_event.wait(self.check_interval_s)
