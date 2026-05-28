"""Token-keyed registry mapping opaque IDs to (source_url, headers) tuples.

Used by the source-proxy endpoint (Pillar 3.6) to inject HTTP headers
(Cookie, User-Agent, Referer) into upstream fetches without exposing the
raw upstream URL to BestSource. BestSource's Python API does not accept
libavformat options, so we route every fetch through a local FastAPI
endpoint that looks up the headers by token and injects them.

Singleton attached to FastAPI's app.state.source_registry at startup.
Thread-safe (RLock) — start/cleanup happens on the pipeline-worker
thread while /source_proxy handlers run on the FastAPI event loop.
"""
from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceEntry:
    source_url: str
    headers: dict[str, str]


class SourceRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, SourceEntry] = {}

    def register(self, source_url: str, headers: dict[str, str]) -> str:
        """Generate a fresh opaque token and store the (url, headers) pair.

        Returns the token (URL-safe base64, 22+ chars) the caller must
        pass to /source_proxy?token=... to retrieve the upstream.
        """
        token = secrets.token_urlsafe(16)
        with self._lock:
            self._entries[token] = SourceEntry(
                source_url=source_url,
                headers=dict(headers),
            )
        return token

    def get(self, token: str) -> SourceEntry | None:
        with self._lock:
            return self._entries.get(token)

    def unregister(self, token: str) -> None:
        with self._lock:
            self._entries.pop(token, None)
