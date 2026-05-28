"""In-pod source proxy: inject Cookie/User-Agent/Referer for BestSource.

BestSource's Python API (`core.bs.VideoSource`) does not expose
libavformat's `headers`/`http-header-fields` options — only the C++
API does. To unblock egydead-class sources (Pillar 3.6), the cloud
worker routes every BestSource fetch through this loopback endpoint:

    rife pipeline -> 127.0.0.1:8001/source_proxy?token=X
                  -> upstream with registered headers injected

Behavior:
    GET /source_proxy?token=X           — top-level playlist or single
                                          file. URL + headers come from
                                          app.state.source_registry[X].
    GET /source_proxy?token=X&u=<b64>   — child URL (segment or nested
                                          playlist) discovered while
                                          rewriting a parent playlist.
                                          base64url-decoded absolute URL.

Playlist responses (Content-Type application/vnd.apple.mpegurl or
URL ending .m3u8) get their URIs rewritten so child fetches come back
through this same endpoint with the same registered headers.

Segment responses get a PNG-wrapper strip applied — see
detect_png_ts_wrapper_offset() docstring for the upstream pirate-CDN
pattern. Ported verbatim from Chrome-cast-extension/app/castbooster/
proxy.py (the laptop's HTTP-proxy path) so the cloud worker handles
the same egydead/masukestin/audinifer-class wrappers without needing
the laptop in the loop.
"""
from __future__ import annotations

import base64
from urllib.parse import urljoin

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import PlainTextResponse, Response


# --- PNG-wrapped MPEG-TS detection (ported from chrome-cast-extension proxy.py) ---

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_HEADER_SCAN_LIMIT = 4096
_TS_PACKET_SIZE = 188


def detect_png_ts_wrapper_offset(head: bytes) -> int:
    """Detect the anti-adblocker 'PNG-wrapped MPEG-TS' pattern.

    Some pirate streaming CDNs (TikTok ad-CDN under masukestin/hanerix/audinifer
    et al., observed 2026-05-18) prefix MPEG-TS segments with a ~62-byte fake
    PNG header. The Chromecast tolerantly skips ahead to the TS sync byte and
    plays the stream; ffmpeg's strict TS demuxer sees the PNG magic and bails
    with 'Invalid data found when processing input'.

    Returns the byte offset where the MPEG-TS stream begins inside `head`, or
    0 if no wrapper is detected (caller serves bytes unchanged). The check is
    deliberately strict: must be a valid PNG signature, must contain IEND, AND
    the byte right after the IEND chunk's CRC must be the TS sync byte 0x47,
    AND the next-expected TS sync byte (188 bytes later) must also be 0x47.
    Three independent signals — vanishingly low false-positive rate.
    """
    if not head.startswith(_PNG_SIGNATURE):
        return 0
    iend = head.find(b"IEND", 0, _PNG_HEADER_SCAN_LIMIT)
    if iend < 0:
        return 0
    # PNG IEND chunk: 4-byte type ('IEND') + 4-byte CRC. TS data follows.
    ts_start = iend + 4 + 4
    if ts_start >= len(head):
        return 0
    if head[ts_start] != 0x47:
        return 0
    second_sync = ts_start + _TS_PACKET_SIZE
    if second_sync < len(head) and head[second_sync] != 0x47:
        return 0
    return ts_start


# --- URL encoding (compat with chrome-cast-extension/app/castbooster/hls_rewriter.py) ---

def _encode_url(absolute_url: str) -> str:
    """base64url-encode without padding — URL-safe for query strings."""
    return (
        base64.urlsafe_b64encode(absolute_url.encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )


def _decode_url(encoded: str) -> str:
    pad = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode((encoded + pad).encode("ascii")).decode("utf-8")


# --- Playlist rewriting ---

_PLAYLIST_CONTENT_TYPES = (
    "application/vnd.apple.mpegurl",
    "application/x-mpegurl",
    "audio/mpegurl",
    "audio/x-mpegurl",
)


def _looks_like_playlist(url: str, content_type: str | None) -> bool:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in _PLAYLIST_CONTENT_TYPES:
        return True
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    return path.endswith(".m3u8")


def _rewrite_playlist(body: str, base_url: str, token: str) -> str:
    """Rewrite URI lines in an HLS playlist so child fetches loop back
    through /source_proxy with the same registered headers.

    Comments (lines starting with '#') and blank lines pass through
    verbatim. Non-comment lines are URIs (relative or absolute) — we
    resolve them against `base_url` and encode the result into a fresh
    /source_proxy?token=...&u=... URL.
    """
    out_lines: list[str] = []
    for raw in body.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(raw)
            continue
        absolute = urljoin(base_url, stripped)
        out_lines.append(f"/source_proxy?token={token}&u={_encode_url(absolute)}")
    return "\n".join(out_lines) + "\n"


# --- Endpoint handler ---

_SEGMENT_PEEK_BYTES = 8192


async def source_proxy(request: Request, token: str, u: str | None = None) -> Response:
    """Loopback fetcher: resolve token + (optional) u to an upstream URL,
    fetch with the registered headers injected, and return the body.

    Returns:
        - 404 if `token` isn't in the registry.
        - 502 if the upstream connect/read fails.
        - 200 with rewritten playlist body if the response is a playlist.
        - 200 with byte-identical segment body (modulo PNG-wrapper strip)
          otherwise.
    """
    registry = request.app.state.source_registry
    entry = registry.get(token)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown token")

    if u is not None:
        try:
            target_url = _decode_url(u)
        except (ValueError, UnicodeDecodeError) as e:
            raise HTTPException(status_code=400, detail=f"invalid u param: {e}")
    else:
        target_url = entry.source_url

    headers = dict(entry.headers)

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
        ) as client:
            upstream = await client.get(target_url, headers=headers)
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"upstream fetch failed: {type(e).__name__}: {e}",
        )

    upstream_ct = upstream.headers.get("content-type")
    if _looks_like_playlist(target_url, upstream_ct):
        rewritten = _rewrite_playlist(
            upstream.text,
            base_url=str(upstream.url),  # post-redirect URL — relative URIs resolve from here
            token=token,
        )
        return PlainTextResponse(
            content=rewritten,
            status_code=upstream.status_code,
            media_type=upstream_ct or "application/vnd.apple.mpegurl",
        )

    # Segment path: peek the head, strip any PNG wrapper, return the rest.
    body = upstream.content
    if body:
        head = body[:_SEGMENT_PEEK_BYTES]
        offset = detect_png_ts_wrapper_offset(head)
        if offset > 0:
            body = body[offset:]

    return Response(
        content=body,
        status_code=upstream.status_code,
        media_type=upstream_ct or "application/octet-stream",
    )
