"""/source_proxy endpoint: header injection, playlist rewrite, PNG strip."""
import base64

import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from source_proxy import (
    _PNG_SIGNATURE,
    _TS_PACKET_SIZE,
    source_proxy,
)
from source_registry import SourceRegistry


@pytest.fixture
def app_with_proxy():
    """Bare FastAPI app with just the /source_proxy route + a fresh
    SourceRegistry on app.state — no auth, no startup hooks, so we
    can hit the proxy directly without provisioning the full pod app."""
    app = FastAPI()
    app.state.source_registry = SourceRegistry()
    app.add_api_route("/source_proxy", source_proxy, methods=["GET"])
    return app


@pytest.fixture
def client(app_with_proxy):
    return TestClient(app_with_proxy)


def _b64url(absolute_url: str) -> str:
    return (
        base64.urlsafe_b64encode(absolute_url.encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )


def _b64url_decode(encoded: str) -> str:
    pad = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode((encoded + pad).encode("ascii")).decode("utf-8")


def test_source_proxy_404_on_unknown_token(client):
    """Unknown token → 404, not 500. Defensive: never fetch upstream
    without a valid registry entry."""
    response = client.get("/source_proxy?token=does-not-exist")
    assert response.status_code == 404


@respx.mock
def test_source_proxy_fetches_with_registered_headers(client, app_with_proxy):
    """The upstream request MUST carry the registered Cookie header — this
    is the whole point of the proxy. BestSource can't inject headers, so we
    do it here."""
    upstream_url = "https://upstream.example/anime.m3u8"
    headers = {"Cookie": "session=abc123", "Referer": "https://egydead.example/"}
    token = app_with_proxy.state.source_registry.register(upstream_url, headers)

    route = respx.get(upstream_url).mock(
        return_value=Response(
            200,
            content=b"#EXTM3U\n#EXT-X-ENDLIST\n",
            headers={"content-type": "application/vnd.apple.mpegurl"},
        )
    )

    r = client.get(f"/source_proxy?token={token}")
    assert r.status_code == 200
    assert route.called, "upstream never fetched"
    sent_request = route.calls[0].request
    assert sent_request.headers.get("cookie") == "session=abc123"
    assert sent_request.headers.get("referer") == "https://egydead.example/"


@respx.mock
def test_source_proxy_rewrites_playlist_urls(client, app_with_proxy):
    """Relative segment URI in upstream playlist must be rewritten to
    /source_proxy?token=X&u=<b64-of-absolute-url>. Comment lines pass
    through verbatim."""
    upstream_url = "https://upstream.example/path/master.m3u8"
    token = app_with_proxy.state.source_registry.register(upstream_url, {"Cookie": "s=1"})

    upstream_body = (
        "#EXTM3U\n"
        "#EXT-X-VERSION:3\n"
        "seg_000.ts\n"
    )
    respx.get(upstream_url).mock(
        return_value=Response(
            200,
            content=upstream_body.encode("utf-8"),
            headers={"content-type": "application/vnd.apple.mpegurl"},
        )
    )

    r = client.get(f"/source_proxy?token={token}")
    assert r.status_code == 200
    body = r.text
    lines = [ln for ln in body.splitlines() if ln]
    # Comments preserved unchanged.
    assert lines[0] == "#EXTM3U"
    assert lines[1] == "#EXT-X-VERSION:3"
    # Segment URL rewritten as /source_proxy?token=...&u=<base64url>.
    seg_line = lines[2]
    assert seg_line.startswith(f"/source_proxy?token={token}&u=")
    encoded = seg_line.split("&u=", 1)[1]
    decoded = _b64url_decode(encoded)
    # urljoin("https://upstream.example/path/master.m3u8", "seg_000.ts") ->
    # "https://upstream.example/path/seg_000.ts"
    assert decoded == "https://upstream.example/path/seg_000.ts"


@respx.mock
def test_source_proxy_strips_png_wrapper_from_segment(client, app_with_proxy):
    """Anti-adblocker PNG-wrapped TS: the upstream serves bytes that start
    with a PNG signature, contain IEND, and then have the real TS sync
    byte 0x47 (and another sync byte 188 bytes later). The proxy must
    detect and strip so ffmpeg's TS demuxer sees a clean stream."""
    upstream_url = "https://upstream.example/seg.ts"
    token = app_with_proxy.state.source_registry.register(upstream_url, {})

    # Build a believable PNG-wrapped TS payload:
    # [PNG sig][some bytes][IEND][CRC]…[TS packet starting 0x47][TS packet starting 0x47]
    png_header = (
        _PNG_SIGNATURE
        + b"\x00\x00\x00\x0dIHDR"  # 13-byte IHDR header
        + b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"  # tiny IHDR data
        + b"\x1f\x15\xc4\x89"  # IHDR CRC (fake)
        + b"IEND"
        + b"\xae\x42\x60\x82"  # IEND CRC (fake)
    )
    ts_packet_1 = b"\x47" + b"\x00" * (_TS_PACKET_SIZE - 1)
    ts_packet_2 = b"\x47" + b"\x11" * (_TS_PACKET_SIZE - 1)
    upstream_body = png_header + ts_packet_1 + ts_packet_2

    respx.get(upstream_url).mock(
        return_value=Response(
            200,
            content=upstream_body,
            headers={"content-type": "video/mp2t"},
        )
    )

    r = client.get(f"/source_proxy?token={token}")
    assert r.status_code == 200
    returned = r.content
    # Must NOT start with the PNG magic — wrapper got stripped.
    assert not returned.startswith(_PNG_SIGNATURE), (
        "PNG wrapper still in response — detect_png_ts_wrapper_offset failed"
    )
    # Must start with the TS sync byte.
    assert returned[0] == 0x47, (
        f"first byte is {returned[0]:#x}, expected 0x47 (TS sync)"
    )
    # Both planted TS packets must be present in-tact.
    assert returned == ts_packet_1 + ts_packet_2


@respx.mock
def test_source_proxy_passes_clean_segment_unchanged(client, app_with_proxy):
    """Negative case: a clean TS segment (starts with 0x47) must come back
    byte-identical. We never want the PNG detector tripping on a clean
    stream and lopping off real video bytes."""
    upstream_url = "https://upstream.example/clean.ts"
    token = app_with_proxy.state.source_registry.register(upstream_url, {})

    clean_ts = b"\x47" + b"\x10" * (_TS_PACKET_SIZE - 1) + b"\x47" + b"\x20" * (_TS_PACKET_SIZE - 1)
    respx.get(upstream_url).mock(
        return_value=Response(
            200,
            content=clean_ts,
            headers={"content-type": "video/mp2t"},
        )
    )

    r = client.get(f"/source_proxy?token={token}")
    assert r.status_code == 200
    assert r.content == clean_ts, (
        "clean segment was modified — PNG-strip false-positive on TS stream"
    )
