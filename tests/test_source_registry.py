"""SourceRegistry: token issuance, lookup, and cleanup behavior."""
from source_registry import SourceRegistry


def test_register_returns_unique_tokens():
    """Same URL+headers must produce distinct tokens — the registry is a
    map of opaque IDs, not a content hash, so concurrent /process calls
    with identical inputs each get an independent token + cleanup hook.
    """
    reg = SourceRegistry()
    t1 = reg.register("https://example.com/x.m3u8", {"Cookie": "s=1"})
    t2 = reg.register("https://example.com/x.m3u8", {"Cookie": "s=1"})
    assert t1 != t2
    assert reg.get(t1) is not None
    assert reg.get(t2) is not None


def test_get_returns_none_for_unknown_token():
    reg = SourceRegistry()
    assert reg.get("nope") is None


def test_unregister_drops_entry():
    reg = SourceRegistry()
    token = reg.register("https://example.com/x.m3u8", {"Cookie": "s=1"})
    assert reg.get(token) is not None
    reg.unregister(token)
    assert reg.get(token) is None
    # Idempotent — unregistering twice is fine (cleanup callbacks may
    # fire after a manager-driven terminal transition has already cleared).
    reg.unregister(token)
    assert reg.get(token) is None
