import os
from pathlib import Path

import pytest

from pipeline import run_passthrough


@pytest.fixture
def sample_path():
    return Path(__file__).parent / "fixtures" / "sample_5s.ts"


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "hls_out"
    d.mkdir()
    return d


def test_passthrough_produces_hls_files(sample_path, output_dir):
    """ffmpeg passthrough should produce playlist.m3u8 + at least one .ts segment."""
    # NOTE: spec uses f"file://{sample_path.as_posix()}" but ffmpeg 8.1 on
    # Windows fails to resolve drive-letter paths via any file:// URI form
    # (2-, 3-, 4-slash variants, percent-encoded or not — all return ENOENT).
    # Bare absolute path works identically on Windows + Linux ffmpeg and
    # exercises the same passthrough code path. Production callers pass
    # http(s):// HLS URLs, so file:// support is purely a local-test convenience.
    result = run_passthrough(
        source_url=str(sample_path),
        output_dir=output_dir,
        timeout_s=30,
    )
    assert result.returncode == 0, f"ffmpeg failed: {result.stderr[-500:]}"
    assert (output_dir / "playlist.m3u8").exists()
    ts_segments = list(output_dir.glob("*.ts"))
    assert len(ts_segments) >= 1, "Expected at least one .ts segment"


def test_passthrough_fails_on_invalid_url(output_dir):
    """Invalid source URL should return non-zero exit code."""
    result = run_passthrough(
        source_url="file:///nonexistent.ts",
        output_dir=output_dir,
        timeout_s=10,
    )
    assert result.returncode != 0


def test_passthrough_rejects_crlf_in_headers(output_dir, sample_path):
    """CRLF in header keys or values must be rejected (HTTP smuggling guard)."""
    with pytest.raises(ValueError, match="CR/LF"):
        run_passthrough(
            source_url=str(sample_path),
            output_dir=output_dir,
            extra_input_headers={"X-Injected": "value\r\nX-Smuggled: pwned"},
        )

    with pytest.raises(ValueError, match="CR/LF"):
        run_passthrough(
            source_url=str(sample_path),
            output_dir=output_dir,
            extra_input_headers={"X-Bad\r\nKey": "ok"},
        )
