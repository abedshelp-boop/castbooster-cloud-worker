# tests/test_run_rife.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_run_rife_writes_vpy_and_spawns_processes(tmp_path, monkeypatch):
    from run_rife import run_rife

    fake_vspipe = MagicMock()
    fake_vspipe.stdout = MagicMock()
    fake_vspipe.returncode = 0
    fake_vspipe.wait.return_value = 0
    fake_vspipe.stderr.read.return_value = b""

    fake_ffmpeg = MagicMock()
    fake_ffmpeg.returncode = 0
    fake_ffmpeg.communicate.return_value = (b"ok", b"")

    popen_calls = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        if cmd[0] == "vspipe":
            return fake_vspipe
        elif cmd[0] == "ffmpeg":
            return fake_ffmpeg
        raise RuntimeError(f"unexpected popen: {cmd}")

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    result = run_rife(
        source_url="https://egydead.example/anime.m3u8",
        output_dir=tmp_path,
        timeout_s=60,
    )

    assert result.returncode == 0
    assert (tmp_path / "pipeline.vpy").exists()
    vpy_content = (tmp_path / "pipeline.vpy").read_text()
    assert "https://egydead.example/anime.m3u8" in vpy_content
    assert "RIFE" in vpy_content
    assert any(cmd[0] == "vspipe" for cmd in popen_calls)
    assert any(cmd[0] == "ffmpeg" for cmd in popen_calls)
    # Verify NVENC is in the ffmpeg command
    ffmpeg_cmd = [c for c in popen_calls if c[0] == "ffmpeg"][0]
    assert "h264_nvenc" in ffmpeg_cmd


def test_run_rife_returns_502_on_ffmpeg_failure(tmp_path, monkeypatch):
    from run_rife import run_rife

    fake_vspipe = MagicMock()
    fake_vspipe.stdout = MagicMock()
    fake_vspipe.returncode = 0
    fake_vspipe.wait.return_value = 0
    fake_vspipe.stderr.read.return_value = b""

    fake_ffmpeg = MagicMock()
    fake_ffmpeg.returncode = 1
    fake_ffmpeg.communicate.return_value = (b"", b"nvenc init failed")

    def fake_popen(cmd, **kwargs):
        if cmd[0] == "vspipe":
            return fake_vspipe
        return fake_ffmpeg

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    result = run_rife(
        source_url="x",
        output_dir=tmp_path,
        timeout_s=60,
    )
    assert result.returncode == 1
    assert "nvenc init failed" in result.stderr


def test_run_rife_rejects_extra_input_headers(tmp_path, monkeypatch):
    """RIFE pipeline reads source via vapoursynth/lsmas, not ffmpeg, so the
    ffmpeg -headers path is dead. Caller passing headers must get a clear
    NotImplementedError until lsmas format_opts wiring lands in Phase 2."""
    from run_rife import run_rife

    # subprocess.Popen should never be called — we should raise before then.
    def fake_popen_should_not_run(cmd, **kwargs):
        raise AssertionError(
            f"subprocess.Popen called with {cmd!r} — "
            "run_rife should raise NotImplementedError before spawning"
        )

    monkeypatch.setattr("subprocess.Popen", fake_popen_should_not_run)

    with pytest.raises(NotImplementedError, match="extra_input_headers"):
        run_rife(
            source_url="https://egydead.example/anime.m3u8",
            output_dir=tmp_path,
            timeout_s=60,
            extra_input_headers={"Cookie": "session=abc"},
        )
