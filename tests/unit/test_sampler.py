"""Tests for ffmpeg sampling: cancellation and an end-to-end sample of a synthetic clip."""
import shutil
import subprocess
import sys
import time

import pytest

from studio import sampler

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def test_run_cancellable_terminates_process():
    started = time.monotonic()
    rc, cancelled, _ = sampler.run_cancellable(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cancel_check=lambda: time.monotonic() - started > 0.3)
    assert cancelled is True
    assert time.monotonic() - started < 10


def test_run_cancellable_returns_stderr_on_failure():
    rc, cancelled, err = sampler.run_cancellable(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"])
    assert (rc, cancelled) == (3, False) and "boom" in err


def test_sample_frames_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sampler.sample_frames(tmp_path / "missing.mp4", tmp_path / "out", fps=1.0)


@needs_ffmpeg
def test_sample_frames_end_to_end_with_progress(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10", str(clip)], check=True)
    seen = []
    res = sampler.sample_frames(clip, tmp_path / "out", fps=2.0,
                                progress_callback=lambda f, m: seen.append((f, m)))
    assert res["cancelled"] is False and res["videos"] == [clip]
    assert 3 <= res["frames"] <= 5
    assert len(list((tmp_path / "out").glob("clip_*.jpg"))) == res["frames"]
    assert seen[0][0] == 0.0 and seen[-1][0] == 1.0


@needs_ffmpeg
def test_sample_frames_cancel_before_start(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10", str(clip)], check=True)
    res = sampler.sample_frames(clip, tmp_path / "out", fps=2.0, cancel_check=lambda: True)
    assert res["cancelled"] is True and res["frames"] == 0
