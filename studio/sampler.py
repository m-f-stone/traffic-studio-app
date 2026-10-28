# studio/sampler.py
"""Frame sampling via ffmpeg with cancellation and per-video progress."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from .logutil import setup_logger

log = setup_logger()


def have_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH.")
    return exe


def run_cancellable(cmd: list[str], cancel_check: Optional[Callable[[], bool]] = None,
                    poll_s: float = 0.2) -> tuple[int, bool, str]:
    """Run a subprocess, terminating it if cancel_check() turns true.

    Returns (returncode, cancelled, stderr_text). stderr is spooled to a temp
    file so a chatty process can never block on a full pipe.
    """
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)
        cancelled = False
        while proc.poll() is None:
            if cancel_check and cancel_check():
                cancelled = True
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(poll_s)
        err.seek(0)
        return proc.returncode, cancelled, err.read()


def sample_frames(video_or_dir: Path, out_dir: Path, *,
                  every=None, fps=None, method="fps", tolerance=0.002,
                  exts=".mp4,.mov,.mkv,.avi",
                  progress_callback: Optional[Callable[[float, str], None]] = None,
                  cancel_check: Optional[Callable[[], bool]] = None) -> dict:
    """Sample frames from one video or every video in a folder.

    Returns {"frames": int, "videos": [Path], "cancelled": bool}.
    """
    from third_party.frame_sampler import build_cmd, collect_sources  # reuse the CLI's command builder

    src_path = Path(video_or_dir)
    if not src_path.exists():
        raise FileNotFoundError(f"Input path not found: {src_path}")
    sources = collect_sources(src_path, exts)
    if not sources:
        raise FileNotFoundError(f"No videos ({exts}) found in {src_path}")
    ffmpeg = have_ffmpeg()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    processed: list[Path] = []
    cancelled = False
    n = len(sources)
    for i, src in enumerate(sources):
        if cancel_check and cancel_check():
            cancelled = True
            break
        if progress_callback:
            progress_callback(i / n, f"Sampling {src.name} ({i + 1}/{n})")
        cmd = build_cmd(ffmpeg, src, out_dir, every, fps, method, tolerance)
        log.info("Sampling %s -> %s (%s)", src.name, out_dir, method)
        rc, was_cancelled, err = run_cancellable(cmd, cancel_check)
        if was_cancelled:
            cancelled = True
            log.warning("Sampling cancelled during %s", src.name)
            break
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed on {src.name} (exit {rc}): {err.strip()[-500:]}")
        frames = len(list(out_dir.glob(f"{src.stem}_*.jpg")))
        total_frames += frames
        processed.append(src)
        log.info("  %d frames from %s", frames, src.name)

    if progress_callback and not cancelled:
        progress_callback(1.0, f"Done: {total_frames} frames from {len(processed)} video(s)")
    return {"frames": total_frames, "videos": processed, "cancelled": cancelled}
