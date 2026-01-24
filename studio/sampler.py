# studio/sampler.py
from pathlib import Path
import subprocess, shutil
from .logutil import setup_logger
log = setup_logger()

def have_ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH.")
    return exe

def sample_frames(video_or_dir: Path, out_dir: Path, *,
                  every=None, fps=None, method="fps", tolerance=0.002,
                  exts=".mp4,.mov,.mkv,.avi"):
    """
    Wraps your sub-second sampler. method: 'fps' (default) or 'select'.  :contentReference[oaicite:2]{index=2}
    """
    ffmpeg = have_ffmpeg()
    from third_party.frame_sampler import build_cmd, collect_sources  # reuse your code
    sources = collect_sources(video_or_dir, exts)
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        cmd = build_cmd(ffmpeg, src, out_dir, every, fps, method, tolerance)
        log.info("Sampling %s -> %s (%s)", src.name, out_dir, method)
        subprocess.run(cmd, check=True)
