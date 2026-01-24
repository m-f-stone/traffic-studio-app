#!/usr/bin/env python3
"""
frame_sampler.py

Extract still frames at fixed time intervals (supports sub-second) from one
video or an entire directory of videos. Requires ffmpeg on PATH.

Examples
--------
# One video, every 0.5 seconds, into data/images/train
python frame_sampler.py videos/clip.mp4 data/images/train --every 0.5

# All MP4/MOV in a folder, every 0.2 seconds (i.e., 5 fps effective)
python frame_sampler.py videos data/images/train --every 0.2

# Use an explicit sampling rate instead of an interval (2 fps == every 0.5s)
python frame_sampler.py videos/clip.mp4 data/images/train --fps 2.0

# Optional: use 'select' method with a small tolerance window
python frame_sampler.py videos/clip.mp4 data/images/train --every 0.5 --method select --tolerance 0.005

Usage cheat-sheet

Every 100 ms: --every 0.1

Every 250 ms: --every 0.25

5 fps sampling: --fps 5

If you see slight drift with --method select, tighten/loosen --tolerance (default 2 ms). For most cases, stick with the default --method fps.
"""
import argparse, shutil, subprocess, sys
from pathlib import Path

def have_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        print("Error: ffmpeg not found on PATH.", file=sys.stderr)
        sys.exit(1)
    return exe

def build_cmd(ffmpeg: str,
              src: Path,
              outdir: Path,
              interval_s: float | None,
              fps: float | None,
              method: str,
              tolerance: float) -> list[str]:
    """
    Build the ffmpeg command.

    - method='fps':   uses fps=RATE (RATE = fps or 1/interval)
    - method='select': uses select=between(mod(t,interval),0,tolerance)
                       + -vsync vfr to output only selected frames

    Notes:
      * Sub-second intervals are fully supported (e.g., interval=0.25).
      * For Windows we avoid shell quoting; the filter expressions are passed as raw args.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = src.stem
    out = outdir / f"{prefix}_%010d.jpg"

    if fps is None:
        if interval_s is None or interval_s <= 0:
            raise ValueError("Interval must be > 0 when --fps is not provided.")
        fps_val = 1.0 / interval_s
    else:
        if fps <= 0:
            raise ValueError("--fps must be > 0.")
        fps_val = fps

    if method == "fps":
        vf = f"fps={fps_val:.6f}"
        return [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src),
            "-vf", vf,
            "-vsync", "vfr",
            "-frame_pts", "1",
            str(out)
        ]
    elif method == "select":
        if interval_s is None:
            # Derive interval from fps if only fps given
            interval_s = 1.0 / fps_val
        tol = max(1e-6, float(tolerance))
        # Select frames whose timestamp modulo interval is within [0, tol]
        # This gives a small window to avoid floating-point mismatches.
        vf = f"select=between(mod(t\\,{interval_s:.9f})\\,0\\,{tol:.9f})"
        return [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src),
            "-vf", vf,
            "-vsync", "vfr",
            "-frame_pts", "1",
            str(out)
        ]
    else:
        raise ValueError(f"Unknown method: {method}")

def collect_sources(in_path: Path, exts_csv: str) -> list[Path]:
    if in_path.is_dir():
        exts = {e.lower().strip() for e in exts_csv.split(",")}
        return [p for p in sorted(in_path.iterdir()) if p.suffix.lower() in exts]
    elif in_path.is_file():
        return [in_path]
    else:
        print(f"Error: input path not found: {in_path}", file=sys.stderr)
        sys.exit(2)

def main():
    ap = argparse.ArgumentParser(description="Extract frames at fixed intervals (supports sub-second) with ffmpeg.")
    ap.add_argument("input", help="Video file OR directory containing videos.")
    ap.add_argument("output_dir", help="Where to write JPGs.")
    ap.add_argument("--every", type=float, default=None,
                    help="Seconds between frames (e.g., 0.5). If not set, you must provide --fps.")
    ap.add_argument("--fps", type=float, default=None,
                    help="Sampling rate in frames per second (e.g., 2.0). Overrides --every if both are set.")
    ap.add_argument("--method", choices=("fps", "select"), default="fps",
                    help="Extraction method: 'fps' (recommended default) or 'select' (modulo-time gate).")
    ap.add_argument("--tolerance", type=float, default=0.002,
                    help="Only for --method select: acceptance window in seconds around each interval boundary (default: 0.002).")
    ap.add_argument("--exts", default=".mp4,.mov,.mkv,.avi",
                    help="Extensions to scan when input is a directory.")
    args = ap.parse_args()

    ffmpeg = have_ffmpeg()

    # Resolve interval/fps
    interval_s = args.every
    fps = args.fps
    if fps is None and interval_s is None:
        # sensible default if user passed neither
        interval_s = 30.0

    # Gather inputs
    sources = collect_sources(Path(args.input), args.exts)
    if not sources:
        print("No videos found to process.", file=sys.stderr)
        sys.exit(3)

    outdir = Path(args.output_dir)

    # Run ffmpeg per source
    for src in sources:
        try:
            cmd = build_cmd(ffmpeg, src, outdir, interval_s, fps, args.method, args.tolerance)
        except ValueError as e:
            print(f"Config error: {e}", file=sys.stderr)
            sys.exit(4)

        # Pretty print summary
        if fps is not None:
            rate_desc = f"{fps:.6g} fps (every {1.0/fps:.6g}s)"
        else:
            rate_desc = f"every {interval_s:.6g}s"
        print(f"[+] {src.name} -> {outdir} ({args.method}, {rate_desc})")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed on {src} (exit {e.returncode})", file=sys.stderr)
            sys.exit(5)

    print("Done.")

if __name__ == "__main__":
    main()
