"""FFmpeg runner infrastructure.

Wraps ffmpeg execution with proper error handling and logging.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional
from logging import Logger, getLogger


class FFmpegRunner:
    """Wrapper for ffmpeg command execution.

    Provides a clean interface for running ffmpeg commands with proper
    error handling, logging, and path validation.
    """

    def __init__(self, logger: Optional[Logger] = None, ffmpeg_path: Optional[str] = None):
        """Initialize FFmpeg runner.

        Args:
            logger: Optional logger instance
            ffmpeg_path: Optional explicit path to ffmpeg binary.
                        If None, searches system PATH.

        Raises:
            RuntimeError: If ffmpeg is not found
        """
        self.logger = logger or getLogger(__name__)
        self._ffmpeg_path = ffmpeg_path or self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        """Find ffmpeg on system PATH.

        Returns:
            Path to ffmpeg executable

        Raises:
            RuntimeError: If ffmpeg is not found on PATH
        """
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg not found on PATH. "
                "Please install ffmpeg and ensure it's available on your system PATH."
            )
        return ffmpeg

    @property
    def path(self) -> str:
        """Get path to ffmpeg executable.

        Returns:
            Path to ffmpeg binary
        """
        return self._ffmpeg_path

    @property
    def is_available(self) -> bool:
        """Check if ffmpeg is available.

        Returns:
            True if ffmpeg can be executed
        """
        try:
            subprocess.run(
                [self._ffmpeg_path, "-version"],
                capture_output=True,
                check=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_version(self) -> Optional[str]:
        """Get ffmpeg version string.

        Returns:
            Version string or None if unable to get version
        """
        try:
            result = subprocess.run(
                [self._ffmpeg_path, "-version"],
                capture_output=True,
                check=True,
                text=True,
                timeout=5
            )
            # First line contains version info
            return result.stdout.split('\n')[0]
        except Exception as e:
            self.logger.warning(f"Could not get ffmpeg version: {e}")
            return None

    def run(
        self,
        args: list[str],
        check: bool = True,
        timeout: Optional[int] = None,
        capture_output: bool = False
    ) -> subprocess.CompletedProcess:
        """Run ffmpeg command.

        Args:
            args: Command arguments (excluding ffmpeg executable path)
            check: Raise exception if command fails
            timeout: Optional timeout in seconds
            capture_output: Capture stdout/stderr

        Returns:
            CompletedProcess instance

        Raises:
            subprocess.CalledProcessError: If check=True and command fails
            subprocess.TimeoutExpired: If timeout is exceeded
        """
        cmd = [self._ffmpeg_path] + args

        self.logger.debug(f"Running ffmpeg: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=check,
                timeout=timeout,
                capture_output=capture_output,
                text=True if capture_output else False
            )
            return result

        except subprocess.CalledProcessError as e:
            self.logger.error(f"ffmpeg command failed with code {e.returncode}")
            if capture_output and e.stderr:
                self.logger.error(f"ffmpeg stderr: {e.stderr}")
            raise

        except subprocess.TimeoutExpired as e:
            self.logger.error(f"ffmpeg command timed out after {timeout}s")
            raise

    def extract_frames_fps(
        self,
        video: Path,
        output_dir: Path,
        fps: float,
        output_pattern: str = "%08d.jpg"
    ):
        """Extract frames at specified FPS.

        Args:
            video: Input video file
            output_dir: Directory to save frames
            fps: Frames per second to extract
            output_pattern: Output filename pattern (ffmpeg format)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_pattern

        args = [
            "-i", str(video),
            "-vf", f"fps={fps}",
            "-q:v", "2",  # High quality JPEG
            str(output_path)
        ]

        self.run(args)
        self.logger.info(f"Extracted frames from {video.name} at {fps} FPS")

    def extract_frames_interval(
        self,
        video: Path,
        output_dir: Path,
        interval: float,
        output_pattern: str = "%08d.jpg"
    ):
        """Extract frames at specified time interval.

        Args:
            video: Input video file
            output_dir: Directory to save frames
            interval: Seconds between frames
            output_pattern: Output filename pattern
        """
        # interval to fps: fps = 1 / interval
        fps = 1.0 / interval
        self.extract_frames_fps(video, output_dir, fps, output_pattern)


# ============================================================================
# Utility functions for backwards compatibility
# ============================================================================

def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on system PATH.

    Returns:
        True if ffmpeg is available
    """
    try:
        FFmpegRunner()
        return True
    except RuntimeError:
        return False
