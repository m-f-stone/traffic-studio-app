"""Domain models for video frame sampling."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import Enum


class SamplingMethod(Enum):
    """Method for frame sampling."""
    FPS = "fps"
    SELECT = "select"


@dataclass
class SamplerConfig:
    """Configuration for video frame sampling.

    Attributes:
        video_path: Path to video file or directory containing videos
        output_dir: Directory to save extracted frames
        interval: Seconds between frames (mutually exclusive with fps)
        fps: Frames per second to extract (mutually exclusive with interval)
        method: Sampling method (fps or select)
        tolerance: Tolerance for select method (default 0.002)
        file_extensions: Video file extensions to process
    """
    video_path: Path
    output_dir: Path
    interval: Optional[float] = None
    fps: Optional[float] = None
    method: SamplingMethod = SamplingMethod.FPS
    tolerance: float = 0.002
    file_extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi")

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.video_path.exists():
            errors.append(f"Video path does not exist: {self.video_path}")

        if self.interval is None and self.fps is None:
            errors.append("Must specify either interval or fps")

        if self.interval is not None and self.fps is not None:
            errors.append("Cannot specify both interval and fps")

        if self.interval is not None and self.interval <= 0:
            errors.append(f"Interval must be positive (got {self.interval})")

        if self.fps is not None and self.fps <= 0:
            errors.append(f"FPS must be positive (got {self.fps})")

        if self.tolerance <= 0:
            errors.append(f"Tolerance must be positive (got {self.tolerance})")

        return errors

    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


@dataclass
class SamplerResult:
    """Result of frame sampling operation.

    Attributes:
        frames_extracted: Total number of frames extracted
        output_dir: Directory containing extracted frames
        videos_processed: List of video files successfully processed
        errors: List of error messages encountered during processing
    """
    frames_extracted: int
    output_dir: Path
    videos_processed: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if operation completed successfully."""
        return len(self.errors) == 0 and self.frames_extracted > 0
