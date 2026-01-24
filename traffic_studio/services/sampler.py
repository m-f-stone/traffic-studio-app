"""Sampler service implementation.

Wraps video frame sampling logic with proper dependency injection,
progress reporting, and cancellation support.
"""

from pathlib import Path
from typing import Optional, Callable
from logging import Logger, getLogger

from ..domain.sampler import SamplerConfig, SamplerResult, SamplingMethod
from ..infrastructure.ffmpeg import FFmpegRunner


class SamplerService:
    """Service for sampling video frames using ffmpeg.

    This service provides a clean, testable interface for frame sampling
    with proper error handling, progress reporting, and cancellation.
    """

    def __init__(self, ffmpeg: FFmpegRunner, logger: Optional[Logger] = None):
        """Initialize sampler service.

        Args:
            ffmpeg: FFmpeg runner instance
            logger: Optional logger instance
        """
        self.ffmpeg = ffmpeg
        self.logger = logger or getLogger(__name__)

    def sample_frames(
        self,
        config: SamplerConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> SamplerResult:
        """Sample frames from video(s) based on configuration.

        Args:
            config: Sampling configuration
            progress_callback: Optional callback for progress (fraction, message)
            cancel_check: Optional callback to check if cancelled

        Returns:
            Sampling result with extracted frames count and errors
        """
        # Validate configuration
        errors = config.validate()
        if errors:
            self.logger.error(f"Invalid configuration: {errors}")
            return SamplerResult(
                frames_extracted=0,
                output_dir=config.output_dir,
                errors=errors
            )

        # Collect video sources
        sources = self._collect_sources(config)
        if not sources:
            return SamplerResult(
                frames_extracted=0,
                output_dir=config.output_dir,
                errors=["No video files found matching criteria"]
            )

        self.logger.info(f"Found {len(sources)} video(s) to process")

        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Process each video
        total_frames = 0
        processed = []
        errors = []

        for i, video in enumerate(sources):
            # Check for cancellation
            if cancel_check and cancel_check():
                self.logger.info("Sampling cancelled by user")
                break

            # Report progress
            if progress_callback:
                progress_callback(
                    fraction=i / len(sources),
                    message=f"Processing {video.name} ({i+1}/{len(sources)})"
                )

            # Sample video
            try:
                frames = self._sample_video(video, config)
                total_frames += frames
                processed.append(video)
                self.logger.info(f"Extracted {frames} frames from {video.name}")

            except Exception as e:
                error_msg = f"{video.name}: {str(e)}"
                self.logger.exception(f"Failed to sample {video.name}")
                errors.append(error_msg)

        # Final progress update
        if progress_callback and not (cancel_check and cancel_check()):
            progress_callback(1.0, f"Complete - {total_frames} frames extracted")

        return SamplerResult(
            frames_extracted=total_frames,
            output_dir=config.output_dir,
            videos_processed=processed,
            errors=errors
        )

    def _collect_sources(self, config: SamplerConfig) -> list[Path]:
        """Collect video files from configured path.

        Args:
            config: Sampling configuration

        Returns:
            List of video file paths
        """
        if config.video_path.is_file():
            # Single video file
            return [config.video_path]

        # Directory - collect all matching video files
        sources = []
        for ext in config.file_extensions:
            # Use glob for each extension
            pattern = f"*{ext}"
            sources.extend(config.video_path.glob(pattern))

        # Sort for consistent ordering
        return sorted(sources)

    def _sample_video(self, video: Path, config: SamplerConfig) -> int:
        """Sample a single video file.

        Args:
            video: Path to video file
            config: Sampling configuration

        Returns:
            Number of frames extracted

        Raises:
            Exception: If sampling fails
        """
        # Build ffmpeg command using existing third_party logic
        # This delegates to the existing well-tested frame_sampler code
        from third_party.frame_sampler import build_cmd

        cmd = build_cmd(
            self.ffmpeg.path,
            video,
            config.output_dir,
            config.interval,
            config.fps,
            config.method.value,
            config.tolerance
        )

        self.logger.debug(f"ffmpeg command: {' '.join(cmd)}")

        # Run ffmpeg (excluding the ffmpeg path since FFmpegRunner adds it)
        self.ffmpeg.run(cmd[1:])  # Skip first element (ffmpeg path)

        # Count extracted frames for this video
        # Pattern: {video_stem}_*.jpg
        pattern = f"{video.stem}_*.jpg"
        frames = list(config.output_dir.glob(pattern))

        return len(frames)
