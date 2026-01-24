"""Unit tests for SamplerService.

Example of how to test services with mocks.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from traffic_studio.domain.sampler import SamplerConfig, SamplingMethod
from traffic_studio.services.sampler import SamplerService
from traffic_studio.infrastructure.ffmpeg import FFmpegRunner


class TestSamplerService:
    """Test cases for SamplerService."""

    def test_validates_config(self, logger, temp_dir):
        """Test that invalid config is rejected."""
        # Create mock FFmpeg runner
        mock_ffmpeg = Mock(spec=FFmpegRunner)

        # Create service
        service = SamplerService(ffmpeg=mock_ffmpeg, logger=logger)

        # Create invalid config (non-existent video path)
        config = SamplerConfig(
            video_path=Path("/nonexistent/video.mp4"),
            output_dir=temp_dir,
            fps=1.0
        )

        # Run sampling
        result = service.sample_frames(config)

        # Should fail validation
        assert not result.success
        assert len(result.errors) > 0
        assert result.frames_extracted == 0

        # FFmpeg should not have been called
        mock_ffmpeg.run.assert_not_called()

    def test_collects_single_video(self, logger, temp_dir):
        """Test that single video file is collected correctly."""
        # Create mock FFmpeg runner
        mock_ffmpeg = Mock(spec=FFmpegRunner)
        mock_ffmpeg.path = "/usr/bin/ffmpeg"
        mock_ffmpeg.run = MagicMock()

        # Create a dummy video file
        video_file = temp_dir / "test.mp4"
        video_file.write_text("dummy")

        # Create output directory
        output_dir = temp_dir / "output"

        # Create service
        service = SamplerService(ffmpeg=mock_ffmpeg, logger=logger)

        # Create config
        config = SamplerConfig(
            video_path=video_file,
            output_dir=output_dir,
            fps=1.0,
            method=SamplingMethod.FPS
        )

        # Collect sources
        sources = service._collect_sources(config)

        # Should find one source
        assert len(sources) == 1
        assert sources[0] == video_file

    def test_collects_directory_videos(self, logger, temp_dir):
        """Test that videos are collected from directory."""
        # Create mock FFmpeg runner
        mock_ffmpeg = Mock(spec=FFmpegRunner)

        # Create dummy video files
        (temp_dir / "video1.mp4").write_text("dummy")
        (temp_dir / "video2.mov").write_text("dummy")
        (temp_dir / "video3.avi").write_text("dummy")
        (temp_dir / "not_a_video.txt").write_text("dummy")

        # Create output directory
        output_dir = temp_dir / "output"

        # Create service
        service = SamplerService(ffmpeg=mock_ffmpeg, logger=logger)

        # Create config pointing to directory
        config = SamplerConfig(
            video_path=temp_dir,
            output_dir=output_dir,
            fps=1.0
        )

        # Collect sources
        sources = service._collect_sources(config)

        # Should find 3 video files (not the .txt)
        assert len(sources) == 3
        assert all(s.suffix in config.file_extensions for s in sources)

    def test_config_validation(self):
        """Test SamplerConfig validation logic."""
        # Valid config
        config = SamplerConfig(
            video_path=Path(__file__).parent,  # Use test file directory (exists)
            output_dir=Path("/tmp/output"),
            fps=1.0
        )
        assert config.is_valid
        assert len(config.validate()) == 0

        # Invalid: no interval or fps
        config = SamplerConfig(
            video_path=Path(__file__).parent,
            output_dir=Path("/tmp/output")
        )
        assert not config.is_valid
        errors = config.validate()
        assert any("interval" in e.lower() and "fps" in e.lower() for e in errors)

        # Invalid: negative fps
        config = SamplerConfig(
            video_path=Path(__file__).parent,
            output_dir=Path("/tmp/output"),
            fps=-1.0
        )
        assert not config.is_valid
        errors = config.validate()
        assert any("positive" in e.lower() for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
