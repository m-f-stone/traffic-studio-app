"""Dependency checking service.

Validates that required external dependencies (ffmpeg, CUDA) are available.
"""

from typing import Optional
from logging import Logger, getLogger

from ..infrastructure.ffmpeg import FFmpegRunner, check_ffmpeg_available
from ..infrastructure.yolo import check_cuda_available


class DependencyCheckService:
    """Service for checking system dependencies.

    Validates availability of ffmpeg, CUDA, and other required tools.
    """

    def __init__(self, logger: Optional[Logger] = None):
        """Initialize dependency check service.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or getLogger(__name__)

    def check_ffmpeg(self) -> tuple[bool, Optional[str]]:
        """Check if ffmpeg is available.

        Returns:
            Tuple of (is_available, version_string_or_error_message)
        """
        try:
            runner = FFmpegRunner(logger=self.logger)
            version = runner.get_version()
            return True, version

        except RuntimeError as e:
            return False, str(e)

        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def check_cuda(self) -> tuple[bool, Optional[str]]:
        """Check if CUDA is available.

        Returns:
            Tuple of (is_available, device_info_or_message)
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return False, "CUDA not available (CPU-only PyTorch or no GPU detected)"

            # Get CUDA device info
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            cuda_version = torch.version.cuda or "Unknown"

            info = f"{device_count} device(s): {device_name} (CUDA {cuda_version})"
            return True, info

        except ImportError:
            return False, "PyTorch not installed"

        except Exception as e:
            return False, f"Error checking CUDA: {str(e)}"

    def check_all(self) -> dict[str, tuple[bool, Optional[str]]]:
        """Check all dependencies.

        Returns:
            Dictionary mapping dependency name to (is_available, info)
        """
        results = {
            'ffmpeg': self.check_ffmpeg(),
            'cuda': self.check_cuda(),
        }

        # Log results
        for dep, (available, info) in results.items():
            if available:
                self.logger.info(f"{dep}: Available - {info}")
            else:
                self.logger.warning(f"{dep}: NOT available - {info}")

        return results

    def get_warnings(self) -> list[str]:
        """Get list of dependency warnings.

        Returns:
            List of warning messages for missing dependencies
        """
        warnings = []
        results = self.check_all()

        if not results['ffmpeg'][0]:
            warnings.append(
                "⚠ ffmpeg not found on PATH\n"
                "Video sampling will not work.\n"
                "Install ffmpeg and ensure it's on your system PATH."
            )

        if not results['cuda'][0]:
            warnings.append(
                "⚠ CUDA not available\n"
                "Training and inference will use CPU (slow).\n"
                "Install CUDA toolkit and compatible PyTorch for GPU acceleration."
            )

        return warnings
