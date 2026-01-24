"""Service interfaces (protocols) for dependency injection and testing.

These protocols define the contract that service implementations must follow.
They enable:
1. Dependency injection
2. Mock implementations for testing
3. Interface-based programming
"""

from typing import Protocol, Optional, Callable
from ..domain import (
    SamplerConfig,
    SamplerResult,
    AutolabelConfig,
    AutolabelResult,
    TrainerConfig,
    TrainerResult,
    AnalyzerConfig,
    AnalyzerResult,
)


class ISamplerService(Protocol):
    """Protocol for video frame sampling service."""

    def sample_frames(
        self,
        config: SamplerConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> SamplerResult:
        """Sample frames from video(s).

        Args:
            config: Sampling configuration
            progress_callback: Optional callback for progress updates (fraction, message)
            cancel_check: Optional callback to check if operation should be cancelled

        Returns:
            Sampling result
        """
        ...


class IAutolabelService(Protocol):
    """Protocol for auto-labeling service."""

    def autolabel(
        self,
        config: AutolabelConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> AutolabelResult:
        """Auto-label images using YOLO.

        Args:
            config: Auto-labeling configuration
            progress_callback: Optional callback for progress updates
            cancel_check: Optional callback to check for cancellation

        Returns:
            Auto-labeling result
        """
        ...


class ITrainerService(Protocol):
    """Protocol for model training service."""

    def train(
        self,
        config: TrainerConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> TrainerResult:
        """Train YOLO model.

        Args:
            config: Training configuration
            progress_callback: Optional callback for progress updates
            cancel_check: Optional callback to check for cancellation

        Returns:
            Training result
        """
        ...


class IAnalyzerService(Protocol):
    """Protocol for traffic analysis service."""

    def analyze(
        self,
        config: AnalyzerConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> AnalyzerResult:
        """Analyze traffic video.

        Args:
            config: Analysis configuration
            progress_callback: Optional callback for progress updates
            cancel_check: Optional callback to check for cancellation

        Returns:
            Analysis result
        """
        ...


class IDependencyCheckService(Protocol):
    """Protocol for dependency checking service."""

    def check_ffmpeg(self) -> tuple[bool, Optional[str]]:
        """Check if ffmpeg is available.

        Returns:
            Tuple of (is_available, version_string_or_error_message)
        """
        ...

    def check_cuda(self) -> tuple[bool, Optional[str]]:
        """Check if CUDA is available.

        Returns:
            Tuple of (is_available, version_or_device_info)
        """
        ...

    def check_all(self) -> dict[str, tuple[bool, Optional[str]]]:
        """Check all dependencies.

        Returns:
            Dictionary mapping dependency name to (is_available, info)
        """
        ...
