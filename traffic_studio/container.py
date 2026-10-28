"""Dependency injection container.

Provides centralized service creation and dependency management.
This enables:
1. Easy service instantiation with proper dependencies
2. Testability via mock containers
3. Consistent configuration across the application
"""

from logging import Logger
from typing import Optional

from .infrastructure import UserConfigManager, FFmpegRunner, YOLORunner
from .services import (
    SamplerService,
    AutolabelService,
    TrainerService,
    AnalyzerService,
    DependencyCheckService,
)
from .utils.logging import create_logger


class Container:
    """Dependency injection container for Traffic Studio services.

    This container manages the lifecycle and dependencies of all services.
    It ensures that services are created with proper dependencies injected.

    Example:
        container = Container()
        sampler = container.sampler_service()
        result = sampler.sample_frames(config)
    """

    def __init__(
        self,
        logger: Optional[Logger] = None,
        config_manager: Optional[UserConfigManager] = None
    ):
        """Initialize container.

        Args:
            logger: Optional logger instance (creates default if None)
            config_manager: Optional config manager (creates default if None)
        """
        self._logger = logger or create_logger("traffic_studio")
        self._config_manager = config_manager or UserConfigManager(logger=self._logger)

        # Lazy-initialized infrastructure
        self._ffmpeg: Optional[FFmpegRunner] = None
        self._yolo: Optional[YOLORunner] = None

        # Lazy-initialized services
        self._sampler: Optional[SamplerService] = None
        self._autolabel: Optional[AutolabelService] = None
        self._trainer: Optional[TrainerService] = None
        self._analyzer: Optional[AnalyzerService] = None
        self._dependency_check: Optional[DependencyCheckService] = None

    # ========================================================================
    # Infrastructure
    # ========================================================================

    @property
    def logger(self) -> Logger:
        """Get logger instance."""
        return self._logger

    @property
    def config_manager(self) -> UserConfigManager:
        """Get config manager instance."""
        return self._config_manager

    def ffmpeg_runner(self) -> FFmpegRunner:
        """Get FFmpeg runner instance (singleton).

        Returns:
            FFmpegRunner instance

        Raises:
            RuntimeError: If ffmpeg is not available
        """
        if self._ffmpeg is None:
            self._ffmpeg = FFmpegRunner(logger=self._logger)
        return self._ffmpeg

    def yolo_runner(self) -> YOLORunner:
        """Get YOLO runner instance (singleton).

        Returns:
            YOLORunner instance
        """
        if self._yolo is None:
            self._yolo = YOLORunner(logger=self._logger)
        return self._yolo

    # ========================================================================
    # Services
    # ========================================================================

    def sampler_service(self) -> SamplerService:
        """Get sampler service instance (singleton).

        Returns:
            SamplerService instance
        """
        if self._sampler is None:
            self._sampler = SamplerService(
                ffmpeg=self.ffmpeg_runner(),
                logger=self._logger
            )
        return self._sampler

    def autolabel_service(self) -> AutolabelService:
        """Get autolabel service instance (singleton).

        Returns:
            AutolabelService instance
        """
        if self._autolabel is None:
            self._autolabel = AutolabelService(
                yolo=self.yolo_runner(),
                logger=self._logger
            )
        return self._autolabel

    def trainer_service(self) -> TrainerService:
        """Get trainer service instance (singleton).

        Returns:
            TrainerService instance
        """
        if self._trainer is None:
            self._trainer = TrainerService(
                yolo=self.yolo_runner(),
                logger=self._logger
            )
        return self._trainer

    def analyzer_service(self) -> AnalyzerService:
        """Get analyzer service instance (singleton).

        Returns:
            AnalyzerService instance
        """
        if self._analyzer is None:
            self._analyzer = AnalyzerService(logger=self._logger)
        return self._analyzer

    def dependency_check_service(self) -> DependencyCheckService:
        """Get dependency check service instance (singleton).

        Returns:
            DependencyCheckService instance
        """
        if self._dependency_check is None:
            self._dependency_check = DependencyCheckService(logger=self._logger)
        return self._dependency_check


class MockContainer(Container):
    """Mock container for testing.

    Provides mock implementations of services for unit testing.
    Override service methods to return mocks, or inject real
    services via the set_*() methods.

    Example:
        container = MockContainer()
        service = container.sampler_service()  # returns MagicMock
    """

    def __init__(self):
        """Initialize mock container."""
        from unittest.mock import MagicMock

        # Don't call super().__init__() to avoid creating real infrastructure
        self._logger = create_logger("test")
        self._config_manager = MagicMock(spec=UserConfigManager)

        # Mock infrastructure
        self._ffmpeg = MagicMock(spec=FFmpegRunner)
        self._yolo = MagicMock(spec=YOLORunner)

        # Mock services (created on first access)
        self._sampler = None
        self._autolabel = None
        self._trainer = None
        self._analyzer = None
        self._dependency_check = None

    def sampler_service(self) -> SamplerService:
        """Get mock sampler service."""
        if self._sampler is None:
            from unittest.mock import MagicMock
            self._sampler = MagicMock(spec=SamplerService)
        return self._sampler

    def autolabel_service(self) -> AutolabelService:
        """Get mock autolabel service."""
        if self._autolabel is None:
            from unittest.mock import MagicMock
            self._autolabel = MagicMock(spec=AutolabelService)
        return self._autolabel

    def trainer_service(self) -> TrainerService:
        """Get mock trainer service."""
        if self._trainer is None:
            from unittest.mock import MagicMock
            self._trainer = MagicMock(spec=TrainerService)
        return self._trainer

    def analyzer_service(self) -> AnalyzerService:
        """Get mock analyzer service."""
        if self._analyzer is None:
            from unittest.mock import MagicMock
            self._analyzer = MagicMock(spec=AnalyzerService)
        return self._analyzer

    def dependency_check_service(self) -> DependencyCheckService:
        """Get mock dependency check service."""
        if self._dependency_check is None:
            from unittest.mock import MagicMock
            self._dependency_check = MagicMock(spec=DependencyCheckService)
        return self._dependency_check
