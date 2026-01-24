"""YOLO model runner infrastructure.

Wraps ultralytics YOLO with proper error handling and device management.
"""

from pathlib import Path
from typing import Optional, Any
from logging import Logger, getLogger


class YOLORunner:
    """Wrapper for YOLO model operations.

    Provides a clean interface for YOLO inference and training with
    device management, error handling, and logging.
    """

    def __init__(self, logger: Optional[Logger] = None):
        """Initialize YOLO runner.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or getLogger(__name__)
        self._cuda_available = self._check_cuda()

    def _check_cuda(self) -> bool:
        """Check if CUDA is available via PyTorch.

        Returns:
            True if CUDA is available
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            self.logger.warning("PyTorch not available")
            return False

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available.

        Returns:
            True if CUDA GPU acceleration is available
        """
        return self._cuda_available

    def resolve_device(self, requested: Optional[str | int] = None) -> tuple[str | int, bool]:
        """Resolve device configuration for YOLO.

        Args:
            requested: Requested device ('cpu', 'auto', 0, '0,1', etc.)

        Returns:
            Tuple of (device_for_ultralytics, can_use_half_precision)
        """
        if requested is None or str(requested).lower() in ("", "auto"):
            if self._cuda_available:
                self.logger.info("Using CUDA device 0 (auto-detected)")
                return 0, True
            else:
                self.logger.info("CUDA not available; using CPU")
                return "cpu", False

        # Explicit CPU request
        req_str = str(requested).lower()
        if req_str == "cpu":
            return "cpu", False

        # GPU request - check if CUDA is available
        if self._cuda_available:
            return requested, True
        else:
            self.logger.warning(
                f"Requested CUDA device {requested} but CUDA is not available; "
                f"falling back to CPU"
            )
            return "cpu", False

    def load_model(self, weights: str | Path):
        """Load YOLO model from weights.

        Args:
            weights: Path to model weights file or model name

        Returns:
            YOLO model instance
        """
        from ultralytics import YOLO

        self.logger.info(f"Loading YOLO model: {weights}")
        try:
            model = YOLO(str(weights))
            return model
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def predict(
        self,
        model: Any,
        source: str | Path,
        device: Optional[str | int] = None,
        **kwargs
    ) -> Any:
        """Run YOLO prediction.

        Args:
            model: YOLO model instance
            source: Input source (image/video/directory path)
            device: Device override (None = auto)
            **kwargs: Additional arguments for model.predict()

        Returns:
            Prediction results
        """
        dev, can_half = self.resolve_device(device)

        # Auto-set half precision if not specified
        if 'half' not in kwargs:
            kwargs['half'] = can_half

        self.logger.info(f"Running prediction on {source} (device={dev}, half={kwargs['half']})")

        try:
            results = model.predict(
                source=str(source),
                device=dev,
                **kwargs
            )
            return results

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def train(
        self,
        model: Any,
        data: str | Path,
        device: Optional[str | int] = None,
        **kwargs
    ) -> Any:
        """Run YOLO training.

        Args:
            model: YOLO model instance
            data: Path to dataset YAML
            device: Device override (None = auto)
            **kwargs: Additional training arguments

        Returns:
            Training results
        """
        dev, _ = self.resolve_device(device)

        self.logger.info(f"Starting training on {data} (device={dev})")
        self.logger.debug(f"Training parameters: {kwargs}")

        try:
            results = model.train(
                data=str(data),
                device=dev,
                **kwargs
            )
            return results

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def validate(
        self,
        model: Any,
        data: str | Path,
        device: Optional[str | int] = None,
        **kwargs
    ) -> Any:
        """Run YOLO validation.

        Args:
            model: YOLO model instance
            data: Path to dataset YAML
            device: Device override (None = auto)
            **kwargs: Additional validation arguments

        Returns:
            Validation metrics
        """
        dev, _ = self.resolve_device(device)

        self.logger.info(f"Running validation on {data} (device={dev})")

        try:
            metrics = model.val(
                data=str(data),
                device=dev,
                **kwargs
            )
            return metrics

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise


# ============================================================================
# Utility functions
# ============================================================================

def check_cuda_available() -> bool:
    """Check if CUDA is available.

    Returns:
        True if CUDA GPU acceleration is available
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def find_latest_weights(patterns: Optional[list[str]] = None) -> Optional[Path]:
    """Find the most recent training weights.

    Args:
        patterns: Glob patterns to search (default: common YOLO output paths)

    Returns:
        Path to most recent weights file, or None if not found
    """
    from glob import glob

    patterns = patterns or [
        "runs/detect/*/weights/best.pt",
        "runs/detect/*/weights/last.pt",
        "runs/*/weights/best.pt",
        "runs/*/weights/last.pt",
    ]

    hits = []
    for pattern in patterns:
        hits.extend(glob(pattern))

    if not hits:
        return None

    # Sort by modification time
    hits = sorted(hits, key=lambda p: Path(p).stat().st_mtime)
    return Path(hits[-1])
