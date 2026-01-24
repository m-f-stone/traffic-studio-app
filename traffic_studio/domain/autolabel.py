"""Domain models for auto-labeling with YOLO."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AutolabelConfig:
    """Configuration for YOLO auto-labeling.

    Attributes:
        images_dir: Directory containing images to label
        model_path: Path to YOLO model weights (.pt file), None for auto-detect
        imgsz: Image size for inference (width/height in pixels)
        conf: Confidence threshold (0.0-1.0)
        iou: IOU threshold for NMS (0.0-1.0)
        device: Device for inference ('cpu', 'auto', or GPU index like 0)
        half: Use FP16 half-precision (None = auto-detect based on device)
        max_det: Maximum detections per image
        augment: Use test-time augmentation
        classes: Filter by class indices (None = all classes)
        project: Project directory for output
        name: Run name for output subdirectory
    """
    images_dir: Path
    model_path: Optional[Path] = None
    imgsz: int = 960
    conf: float = 0.70
    iou: float = 0.60
    device: str | int = "auto"
    half: Optional[bool] = None
    max_det: int = 800
    augment: bool = False
    classes: Optional[list[int]] = None
    project: str = "runs/autolabel"
    name: str = "autolabel_run"

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.images_dir.exists():
            errors.append(f"Images directory does not exist: {self.images_dir}")
        elif not self.images_dir.is_dir():
            errors.append(f"Images path is not a directory: {self.images_dir}")

        if self.model_path is not None and not self.model_path.exists():
            errors.append(f"Model path does not exist: {self.model_path}")

        if not 320 <= self.imgsz <= 2048:
            errors.append(f"Image size must be between 320 and 2048 (got {self.imgsz})")

        if not 0.0 <= self.conf <= 1.0:
            errors.append(f"Confidence must be between 0.0 and 1.0 (got {self.conf})")

        if not 0.0 <= self.iou <= 1.0:
            errors.append(f"IOU threshold must be between 0.0 and 1.0 (got {self.iou})")

        if self.max_det < 1:
            errors.append(f"max_det must be at least 1 (got {self.max_det})")

        return errors

    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


@dataclass
class AutolabelResult:
    """Result of auto-labeling operation.

    Attributes:
        labels_dir: Directory containing generated label files
        images_labeled: Number of images successfully labeled
        detections_created: Total number of detections created
        model_used: Path to model weights used
        errors: List of error messages encountered
    """
    labels_dir: Path
    images_labeled: int = 0
    detections_created: int = 0
    model_used: Optional[Path] = None
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if operation completed successfully."""
        return len(self.errors) == 0 and self.images_labeled > 0
