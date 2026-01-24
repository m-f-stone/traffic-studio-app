"""Domain models for YOLO model training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainerConfig:
    """Configuration for YOLO model training.

    Attributes:
        data_yaml: Path to dataset configuration YAML
        weights: Base model weights (e.g., 'yolov8s.pt')
        imgsz: Image size for training
        epochs: Number of training epochs
        batch: Batch size (-1 for auto)
        device: Device for training ('cpu', 0, '0,1', etc.)
        workers: Number of dataloader workers
        cache: Cache strategy ('ram', 'disk', False)
        name: Run name for output
        mosaic: Mosaic augmentation probability (0.0-1.0)
        mixup: Mixup augmentation probability (0.0-1.0)
        copy_paste: Copy-paste augmentation probability (0.0-1.0)
        degrees: Rotation augmentation degrees
        hsv_h: HSV hue augmentation
        hsv_s: HSV saturation augmentation
        hsv_v: HSV value augmentation
        fl_gamma: Focal loss gamma (None = disabled)
        close_mosaic: Disable mosaic for final N epochs
        lr0: Initial learning rate (None = auto)
        lrf: Final learning rate factor (None = auto)
        cos_lr: Use cosine learning rate scheduler
        warmup_epochs: Warmup epochs
        freeze: Freeze first N layers (None = no freezing)
        optimizer: Optimizer ('SGD', 'Adam', 'AdamW', None = auto)
    """
    data_yaml: Path
    weights: str = "yolov8s.pt"
    imgsz: int = 640
    epochs: int = 60
    batch: int = -1
    device: str | int = 0
    workers: int = 2
    cache: str | bool = "disk"
    name: str = "traffic_v8s"

    # Augmentation parameters
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    degrees: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    # Advanced training parameters
    fl_gamma: Optional[float] = None
    close_mosaic: int = 10
    lr0: Optional[float] = None
    lrf: Optional[float] = None
    cos_lr: bool = True
    warmup_epochs: float = 3.0
    freeze: Optional[int] = None
    optimizer: Optional[str] = None

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.data_yaml.exists():
            errors.append(f"Data YAML does not exist: {self.data_yaml}")

        if not 320 <= self.imgsz <= 2048:
            errors.append(f"Image size must be between 320 and 2048 (got {self.imgsz})")

        if self.epochs < 1 or self.epochs > 1000:
            errors.append(f"Epochs must be between 1 and 1000 (got {self.epochs})")

        if self.batch < -1 or self.batch == 0:
            errors.append(f"Batch must be -1 (auto) or positive (got {self.batch})")

        if self.workers < 0:
            errors.append(f"Workers must be non-negative (got {self.workers})")

        # Validate augmentation probabilities
        for param_name, param_value in [
            ("mosaic", self.mosaic),
            ("mixup", self.mixup),
            ("copy_paste", self.copy_paste),
        ]:
            if not 0.0 <= param_value <= 1.0:
                errors.append(f"{param_name} must be between 0.0 and 1.0 (got {param_value})")

        if self.warmup_epochs < 0:
            errors.append(f"Warmup epochs must be non-negative (got {self.warmup_epochs})")

        if self.close_mosaic < 0:
            errors.append(f"close_mosaic must be non-negative (got {self.close_mosaic})")

        return errors

    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


@dataclass
class TrainerResult:
    """Result of model training operation.

    Attributes:
        model_path: Path to trained model weights
        best_map50_95: Best mAP50-95 achieved during training
        final_map50_95: Final validation mAP50-95
        epochs_completed: Number of epochs completed
        training_time_seconds: Total training time in seconds
        errors: List of error messages encountered
    """
    model_path: Optional[Path] = None
    best_map50_95: float = 0.0
    final_map50_95: float = 0.0
    epochs_completed: int = 0
    training_time_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if operation completed successfully."""
        return len(self.errors) == 0 and self.model_path is not None
