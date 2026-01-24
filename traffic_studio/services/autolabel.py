"""Autolabel service implementation.

Provides auto-labeling of images using YOLO models with proper
dependency injection and error handling.
"""

from pathlib import Path
from typing import Optional, Callable
from logging import Logger, getLogger

from ..domain.autolabel import AutolabelConfig, AutolabelResult
from ..infrastructure.yolo import YOLORunner, find_latest_weights


class AutolabelService:
    """Service for auto-labeling images using YOLO.

    This service wraps YOLO inference with proper configuration management,
    error handling, and progress reporting.
    """

    def __init__(self, yolo: YOLORunner, logger: Optional[Logger] = None):
        """Initialize autolabel service.

        Args:
            yolo: YOLO runner instance
            logger: Optional logger instance
        """
        self.yolo = yolo
        self.logger = logger or getLogger(__name__)

    def autolabel(
        self,
        config: AutolabelConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> AutolabelResult:
        """Auto-label images using YOLO model.

        Args:
            config: Auto-labeling configuration
            progress_callback: Optional progress callback
            cancel_check: Optional cancellation check

        Returns:
            Auto-labeling result
        """
        # Validate configuration
        errors = config.validate()
        if errors:
            self.logger.error(f"Invalid configuration: {errors}")
            return AutolabelResult(
                labels_dir=Path(config.project) / config.name / "labels",
                errors=errors
            )

        # Resolve model path
        model_path = self._resolve_model_path(config)
        if not model_path:
            return AutolabelResult(
                labels_dir=Path(config.project) / config.name / "labels",
                errors=["No model weights found. Specify model_path or train a model first."]
            )

        if progress_callback:
            progress_callback(0.1, f"Loading model: {model_path.name}")

        # Load model
        try:
            model = self.yolo.load_model(model_path)
        except Exception as e:
            self.logger.exception("Failed to load YOLO model")
            return AutolabelResult(
                labels_dir=Path(config.project) / config.name / "labels",
                errors=[f"Failed to load model: {str(e)}"]
            )

        # Check for cancellation
        if cancel_check and cancel_check():
            return AutolabelResult(
                labels_dir=Path(config.project) / config.name / "labels",
                errors=["Operation cancelled by user"]
            )

        if progress_callback:
            progress_callback(0.2, "Running inference...")

        # Run inference
        try:
            results = self.yolo.predict(
                model=model,
                source=config.images_dir,
                device=config.device,
                half=config.half,
                imgsz=config.imgsz,
                conf=config.conf,
                iou=config.iou,
                max_det=config.max_det,
                augment=config.augment,
                classes=config.classes,
                save_txt=True,
                save_conf=False,  # 5-column format for compatibility
                project=config.project,
                name=config.name,
            )

            if progress_callback:
                progress_callback(0.9, "Finalizing labels...")

        except Exception as e:
            self.logger.exception("Inference failed")
            return AutolabelResult(
                labels_dir=Path(config.project) / config.name / "labels",
                errors=[f"Inference failed: {str(e)}"]
            )

        # Determine output paths
        # Ultralytics saves to: {project}/{name}/labels/
        labels_dir = Path(config.project) / config.name / "labels"

        # Count results
        images_labeled = 0
        detections_created = 0

        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.txt"))
            images_labeled = len(label_files)

            # Count total detections
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        detections_created += len(lines)
                except Exception:
                    pass

        if progress_callback:
            progress_callback(
                1.0,
                f"Complete - {images_labeled} images labeled, {detections_created} detections"
            )

        self.logger.info(
            f"Auto-labeling complete: {images_labeled} images, "
            f"{detections_created} detections"
        )

        return AutolabelResult(
            labels_dir=labels_dir,
            images_labeled=images_labeled,
            detections_created=detections_created,
            model_used=model_path,
            errors=[]
        )

    def _resolve_model_path(self, config: AutolabelConfig) -> Optional[Path]:
        """Resolve model path from config or auto-detect.

        Args:
            config: Auto-labeling configuration

        Returns:
            Path to model weights or None if not found
        """
        if config.model_path is not None:
            return config.model_path

        # Auto-detect latest weights
        latest = find_latest_weights()
        if latest:
            self.logger.info(f"Auto-detected model weights: {latest}")
            return latest

        self.logger.warning("No model weights found")
        return None
