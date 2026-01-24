"""Trainer service implementation.

Provides YOLO model training with proper configuration management
and progress reporting.
"""

from pathlib import Path
from typing import Optional, Callable
from logging import Logger, getLogger
import time

from ..domain.trainer import TrainerConfig, TrainerResult
from ..infrastructure.yolo import YOLORunner


class TrainerService:
    """Service for training YOLO models.

    Wraps YOLO training with proper error handling and result reporting.
    """

    def __init__(self, yolo: YOLORunner, logger: Optional[Logger] = None):
        """Initialize trainer service.

        Args:
            yolo: YOLO runner instance
            logger: Optional logger instance
        """
        self.yolo = yolo
        self.logger = logger or getLogger(__name__)

    def train(
        self,
        config: TrainerConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> TrainerResult:
        """Train YOLO model.

        Args:
            config: Training configuration
            progress_callback: Optional progress callback
            cancel_check: Optional cancellation check (Note: YOLO training can't be cancelled mid-epoch)

        Returns:
            Training result
        """
        # Validate configuration
        errors = config.validate()
        if errors:
            self.logger.error(f"Invalid configuration: {errors}")
            return TrainerResult(errors=errors)

        # Check for early cancellation
        if cancel_check and cancel_check():
            return TrainerResult(errors=["Operation cancelled by user"])

        if progress_callback:
            progress_callback(0.05, "Loading base model...")

        # Load base model
        try:
            model = self.yolo.load_model(config.weights)
        except Exception as e:
            self.logger.exception("Failed to load base model")
            return TrainerResult(errors=[f"Failed to load model: {str(e)}"])

        if progress_callback:
            progress_callback(0.1, f"Starting training ({config.epochs} epochs)...")

        # Build training arguments
        train_args = {
            'data': str(config.data_yaml),
            'imgsz': config.imgsz,
            'epochs': config.epochs,
            'batch': config.batch,
            'workers': config.workers,
            'cache': config.cache,
            'name': config.name,
            # Augmentation
            'mosaic': config.mosaic,
            'mixup': config.mixup,
            'copy_paste': config.copy_paste,
            'degrees': config.degrees,
            'hsv_h': config.hsv_h,
            'hsv_s': config.hsv_s,
            'hsv_v': config.hsv_v,
            # Learning rate scheduling
            'close_mosaic': config.close_mosaic,
            'cos_lr': config.cos_lr,
            'warmup_epochs': config.warmup_epochs,
        }

        # Add optional parameters
        if config.fl_gamma is not None:
            train_args['fl_gamma'] = config.fl_gamma
        if config.lr0 is not None:
            train_args['lr0'] = config.lr0
        if config.lrf is not None:
            train_args['lrf'] = config.lrf
        if config.freeze is not None:
            train_args['freeze'] = config.freeze
        if config.optimizer is not None:
            train_args['optimizer'] = config.optimizer

        # Log training configuration
        self.logger.info(f"Training configuration: {train_args}")

        # Run training
        start_time = time.time()
        try:
            results = self.yolo.train(
                model=model,
                data=config.data_yaml,
                device=config.device,
                **train_args
            )

            training_time = time.time() - start_time

            if progress_callback:
                progress_callback(0.9, "Running validation...")

            # Run validation
            best_map = 0.0
            final_map = 0.0
            try:
                metrics = self.yolo.validate(
                    model=model,
                    data=config.data_yaml,
                    device=config.device,
                    imgsz=config.imgsz,
                    plots=True
                )

                # Extract mAP50-95
                if hasattr(metrics, 'box') and hasattr(metrics.box, 'map'):
                    final_map = float(metrics.box.map)
                elif hasattr(metrics, 'map'):
                    final_map = float(metrics.map)

                self.logger.info(f"Final mAP50-95: {final_map:.4f}")

            except Exception as e:
                self.logger.warning(f"Validation after training failed: {e}")

            # Determine model path (Ultralytics saves to runs/detect/{name}/weights/best.pt)
            model_dir = Path(f"runs/detect/{config.name}/weights")
            best_weights = model_dir / "best.pt"

            if not best_weights.exists():
                # Try last.pt as fallback
                last_weights = model_dir / "last.pt"
                model_path = last_weights if last_weights.exists() else None
            else:
                model_path = best_weights

            if progress_callback:
                progress_callback(1.0, f"Training complete ({config.epochs} epochs)")

            return TrainerResult(
                model_path=model_path,
                best_map50_95=best_map,  # TODO: Extract from training results
                final_map50_95=final_map,
                epochs_completed=config.epochs,
                training_time_seconds=training_time,
                errors=[]
            )

        except Exception as e:
            self.logger.exception("Training failed")
            training_time = time.time() - start_time

            return TrainerResult(
                epochs_completed=0,  # Unknown how many completed
                training_time_seconds=training_time,
                errors=[f"Training failed: {str(e)}"]
            )
