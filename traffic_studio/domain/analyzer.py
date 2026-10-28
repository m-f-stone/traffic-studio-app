"""Domain models for traffic video analysis."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AnalyzerConfig:
    """Configuration for traffic video analysis.

    Attributes:
        video: Path to video file to analyze
        model_path: Path to YOLO model weights
        roi_json: Path to ROI JSON configuration
        output_dir: Directory for analysis outputs
        run_name: Name for this analysis run
        tracker: Tracker configuration (bytetrack.yaml or botsort.yaml)
        imgsz: Image size for inference
        conf: Confidence threshold
        iou: IOU threshold
        device: Device for inference
        include_lane_rois: Include lane ROIs in tracking
        lane_pattern: Regex pattern to identify lane ROIs
        bin_minutes: Time bin size for temporal analysis (minutes)
        detect_cardinal: Detect cardinal directions from ROI names
    """
    video: Path
    model_path: Path
    roi_json: Path
    output_dir: Path
    run_name: str = "analysis"
    tracker: str = "bytetrack.yaml"
    imgsz: int = 640
    conf: float = 0.30
    iou: float = 0.50
    device: str | int = "auto"
    include_lane_rois: bool = False
    lane_pattern: str = r"(lane|_ln|\b[ltr]\b|through|left|right)"
    bin_minutes: int = 15
    detect_cardinal: bool = True

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.video.exists():
            errors.append(f"Video file does not exist: {self.video}")

        if not self.model_path.exists():
            errors.append(f"Model path does not exist: {self.model_path}")

        if not self.roi_json.exists():
            errors.append(f"ROI JSON does not exist: {self.roi_json}")

        if not 320 <= self.imgsz <= 2048:
            errors.append(f"Image size must be between 320 and 2048 (got {self.imgsz})")

        if not 0.0 <= self.conf <= 1.0:
            errors.append(f"Confidence must be between 0.0 and 1.0 (got {self.conf})")

        if not 0.0 <= self.iou <= 1.0:
            errors.append(f"IOU must be between 0.0 and 1.0 (got {self.iou})")

        if self.bin_minutes < 1:
            errors.append(f"bin_minutes must be at least 1 (got {self.bin_minutes})")

        return errors

    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


@dataclass
class AnalyzerResult:
    """Result of traffic analysis operation.

    Attributes:
        tracks_csv: Path to tracks CSV file
        movements_csv: Path to movements/LTR CSV file
        tracks_ltr_csv: Path to per-track LTR CSV file
        total_tracks: Total number of unique tracks
        movements_detected: Number of movements detected
        errors: List of error messages encountered
    """
    tracks_csv: Optional[Path] = None
    movements_csv: Optional[Path] = None
    tracks_ltr_csv: Optional[Path] = None
    total_tracks: int = 0
    movements_detected: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if operation completed successfully."""
        return len(self.errors) == 0 and self.tracks_csv is not None
