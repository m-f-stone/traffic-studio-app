"""Analyzer service implementation.

Provides traffic video analysis with ROI-based tracking and
left-turn-right movement classification.
"""

from pathlib import Path
from typing import Optional, Callable
from logging import Logger, getLogger

from ..domain.analyzer import AnalyzerConfig, AnalyzerResult


class AnalyzerService:
    """Service for traffic video analysis.

    Wraps traffic analysis logic with proper configuration management
    and progress reporting.
    """

    def __init__(self, logger: Optional[Logger] = None):
        """Initialize analyzer service.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or getLogger(__name__)

    def analyze(
        self,
        config: AnalyzerConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> AnalyzerResult:
        """Analyze traffic video with ROI-based tracking.

        Args:
            config: Analysis configuration
            progress_callback: Optional progress callback
            cancel_check: Optional cancellation check

        Returns:
            Analysis result
        """
        # Validate configuration
        errors = config.validate()
        if errors:
            self.logger.error(f"Invalid configuration: {errors}")
            return AnalyzerResult(errors=errors)

        # Check for early cancellation
        if cancel_check and cancel_check():
            return AnalyzerResult(errors=["Operation cancelled by user"])

        if progress_callback:
            progress_callback(0.1, "Preparing ROI configuration...")

        # Prepare output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Normalize ROI JSON for tracking
        from studio.roi_utils import normalize_for_tracking

        flat_json = config.output_dir / f"{config.run_name}_rois_flat.json"

        try:
            normalize_for_tracking(
                config.roi_json,
                flat_json,
                include_lanes=config.include_lane_rois,
                lane_pattern=config.lane_pattern
            )
        except Exception as e:
            self.logger.exception("Failed to normalize ROI JSON")
            return AnalyzerResult(errors=[f"ROI processing failed: {str(e)}"])

        if progress_callback:
            progress_callback(0.2, "Running tracking analysis...")

        # Run tracking analysis via direct-call API (thread-safe)
        try:
            from third_party.traffic_analysis_tracking import run_analysis

            run_analysis(
                input_video=str(config.video),
                model=str(config.model_path),
                output=str(config.output_dir),
                run=config.run_name,
                rois_path=str(flat_json),
                tracker=config.tracker,
                imgsz=config.imgsz,
                conf=config.conf,
                iou=config.iou,
                device=config.device
            )

        except Exception as e:
            self.logger.exception("Tracking analysis failed")
            return AnalyzerResult(errors=[f"Tracking failed: {str(e)}"])

        if progress_callback:
            progress_callback(0.8, "Post-processing movement classification...")

        # Post-process for L/T/R movement classification
        try:
            from studio.analyzer import _postprocess_ltr
            _postprocess_ltr(config.output_dir, config.run_name)
        except Exception as e:
            self.logger.exception("L/T/R post-processing failed")
            return AnalyzerResult(errors=[f"Post-processing failed: {str(e)}"])

        # Gather results
        tracks_csv = config.output_dir / f"{config.run_name}_tracks.csv"
        tracks_ltr_csv = config.output_dir / f"{config.run_name}_tracks_ltr.csv"
        movements_csv = config.output_dir / f"{config.run_name}_movements_ltr.csv"

        # Count tracks and movements
        total_tracks = 0
        movements_detected = 0

        if tracks_csv.exists():
            try:
                with open(tracks_csv, 'r') as f:
                    # -1 for header
                    total_tracks = sum(1 for _ in f) - 1
            except Exception:
                pass

        if movements_csv.exists():
            try:
                with open(movements_csv, 'r') as f:
                    import csv
                    reader = csv.DictReader(f)
                    for row in reader:
                        movements_detected += int(row.get('count', 0))
            except Exception:
                pass

        if progress_callback:
            progress_callback(
                1.0,
                f"Complete - {total_tracks} tracks, {movements_detected} movements"
            )

        self.logger.info(
            f"Analysis complete: {total_tracks} tracks, {movements_detected} movements"
        )

        return AnalyzerResult(
            tracks_csv=tracks_csv if tracks_csv.exists() else None,
            movements_csv=movements_csv if movements_csv.exists() else None,
            tracks_ltr_csv=tracks_ltr_csv if tracks_ltr_csv.exists() else None,
            total_tracks=total_tracks,
            movements_detected=movements_detected,
            errors=[]
        )
