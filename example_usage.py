#!/usr/bin/env python3
"""
Example usage of the refactored Traffic Studio architecture.

This script demonstrates how to use the new service layer architecture
for common workflows without needing the GUI.
"""

from pathlib import Path
import logging

from traffic_studio.container import Container
from traffic_studio.domain.sampler import SamplerConfig, SamplingMethod
from traffic_studio.domain.autolabel import AutolabelConfig
from traffic_studio.domain.trainer import TrainerConfig
from traffic_studio.domain.analyzer import AnalyzerConfig


def setup_example():
    """Set up container and logger for examples."""
    # Create container (manages all service dependencies)
    container = Container()

    # Set logger to INFO level
    container.logger.setLevel(logging.INFO)

    return container


def example_1_sampling(container: Container):
    """Example 1: Sample frames from a video."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Frame Sampling")
    print("=" * 60)

    # Create configuration
    config = SamplerConfig(
        video_path=Path("path/to/video.mp4"),
        output_dir=Path("output/frames"),
        fps=1.0,  # 1 frame per second
        method=SamplingMethod.FPS
    )

    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return

    # Create progress callback
    def on_progress(fraction: float, message: str):
        print(f"  [{int(fraction * 100):3d}%] {message}")

    # Run sampling
    print("\nSampling frames...")
    result = container.sampler_service().sample_frames(
        config,
        progress_callback=on_progress
    )

    # Check result
    if result.success:
        print(f"\n✓ Success!")
        print(f"  Extracted: {result.frames_extracted} frames")
        print(f"  Output: {result.output_dir}")
        print(f"  Videos processed: {len(result.videos_processed)}")
    else:
        print(f"\n✗ Failed!")
        for error in result.errors:
            print(f"  - {error}")


def example_2_autolabel(container: Container):
    """Example 2: Auto-label images with YOLO."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Auto-Labeling")
    print("=" * 60)

    # Create configuration
    config = AutolabelConfig(
        images_dir=Path("output/frames"),
        model_path=Path("weights/best.pt"),  # Or None for auto-detect
        conf=0.70,  # High confidence threshold
        imgsz=960,
        iou=0.60
    )

    # Validate
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return

    # Progress callback
    def on_progress(fraction: float, message: str):
        print(f"  [{int(fraction * 100):3d}%] {message}")

    # Run auto-labeling
    print("\nAuto-labeling images...")
    result = container.autolabel_service().autolabel(
        config,
        progress_callback=on_progress
    )

    # Check result
    if result.success:
        print(f"\n✓ Success!")
        print(f"  Images labeled: {result.images_labeled}")
        print(f"  Detections created: {result.detections_created}")
        print(f"  Labels directory: {result.labels_dir}")
        print(f"  Model used: {result.model_used}")
    else:
        print(f"\n✗ Failed!")
        for error in result.errors:
            print(f"  - {error}")


def example_3_training(container: Container):
    """Example 3: Train a YOLO model."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Model Training")
    print("=" * 60)

    # Create configuration
    config = TrainerConfig(
        data_yaml=Path("data/data.yaml"),
        weights="yolov8s.pt",  # Base model
        imgsz=640,
        epochs=50,
        batch=16,
        device=0,  # GPU 0 (or 'cpu')
        # Augmentation
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # Learning rate
        cos_lr=True,
        warmup_epochs=3.0
    )

    # Validate
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return

    # Progress callback
    def on_progress(fraction: float, message: str):
        print(f"  [{int(fraction * 100):3d}%] {message}")

    # Run training
    print("\nTraining model...")
    print("  (This may take hours depending on dataset size)")

    result = container.trainer_service().train(
        config,
        progress_callback=on_progress
    )

    # Check result
    if result.success:
        print(f"\n✓ Success!")
        print(f"  Model saved: {result.model_path}")
        print(f"  Best mAP50-95: {result.best_map50_95:.4f}")
        print(f"  Final mAP50-95: {result.final_map50_95:.4f}")
        print(f"  Epochs: {result.epochs_completed}")
        print(f"  Training time: {result.training_time_seconds:.1f}s")
    else:
        print(f"\n✗ Failed!")
        for error in result.errors:
            print(f"  - {error}")


def example_4_analysis(container: Container):
    """Example 4: Analyze traffic video."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Traffic Analysis")
    print("=" * 60)

    # Create configuration
    config = AnalyzerConfig(
        video=Path("path/to/traffic_video.mp4"),
        model_path=Path("weights/best.pt"),
        roi_json=Path("config/rois.json"),
        output_dir=Path("output/analysis"),
        run_name="intersection_analysis",
        conf=0.30,
        imgsz=640,
        include_lane_rois=True,
        detect_cardinal=True
    )

    # Validate
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return

    # Progress callback
    def on_progress(fraction: float, message: str):
        print(f"  [{int(fraction * 100):3d}%] {message}")

    # Run analysis
    print("\nAnalyzing traffic video...")
    result = container.analyzer_service().analyze(
        config,
        progress_callback=on_progress
    )

    # Check result
    if result.success:
        print(f"\n✓ Success!")
        print(f"  Total tracks: {result.total_tracks}")
        print(f"  Movements detected: {result.movements_detected}")
        print(f"  Tracks CSV: {result.tracks_csv}")
        print(f"  Movements CSV: {result.movements_csv}")
        print(f"  Tracks with L/T/R: {result.tracks_ltr_csv}")
    else:
        print(f"\n✗ Failed!")
        for error in result.errors:
            print(f"  - {error}")


def example_5_dependency_check(container: Container):
    """Example 5: Check system dependencies."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Dependency Check")
    print("=" * 60)

    service = container.dependency_check_service()

    # Check all dependencies
    print("\nChecking dependencies...")
    results = service.check_all()

    # Display results
    for dep, (available, info) in results.items():
        status = "✓ Available" if available else "✗ NOT available"
        print(f"\n{dep}: {status}")
        print(f"  {info}")

    # Get warnings
    warnings = service.get_warnings()
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"\n{warning}")


def example_6_config_management(container: Container):
    """Example 6: User configuration management."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Configuration Management")
    print("=" * 60)

    config = container.config_manager

    print(f"\nConfig directory: {config.config_dir}")

    # Recent projects
    print("\nRecent projects:")
    recent = config.get_recent_projects()
    if recent:
        for i, path in enumerate(recent, 1):
            print(f"  {i}. {path}")
    else:
        print("  (none)")

    # Add a project
    config.add_recent_project(Path("example_project.traffproj"))
    print("\nAdded example project to recent list")

    # Presets
    print("\nSaving preset...")
    config.save_preset(
        category="sampler",
        name="high_quality",
        data={
            "fps": 2.0,
            "method": "fps",
            "tolerance": 0.001
        }
    )

    print("Loading preset...")
    preset = config.get_preset("sampler", "high_quality")
    print(f"  {preset}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Traffic Studio - Refactored Architecture Examples")
    print("=" * 60)

    # Set up container
    container = setup_example()

    # Note: These examples use placeholder paths.
    # Uncomment and modify paths to run with real data.

    # example_1_sampling(container)
    # example_2_autolabel(container)
    # example_3_training(container)
    # example_4_analysis(container)
    example_5_dependency_check(container)
    example_6_config_management(container)

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nTo run specific examples:")
    print("  1. Uncomment example functions in main()")
    print("  2. Update file paths to point to your data")
    print("  3. Run: python example_usage.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
