#!/usr/bin/env python3
"""
Quick test to verify PrerequisiteChecker logic works correctly.
"""

from pathlib import Path
from app import ProjectModel, PrerequisiteChecker


def test_prerequisites():
    """Test prerequisite checking for all workflow steps."""

    print("=" * 60)
    print("Testing PrerequisiteChecker")
    print("=" * 60)

    # Test 1: Fresh project (no workflow state)
    print("\n[Test 1] Fresh Project - No prior work")
    project = ProjectModel()

    can_sample, msg, severity = PrerequisiteChecker.can_sample(project)
    print(f"  Can sample? {can_sample} - {severity}: {msg}")

    can_autolabel, msg, severity = PrerequisiteChecker.can_autolabel(project)
    print(f"  Can autolabel? {can_autolabel} - {severity}: {msg}")

    can_train, msg, severity = PrerequisiteChecker.can_train(project)
    print(f"  Can train? {can_train} - {severity}: {msg}")

    can_analyze, msg, severity = PrerequisiteChecker.can_analyze(project)
    print(f"  Can analyze? {can_analyze} - {severity}: {msg}")

    # Test 2: After sampling (simulate)
    print("\n[Test 2] After Sampling")
    project = ProjectModel()
    project.video = str(Path(__file__).parent / "test_video.mp4")  # Fake path
    project.sample_out = str(Path(__file__).parent / "data" / "images")  # Fake path
    # Simulate that sampling was completed
    project.workflow_state["has_sampled_frames"] = True
    project.workflow_state["sample_output_dir"] = project.sample_out

    can_autolabel, msg, severity = PrerequisiteChecker.can_autolabel(project)
    print(f"  Can autolabel? {can_autolabel} - {severity}: {msg}")

    # Test 3: After autolabeling (simulate)
    print("\n[Test 3] After Autolabeling")
    project.workflow_state["has_labels"] = True
    project.workflow_state["labels_dir"] = "runs/autolabel/exp1/labels"

    can_train, msg, severity = PrerequisiteChecker.can_train(project)
    print(f"  Can train? {can_train} - {severity}: {msg}")

    # Test 4: After training (simulate)
    print("\n[Test 4] After Training")
    project.workflow_state["has_trained_model"] = True
    project.workflow_state["trained_weights"] = "runs/detect/traffic_v8s/weights/best.pt"
    project.an_video = "test_video.mp4"
    project.an_roi = "rois.json"

    can_analyze, msg, severity = PrerequisiteChecker.can_analyze(project)
    print(f"  Can analyze? {can_analyze} - {severity}: {msg}")

    # Test 5: Low confidence warning
    print("\n[Test 5] Low Confidence Threshold")
    project.auto_images = project.sample_out
    project.conf = "0.30"  # Low confidence

    can_autolabel, msg, severity = PrerequisiteChecker.can_autolabel(project)
    print(f"  Can autolabel with conf=0.30? {can_autolabel} - {severity}: {msg}")

    # Test 6: High epoch count warning
    print("\n[Test 6] High Epoch Count")
    project.train_data_yaml = "data.yaml"
    project.train_epochs = "200"  # High epoch count

    can_train, msg, severity = PrerequisiteChecker.can_train(project)
    print(f"  Can train with 200 epochs? {can_train} - {severity}: {msg}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_prerequisites()
