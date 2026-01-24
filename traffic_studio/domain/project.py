"""Project model for Traffic Studio.

This is a refactored version that will eventually replace the ProjectModel in app.py.
Uses proper typing and delegates to domain-specific config objects where appropriate.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


@dataclass
class ProjectMetadata:
    """Metadata about the project."""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "2.0.0"


@dataclass
class StepCompletion:
    """Track which workflow steps have been completed."""
    sample: bool = False
    roi_mask: bool = False
    autolabel: bool = False
    qaqc: bool = False
    train: bool = False
    analyze: bool = False

    def mark_complete(self, step: str):
        """Mark a step as completed."""
        if hasattr(self, step):
            setattr(self, step, True)

    def is_complete(self, step: str) -> bool:
        """Check if a step is completed."""
        return getattr(self, step, False)


@dataclass
class ProjectModel:
    """Main project model encapsulating all application state.

    This model serves as the single source of truth for all project data.
    It can be serialized to/from JSON for persistence.

    Note: This is a transition model. Eventually, UI state (StringVars) will
    be completely separated, and this model will only contain domain data.
    """
    # Project metadata
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    steps: StepCompletion = field(default_factory=StepCompletion)

    # Project file path (not serialized)
    project_file: Optional[Path] = field(default=None, repr=False)

    # Sampling configuration (legacy string-based for now)
    video: str = ""
    sample_out: str = "data/images/new_site"
    every: str = "1.0"
    fps: str = ""
    smethod: str = "fps"
    stol: str = "0.002"

    # ROI & Mask configuration
    roi_json: str = ""
    mask_out: str = "data/images/new_site_masked"
    crop_toggle: bool = False
    crop_margin: str = "0.10"
    crop_labels_dir: str = ""

    # Autolabel configuration
    auto_images: str = ""
    imgsz: str = "960"
    conf: str = "0.70"
    iou: str = "0.60"
    autolabel_out: str = "runs/autolabel"
    classes_file: str = "data/classes.txt"
    labels_dir: str = ""
    autolabel_weights: str = ""

    # Train configuration
    train_data_yaml: str = "data.yaml"
    train_weights: str = "yolov8s.pt"
    train_out_name: str = "traffic_v8s"
    train_epochs: str = "50"
    train_imgsz: str = "640"
    train_batch: str = "16"
    train_device: str = "0"
    train_cache: bool = False
    train_workers: str = "4"
    train_mosaic: bool = True
    train_mixup: bool = False
    train_copy_paste: bool = False
    train_cos_lr: bool = True
    train_warmup_epochs: str = "3.0"
    train_hsv_h: str = "0.015"
    train_hsv_s: str = "0.7"
    train_hsv_v: str = "0.4"

    # QA/QC configuration
    qalabels_dir: str = ""
    qa_delete_line: str = ""
    qa_delete_class: str = ""
    qa_delete_regex: str = ""
    qa_delete_near: str = ""
    qa_eps: str = "0.0"
    qa_dry: bool = True

    # CLI bulk operations
    cli_glob: str = "**/*.txt"
    cli_start: str = ""
    cli_end: str = ""
    cli_step: str = "1"
    cli_prefix: str = "frame_"
    cli_pad: str = "5"
    cli_ext: str = ".txt"
    cli_mode: str = "delete"
    cli_lines: str = ""
    cli_del_line: str = ""
    cli_del_class: str = ""
    cli_del_regex: str = ""
    cli_del_near: str = ""
    cli_eps_bulk: str = "0.0"
    cli_dry: bool = True

    # Split configuration
    split_images_root: str = ""
    split_labels_root: str = ""
    split_out: str = "data"
    split_site_regex: str = r"site_\d+"
    split_train_ratio: str = "0.7"
    split_val_ratio: str = "0.2"
    split_test_ratio: str = "0.1"

    # Analyze configuration
    an_video: str = ""
    an_roi: str = ""
    an_weights: str = ""
    an_out: str = "runs/traffic_analysis"
    an_bin_minutes: str = "15"
    an_do_lane: bool = False
    an_cardinal: bool = False

    def to_dict(self) -> dict:
        """Serialize project to dictionary for JSON persistence.

        Excludes non-serializable fields like project_file.
        """
        data = asdict(self)
        # Remove project_file as it's context-dependent
        data.pop('project_file', None)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'ProjectModel':
        """Load project from dictionary.

        Args:
            data: Dictionary containing project data

        Returns:
            Loaded ProjectModel instance
        """
        # Handle nested dataclasses
        if 'metadata' in data and isinstance(data['metadata'], dict):
            data['metadata'] = ProjectMetadata(**data['metadata'])

        if 'steps' in data and isinstance(data['steps'], dict):
            data['steps'] = StepCompletion(**data['steps'])

        # Create instance with all fields
        return cls(**{k: v for k, v in data.items() if k != 'project_file'})

    def save(self, path: Optional[Path] = None):
        """Save project to JSON file.

        Args:
            path: Optional path to save to. If not provided, uses self.project_file.

        Raises:
            ValueError: If no path is specified and project_file is None
        """
        if path:
            self.project_file = Path(path)

        if not self.project_file:
            raise ValueError("No project file path specified")

        # Update modified timestamp
        self.metadata.modified_at = datetime.now().isoformat()

        # Ensure parent directory exists
        self.project_file.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        with open(self.project_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ProjectModel':
        """Load project from JSON file.

        Args:
            path: Path to project file

        Returns:
            Loaded ProjectModel instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        project = cls.from_dict(data)
        project.project_file = Path(path)
        return project
