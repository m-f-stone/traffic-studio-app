#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import queue, shutil, subprocess, glob
import sys, re, os, json
from typing import Optional, Callable, Any
from datetime import datetime

from studio.logutil import setup_logger
from studio.threads import Worker
from studio import sampler, roi_utils, autolabel, trainer, analyzer, qaqc


# ============================================================================
# Project Model & Validation
# ============================================================================

class ProjectModel:
    """
    Encapsulates all project state: paths, settings, and step completion status.
    Serializes to/from a single .json project file.
    """
    def __init__(self):
        self.project_file: Optional[Path] = None
        self.created_at = datetime.now().isoformat()
        self.modified_at = self.created_at

        # Step completion tracking
        self.steps_completed = {
            "sample": False,
            "roi_mask": False,
            "autolabel": False,
            "qaqc": False,
            "train": False,
            "analyze": False
        }

        # Sampling
        self.video = ""
        self.sample_out = "data/images/new_site"
        self.every = "1.0"
        self.fps = ""
        self.smethod = "fps"
        self.stol = "0.002"

        # ROI & Mask
        self.roi_json = ""
        self.mask_out = "data/images/new_site_masked"
        self.crop_toggle = False
        self.crop_margin = "0.10"
        self.crop_labels_dir = ""

        # Autolabel
        self.auto_images = ""
        self.imgsz = "960"
        self.conf = "0.70"
        self.iou = "0.60"
        self.autolabel_out = "runs/autolabel"
        self.classes_file = "data/classes.txt"
        self.labels_dir = ""
        self.autolabel_weights = ""

        # Train
        self.train_data_yaml = "data.yaml"
        self.train_weights = "yolov8s.pt"
        self.train_out_name = "traffic_v8s"
        self.train_epochs = "50"
        self.train_imgsz = "640"
        self.train_batch = "16"
        self.train_device = "0"
        self.train_cache = False
        self.train_workers = "4"
        # Advanced augmentation
        self.train_mosaic = True
        self.train_mixup = False
        self.train_copy_paste = False
        self.train_cos_lr = True
        self.train_warmup_epochs = "3.0"
        self.train_hsv_h = "0.015"
        self.train_hsv_s = "0.7"
        self.train_hsv_v = "0.4"

        # QA/QC
        self.qalabels_dir = ""
        self.qa_delete_line = ""
        self.qa_delete_class = ""
        self.qa_delete_regex = ""
        self.qa_delete_near = ""
        self.qa_eps = "0.0"
        self.qa_dry = True

        # CLI bulk
        self.cli_glob = "**/*.txt"
        self.cli_start = ""
        self.cli_end = ""
        self.cli_step = "1"
        self.cli_prefix = "frame_"
        self.cli_pad = "5"
        self.cli_ext = ".txt"
        self.cli_mode = "delete"
        self.cli_lines = ""
        self.cli_del_line = ""
        self.cli_del_class = ""
        self.cli_del_regex = ""
        self.cli_del_near = ""
        self.cli_eps_bulk = "0.0"
        self.cli_dry = True

        # Split
        self.split_images_root = ""
        self.split_labels_root = ""
        self.split_out = "data"
        self.split_site_regex = r"site_\d+"
        self.split_train_ratio = "0.7"
        self.split_val_ratio = "0.2"
        self.split_test_ratio = "0.1"

        # Analyze
        self.an_video = ""
        self.an_roi = ""
        self.an_weights = ""
        self.an_out = "runs/traffic_analysis"
        self.an_bin_minutes = "15"
        self.an_do_lane = False
        self.an_cardinal = False

        # Workflow State (for prerequisite tracking)
        self.workflow_state = {
            "has_sampled_frames": False,
            "has_roi_json": False,
            "has_labels": False,
            "has_trained_model": False,
            "sample_output_dir": "",
            "roi_json_path": "",
            "labels_dir": "",
            "trained_weights": "",
            "last_autolabel_run": "",
        }

    def detect_workflow_state(self):
        """
        Auto-detect workflow state from existing paths.
        Called after loading a project to ensure backward compatibility.
        """
        # Detect sampled frames
        if self.sample_out and Path(self.sample_out).exists():
            image_count = len(list(Path(self.sample_out).glob("*.jpg")))
            if image_count > 0:
                self.workflow_state["has_sampled_frames"] = True
                self.workflow_state["sample_output_dir"] = self.sample_out

        # Detect ROI JSON
        if self.roi_json and Path(self.roi_json).exists():
            self.workflow_state["has_roi_json"] = True
            self.workflow_state["roi_json_path"] = self.roi_json

        # Detect labels
        if self.labels_dir and Path(self.labels_dir).exists():
            label_count = len(list(Path(self.labels_dir).glob("*.txt")))
            if label_count > 0:
                self.workflow_state["has_labels"] = True

        # Detect trained model
        if self.train_out_name:
            # Look for typical YOLO training output structure
            weights_path = Path(f"runs/detect/{self.train_out_name}/weights/best.pt")
            if weights_path.exists():
                self.workflow_state["has_trained_model"] = True
                self.workflow_state["trained_weights"] = str(weights_path)

    def to_dict(self) -> dict:
        """Serialize to dictionary (excluding project_file path)."""
        return {k: v for k, v in self.__dict__.items() if k != 'project_file'}

    def from_dict(self, data: dict):
        """Load from dictionary."""
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def save(self, path: Optional[Path] = None):
        """Save project to JSON file."""
        if path:
            self.project_file = Path(path)
        if not self.project_file:
            raise ValueError("No project file path specified")

        self.modified_at = datetime.now().isoformat()
        self.project_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.project_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: Path) -> 'ProjectModel':
        """Load project from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        project = ProjectModel()
        project.from_dict(data)
        project.project_file = Path(path)
        # Auto-detect workflow state for backward compatibility
        project.detect_workflow_state()
        return project


class PrerequisiteChecker:
    """
    Validates prerequisites for each workflow step.
    Returns (can_proceed: bool, message: str, severity: str)
    """

    @staticmethod
    def can_sample(project: ProjectModel) -> tuple[bool, str, str]:
        """Check if sampling can proceed."""
        if not project.video or not Path(project.video).exists():
            return False, "⚠️ Select a valid video file or directory", "error"
        if not project.sample_out:
            return False, "⚠️ Specify an output directory", "error"
        return True, "✓ Ready to sample frames", "ok"

    @staticmethod
    def can_autolabel(project: ProjectModel) -> tuple[bool, str, str]:
        """Check if autolabeling can proceed."""
        # Check for images directory
        images_dir = project.auto_images or project.workflow_state.get("sample_output_dir", "")
        if not images_dir or not Path(images_dir).exists():
            return False, "⚠️ No images found. Run Sampler first or select an images directory.", "warning"

        image_count = len(list(Path(images_dir).glob("*.jpg"))) + len(list(Path(images_dir).glob("*.png")))
        if image_count == 0:
            return False, "⚠️ Images directory is empty", "warning"

        # Check confidence threshold
        try:
            conf = float(project.conf)
            if conf < 0.5:
                return True, f"⚠️ Low confidence ({conf}). Proceed anyway?", "warning"
        except ValueError:
            return False, "⚠️ Invalid confidence value", "error"

        return True, f"✓ Ready to autolabel ({image_count} images)", "ok"

    @staticmethod
    def can_train(project: ProjectModel) -> tuple[bool, str, str]:
        """Check if training can proceed."""
        # Check for data.yaml
        if not project.train_data_yaml or not Path(project.train_data_yaml).exists():
            return False, "⚠️ data.yaml not found. Create labels first.", "warning"

        # Check for labels
        if not project.workflow_state.get("has_labels", False):
            return True, "⚠️ No labels detected. Proceed anyway?", "warning"

        # Check epochs
        try:
            epochs = int(project.train_epochs)
            if epochs > 100:
                return True, f"⚠️ Training for {epochs} epochs may take hours. Proceed?", "warning"
        except ValueError:
            return False, "⚠️ Invalid epochs value", "error"

        return True, "✓ Ready to train", "ok"

    @staticmethod
    def can_analyze(project: ProjectModel) -> tuple[bool, str, str]:
        """Check if traffic analysis can proceed."""
        # Check for video
        if not project.an_video or not Path(project.an_video).exists():
            return False, "⚠️ Select a video to analyze", "error"

        # Check for ROI JSON
        if not project.an_roi or not Path(project.an_roi).exists():
            return False, "⚠️ ROI JSON required. Use ROI Tool to create one.", "warning"

        # Check for trained weights
        if not project.an_weights or not Path(project.an_weights).exists():
            trained_weights = project.workflow_state.get("trained_weights", "")
            if trained_weights and Path(trained_weights).exists():
                return True, f"ℹ️ No weights specified. Use trained model ({Path(trained_weights).name})?", "info"
            else:
                return True, "⚠️ No weights specified. Will use default model. Proceed?", "warning"

        return True, "✓ Ready to analyze", "ok"

    @staticmethod
    def can_do_roi_mask(project: ProjectModel) -> tuple[bool, str, str]:
        """Check if ROI/mask operations can proceed."""
        # ROI tool can always run (creates new ROI)
        # Mask application needs images
        images_dir = project.workflow_state.get("sample_output_dir", "")
        if images_dir and Path(images_dir).exists():
            return True, "✓ Ready", "ok"
        else:
            return True, "ℹ️ ROI Tool can run without images", "info"

    @staticmethod
    def can_do_qaqc(project: ProjectModel) -> tuple[bool, str, str]:
        """Check if QA/QC can proceed."""
        if not project.qalabels_dir or not Path(project.qalabels_dir).exists():
            labels_dir = project.workflow_state.get("labels_dir", "")
            if labels_dir and Path(labels_dir).exists():
                return True, f"ℹ️ Use detected labels directory ({Path(labels_dir).name})?", "info"
            else:
                return False, "⚠️ Specify a labels directory", "warning"
        return True, "✓ Ready", "ok"


class FieldValidator:
    """
    Validates Entry fields and manages visual feedback (red/green indicators).
    """
    @staticmethod
    def path_exists(path_str: str, must_be_dir: bool = False, must_be_file: bool = False) -> bool:
        """Check if path exists and optionally if it's a file or directory."""
        if not path_str or not path_str.strip():
            return False
        p = Path(path_str)
        if not p.exists():
            return False
        if must_be_dir and not p.is_dir():
            return False
        if must_be_file and not p.is_file():
            return False
        return True

    @staticmethod
    def is_float(val: str, min_val: float = None, max_val: float = None) -> bool:
        """Check if string is a valid float within optional bounds."""
        try:
            f = float(val)
            if min_val is not None and f < min_val:
                return False
            if max_val is not None and f > max_val:
                return False
            return True
        except ValueError:
            return False

    @staticmethod
    def is_int(val: str, min_val: int = None, max_val: int = None) -> bool:
        """Check if string is a valid integer within optional bounds."""
        try:
            i = int(val)
            if min_val is not None and i < min_val:
                return False
            if max_val is not None and i > max_val:
                return False
            return True
        except ValueError:
            return False

    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if ffmpeg is available on PATH."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def check_cuda() -> bool:
        """Check if CUDA is available via PyTorch."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


class ValidatedEntry(ttk.Entry):
    """
    Entry widget with built-in validation and visual feedback.
    Shows green checkmark when valid, red X when invalid.
    """
    def __init__(self, parent, textvariable, validator: Callable[[str], bool],
                 required: bool = True, *args, **kwargs):
        super().__init__(parent, textvariable=textvariable, *args, **kwargs)
        self.validator = validator
        self.required = required
        self.textvariable = textvariable
        self.indicator = ttk.Label(parent, text="", width=2)

        # Validate on focus out
        self.bind("<FocusOut>", self._validate)
        # Also validate when text changes (for immediate feedback)
        textvariable.trace_add("write", lambda *args: self._validate())

        # Initial validation
        self.after(100, self._validate)

    def _validate(self, event=None):
        """Run validator and update visual indicator."""
        val = self.textvariable.get()

        # Empty is OK if not required
        if not self.required and not val.strip():
            self.indicator.config(text="", foreground="gray")
            self.is_valid = True
            return

        # Run validator
        valid = self.validator(val)
        self.is_valid = valid

        if valid:
            self.indicator.config(text="✓", foreground="green")
        else:
            self.indicator.config(text="✗", foreground="red")

    def grid(self, **kwargs):
        """Override grid to auto-place indicator to the right."""
        row = kwargs.get('row', 0)
        col = kwargs.get('column', 1)
        super().grid(**kwargs)
        self.indicator.grid(row=row, column=col+1, padx=2)
        return self


class ToolTip:
    """
    Create a tooltip for a given widget with hover support.
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)

    def on_enter(self, event=None):
        if self.tooltip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                        background="#ffffe0", relief='solid', borderwidth=1,
                        font=("sans-serif", "9", "normal"), padx=6, pady=4)
        label.pack()

    def on_leave(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None



class AccordionSection(ttk.Frame):
    """A collapsible section with a header and a body frame (self.body)."""
    def __init__(self, parent, title: str, open: bool=True, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.columnconfigure(0, weight=1)
        self._open = open
        self._title = title

        self._hdr = ttk.Frame(self)
        self._hdr.grid(row=0, column=0, sticky="ew")
        self._hdr.columnconfigure(0, weight=1)

        self._btn = ttk.Button(self._hdr, text=self._label_text(), command=self._toggle)
        self._btn.grid(row=0, column=0, sticky="ew")

        self.body = ttk.Frame(self)
        self.body.grid(row=1, column=0, sticky="nsew")

        if not self._open:
            self.body.grid_remove()

    def _label_text(self):
        return ("▼ " if self._open else "► ") + self._title

    def _toggle(self):
        self._open = not self._open
        if self._open:
            self.body.grid()
        else:
            self.body.grid_remove()
        self._btn.configure(text=self._label_text())

class VScrolledFrame(ttk.Frame):
    """A simple vertical scrolled container: use .interior as the parent for your widgets."""
    def __init__(self, parent, height=260, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, height=height)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")

        self.interior = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")

        def _configure_interior(event=None):
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=bbox)
            cw = self.canvas.winfo_width()
            iw = self.interior.winfo_reqwidth()
            if iw != cw:
                self.canvas.itemconfigure(self._win, width=cw)

        def _configure_canvas(event=None):
            _configure_interior()

        self.interior.bind("<Configure>", _configure_interior)
        self.canvas.bind("<Configure>", _configure_canvas)

        def _on_mousewheel(event):
            delta = event.delta
            if delta == 0:
                return
            self.canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        def _on_linux_up(event): self.canvas.yview_scroll(-1, "units")
        def _on_linux_down(event): self.canvas.yview_scroll(1, "units")

        def _bind_wheel(_):
            self.interior.bind_all("<MouseWheel>", _on_mousewheel)
            self.interior.bind_all("<Button-4>", _on_linux_up)
            self.interior.bind_all("<Button-5>", _on_linux_down)

        def _unbind_wheel(_):
            self.interior.unbind_all("<MouseWheel>")
            self.interior.unbind_all("<Button-4>")
            self.interior.unbind_all("<Button-5>")

        self.interior.bind("<Enter>", _bind_wheel)
        self.interior.bind("<Leave>", _unbind_wheel)

from studio.theme import apply_theme_base

LOGQ = queue.Queue()
log = setup_logger(LOGQ)

PRESET_FILE = Path(__file__).parent / "config" / "bulk_presets.json"
RECENT_PROJECTS_FILE = Path(__file__).parent / "config" / "recent_projects.json"
MAX_RECENT_PROJECTS = 10


def load_recent_projects() -> list:
    """Load list of recent project paths."""
    try:
        if RECENT_PROJECTS_FILE.exists():
            with open(RECENT_PROJECTS_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_recent_projects(projects: list):
    """Save list of recent project paths."""
    try:
        RECENT_PROJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RECENT_PROJECTS_FILE, 'w') as f:
            json.dump(projects[:MAX_RECENT_PROJECTS], f, indent=2)
    except Exception:
        pass


def add_recent_project(path: Path):
    """Add a project to recent projects list."""
    path_str = str(path.resolve())
    recent = load_recent_projects()
    # Remove if exists (to move to front)
    recent = [p for p in recent if p != path_str]
    recent.insert(0, path_str)
    save_recent_projects(recent[:MAX_RECENT_PROJECTS])

class TrafficStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Traffic Studio — Sampling · ROI · Autolabel · Train · Analyze")
        self.geometry("1280x780"); self.minsize(1100, 680)

        # ---- Apply consistent style (ROI Tool base / fallback) ----
        try:
            # If ROI tool exposes a style helper, prefer that
            from third_party.roi_tool_style import apply_roi_style
            apply_roi_style(self)
        except Exception:
            apply_theme_base(self)  # fallback theme for consistency

        # ---- Project Model ----
        self.project = ProjectModel()
        self.current_worker: Optional[Worker] = None  # Track running task

        # ---- State ----
        # Sampling
        self.video = tk.StringVar()
        self.sample_out = tk.StringVar(value="data/images/new_site")
        self.every = tk.StringVar(value="1.0")
        self.fps = tk.StringVar(value="")
        self.smethod = tk.StringVar(value="fps")     # fps/select
        self.stol = tk.StringVar(value="0.002")      # tolerance

        # ROI & Mask
        self.roi_json = tk.StringVar()
        self.mask_out = tk.StringVar(value="data/images/new_site_masked")
        self.crop_toggle = tk.BooleanVar(value=False)     # border crop
        self.crop_margin = tk.StringVar(value="0.10")
        self.crop_labels_dir = tk.StringVar(value="")     # optional

        # Autolabel
        self.auto_images = tk.StringVar(value=self.mask_out.get())
        self.imgsz = tk.StringVar(value="960")
        self.conf = tk.StringVar(value="0.70")
        self.iou = tk.StringVar(value="0.60")
        self.autolabel_out = tk.StringVar(value="runs/autolabel")
        self.classes_file = tk.StringVar(value="data/classes.txt")
        self.labels_dir = tk.StringVar(value="")                       # set after run
        self.autolabel_weights = tk.StringVar(value="")                # optional explicit .pt

        # Train
        self.train_data_yaml = tk.StringVar(value="data.yaml")
        self.train_weights = tk.StringVar(value="yolov8s.pt")
        self.train_out_name = tk.StringVar(value="traffic_v8s")

        # QA/QC
        self.qalabels_dir = tk.StringVar(value="")
        self.qa_delete_line = tk.StringVar(value="")
        self.qa_delete_class = tk.StringVar(value="")
        self.qa_delete_regex = tk.StringVar(value="")
        self.qa_delete_near = tk.StringVar(value="")
        self.qa_eps = tk.StringVar(value="0.0")
        self.qa_dry = tk.BooleanVar(value=True)

        # CLI bulk tool (wraps user's script)
        self.cli_glob = tk.StringVar(value="**/*.txt")
        self.cli_start = tk.StringVar(value="")
        self.cli_end = tk.StringVar(value="")
        self.cli_step = tk.StringVar(value="1")
        self.cli_prefix = tk.StringVar(value="frame_")
        self.cli_pad = tk.StringVar(value="5")
        self.cli_ext = tk.StringVar(value=".txt")
        self.cli_mode = tk.StringVar(value="delete")
        self.cli_lines = tk.StringVar(value="")         # for write modes
        self.cli_del_line = tk.StringVar(value="")      # for delete
        self.cli_del_class = tk.StringVar(value="")
        self.cli_del_regex = tk.StringVar(value="")
        self.cli_del_near = tk.StringVar(value="")
        self.cli_cli_eps = tk.StringVar(value="1e-6")
        self.cli_dry = tk.BooleanVar(value=True)
        self.cli_quiet = tk.BooleanVar(value=False)
        self.cli_backup = tk.BooleanVar(value=False)    # NEW: backup originals

        # Presets
        self.cli_preset_name = tk.StringVar(value="")
        self.cli_preset_sel  = tk.StringVar(value="")
        self._cli_presets_dir = Path.cwd() / "presets" / "bulk_tool"
        self.preset_name = tk.StringVar(value="")
        self.preset_selected = tk.StringVar(value="")
        self._ensure_preset_file()
        self._presets_cache = self._load_presets()

        # Bulk Tool Results
        self.bulk_files_modified = tk.StringVar(value="0")
        self.bulk_lines_removed = tk.StringVar(value="0")
        self.bulk_updated_files = tk.StringVar(value="0")
        self.bulk_dry_note = tk.StringVar(value="")

        # Splits (site-disjoint)
        self.split_images_root = tk.StringVar(value=self.mask_out.get())
        self.split_labels_root = tk.StringVar(value="")
        self.split_out_root = tk.StringVar(value="dataset_site_disjoint")
        self.split_site_regex = tk.StringVar(value="")
        self.split_train = tk.StringVar(value="0.7")
        self.split_val = tk.StringVar(value="0.2")
        self.split_test = tk.StringVar(value="0.1")

        # Analyze
        self.an_video = tk.StringVar()
        self.an_weights = tk.StringVar(value="")
        self.an_run = tk.StringVar(value="analysis_run")
        self.an_out = tk.StringVar(value="outputs")
        self.include_lanes = tk.BooleanVar(value=True)
        self.lane_pattern = tk.StringVar(value=r"(lane|_ln|\b[ltr]\b|through|left|right)")

        self._build_ui()
        self.after(100, self._drain_logs)
        self.after(500, self._update_workflow_status)  # Initial status update after UI is built

    # ----------------- Presets helpers -----------------
    def _ensure_preset_file(self):
        PRESET_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not PRESET_FILE.exists():
            PRESET_FILE.write_text("{}", encoding="utf-8")

    def _load_presets(self):
        try:
            return json.loads(PRESET_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_presets(self, d):
        PRESET_FILE.parent.mkdir(parents=True, exist_ok=True)
        PRESET_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")

    def _get_current_bulk_settings(self):
        return {
            "labels_dir": self.qalabels_dir.get().strip(),
            "glob": self.cli_glob.get().strip(),
            "start": self.cli_start.get().strip(),
            "end": self.cli_end.get().strip(),
            "step": self.cli_step.get().strip(),
            "prefix": self.cli_prefix.get().strip(),
            "pad": self.cli_pad.get().strip(),
            "ext": self.cli_ext.get().strip(),
            "mode": self.cli_mode.get().strip(),
            "lines": self.cli_lines.get().strip(),
            "delete_line": self.cli_del_line.get().strip(),
            "delete_class": self.cli_del_class.get().strip(),
            "delete_regex": self.cli_del_regex.get().strip(),
            "delete_near": self.cli_del_near.get().strip(),
            "eps": self.cli_cli_eps.get().strip(),
            "dry_run": bool(self.cli_dry.get()),
            "quiet": bool(self.cli_quiet.get()),
            "backup": bool(self.cli_backup.get()),
        }

    def _apply_bulk_settings(self, d):
        self.qalabels_dir.set(d.get("labels_dir",""))
        self.cli_glob.set(d.get("glob",""))
        self.cli_start.set(d.get("start",""))
        self.cli_end.set(d.get("end",""))
        self.cli_step.set(d.get("step","1"))
        self.cli_prefix.set(d.get("prefix","frame_"))
        self.cli_pad.set(d.get("pad","5"))
        self.cli_ext.set(d.get("ext",".txt"))
        self.cli_mode.set(d.get("mode","delete"))
        self.cli_lines.set(d.get("lines",""))
        self.cli_del_line.set(d.get("delete_line",""))
        self.cli_del_class.set(d.get("delete_class",""))
        self.cli_del_regex.set(d.get("delete_regex",""))
        self.cli_del_near.set(d.get("delete_near",""))
        self.cli_cli_eps.set(d.get("eps","1e-6"))
        self.cli_dry.set(d.get("dry_run", True))
        self.cli_quiet.set(d.get("quiet", False))
        self.cli_backup.set(d.get("backup", False))

    def _save_preset_clicked(self):
        name = self.preset_name.get().strip()
        if not name:
            messagebox.showerror("Save Preset", "Please provide a preset name.")
            return
        presets = self._load_presets()
        presets[name] = self._get_current_bulk_settings()
        self._save_presets(presets)
        self._presets_cache = presets
        self._refresh_presets_combo(select=name)
        messagebox.showinfo("Save Preset", f"Saved preset “{name}”.")

    def _load_preset_clicked(self):
        name = self.preset_selected.get().strip()
        presets = self._load_presets()
        if not name or name not in presets:
            messagebox.showerror("Load Preset", "Choose a preset to load.")
            return
        self._apply_bulk_settings(presets[name])
        messagebox.showinfo("Load Preset", f"Loaded preset “{name}”.")

    def _refresh_presets_combo(self, select=None):
        names = sorted(self._presets_cache.keys())
        self.preset_combo["values"] = names
        if select and select in names:
            self.preset_selected.set(select)

    # ----------------- Project Management -----------------
    def _build_menu(self):
        """Create menu bar with File menu for project management."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="New Project", command=self._new_project, accelerator="Ctrl+N")
        file_menu.add_command(label="Open Project...", command=self._open_project, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Project", command=self._save_project, accelerator="Ctrl+S")
        file_menu.add_command(label="Save Project As...", command=self._save_project_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()

        # Recent projects submenu
        self.recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Projects", menu=self.recent_menu)
        self._update_recent_menu()

        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Bind keyboard shortcuts
        self.bind("<Control-n>", lambda e: self._new_project())
        self.bind("<Control-o>", lambda e: self._open_project())
        self.bind("<Control-s>", lambda e: self._save_project())
        self.bind("<Control-Shift-S>", lambda e: self._save_project_as())

    def _update_recent_menu(self):
        """Refresh the recent projects menu."""
        self.recent_menu.delete(0, "end")
        recent = load_recent_projects()

        if not recent:
            self.recent_menu.add_command(label="(No recent projects)", state="disabled")
            return

        for proj_path in recent:
            p = Path(proj_path)
            if p.exists():
                self.recent_menu.add_command(
                    label=f"{p.name} — {p.parent}",
                    command=lambda path=p: self._open_project_file(path)
                )

    def _new_project(self):
        """Create a new project (reset state)."""
        if self.project.project_file or any(self.project.steps_completed.values()):
            response = messagebox.askyesnocancel(
                "New Project",
                "Save current project before creating a new one?"
            )
            if response is None:  # Cancel
                return
            if response:  # Yes
                self._save_project()

        self.project = ProjectModel()
        self._sync_ui_from_project()
        self.title("Traffic Studio — (New Project)")
        log.info("Created new project")

    def _open_project(self):
        """Open an existing project file."""
        path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("Traffic Studio Project", "*.traffproj"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            self._open_project_file(Path(path))

    def _open_project_file(self, path: Path):
        """Load project from file and sync UI."""
        try:
            self.project = ProjectModel.load(path)
            self._sync_ui_from_project()
            self._update_workflow_status()  # Update status indicators after loading
            add_recent_project(path)
            self._update_recent_menu()
            self.title(f"Traffic Studio — {path.name}")
            log.info(f"Loaded project: {path}")
        except Exception as e:
            messagebox.showerror("Open Project", f"Failed to load project:\n{e}")
            log.error(f"Failed to load project {path}: {e}")

    def _save_project(self):
        """Save current project (prompt for location if new)."""
        if not self.project.project_file:
            self._save_project_as()
        else:
            self._sync_project_from_ui()
            self.project.save()
            log.info(f"Saved project: {self.project.project_file}")

    def _save_project_as(self):
        """Save project to a new file."""
        path = filedialog.asksaveasfilename(
            title="Save Project As",
            defaultextension=".traffproj",
            filetypes=[("Traffic Studio Project", "*.traffproj"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            self._sync_project_from_ui()
            self.project.save(Path(path))
            add_recent_project(Path(path))
            self._update_recent_menu()
            self.title(f"Traffic Studio — {Path(path).name}")
            log.info(f"Saved project as: {path}")

    def _sync_ui_from_project(self):
        """Load project state into UI widgets."""
        p = self.project

        # Sampling
        self.video.set(p.video)
        self.sample_out.set(p.sample_out)
        self.every.set(p.every)
        self.fps.set(p.fps)
        self.smethod.set(p.smethod)
        self.stol.set(p.stol)

        # ROI & Mask
        self.roi_json.set(p.roi_json)
        self.mask_out.set(p.mask_out)
        self.crop_toggle.set(p.crop_toggle)
        self.crop_margin.set(p.crop_margin)
        self.crop_labels_dir.set(p.crop_labels_dir)

        # Autolabel
        self.auto_images.set(p.auto_images)
        self.imgsz.set(p.imgsz)
        self.conf.set(p.conf)
        self.iou.set(p.iou)
        self.autolabel_out.set(p.autolabel_out)
        self.classes_file.set(p.classes_file)
        self.labels_dir.set(p.labels_dir)
        self.autolabel_weights.set(p.autolabel_weights)

        # Train
        self.train_data_yaml.set(p.train_data_yaml)
        self.train_weights.set(p.train_weights)
        self.train_out_name.set(p.train_out_name)
        # Note: Train advanced params will be loaded when UI is built

        # QA/QC
        self.qalabels_dir.set(p.qalabels_dir)

        # Split
        # Note: These will be loaded when UI is built

        # Analyze
        # Note: These will be loaded when UI is built

    def _sync_project_from_ui(self):
        """Save current UI state to project model."""
        p = self.project

        # Sampling
        p.video = self.video.get()
        p.sample_out = self.sample_out.get()
        p.every = self.every.get()
        p.fps = self.fps.get()
        p.smethod = self.smethod.get()
        p.stol = self.stol.get()

        # ROI & Mask
        p.roi_json = self.roi_json.get()
        p.mask_out = self.mask_out.get()
        p.crop_toggle = self.crop_toggle.get()
        p.crop_margin = self.crop_margin.get()
        p.crop_labels_dir = self.crop_labels_dir.get()

        # Autolabel
        p.auto_images = self.auto_images.get()
        p.imgsz = self.imgsz.get()
        p.conf = self.conf.get()
        p.iou = self.iou.get()
        p.autolabel_out = self.autolabel_out.get()
        p.classes_file = self.classes_file.get()
        p.labels_dir = self.labels_dir.get()
        p.autolabel_weights = self.autolabel_weights.get()

        # Train
        p.train_data_yaml = self.train_data_yaml.get()
        p.train_weights = self.train_weights.get()
        p.train_out_name = self.train_out_name.get()

        # QA/QC
        p.qalabels_dir = self.qalabels_dir.get()

        # ... Add more fields as needed

    def _update_workflow_status(self):
        """
        Update all workflow status indicators based on current project state.
        Call this after loading a project, after completing a task, or when paths change.
        """
        # Update workflow state from UI
        self._sync_project_from_ui()
        self.project.detect_workflow_state()

        # Define color mapping for severity levels
        severity_colors = {
            "ok": "green",
            "info": "blue",
            "warning": "orange",
            "error": "red"
        }

        # Update each step's status indicator
        if hasattr(self, 'sample_status_label'):
            can_proceed, message, severity = PrerequisiteChecker.can_sample(self.project)
            self.sample_status_label.config(
                text=message,
                foreground=severity_colors.get(severity, "gray")
            )

        if hasattr(self, 'autolabel_status_label'):
            can_proceed, message, severity = PrerequisiteChecker.can_autolabel(self.project)
            self.autolabel_status_label.config(
                text=message,
                foreground=severity_colors.get(severity, "gray")
            )

        if hasattr(self, 'train_status_label'):
            can_proceed, message, severity = PrerequisiteChecker.can_train(self.project)
            self.train_status_label.config(
                text=message,
                foreground=severity_colors.get(severity, "gray")
            )

        if hasattr(self, 'analyze_status_label'):
            can_proceed, message, severity = PrerequisiteChecker.can_analyze(self.project)
            self.analyze_status_label.config(
                text=message,
                foreground=severity_colors.get(severity, "gray")
            )

    def _check_dependencies(self):
        """Check for required dependencies and warn user if missing."""
        warnings = []

        if not FieldValidator.check_ffmpeg():
            warnings.append("⚠ ffmpeg not found on PATH — Video sampling will not work.\n"
                           "Install ffmpeg and ensure it's on your system PATH.")

        if not FieldValidator.check_cuda():
            warnings.append("⚠ CUDA not available — Training and inference will use CPU (slow).\n"
                           "Install CUDA toolkit and compatible PyTorch for GPU acceleration.")

        if warnings:
            messagebox.showwarning(
                "Dependency Check",
                "The following issues were detected:\n\n" + "\n\n".join(warnings)
            )

    def _create_output_discovery_buttons(self, parent, output_path_var: tk.StringVar, row: int, col: int = 1):
        """Create 'Open Folder' and 'Copy Path' buttons for output discovery."""
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=row, column=col, sticky="w", padx=4, pady=4)

        def open_folder():
            path = Path(output_path_var.get())
            if path.exists():
                import platform
                if platform.system() == "Windows":
                    os.startfile(path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", path])
                else:  # Linux
                    subprocess.run(["xdg-open", path])
            else:
                messagebox.showwarning("Open Folder", f"Path does not exist:\n{path}")

        def copy_path():
            path_str = str(Path(output_path_var.get()).resolve())
            self.clipboard_clear()
            self.clipboard_append(path_str)
            messagebox.showinfo("Copy Path", f"Copied to clipboard:\n{path_str}")

        ttk.Button(btn_frame, text="📁 Open Folder", command=open_folder).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="📋 Copy Path", command=copy_path).pack(side="left", padx=2)

        return btn_frame

    # ----------------- UI -----------------
    def _build_ui(self):
        # Create menu bar
        self._build_menu()

        # Check dependencies on startup
        self._check_dependencies()

        # Root grid: one column, two rows (tabs on top, log at bottom)
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)
        main = ttk.Frame(self); main.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        main.columnconfigure(0, weight=1); main.rowconfigure(0, weight=1); main.rowconfigure(1, weight=0)

        # Tabs occupy the top row
        self.steps = ttk.Notebook(main)
        self.steps.grid(row=0, column=0, sticky="nsew")
        self.steps.add(self._step_sample(self.steps), text="1) Sample")
        self.steps.add(self._step_roi_mask(self.steps), text="2) ROI & Mask")
        self.steps.add(self._step_autolabel(self.steps), text="3) Autolabel")
        self.steps.add(self._step_qaqc(self.steps), text="4) QA/QC + Splits")
        self.steps.add(self._step_train(self.steps), text="5) Train")
        self.steps.add(self._step_analyze(self.steps), text="6) Analyze")

        # Log console spans full width at the bottom row
        console = ttk.LabelFrame(main, text="Process Log")
        console.grid(row=1, column=0, sticky="nsew", pady=(8,0))
        console.columnconfigure(0, weight=1); console.rowconfigure(0, weight=1)
        self.logbox = tk.Text(console, height=12, wrap="none")
        self.logbox.grid(row=0, column=0, sticky="nsew")
        ybar = ttk.Scrollbar(console, orient="vertical", command=self.logbox.yview)
        ybar.grid(row=0, column=1, sticky="ns")
        self.logbox.configure(yscrollcommand=ybar.set)

    # ---------- Step 1: Sample ----------
    def _step_sample(self, parent):
        f = ttk.Frame(parent); f.columnconfigure(1, weight=1)

        # Video/Folder input with validation
        lbl_video = ttk.Label(f, text="Video or Folder")
        lbl_video.grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_video, "Select a single video file (.mp4, .mov, .avi) or a folder containing multiple videos.")
        self.ent_video_sample = ValidatedEntry(
            f, self.video,
            validator=lambda p: FieldValidator.path_exists(p),
            required=True
        )
        self.ent_video_sample.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_file_or_dir(self.video)).grid(row=0, column=3, padx=4)

        # Output directory (create if doesn't exist, so validate parent exists)
        lbl_out = ttk.Label(f, text="Output Images")
        lbl_out.grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_out, "Directory where sampled frames will be saved as images.\nExample: data/images/site_north")
        ent_out = ttk.Entry(f, textvariable=self.sample_out)
        ent_out.grid(row=1, column=1, sticky="ew", padx=4, columnspan=2)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_dir(self.sample_out)).grid(row=1, column=3, padx=4)

        # Output discovery buttons (initially hidden, shown after successful sampling)
        self.sample_output_discovery = self._create_output_discovery_buttons(f, self.sample_out, row=2, col=1)
        self.sample_output_discovery.grid_remove()  # Hide until after sampling

        # Interval validation
        lbl_interval = ttk.Label(f, text="Interval (s)")
        lbl_interval.grid(row=3, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_interval, "Time interval in seconds between sampled frames.\nExample: 1.0 = one frame per second")
        self.ent_interval = ValidatedEntry(
            f, self.every,
            validator=lambda v: FieldValidator.is_float(v, min_val=0.01) if v else True,
            required=False, width=10
        )
        self.ent_interval.grid(row=3, column=1, sticky="w")

        lbl_fps = ttk.Label(f, text="OR FPS")
        lbl_fps.grid(row=4, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_fps, "Alternative: specify frames per second to sample.\nLeave blank to use Interval instead.")
        self.ent_fps = ValidatedEntry(
            f, self.fps,
            validator=lambda v: FieldValidator.is_float(v, min_val=0.1) if v else True,
            required=False, width=10
        )
        self.ent_fps.grid(row=4, column=1, sticky="w")

        # Progress bar (hidden until task starts)
        self.sample_progress = ttk.Progressbar(f, mode='indeterminate', length=300)
        self.sample_progress.grid(row=5, column=1, sticky="ew", padx=4, pady=4)
        self.sample_progress.grid_remove()

        self.sample_progress_label = ttk.Label(f, text="", foreground="blue")
        self.sample_progress_label.grid(row=6, column=1, sticky="w", padx=4)
        self.sample_progress_label.grid_remove()

        # Status indicator
        self.sample_status_label = ttk.Label(f, text="", font=("", 9))
        self.sample_status_label.grid(row=7, column=1, sticky="w", padx=4, pady=(4,0))

        # Buttons: Start and Cancel
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=8, column=1, sticky="e", pady=8)

        self.btn_sample = ttk.Button(btn_frame, text="Start Sampling", command=self._do_sample)
        self.btn_sample.pack(side="left", padx=4)
        ToolTip(self.btn_sample, "Extract frames from video(s) using ffmpeg.\nFrames will be saved to the output directory.")

        self.btn_sample_cancel = ttk.Button(btn_frame, text="Cancel", command=self._cancel_current_task, state="disabled")
        self.btn_sample_cancel.pack(side="left", padx=4)

        # Advanced (sampler method/tolerance)
        adv = ttk.LabelFrame(f, text="Advanced")
        adv.grid(row=5, column=0, columnspan=3, sticky="ew", padx=4, pady=(6,4))
        lbl_method = ttk.Label(adv, text="Method")
        lbl_method.grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_method, "Sampling method:\n• fps: Standard frame rate sampling\n• select: Time-modulo selection for precise control")
        rb_fps = ttk.Radiobutton(adv, text="fps", value="fps", variable=self.smethod)
        rb_fps.grid(row=0, column=1, sticky="w")
        rb_select = ttk.Radiobutton(adv, text="select (t-mod)", value="select", variable=self.smethod)
        rb_select.grid(row=0, column=2, sticky="w")

        lbl_tol = ttk.Label(adv, text="Tolerance (s)")
        lbl_tol.grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_tol, "Timing tolerance in seconds for frame selection.\nDefault: 0.002 (2 milliseconds)")
        ent_tol = ttk.Entry(adv, width=10, textvariable=self.stol)
        ent_tol.grid(row=1, column=1, sticky="w")
        return f

    def _do_sample(self):
        """Run frame sampling with validation and progress tracking."""
        # Check prerequisites with soft warnings
        can_proceed, message, severity = PrerequisiteChecker.can_sample(self.project)
        if not can_proceed:
            if severity == "error":
                messagebox.showerror("Cannot Sample", message)
                return
            elif severity == "warning":
                if not messagebox.askyesno("Warning", f"{message}\n\nProceed anyway?"):
                    return

        # Validate inputs
        if not self.ent_video_sample.is_valid:
            messagebox.showerror("Sampling", "Video/folder path is invalid or does not exist.")
            return

        if not self.every.get() and not self.fps.get():
            messagebox.showerror("Sampling", "Specify either Interval or FPS.")
            return

        if self.every.get() and not self.ent_interval.is_valid:
            messagebox.showerror("Sampling", "Interval must be a positive number.")
            return

        if self.fps.get() and not self.ent_fps.is_valid:
            messagebox.showerror("Sampling", "FPS must be a positive number.")
            return

        # Prepare arguments
        video = Path(self.video.get())
        out = Path(self.sample_out.get())
        every = float(self.every.get()) if self.every.get() else None
        fps = float(self.fps.get()) if self.fps.get() else None
        method = self.smethod.get()
        tol = float(self.stol.get() or "0.002")

        # Run with progress tracking
        def on_success_sample(res):
            self.project.steps_completed.update({"sample": True})
            self.project.workflow_state["has_sampled_frames"] = True
            self.project.workflow_state["sample_output_dir"] = str(out)
            self._update_workflow_status()

        self._run_task(
            lambda: sampler.sample_frames(video, out, every=every, fps=fps, method=method, tolerance=tol),
            "Sampling",
            on_success=on_success_sample,
            progress_bar=self.sample_progress,
            progress_label=self.sample_progress_label,
            run_button=self.btn_sample,
            cancel_button=self.btn_sample_cancel,
            output_discovery=self.sample_output_discovery
        )

    # ---------- Step 2: ROI & Mask ----------
    def _step_roi_mask(self, parent):
        f = ttk.Frame(parent); f.columnconfigure(1, weight=1)
        lbl_roi = ttk.Label(f, text="ROI JSON (template or legacy)")
        lbl_roi.grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_roi, "JSON file defining regions of interest (ROI) for detection zones.\nCreate this using the ROI Tool below.")
        ent_roi = ttk.Entry(f, textvariable=self.roi_json)
        ent_roi.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_file(self.roi_json,[("JSON","*.json")])).grid(row=0, column=2, padx=4)

        lbl_mask_out = ttk.Label(f, text="Mask Output Folder")
        lbl_mask_out.grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_mask_out, "Directory where masked images will be saved.\nMasking applies ROI boundaries to focus on specific areas.")
        ent_mask_out = ttk.Entry(f, textvariable=self.mask_out)
        ent_mask_out.grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_dir(self.mask_out)).grid(row=1, column=2, padx=4)

        # Border crop & relabel
        crop_box = ttk.LabelFrame(f, text="Border Crop (optional)")
        crop_box.grid(row=2, column=0, columnspan=3, sticky="ew", padx=4, pady=(6,4))
        self.crop_toggle = tk.BooleanVar(value=False)
        chk_crop = ttk.Checkbutton(crop_box, text="Border crop & relabel before masking", variable=self.crop_toggle)
        chk_crop.grid(row=0, column=0, columnspan=3, sticky="w", padx=4)
        ToolTip(chk_crop, "Crop image borders before masking to remove edge artifacts.\nAutomatically adjusts any existing labels.")

        lbl_margin = ttk.Label(crop_box, text="Margin (fraction)")
        lbl_margin.grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_margin, "Fraction of image to crop from borders (0.0-1.0).\nExample: 0.10 = remove 10% from each edge")
        ent_margin = ttk.Entry(crop_box, width=10, textvariable=self.crop_margin)
        ent_margin.grid(row=1, column=1, sticky="w")

        lbl_crop_lbl = ttk.Label(crop_box, text="Labels (optional)")
        lbl_crop_lbl.grid(row=2, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_crop_lbl, "Optional: Directory containing YOLO labels to adjust during cropping.\nLeave blank if no labels exist yet.")
        ent_crop_lbl = ttk.Entry(crop_box, width=32, textvariable=self.crop_labels_dir)
        ent_crop_lbl.grid(row=2, column=1, sticky="ew")
        ttk.Button(crop_box, text="Browse…", command=lambda: self._choose_dir(self.crop_labels_dir)).grid(row=2, column=2)

        btn_roi_tool = ttk.Button(f, text="Open ROI Tool", command=self._launch_roi_tool)
        btn_roi_tool.grid(row=3, column=1, sticky="w", pady=6)
        ToolTip(btn_roi_tool, "Launch interactive ROI editor to draw detection zones on video frames.\nSaves ROI configuration to JSON file.")

        btn_mask = ttk.Button(f, text="Apply Mask to Sampled Images", command=self._do_mask)
        btn_mask.grid(row=4, column=1, sticky="e", pady=6)
        ToolTip(btn_mask, "Apply ROI mask to all sampled images.\nPixels outside ROI zones will be masked out.")

        btn_one_click = ttk.Button(f, text="One-Click: Crop → Mask → Autolabel", command=self._do_crop_mask_autolabel)
        btn_one_click.grid(row=5, column=1, sticky="e", pady=10)
        ToolTip(btn_one_click, "Automated workflow: crop borders, apply mask, then run autolabeling.\nRequires ROI JSON to be configured.")
        return f

    def _launch_roi_tool(self):
        vid_or_img = self.video.get() or filedialog.askopenfilename(
            title="Select a video or image",
            filetypes=[("Media","*.mp4 *.mov *.mkv *.avi *.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All","*.*")]
        )
        if not vid_or_img:
            return
        args = [sys.executable, "-u", "third_party/roi_tool_gui.py", vid_or_img]
        try:
            subprocess.Popen(args)
            self.focus_set()
        except Exception as e:
            messagebox.showerror("ROI Tool", f"Failed to launch ROI tool:\n{e}")

    def _do_mask(self):
        from studio import roi_utils
        import cv2
        src_images = Path(self.sample_out.get())
        images_for_mask = src_images
        if self.crop_toggle.get():
            margin = float(self.crop_margin.get() or "0.10")
            labels_dir = Path(self.crop_labels_dir.get()) if self.crop_labels_dir.get() else None
            cropped_dir = src_images.with_name(src_images.name + "_cropped")
            out_lbl = None
            if labels_dir:
                out_lbl = cropped_dir.parent / "labels" / cropped_dir.name
            n_img, n_lbl = roi_utils.crop_margin_and_relabel(src_images, cropped_dir, labels_dir, out_lbl, margin_frac=margin)
            log.info("Border-cropped %d images, rewrote %d label files -> %s", n_img, n_lbl, cropped_dir)
            images_for_mask = cropped_dir

        # Build union mask and apply to folder
        probe = next(iter(sorted(images_for_mask.glob("*.jpg"))), None) or next(iter(sorted(images_for_mask.glob("*.png"))), None)
        if not probe:
            messagebox.showwarning("No images", f"No images in {images_for_mask}"); return
        im = cv2.imread(str(probe)); h, w = im.shape[:2]
        rois = roi_utils.load_roi_any(Path(self.roi_json.get()))
        flat = roi_utils.normalize_for_tracking(rois, w, h)
        mask = roi_utils.mask_from_rois(flat, w, h)
        out_dir = Path(self.mask_out.get()); out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(images_for_mask.glob("*.*")):
            if p.suffix.lower() not in (".jpg",".jpeg",".png",".bmp",".tif",".tiff"): continue
            roi_utils.apply_mask_to_image(p, out_dir / p.name, mask)
        log.info("Masked images -> %s", out_dir)

    def _do_crop_mask_autolabel(self):
        self._do_mask()
        self.auto_images.set(self.mask_out.get())
        self._do_autolabel()

    # ---------- Step 3: Autolabel ----------
    def _step_autolabel(self, parent):
        f = ttk.Frame(parent); f.columnconfigure(1, weight=1)

        # Images folder with validation
        lbl_auto_img = ttk.Label(f, text="Images Folder")
        lbl_auto_img.grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_auto_img, "Directory containing images to label automatically.\nUse masked images for better results in specific ROIs.")
        self.ent_auto_img = ValidatedEntry(
            f, self.auto_images,
            validator=lambda p: FieldValidator.path_exists(p, must_be_dir=True),
            required=True
        )
        self.ent_auto_img.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_dir(self.auto_images)).grid(row=0, column=3)

        # Confidence threshold (always visible - most important parameter)
        lbl_conf = ttk.Label(f, text="Confidence ≥")
        lbl_conf.grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_conf, "Minimum confidence threshold (0.0-1.0).\nOnly detections above this score are kept.\nRecommended: 0.70 or higher for autolabeling")
        self.ent_conf = ValidatedEntry(
            f, self.conf,
            validator=lambda v: FieldValidator.is_float(v, min_val=0.0, max_val=1.0),
            required=True, width=10
        )
        self.ent_conf.grid(row=1, column=1, sticky="w")

        # Weights (optional, visible)
        lbl_weights = ttk.Label(f, text="Weights (.pt)")
        lbl_weights.grid(row=2, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_weights, "YOLO model weights file (.pt) for detection.\nLeave blank to use default YOLOv8 model.\nUse custom trained models for better accuracy.")
        ent_weights = ttk.Entry(f, textvariable=self.autolabel_weights)
        ent_weights.grid(row=2, column=1, sticky="ew", padx=4, columnspan=2)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_file(self.autolabel_weights,[("PT","*.pt")])).grid(row=2, column=3)

        # Advanced inference options (collapsed by default)
        adv_infer = AccordionSection(f, "Advanced Inference Options", open=False)
        adv_infer.grid(row=3, column=0, columnspan=4, sticky="ew", padx=4, pady=(6,4))
        adv = adv_infer.body
        adv.columnconfigure(1, weight=1)

        lbl_imgsz = ttk.Label(adv, text="Image size")
        lbl_imgsz.grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_imgsz, "Image size for inference (width/height in pixels).\nHigher values = more detail but slower.\nRecommended: 640-960")
        self.ent_imgsz = ValidatedEntry(
            adv, self.imgsz,
            validator=lambda v: FieldValidator.is_int(v, min_val=320, max_val=2048),
            required=True, width=10
        )
        self.ent_imgsz.grid(row=0, column=1, sticky="w")

        lbl_iou = ttk.Label(adv, text="IOU threshold")
        lbl_iou.grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_iou, "Intersection over Union threshold for NMS (0.0-1.0).\nHigher values = allow more overlapping boxes.\nRecommended: 0.50-0.70")
        self.ent_iou = ValidatedEntry(
            adv, self.iou,
            validator=lambda v: FieldValidator.is_float(v, min_val=0.0, max_val=1.0),
            required=True, width=10
        )
        self.ent_iou.grid(row=1, column=1, sticky="w")

        # Output discovery (hidden until after run)
        self.autolabel_output_discovery = self._create_output_discovery_buttons(f, self.labels_dir, row=4, col=1)
        self.autolabel_output_discovery.grid_remove()

        # Progress bar
        self.autolabel_progress = ttk.Progressbar(f, mode='indeterminate', length=300)
        self.autolabel_progress.grid(row=5, column=1, sticky="ew", padx=4, pady=4, columnspan=2)
        self.autolabel_progress.grid_remove()

        self.autolabel_progress_label = ttk.Label(f, text="", foreground="blue")
        self.autolabel_progress_label.grid(row=6, column=1, sticky="w", padx=4, columnspan=2)
        self.autolabel_progress_label.grid_remove()

        # Status indicator
        self.autolabel_status_label = ttk.Label(f, text="", font=("", 9))
        self.autolabel_status_label.grid(row=7, column=1, sticky="w", padx=4, pady=(4,0), columnspan=2)

        # Buttons: Run and Cancel
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=8, column=1, sticky="e", pady=8, columnspan=2)

        self.btn_autolabel = ttk.Button(btn_frame, text="Run Autolabel", command=self._do_autolabel)
        self.btn_autolabel.pack(side="left", padx=4)
        ToolTip(self.btn_autolabel, "Generate YOLO format labels automatically using the model.\nLabels will be saved in runs/autolabel/*/labels/")

        self.btn_autolabel_cancel = ttk.Button(btn_frame, text="Cancel", command=self._cancel_current_task, state="disabled")
        self.btn_autolabel_cancel.pack(side="left", padx=4)

        # LabelImg helper
        li = ttk.LabelFrame(f, text="LabelImg")
        li.grid(row=6, column=0, columnspan=3, sticky="ew", padx=4, pady=(8,2))
        lbl_classes = ttk.Label(li, text="classes.txt")
        lbl_classes.grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_classes, "Text file containing class names (one per line).\nRequired for LabelImg to display class labels properly.")
        ent_classes = ttk.Entry(li, textvariable=self.classes_file)
        ent_classes.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(li, text="Browse…", command=lambda: self._choose_file(self.classes_file,[("TXT","*.txt")])).grid(row=0, column=2)

        lbl_labels = ttk.Label(li, text="Labels dir")
        lbl_labels.grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_labels, "Directory containing YOLO label files (.txt).\nAuto-populated after running autolabel.")
        ent_labels = ttk.Entry(li, textvariable=self.labels_dir)
        ent_labels.grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(li, text="Browse…", command=lambda: self._choose_dir(self.labels_dir)).grid(row=1, column=2)

        btn_labelimg = ttk.Button(li, text="Launch LabelImg", command=self._launch_labelimg)
        btn_labelimg.grid(row=2, column=1, sticky="e", pady=6)
        ToolTip(btn_labelimg, "Open LabelImg for manual review and correction of labels.\nUse this to fix autolabel errors and add missing annotations.")
        return f

    def _do_autolabel(self):
        """Run autolabeling with validation and progress tracking."""
        # Check prerequisites with soft warnings
        can_proceed, message, severity = PrerequisiteChecker.can_autolabel(self.project)
        if not can_proceed:
            if severity == "error":
                messagebox.showerror("Cannot Autolabel", message)
                return
            elif severity == "warning":
                if not messagebox.askyesno("Warning", f"{message}\n\nProceed anyway?"):
                    return

        # Validate inputs
        if not self.ent_auto_img.is_valid:
            messagebox.showerror("Autolabel", "Images folder path is invalid or does not exist.")
            return

        if not self.ent_conf.is_valid:
            messagebox.showerror("Autolabel", "Confidence must be between 0.0 and 1.0.")
            return

        conf = float(self.conf.get())
        if conf < 0.70:
            if not messagebox.askyesno("Low Confidence", "Confidence threshold < 0.70 may produce many false positives.\nRecommended: 0.70 or higher.\n\nProceed anyway?"):
                return

        if not self.ent_imgsz.is_valid or not self.ent_iou.is_valid:
            messagebox.showerror("Autolabel", "Check advanced inference parameters (imgsz, IOU).")
            return

        # Prepare arguments
        images_dir = Path(self.auto_images.get())
        model_path = self.autolabel_weights.get().strip() or None
        imgsz = int(self.imgsz.get())
        iou = float(self.iou.get())

        def _run():
            res = autolabel.autolabel(images_dir, model_path=model_path, imgsz=imgsz, conf=conf, iou=iou)
            base = Path("runs/autolabel")
            latest = sorted(base.glob("*/labels"), key=lambda p: p.stat().st_mtime)
            if latest:
                self.labels_dir.set(str(latest[-1]))
                log.info("Latest labels -> %s", latest[-1])
            return res

        def on_success_autolabel(res):
            self.project.steps_completed.update({"autolabel": True})
            self.project.workflow_state["has_labels"] = True
            if self.labels_dir.get():
                self.project.workflow_state["labels_dir"] = self.labels_dir.get()
            self._update_workflow_status()

        self._run_task(
            _run,
            "Autolabel",
            on_success=on_success_autolabel,
            progress_bar=self.autolabel_progress,
            progress_label=self.autolabel_progress_label,
            run_button=self.btn_autolabel,
            cancel_button=self.btn_autolabel_cancel,
            output_discovery=self.autolabel_output_discovery
        )

    def _launch_labelimg(self):
        labels_dir = self.labels_dir.get() or filedialog.askdirectory(title="Choose labels directory")
        if not labels_dir:
            return
        classes = self.classes_file.get()
        args = [sys.executable, "-m", "labelImg", labels_dir, classes] if classes else [sys.executable, "-m", "labelImg", labels_dir]
        try:
            subprocess.Popen(args)
        except Exception as e:
            messagebox.showerror("LabelImg", f"Failed to launch LabelImg:\n{e}")

    # ---------- Step 4: QA/QC + Splits ----------

    def _step_qaqc(self, parent):
        f = ttk.Frame(parent);
        f.columnconfigure(0, weight=1)
        # --- Section 1: Label QA/Cleanup (native) ---
        sect_qa = AccordionSection(f, "Label QA/Cleanup (native)", open=True)
        sect_qa.grid(row=0, column=0, sticky="ew", padx=4, pady=(6,4))
        qa = sect_qa.body
        qa.columnconfigure(1, weight=1)

        lbl_qa_dir = ttk.Label(qa, text="Labels dir")
        lbl_qa_dir.grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_qa_dir, "Directory containing YOLO label files to validate/clean.\nTypically the output from autolabeling.")
        ent_qa_dir = ttk.Entry(qa, textvariable=self.qalabels_dir)
        ent_qa_dir.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(qa, text="Browse…", command=lambda: self._choose_dir(self.qalabels_dir)).grid(row=0, column=2)
        btn_validate = ttk.Button(qa, text="Validate", command=self._do_validate)
        btn_validate.grid(row=0, column=3, padx=8)
        ToolTip(btn_validate, "Check all label files for formatting errors and invalid values.\nReports issues in the log.")

        lbl_del_line = ttk.Label(qa, text="Delete: exact line(s)")
        lbl_del_line.grid(row=1, column=0, sticky="e")
        ToolTip(lbl_del_line, "Remove exact matching lines from labels.\nEnter YOLO format lines (e.g., '0 0.5 0.5 0.1 0.1')\nSeparate multiple lines with newlines.")
        ent_del_line = ttk.Entry(qa, textvariable=self.qa_delete_line, width=50)
        ent_del_line.grid(row=1, column=1, columnspan=3, sticky="ew", padx=4)

        lbl_del_class = ttk.Label(qa, text="Delete: class id(s)")
        lbl_del_class.grid(row=2, column=0, sticky="e")
        ToolTip(lbl_del_class, "Remove all labels with specific class IDs.\nEnter comma or space-separated IDs (e.g., '0,2,5')")
        ent_del_class = ttk.Entry(qa, textvariable=self.qa_delete_class, width=20)
        ent_del_class.grid(row=2, column=1, sticky="w", padx=4)

        lbl_regex = ttk.Label(qa, text="Regex")
        lbl_regex.grid(row=2, column=2, sticky="e")
        ToolTip(lbl_regex, "Remove lines matching regular expression pattern.\nUseful for complex filtering (e.g., '^0 0\\.[89].*')")
        ent_regex = ttk.Entry(qa, textvariable=self.qa_delete_regex, width=24)
        ent_regex.grid(row=2, column=3, sticky="w")

        lbl_del_near = ttk.Label(qa, text="Delete-near line")
        lbl_del_near.grid(row=3, column=0, sticky="e")
        ToolTip(lbl_del_near, "Remove labels within epsilon distance of this reference line.\nUseful for removing duplicates or nearby false positives.")
        ent_del_near = ttk.Entry(qa, textvariable=self.qa_delete_near, width=50)
        ent_del_near.grid(row=3, column=1, columnspan=2, sticky="ew", padx=4)

        lbl_eps = ttk.Label(qa, text="eps")
        lbl_eps.grid(row=3, column=3, sticky="e")
        ToolTip(lbl_eps, "Epsilon (distance threshold) for delete-near operation.\nSmaller values = stricter matching.\nDefault: 0.0 (exact match)")
        ent_eps = ttk.Entry(qa, width=10, textvariable=self.qa_eps)
        ent_eps.grid(row=3, column=4, sticky="w")

        chk_dry = ttk.Checkbutton(qa, text="Dry run", variable=self.qa_dry)
        chk_dry.grid(row=4, column=1, sticky="w")
        ToolTip(chk_dry, "Preview changes without modifying files.\nAlways test with dry run first!")
        btn_cleanup = ttk.Button(qa, text="Run Cleanup", command=self._do_cleanup)
        btn_cleanup.grid(row=4, column=3, sticky="e", pady=4)
        ToolTip(btn_cleanup, "Apply selected cleanup operations to all label files.\nUse dry run first to preview changes!")

        # --- Section 2: Site-Disjoint Splits ---
        sect_sp = AccordionSection(f, "Site-Disjoint Splits & data.yaml", open=False)
        sect_sp.grid(row=1, column=0, sticky="ew", padx=4, pady=(6,4))
        sp = sect_sp.body
        sp.columnconfigure(1, weight=1)

        lbl_img_root = ttk.Label(sp, text="Images root")
        lbl_img_root.grid(row=0, column=0, sticky="e")
        ToolTip(lbl_img_root, "Root directory containing all images to split.\nAll images in this folder will be organized into train/val/test.")
        ent_img_root = ttk.Entry(sp, textvariable=self.split_images_root)
        ent_img_root.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(sp, text="Browse…", command=lambda: self._choose_dir(self.split_images_root)).grid(row=0, column=2)

        lbl_lbl_root = ttk.Label(sp, text="Labels root")
        lbl_lbl_root.grid(row=1, column=0, sticky="e")
        ToolTip(lbl_lbl_root, "Root directory containing corresponding label files.\nLeave blank to auto-detect from images path.")
        ent_lbl_root = ttk.Entry(sp, textvariable=self.split_labels_root)
        ent_lbl_root.grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(sp, text="Browse…", command=lambda: self._choose_dir(self.split_labels_root)).grid(row=1, column=2)

        lbl_out_root = ttk.Label(sp, text="Output dataset dir")
        lbl_out_root.grid(row=2, column=0, sticky="e")
        ToolTip(lbl_out_root, "Output directory for organized dataset.\nWill create images/ and labels/ subdirectories with train/val/test splits.")
        ent_out_root = ttk.Entry(sp, textvariable=self.split_out_root)
        ent_out_root.grid(row=2, column=1, sticky="ew", padx=4)
        ttk.Button(sp, text="Browse…", command=lambda: self._choose_dir(self.split_out_root)).grid(row=2, column=2)

        lbl_site_regex = ttk.Label(sp, text="Site regex (group 1 captures site id)")
        lbl_site_regex.grid(row=3, column=0, sticky="e")
        ToolTip(lbl_site_regex, "Regular expression to extract site identifier from filenames.\nExample: r'site_(\\w+)_' captures site name.\nEnsures same site doesn't appear in train and test.")
        ent_site_regex = ttk.Entry(sp, textvariable=self.split_site_regex)
        ent_site_regex.grid(row=3, column=1, sticky="ew", padx=4)

        lbl_ratios = ttk.Label(sp, text="Ratios (train/val/test)")
        lbl_ratios.grid(row=4, column=0, sticky="e")
        ToolTip(lbl_ratios, "Split ratios for train/validation/test sets.\nMust sum to 1.0 (e.g., 0.7/0.2/0.1)")
        row4 = ttk.Frame(sp); row4.grid(row=4, column=1, sticky="w")
        ttk.Entry(row4, width=6, textvariable=self.split_train).pack(side="left")
        ttk.Label(row4, text="/").pack(side="left")
        ttk.Entry(row4, width=6, textvariable=self.split_val).pack(side="left")
        ttk.Label(row4, text="/").pack(side="left")
        ttk.Entry(row4, width=6, textvariable=self.split_test).pack(side="left")

        btn_splits = ttk.Button(sp, text="Make Splits + data.yaml", command=self._do_splits)
        btn_splits.grid(row=5, column=1, sticky="e", pady=6)
        ToolTip(btn_splits, "Create train/val/test splits and generate YOLO data.yaml config.\nPreserves site boundaries to prevent data leakage.")
        btn_fhwa = ttk.Button(sp, text="Write FHWA mapping template", command=self._do_write_fhwa_template)
        btn_fhwa.grid(row=5, column=2, sticky="w", pady=6)
        ToolTip(btn_fhwa, "Generate template CSV for FHWA vehicle classification mapping.\nMaps YOLO classes to FHWA standard categories.")

        # --- Section 3: Bulk Write/Delete (scrollable) ---
        sect_cli = AccordionSection(f, "Bulk Write/Delete (wraps your CLI tool)", open=True)
        sect_cli.grid(row=2, column=0, sticky="nsew", padx=4, pady=(6,4))
        f.rowconfigure(2, weight=1)  # let this section expand
        # Put a scrollable frame inside the body
        sect_cli.body.columnconfigure(0, weight=1)
        sf = VScrolledFrame(sect_cli.body, height=280)
        sf.grid(row=0, column=0, sticky="nsew")
        cli = sf.interior
        for i in range(8): cli.columnconfigure(i, weight=1)

        lbl_mode = ttk.Label(cli, text="Mode")
        lbl_mode.grid(row=0, column=0, sticky="e")
        ToolTip(lbl_mode, "Operation mode:\n• append: Add lines to files\n• overwrite: Replace file contents\n• delete: Remove matching lines")
        cmb_mode = ttk.Combobox(cli, textvariable=self.cli_mode, values=("append","overwrite","delete"), state="readonly", width=12)
        cmb_mode.grid(row=0, column=1, sticky="w")

        lbl_glob = ttk.Label(cli, text="Glob (preferred)")
        lbl_glob.grid(row=1, column=0, sticky="e")
        ToolTip(lbl_glob, "Glob pattern to match files (e.g., '**/*.txt' for all .txt files).\nPreferred over Start/End/Step for flexible file selection.")
        ent_glob = ttk.Entry(cli, textvariable=self.cli_glob)
        ent_glob.grid(row=1, column=1, columnspan=2, sticky="ew", padx=4)

        lbl_range = ttk.Label(cli, text="OR Start/End/Step")
        lbl_range.grid(row=1, column=3, sticky="e")
        ToolTip(lbl_range, "Alternative: specify numeric file range.\nExample: Start=0, End=100, Step=1\nUses Prefix/Pad/Ext to build filenames.")
        rowr = ttk.Frame(cli); rowr.grid(row=1, column=4, columnspan=2, sticky="w")
        ttk.Entry(rowr, width=6, textvariable=self.cli_start).pack(side="left")
        ttk.Label(rowr, text="-").pack(side="left")
        ttk.Entry(rowr, width=6, textvariable=self.cli_end).pack(side="left")
        ttk.Label(rowr, text=" / ").pack(side="left")
        ttk.Entry(rowr, width=4, textvariable=self.cli_step).pack(side="left")

        lbl_prefix = ttk.Label(cli, text="Prefix/Pad/Ext")
        lbl_prefix.grid(row=2, column=0, sticky="e")
        ToolTip(lbl_prefix, "Filename template for numeric range:\n• Prefix: filename prefix (e.g., 'frame_')\n• Pad: zero-padding width (e.g., 5 → 00001)\n• Ext: file extension (e.g., '.txt')")
        rowp = ttk.Frame(cli); rowp.grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Entry(rowp, width=10, textvariable=self.cli_prefix).pack(side="left")
        ttk.Entry(rowp, width=5, textvariable=self.cli_pad).pack(side="left")
        ttk.Entry(rowp, width=8, textvariable=self.cli_ext).pack(side="left")

        # Write lines
        lbl_write_lines = ttk.Label(cli, text="Line(s) for write (comma or newline separated)")
        lbl_write_lines.grid(row=3, column=0, sticky="e")
        ToolTip(lbl_write_lines, "YOLO format lines to write in append/overwrite modes.\nExample: '0 0.5 0.5 0.1 0.1' or multiple lines separated by commas/newlines.")
        ent_write_lines = ttk.Entry(cli, textvariable=self.cli_lines)
        ent_write_lines.grid(row=3, column=1, columnspan=7, sticky="ew", padx=4)

        # Delete selectors
        lbl_del_exact = ttk.Label(cli, text="Delete: exact lines")
        lbl_del_exact.grid(row=4, column=0, sticky="e")
        ToolTip(lbl_del_exact, "Remove exact matching YOLO format lines.\nUseful for removing specific known false positives.")
        ent_del_exact = ttk.Entry(cli, textvariable=self.cli_del_line)
        ent_del_exact.grid(row=4, column=1, columnspan=7, sticky="ew", padx=4)

        lbl_del_cls = ttk.Label(cli, text="Delete: class ids")
        lbl_del_cls.grid(row=5, column=0, sticky="e")
        ToolTip(lbl_del_cls, "Remove all labels with these class IDs.\nSpace or comma separated (e.g., '0 2 5').")
        ent_del_cls = ttk.Entry(cli, width=18, textvariable=self.cli_del_class)
        ent_del_cls.grid(row=5, column=1, sticky="w", padx=4)

        lbl_del_regex = ttk.Label(cli, text="Regex")
        lbl_del_regex.grid(row=5, column=2, sticky="e")
        ToolTip(lbl_del_regex, "Remove lines matching regex pattern.\nAdvanced filtering for complex conditions.")
        ent_del_regex = ttk.Entry(cli, textvariable=self.cli_del_regex)
        ent_del_regex.grid(row=5, column=3, columnspan=5, sticky="ew", padx=4)

        lbl_del_near = ttk.Label(cli, text="Delete-near line")
        lbl_del_near.grid(row=6, column=0, sticky="e")
        ToolTip(lbl_del_near, "Remove labels within epsilon distance of reference.\nUseful for duplicate removal.")
        ent_del_near = ttk.Entry(cli, textvariable=self.cli_del_near)
        ent_del_near.grid(row=6, column=1, columnspan=6, sticky="ew", padx=4)

        lbl_cli_eps = ttk.Label(cli, text="eps")
        lbl_cli_eps.grid(row=6, column=7, sticky="e")
        ToolTip(lbl_cli_eps, "Epsilon threshold for delete-near.\nDefault: 1e-6 (near-exact match)")
        ent_cli_eps = ttk.Entry(cli, width=10, textvariable=self.cli_cli_eps)
        ent_cli_eps.grid(row=6, column=8, sticky="w")

        # Presets row
        pre = ttk.Frame(cli); pre.grid(row=7, column=0, columnspan=9, sticky="ew", pady=(6,2))
        pre.columnconfigure(1, weight=1)
        ttk.Label(pre, text="Preset").grid(row=0, column=0, sticky="e")
        self.cli_preset_combo = ttk.Combobox(pre, textvariable=self.cli_preset_sel, state="readonly")
        self.cli_preset_combo.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(pre, text="Load", command=self._cli_load_preset).grid(row=0, column=2, padx=2)
        ttk.Entry(pre, width=24, textvariable=self.cli_preset_name).grid(row=0, column=3, padx=4)
        ttk.Button(pre, text="Save", command=self._cli_save_preset).grid(row=0, column=4, padx=2)
        ttk.Button(pre, text="Refresh", command=self._cli_refresh_presets).grid(row=0, column=5, padx=2)
        self._cli_refresh_presets()

        # Backup + Run row
        opts = ttk.Frame(cli); opts.grid(row=8, column=0, columnspan=9, sticky="ew")
        if not hasattr(self, 'cli_backup'):
            self.cli_backup = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Backup originals (.bak) before run", variable=self.cli_backup).pack(side="left")
        ttk.Checkbutton(opts, text="Dry run", variable=self.cli_dry).pack(side="left", padx=(10,0))
        ttk.Checkbutton(opts, text="Quiet", variable=self.cli_quiet).pack(side="left", padx=(10,0))
        ttk.Button(opts, text="Run Bulk Tool", command=self._do_cli_bulk_tool).pack(side="right")

        # Results pane
        resf = ttk.LabelFrame(cli, text="Results")
        resf.grid(row=9, column=0, columnspan=9, sticky="ew", pady=(6,4))
        for i in range(9): resf.columnconfigure(i, weight=1)
        ttk.Label(resf, text="Files modified:").grid(row=0, column=0, sticky="e")
        self.cli_res_files = tk.StringVar(value="-")
        ttk.Label(resf, textvariable=self.cli_res_files).grid(row=0, column=1, sticky="w")
        ttk.Label(resf, text="Lines removed:").grid(row=0, column=2, sticky="e")
        self.cli_res_lines = tk.StringVar(value="-")
        ttk.Label(resf, textvariable=self.cli_res_lines).grid(row=0, column=3, sticky="w")
        ttk.Label(resf, text="Files updated:").grid(row=0, column=4, sticky="e")
        self.cli_res_updated = tk.StringVar(value="-")
        ttk.Label(resf, textvariable=self.cli_res_updated).grid(row=0, column=5, sticky="w")

        return f


    # ---------- Presets Helpers ----------
    def _cli_presets_dir_ready(self):
        try:
            self._cli_presets_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _cli_list_presets(self):
        self._cli_presets_dir_ready()
        if not self._cli_presets_dir.exists():
            return []
        try:
            return sorted([p.stem for p in self._cli_presets_dir.glob("*.json")])
        except Exception:
            return []

    def _cli_refresh_presets(self):
        try:
            self.cli_preset_combo['values'] = self._cli_list_presets()
        except Exception:
            pass

    def _cli_save_preset(self):
        name = (self.cli_preset_name.get() or self.cli_preset_sel.get()).strip()
        if not name:
            messagebox.showwarning("Save Preset", "Please enter a preset name.")
            return
        data = {
            "mode": self.cli_mode.get().strip(),
            "glob": self.cli_glob.get().strip(),
            "start": self.cli_start.get().strip(),
            "end": self.cli_end.get().strip(),
            "step": self.cli_step.get().strip(),
            "prefix": self.cli_prefix.get().strip(),
            "pad": self.cli_pad.get().strip(),
            "ext": self.cli_ext.get().strip(),
            "lines": self.cli_lines.get().strip(),
            "del_line": self.cli_del_line.get().strip(),
            "del_class": self.cli_del_class.get().strip(),
            "del_regex": self.cli_del_regex.get().strip(),
            "del_near": self.cli_del_near.get().strip(),
            "eps": self.cli_cli_eps.get().strip(),
            "dry": bool(self.cli_dry.get()),
            "quiet": bool(self.cli_quiet.get()),
            "backup": bool(getattr(self, "cli_backup", tk.BooleanVar(value=True)).get()),
        }
        self._cli_presets_dir_ready()
        path = self._cli_presets_dir / f"{name}.json"
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.cli_preset_sel.set(name)
            self._cli_refresh_presets()
            log.info("Saved preset -> %s", path)
        except Exception as e:
            messagebox.showerror("Save Preset", f"Failed to save preset:\n{e}")

    def _cli_load_preset(self):
        name = self.cli_preset_sel.get().strip()
        if not name:
            messagebox.showwarning("Load Preset", "Select a preset to load.")
            return
        path = self._cli_presets_dir / f"{name}.json"
        if not path.exists():
            messagebox.showerror("Load Preset", f"Preset not found:\n{path}")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.cli_mode.set(data.get("mode","delete"))
            self.cli_glob.set(data.get("glob","**/*.txt"))
            self.cli_start.set(data.get("start",""))
            self.cli_end.set(data.get("end",""))
            self.cli_step.set(data.get("step","1"))
            self.cli_prefix.set(data.get("prefix","frame_"))
            self.cli_pad.set(data.get("pad","5"))
            self.cli_ext.set(data.get("ext",".txt"))
            self.cli_lines.set(data.get("lines",""))
            self.cli_del_line.set(data.get("del_line",""))
            self.cli_del_class.set(data.get("del_class",""))
            self.cli_del_regex.set(data.get("del_regex",""))
            self.cli_del_near.set(data.get("del_near",""))
            self.cli_cli_eps.set(data.get("eps","1e-6"))
            self.cli_dry.set(bool(data.get("dry", True)))
            self.cli_quiet.set(bool(data.get("quiet", False)))
            if hasattr(self, "cli_backup"):
                self.cli_backup.set(bool(data.get("backup", True)))
            log.info("Loaded preset: %s", name)
        except Exception as e:
            messagebox.showerror("Load Preset", f"Failed to load preset:\n{e}")

    # ---------- Backup helpers (used by Run Bulk Tool) ----------
    def _cli_resolve_targets(self, labels_dir, glob_pat, start, end, step, prefix, pad, ext):
        targets = []
        if glob_pat:
            pattern = str(Path(labels_dir) / glob_pat)
            recursive = "**" in glob_pat
            import glob as _glob, os as _os
            for f in sorted(_glob.glob(pattern, recursive=recursive)):
                if _os.path.isfile(f):
                    targets.append(Path(f))
        else:
            if not (start and end):
                return []
            start_i, end_i, step_i = int(start), int(end), int(step or 1)
            pad_i = int(pad)
            for i in range(start_i, end_i + 1, step_i):
                fname = f"{prefix}{i:0{pad_i}}{ext}"
                fpath = Path(labels_dir) / fname
                targets.append(fpath)
        return targets

    def _cli_backup_targets(self, targets):
        count = 0
        for p in targets:
            try:
                if p.exists():
                    bak = p.with_suffix(p.suffix + ".bak")
                    if not bak.exists():
                        shutil.copy2(p, bak)
                        count += 1
            except Exception as e:
                log.warning("Backup failed for %s: %s", p, e)
        if count:
            log.info("Backed up %d file(s) with .bak", count)
        return count

    def _do_validate(self):
        labels_dir = Path(self.qalabels_dir.get())
        self._run_task(lambda: qaqc.validate_labels(labels_dir), "Validate Labels")

    def _do_cleanup(self):
        labels_dir = Path(self.qalabels_dir.get())
        delete_lines = [s.strip() for s in self.qa_delete_line.get().split("\n") if s.strip()]
        delete_classes = [int(x) for x in re.split(r"[\s,]+", self.qa_delete_class.get().strip()) if x.strip().isdigit()]
        delete_regex = [s.strip() for s in re.split(r"[\n]+", self.qa_delete_regex.get().strip()) if s.strip()]
        delete_near = [s.strip() for s in self.qa_delete_near.get().split("\n") if s.strip()]
        eps = float(self.qa_eps.get() or "0.0")
        dry = bool(self.qa_dry.get())
        self._run_task(lambda: qaqc.cleanup_labels(labels_dir,
                                                   delete_lines=delete_lines,
                                                   delete_classes=delete_classes,
                                                   delete_regex=delete_regex,
                                                   delete_near=delete_near,
                                                   eps=eps, dry_run=dry), "Cleanup Labels")

    def _do_splits(self):
        images_root = Path(self.split_images_root.get())
        labels_root = Path(self.split_labels_root.get() or images_root.parent / "labels" / images_root.name)
        out_root = Path(self.split_out_root.get())
        site_regex = self.split_site_regex.get().strip() or None
        tr, va, te = float(self.split_train.get()), float(self.split_val.get()), float(self.split_test.get())
        classes_txt = Path(self.classes_file.get()) if self.classes_file.get() else None
        self._run_task(lambda: qaqc.make_site_disjoint_splits(images_root, labels_root, out_root,
                                                              train=tr, val=va, test=te, site_regex=site_regex,
                                                              copy=True, classes_txt=classes_txt), "Make Splits")
        self.train_data_yaml.set(str(out_root / "data.yaml"))

    def _do_write_fhwa_template(self):
        out_csv = Path(self.split_out_root.get()) / "fhwa_mapping.csv"
        self._run_task(lambda: qaqc.write_fhwa_mapping_csv(out_csv), "FHWA Mapping")

    # ---------- Bulk CLI wrapper ----------
    def _do_cli_bulk_tool(self):
        import subprocess, shlex
        labels_dir = self.qalabels_dir.get().strip()
        if not labels_dir:
            messagebox.showwarning("Bulk Tool", "Please select Labels dir."); return
        tool = str(Path(__file__).parent / "tools" / "bulk_write_label.py")
        args = [sys.executable, "-u", tool, "--labels_dir", labels_dir]

        glob_pat = self.cli_glob.get().strip()
        start, end, step = self.cli_start.get().strip(), self.cli_end.get().strip(), self.cli_step.get().strip() or "1"
        if glob_pat:
            args += ["--glob", glob_pat]
        else:
            if not (start and end):
                messagebox.showerror("Bulk Tool", "Provide either a Glob pattern or a Start/End range."); return
            args += ["--start", start, "--end", end, "--step", step,
                     "--prefix", self.cli_prefix.get().strip() or "frame_",
                     "--pad", self.cli_pad.get().strip() or "5",
                     "--ext", self.cli_ext.get().strip() or ".txt"]

        mode = self.cli_mode.get().strip() or "append"
        args += ["--mode", mode]

        # Lines for write modes
        if mode in ("append","overwrite"):
            raw = self.cli_lines.get().strip()
            if not raw:
                messagebox.showerror("Bulk Tool", "In write modes, --line is required."); return
            lines = [s.strip() for s in re.split(r"[\n,]+", raw) if s.strip()]
            for ln in lines:
                args += ["--line", ln]
        else:
            # Delete selectors
            if self.cli_del_line.get().strip():
                for s in [t.strip() for t in re.split(r"[\n,]+", self.cli_del_line.get().strip()) if t.strip()]:
                    args += ["--delete-line", s]
            if self.cli_del_class.get().strip():
                for cls in [t.strip() for t in re.split(r"[ ,]+", self.cli_del_class.get().strip()) if t.strip()]:
                    args += ["--delete-class", cls]
            if self.cli_del_regex.get().strip():
                for rx in [t.strip() for t in self.cli_del_regex.get().splitlines() if t.strip()]:
                    args += ["--delete-regex", rx]
            if self.cli_del_near.get().strip():
                for nr in [t.strip() for t in self.cli_del_near.get().splitlines() if t.strip()]:
                    args += ["--delete-near", nr]
            eps = self.cli_cli_eps.get().strip()
            if eps:
                args += ["--eps", eps]

        if self.cli_dry.get(): args.append("--dry-run")
        if self.cli_quiet.get(): args.append("--quiet")

        # ---- NEW: Pre-copy backups (.bak) of targeted files (simple blanket backup) ----
        if self.cli_backup.get():
            # Derive file list based on scope
            backup_paths = []
            if glob_pat:
                pattern = os.path.join(labels_dir, glob_pat)
                recursive = "**" in glob_pat
                for p in glob.glob(pattern, recursive=recursive):
                    if os.path.isfile(p):
                        backup_paths.append(p)
            else:
                try:
                    s, e, st = int(start), int(end), int(step or "1")
                except Exception:
                    s, e, st = 0, -1, 1
                prefix = self.cli_prefix.get().strip() or "frame_"
                pad = int(self.cli_pad.get().strip() or "5")
                ext = self.cli_ext.get().strip() or ".txt"
                for i in range(s, e+1, st):
                    fname = f"{prefix}{i:0{pad}}{ext}"
                    fpath = os.path.join(labels_dir, fname)
                    if os.path.isfile(fpath):
                        backup_paths.append(fpath)
            # Copy if not already backed up
            for p in backup_paths:
                bak = p + ".bak"
                if not os.path.exists(bak):
                    try:
                        shutil.copy2(p, bak)
                    except Exception as e:
                        self.logbox.insert("end", f"[backup] failed for {p}: {e}\n"); self.logbox.see("end")
            self.logbox.insert("end", f"[backup] Created .bak for {len(backup_paths)} file(s)\n"); self.logbox.see("end")

        # Reset results pane
        self.bulk_files_modified.set("0"); self.bulk_lines_removed.set("0"); self.bulk_updated_files.set("0"); self.bulk_dry_note.set("")

        def _run():
            self.logbox.insert("end", f"$ {' '.join(shlex.quote(a) for a in args)}\n"); self.logbox.see("end")
            out_lines = []
            try:
                p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in p.stdout:
                    out_lines.append(line.rstrip("\n"))
                    self.logbox.insert("end", line); self.logbox.see("end")
                p.wait()
                rc = p.returncode
                # Parse summary for the mini results pane
                text = "\n".join(out_lines)
                import re as _re
                m_del = _re.search(r"Done\. Modified (\d+) file\(s\); removed (\d+) line\(s\)\.", text)
                m_wrt = _re.search(r"Done\. Updated (\d+) file\(s\) in", text)
                m_dry = _re.search(r"Note: dry run—no files were written\.", text)
                if m_del:
                    self.bulk_files_modified.set(m_del.group(1)); self.bulk_lines_removed.set(m_del.group(2))
                if m_wrt:
                    self.bulk_updated_files.set(m_wrt.group(1))
                if m_dry:
                    self.bulk_dry_note.set("Dry run — no files were written.")
                if rc != 0:
                    self.logbox.insert("end", f"[bulk_tool] exited with code {rc}\n"); self.logbox.see("end")
            except Exception as e:
                self.logbox.insert("end", f"[bulk_tool] error: {e}\n"); self.logbox.see("end")

        self._run_task(_run, "Bulk Write/Delete")

    # ---------- Step 5: Train ----------
    def _step_train(self, parent):
        f = ttk.Frame(parent); f.columnconfigure(1, weight=1)
        lbl_data_yaml = ttk.Label(f, text="data.yaml")
        lbl_data_yaml.grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_data_yaml, "YOLO dataset configuration file.\nDefines paths to train/val/test splits and class names.\nAuto-populated after creating splits.")
        ent_data_yaml = ttk.Entry(f, textvariable=self.train_data_yaml)
        ent_data_yaml.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_file(self.train_data_yaml,[("YAML","*.yaml *.yml")])).grid(row=0, column=2)

        lbl_weights = ttk.Label(f, text="weights")
        lbl_weights.grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_weights, "Pre-trained model weights to start from.\nRecommended: yolov8s.pt (small) or yolov8m.pt (medium)\nTransfer learning speeds up training.")
        ent_weights = ttk.Entry(f, textvariable=self.train_weights)
        ent_weights.grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(f, text="Browse…", command=lambda: self._choose_file(self.train_weights,[("PT","*.pt")])).grid(row=1, column=2)

        lbl_run_name = ttk.Label(f, text="run name")
        lbl_run_name.grid(row=2, column=0, sticky="e", padx=4, pady=4)
        ToolTip(lbl_run_name, "Name for this training run.\nResults saved to runs/detect/{run_name}/")
        ent_run_name = ttk.Entry(f, textvariable=self.train_out_name)
        ent_run_name.grid(row=2, column=1, sticky="w", padx=4)

        ft = ttk.LabelFrame(f, text="Fine-tuning & Augmentations (best practices)")
        ft.grid(row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=(8,4))

        # Column 1 - Basic parameters
        lbl_imgsz = ttk.Label(ft, text="imgsz"); lbl_imgsz.grid(row=0, column=0, sticky="e")
        ToolTip(lbl_imgsz, "Training image size (pixels).\nHigher = more detail, slower.\nRecommended: 640-960")
        self.t_imgsz = tk.StringVar(value="960"); ttk.Entry(ft, width=10, textvariable=self.t_imgsz).grid(row=0, column=1, sticky="w")

        lbl_epochs = ttk.Label(ft, text="epochs"); lbl_epochs.grid(row=1, column=0, sticky="e")
        ToolTip(lbl_epochs, "Number of training epochs.\nMore epochs = better learning but risk overfitting.\nRecommended: 50-100")
        self.t_epochs = tk.StringVar(value="60"); ttk.Entry(ft, width=10, textvariable=self.t_epochs).grid(row=1, column=1, sticky="w")

        lbl_batch = ttk.Label(ft, text="batch"); lbl_batch.grid(row=2, column=0, sticky="e")
        ToolTip(lbl_batch, "Batch size (images per iteration).\n-1 = auto-detect max batch for your GPU.\nLower if running out of memory.")
        self.t_batch = tk.StringVar(value="-1"); ttk.Entry(ft, width=10, textvariable=self.t_batch).grid(row=2, column=1, sticky="w")

        lbl_device = ttk.Label(ft, text="device"); lbl_device.grid(row=3, column=0, sticky="e")
        ToolTip(lbl_device, "GPU device ID (0, 1, 2...).\nUse 'cpu' for CPU training (very slow).")
        self.t_device = tk.StringVar(value="0"); ttk.Entry(ft, width=10, textvariable=self.t_device).grid(row=3, column=1, sticky="w")

        lbl_workers = ttk.Label(ft, text="workers"); lbl_workers.grid(row=4, column=0, sticky="e")
        ToolTip(lbl_workers, "Number of data loading workers.\nMore = faster data loading but more RAM.\nRecommended: 2-8")
        self.t_workers = tk.StringVar(value="2"); ttk.Entry(ft, width=10, textvariable=self.t_workers).grid(row=4, column=1, sticky="w")

        # Column 2 - Augmentation
        lbl_mosaic = ttk.Label(ft, text="mosaic"); lbl_mosaic.grid(row=0, column=2, sticky="e")
        ToolTip(lbl_mosaic, "Mosaic augmentation probability (0.0-1.0).\nCombines 4 images into one.\nGreat for small object detection.")
        self.t_mosaic = tk.StringVar(value="1.0"); ttk.Entry(ft, width=10, textvariable=self.t_mosaic).grid(row=0, column=3, sticky="w")

        lbl_mixup = ttk.Label(ft, text="mixup"); lbl_mixup.grid(row=1, column=2, sticky="e")
        ToolTip(lbl_mixup, "MixUp augmentation probability (0.0-1.0).\nBlends images together.\nUseful for regularization.")
        self.t_mixup = tk.StringVar(value="0.1"); ttk.Entry(ft, width=10, textvariable=self.t_mixup).grid(row=1, column=3, sticky="w")

        lbl_copy_paste = ttk.Label(ft, text="copy_paste"); lbl_copy_paste.grid(row=2, column=2, sticky="e")
        ToolTip(lbl_copy_paste, "Copy-paste augmentation probability (0.0-1.0).\nCopies objects between images.\nHelps with class imbalance.")
        self.t_copy_paste = tk.StringVar(value="0.0"); ttk.Entry(ft, width=10, textvariable=self.t_copy_paste).grid(row=2, column=3, sticky="w")

        lbl_degrees = ttk.Label(ft, text="degrees"); lbl_degrees.grid(row=3, column=2, sticky="e")
        ToolTip(lbl_degrees, "Image rotation augmentation (±degrees).\n0 = no rotation.\nUse if camera angles vary.")
        self.t_degrees = tk.StringVar(value="0.0"); ttk.Entry(ft, width=10, textvariable=self.t_degrees).grid(row=3, column=3, sticky="w")

        lbl_fl_gamma = ttk.Label(ft, text="fl_gamma (FocalLoss)"); lbl_fl_gamma.grid(row=4, column=2, sticky="e")
        ToolTip(lbl_fl_gamma, "Focal Loss gamma parameter.\nHelps with class imbalance.\nLeave blank to use default loss.\nTry 1.5-2.0 for imbalanced data.")
        self.t_fl_gamma = tk.StringVar(value=""); ttk.Entry(ft, width=10, textvariable=self.t_fl_gamma).grid(row=4, column=3, sticky="w")

        # Column 3 - Advanced
        lbl_warmup = ttk.Label(ft, text="warmup_epochs"); lbl_warmup.grid(row=0, column=4, sticky="e")
        ToolTip(lbl_warmup, "Learning rate warmup epochs.\nGradually increases LR at start.\nHelps training stability.")
        self.t_warmup = tk.StringVar(value="3.0"); ttk.Entry(ft, width=10, textvariable=self.t_warmup).grid(row=0, column=5, sticky="w")

        lbl_close_mosaic = ttk.Label(ft, text="close_mosaic (last N)"); lbl_close_mosaic.grid(row=1, column=4, sticky="e")
        ToolTip(lbl_close_mosaic, "Disable mosaic for last N epochs.\nHelps model learn normal images.\nRecommended: 10-15")
        self.t_close_mosaic = tk.StringVar(value="10"); ttk.Entry(ft, width=10, textvariable=self.t_close_mosaic).grid(row=1, column=5, sticky="w")

        lbl_lr0 = ttk.Label(ft, text="lr0"); lbl_lr0.grid(row=2, column=4, sticky="e")
        ToolTip(lbl_lr0, "Initial learning rate.\nLeave blank for auto.\nLower if training unstable.")
        self.t_lr0 = tk.StringVar(value=""); ttk.Entry(ft, width=10, textvariable=self.t_lr0).grid(row=2, column=5, sticky="w")

        lbl_lrf = ttk.Label(ft, text="lrf"); lbl_lrf.grid(row=3, column=4, sticky="e")
        ToolTip(lbl_lrf, "Final learning rate (as fraction of lr0).\nLeave blank for auto.\nDefault: 0.01")
        self.t_lrf = tk.StringVar(value=""); ttk.Entry(ft, width=10, textvariable=self.t_lrf).grid(row=3, column=5, sticky="w")

        lbl_freeze = ttk.Label(ft, text="freeze (e.g., 0-9)"); lbl_freeze.grid(row=4, column=4, sticky="e")
        ToolTip(lbl_freeze, "Freeze model layers (e.g., '0-9' freezes first 10 layers).\nUseful for fine-tuning with small datasets.\nLeave blank to train all layers.")
        self.t_freeze = tk.StringVar(value=""); ttk.Entry(ft, width=10, textvariable=self.t_freeze).grid(row=4, column=5, sticky="w")

        self.t_cos_lr = tk.BooleanVar(value=True)
        chk_cos_lr = ttk.Checkbutton(ft, text="Cosine LR", variable=self.t_cos_lr)
        chk_cos_lr.grid(row=5, column=4, sticky="w")
        ToolTip(chk_cos_lr, "Use cosine learning rate schedule.\nRecommended for most cases.\nProvides smooth LR decay.")

        lbl_optimizer = ttk.Label(ft, text="optimizer"); lbl_optimizer.grid(row=5, column=2, sticky="e")
        ToolTip(lbl_optimizer, "Optimizer: Adam, SGD, AdamW, etc.\nLeave blank for auto (AdamW).")
        self.t_optimizer = tk.StringVar(value=""); ttk.Entry(ft, width=10, textvariable=self.t_optimizer).grid(row=5, column=3, sticky="w")

        # Status indicator
        self.train_status_label = ttk.Label(f, text="", font=("", 9))
        self.train_status_label.grid(row=4, column=1, sticky="w", padx=4, pady=(4,0))

        btn_train = ttk.Button(f, text="Start Training", command=self._do_train)
        btn_train.grid(row=5, column=1, sticky="e", pady=8)
        ToolTip(btn_train, "Begin model training.\nMonitor progress in the log.\nBest weights saved to runs/detect/{run_name}/weights/best.pt")
        return f

    def _do_train(self):
        # Check prerequisites with soft warnings
        can_proceed, message, severity = PrerequisiteChecker.can_train(self.project)
        if not can_proceed:
            if severity == "error":
                messagebox.showerror("Cannot Train", message)
                return
            elif severity == "warning":
                if not messagebox.askyesno("Warning", f"{message}\n\nProceed anyway?"):
                    return

        data_yaml = self.train_data_yaml.get()
        weights = self.train_weights.get()
        run_name = self.train_out_name.get()
        imgsz = int(float(self.t_imgsz.get()))
        epochs = int(float(self.t_epochs.get()))
        batch = int(float(self.t_batch.get()))
        device = int(float(self.t_device.get()))
        workers = int(float(self.t_workers.get()))
        mosaic = float(self.t_mosaic.get())
        mixup = float(self.t_mixup.get())
        copy_paste = float(self.t_copy_paste.get())
        degrees = float(self.t_degrees.get())
        fl_gamma = self.t_fl_gamma.get().strip()
        fl_gamma = float(fl_gamma) if fl_gamma else None
        warmup = float(self.t_warmup.get())
        close_mosaic = int(float(self.t_close_mosaic.get()))
        lr0 = self.t_lr0.get().strip(); lr0 = float(lr0) if lr0 else None
        lrf = self.t_lrf.get().strip(); lrf = float(lrf) if lrf else None
        freeze = self.t_freeze.get().strip() or None
        optimizer = self.t_optimizer.get().strip() or None
        cos_lr = bool(self.t_cos_lr.get())

        def on_success_train(res):
            self.project.steps_completed.update({"train": True})
            self.project.workflow_state["has_trained_model"] = True
            weights_path = Path(f"runs/detect/{run_name}/weights/best.pt")
            if weights_path.exists():
                self.project.workflow_state["trained_weights"] = str(weights_path)
            self._update_workflow_status()

        self._run_task(lambda: trainer.train(weights=weights, data_yaml=data_yaml, name=run_name,
                                             imgsz=imgsz, epochs=epochs, batch=batch,
                                             device=device, workers=workers,
                                             mosaic=mosaic, mixup=mixup, copy_paste=copy_paste,
                                             degrees=degrees, fl_gamma=fl_gamma,
                                             warmup_epochs=warmup, close_mosaic=close_mosaic,
                                             lr0=lr0, lrf=lrf, freeze=freeze, optimizer=optimizer,
                                             cos_lr=cos_lr), "Training", on_success=on_success_train)

    # ---------- Step 6: Analyze ----------
    def _step_analyze(self, parent):
        f = ttk.Frame(parent); f.columnconfigure(1, weight=1)
        lbl_an_video = ttk.Label(f, text="Video")
        lbl_an_video.grid(row=0, column=0, sticky="e")
        ToolTip(lbl_an_video, "Video file to analyze for traffic patterns.\nUse the same type of video as training data.")
        ent_an_video = ttk.Entry(f, textvariable=self.an_video)
        ent_an_video.grid(row=0, column=1, sticky="ew")
        ttk.Button(f, text="Browse…", command=lambda: self._choose_file(self.an_video,[("MP4","*.mp4")])).grid(row=0, column=2)

        lbl_an_roi = ttk.Label(f, text="ROI JSON (template)")
        lbl_an_roi.grid(row=1, column=0, sticky="e")
        ToolTip(lbl_an_roi, "ROI template defining entry/exit zones for tracking.\nSame format as used in ROI & Mask step.")
        ent_an_roi = ttk.Entry(f, textvariable=self.roi_json)
        ent_an_roi.grid(row=1, column=1, sticky="ew")
        ttk.Button(f, text="Browse…", command=lambda: self._choose_file(self.roi_json,[("JSON","*.json")])).grid(row=1, column=2)

        lbl_an_weights = ttk.Label(f, text="Weights (best.pt)")
        lbl_an_weights.grid(row=2, column=0, sticky="e")
        ToolTip(lbl_an_weights, "Trained model weights for vehicle detection.\nUse best.pt from your training run for best results.")
        ent_an_weights = ttk.Entry(f, textvariable=self.an_weights)
        ent_an_weights.grid(row=2, column=1, sticky="ew")
        ttk.Button(f, text="Browse…", command=lambda: self._choose_file(self.an_weights,[("PT","*.pt")])).grid(row=2, column=2)

        lbl_an_out = ttk.Label(f, text="Output Folder")
        lbl_an_out.grid(row=3, column=0, sticky="e")
        ToolTip(lbl_an_out, "Directory for analysis outputs.\nWill contain tracking results, CSV reports, and visualizations.")
        ent_an_out = ttk.Entry(f, textvariable=self.an_out)
        ent_an_out.grid(row=3, column=1, sticky="ew")
        ttk.Button(f, text="Browse…", command=lambda: self._choose_dir(self.an_out)).grid(row=3, column=2)

        lbl_an_run = ttk.Label(f, text="Run name")
        lbl_an_run.grid(row=4, column=0, sticky="e")
        ToolTip(lbl_an_run, "Name for this analysis run.\nUsed for output file naming.")
        ent_an_run = ttk.Entry(f, textvariable=self.an_run)
        ent_an_run.grid(row=4, column=1, sticky="w")

        lane_box = ttk.LabelFrame(f, text="Per-lane ROIs")
        lane_box.grid(row=5, column=0, columnspan=3, sticky="ew", padx=4, pady=(6,4))
        chk_lanes = ttk.Checkbutton(lane_box, text="Include per-lane ROIs before analysis",
                        variable=self.include_lanes)
        chk_lanes.grid(row=0, column=0, columnspan=3, sticky="w", padx=4)
        ToolTip(chk_lanes, "Process lane-level ROIs for detailed movement analysis.\nEnables left/through/right turn classification.")

        lbl_lane_pattern = ttk.Label(lane_box, text="Lane regex")
        lbl_lane_pattern.grid(row=1, column=0, sticky="e")
        ToolTip(lbl_lane_pattern, "Regular expression to identify lane ROI names.\nExample: r'(lane|_ln|\\bltr\\b|through|left|right)'\nMatches ROI names containing lane indicators.")
        ent_lane_pattern = ttk.Entry(lane_box, textvariable=self.lane_pattern)
        ent_lane_pattern.grid(row=1, column=1, sticky="ew")

        # Status indicator
        self.analyze_status_label = ttk.Label(lane_box, text="", font=("", 9))
        self.analyze_status_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(4,0))

        btn_analyze = ttk.Button(lane_box, text="Run Analysis", command=self._do_analyze)
        btn_analyze.grid(row=3, column=2, sticky="e", pady=6)
        ToolTip(btn_analyze, "Start traffic analysis with vehicle tracking.\nGenerates movement reports, counts, and CSV outputs.\nMay take several minutes for long videos.")
        return f

    def _do_analyze(self):
        # Check prerequisites with soft warnings
        can_proceed, message, severity = PrerequisiteChecker.can_analyze(self.project)
        if not can_proceed:
            if severity == "error":
                messagebox.showerror("Cannot Analyze", message)
                return
            elif severity == "warning" or severity == "info":
                if not messagebox.askyesno("Confirm", f"{message}\n\nProceed?"):
                    return

        video = Path(self.an_video.get())
        roi = Path(self.roi_json.get())
        weights = self.an_weights.get().strip() or "yolov8n.pt"
        out_dir = Path(self.an_out.get()); out_dir.mkdir(parents=True, exist_ok=True)
        run_name = self.an_run.get().strip() or "analysis_run"
        include_lanes = bool(self.include_lanes.get())
        lane_pattern = self.lane_pattern.get().strip()

        def _run():
            flat_json = roi_utils.flatten_for_analysis(roi, out_dir / f"{run_name}_flat_rois.json",
                                                       include_lanes=include_lanes, lane_pattern=lane_pattern)
            return analyzer.analyze(video, weights, out_dir, run_name, flat_json, imgsz=960, conf=0.25, iou=0.50)

        def on_success_analyze(res):
            self.project.steps_completed.update({"analyze": True})
            self._update_workflow_status()

        self._run_task(_run, "Analyze", on_success=on_success_analyze)

    # ----- Plumbing -----
    def _drain_logs(self):
        try:
            while True:
                t, lvl, msg = LOGQ.get_nowait()
                self.logbox.insert("end", f"[{t}] {lvl:<8} {msg}\n")
                self.logbox.see("end")
        except queue.Empty:
            pass
        self.after(150, self._drain_logs)

    def _run_task(self, fn, title, on_success: Optional[Callable] = None,
                  progress_bar: Optional[ttk.Progressbar] = None,
                  progress_label: Optional[ttk.Label] = None,
                  run_button: Optional[ttk.Button] = None,
                  cancel_button: Optional[ttk.Button] = None,
                  output_discovery: Optional[ttk.Frame] = None):
        """
        Enhanced task runner with progress reporting, cancellation, and UI state management.

        Args:
            fn: Task function to run
            title: Task title for logging
            on_success: Optional callback to run on successful completion
            progress_bar: Optional progress bar widget to animate
            progress_label: Optional label to show progress messages
            run_button: Optional button to disable during task
            cancel_button: Optional cancel button to enable during task
            output_discovery: Optional frame to show after successful completion
        """
        # Show progress UI
        if progress_bar:
            progress_bar.grid()
            progress_bar.start()
        if progress_label:
            progress_label.grid()
            progress_label.config(text="Starting...")
        if run_button:
            run_button.config(state="disabled")
        if cancel_button:
            cancel_button.config(state="normal")

        def on_progress(fraction: float, message: str):
            """Update progress UI on main thread."""
            if progress_label:
                self.after(0, lambda: progress_label.config(text=message))

        def on_done(success, result, cancelled=False):
            """Cleanup and finalize UI state."""
            # Hide progress UI
            if progress_bar:
                self.after(0, lambda: progress_bar.stop())
                self.after(0, lambda: progress_bar.grid_remove())
            if progress_label:
                if cancelled:
                    self.after(0, lambda: progress_label.config(text="Cancelled", foreground="orange"))
                elif success:
                    self.after(0, lambda: progress_label.config(text="✓ Complete", foreground="green"))
                else:
                    self.after(0, lambda: progress_label.config(text="✗ Failed", foreground="red"))

            # Re-enable buttons
            if run_button:
                self.after(0, lambda: run_button.config(state="normal"))
            if cancel_button:
                self.after(0, lambda: cancel_button.config(state="disabled"))

            # Show output discovery buttons on success
            if success and output_discovery:
                self.after(0, lambda: output_discovery.grid())

            # Log result
            if cancelled:
                log.warning("%s — cancelled by user.", title)
            elif success:
                log.info("%s — done.", title)
                if on_success:
                    on_success(result)
            else:
                self.after(0, lambda: messagebox.showerror(title, "Task failed. See log for details."))

            # Clear current worker reference
            self.current_worker = None

        # Start worker
        self.current_worker = Worker(fn, on_done, log, on_progress=on_progress)
        self.current_worker.start()

    def _cancel_current_task(self):
        """Cancel the currently running task."""
        if self.current_worker:
            self.current_worker.cancel()
            log.info("Cancel requested for current task.")

    def _choose_file(self, var, types):
        p = filedialog.askopenfilename(filetypes=types)
        if p: var.set(p)

    def _choose_dir(self, var):
        p = filedialog.askdirectory()
        if p: var.set(p)

    def _choose_file_or_dir(self, var):
        p = filedialog.askopenfilename(title="Select a video or choose a folder", filetypes=[("Media and folders","*.*")])
        if not p:
            d = filedialog.askdirectory()
            if d: var.set(d)
        else:
            var.set(p)

if __name__ == "__main__":
    try:
        import cv2
    except Exception:
        pass
    TrafficStudio().mainloop()
