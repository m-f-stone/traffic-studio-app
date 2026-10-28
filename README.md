# Traffic Studio

A comprehensive GUI application for end-to-end traffic video analysis and YOLO model training. Traffic Studio provides an integrated workflow for sampling video frames, defining regions of interest (ROI), auto-labeling with pre-trained models, training custom YOLO models, and analyzing traffic patterns.

## 🎯 New Architecture (v2.0)

Traffic Studio has been refactored with a modern, maintainable architecture featuring:

- **Typed Domain Models** - Type-safe configurations with built-in validation
- **Service Layer** - Testable business logic with dependency injection
- **Cross-Platform Config** - User configs stored in platform-appropriate directories
- **Protocol-Based Testing** - Mock external dependencies for unit testing
- **Comprehensive Documentation** - Migration guides, packaging strategies, and examples

For details, see:
- [ARCHITECTURE_REFACTOR.md](ARCHITECTURE_REFACTOR.md) - Architecture design and rationale
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Step-by-step migration from old architecture
- [PACKAGING.md](PACKAGING.md) - PyTorch packaging strategies (CPU vs CUDA)
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Complete implementation overview
- [example_usage.py](example_usage.py) - Working code examples using new architecture

## Features

Traffic Studio combines multiple traffic analysis workflows into a single accordion-style interface:

### 1. Video Frame Sampling
- Extract frames from traffic videos using ffmpeg
- Two sampling methods:
  - **FPS-based**: Sample at specific frames per second
  - **Select**: Advanced frame selection with tolerance control
- Support for multiple video formats (.mp4, .mov, .mkv, .avi)
- Batch processing for multiple videos

### 2. ROI (Region of Interest) Management
- Define detection zones using an interactive ROI tool
- Apply masks to focus on specific areas
- Optional border cropping with configurable margins
- Export masked images for downstream processing

### 3. Auto-Labeling
- Automatically label images using pre-trained YOLO models
- Configurable inference parameters:
  - Image size, confidence threshold, IOU threshold
- Generate YOLO-format labels for training data
- Custom class definitions support

### 4. Model Training
- Train YOLOv8 models on traffic datasets
- Comprehensive training configuration:
  - Image size, epochs, batch size, device selection
  - Data augmentation (mosaic, mixup, copy-paste)
  - Learning rate scheduling (cosine LR, warmup)
  - HSV color augmentation
- Automatic validation after training

### 5. Traffic Analysis
- Track vehicles across frames using ROI-based logic
- Generate comprehensive traffic reports:
  - Entry/exit counts by ROI
  - Left-turn/through/right-turn movement classification
  - Temporal analysis with configurable bin sizes
- Cardinal direction detection (North/South/East/West)
- CSV export for further analysis

### 6. QA/QC Tools
- Bulk label editing and cleaning:
  - Delete specific lines or classes
  - Regex-based filtering
  - Proximity-based deletion (remove nearby labels)
- Dry-run mode for safe preview
- Backup option to preserve originals

### 7. Dataset Management
- Site-disjoint train/val/test splitting
- Configurable split ratios
- Automatic dataset organization

## Installation

### Prerequisites
- **Python 3.10 or higher** (required for new architecture type hints)
- **ffmpeg** (must be available on PATH)
- **PyTorch** (see PyTorch Installation below)
- GPU recommended for training and inference

### Setup

1. Clone or download this repository

2. Install core dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch (choose one):

   **For CPU-only** (universal compatibility, ~200MB):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

   **For NVIDIA GPU** (CUDA 11.8, ~2GB, recommended for GPU systems):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

   **For Latest NVIDIA GPU** (CUDA 12.1+):
   ```bash
   pip install torch torchvision
   ```

   For more options, visit: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

4. Verify ffmpeg installation:
```bash
ffmpeg -version
```

5. Verify PyTorch installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Development Setup

For development with testing and type-checking tools:
```bash
pip install -r requirements-dev.txt
```

This includes pytest, mypy, black, and other development tools.

## Usage

### Starting the Application

**GUI Mode** (traditional interface):
```bash
python app.py
```

The GUI features an accordion-style interface where each section can be expanded or collapsed for easy navigation.

Every step runs on a worker thread with a real progress bar and a working **Cancel** button:
sampling terminates ffmpeg, masking and auto-labeling stop after the current image, training stops
at the end of the current epoch (keeping that checkpoint), and analysis stops after the current frame
and still writes the tracks seen so far. Opening or saving a project makes its folder the working
directory, so relative paths such as `data/`, `runs/` and `outputs/` resolve the same way everywhere.

The annotated analysis video is optional (**Write annotated video** in the Analyze step). When
ffmpeg is on PATH it is encoded with libx264, which is several times faster than OpenCV's built-in
encoder at full resolution; untick it for counts-only runs.

**Programmatic Usage** (new architecture):
```python
from pathlib import Path
from traffic_studio.container import Container
from traffic_studio.domain.sampler import SamplerConfig, SamplingMethod

# Create service container
container = Container()

# Configure sampling
config = SamplerConfig(
    video_path=Path("video.mp4"),
    output_dir=Path("frames/"),
    fps=1.0,
    method=SamplingMethod.FPS
)

# Validate configuration
if not config.is_valid:
    print("Errors:", config.validate())
    exit(1)

# Run sampling
result = container.sampler_service().sample_frames(
    config,
    progress_callback=lambda frac, msg: print(f"{int(frac*100)}%: {msg}")
)

# Check results
if result.success:
    print(f"Extracted {result.frames_extracted} frames")
else:
    print("Errors:", result.errors)
```

See [example_usage.py](example_usage.py) for complete examples of all workflows.

### Architecture Overview (v2.0)

The new architecture follows a **layered pattern** with clear separation of concerns:

```
┌─────────────────────────────────┐
│  UI Layer (app.py)              │  ← Tkinter GUI (legacy, to be refactored)
│  Presenters, Views, Widgets     │
├─────────────────────────────────┤
│  Service Layer                  │  ← Business logic (NEW)
│  SamplerService, TrainerService │     - Dependency injection
│  AutolabelService, etc.         │     - Progress callbacks
│                                  │     - Cancellation support
├─────────────────────────────────┤
│  Domain Layer                   │  ← Typed models (NEW)
│  SamplerConfig, TrainerConfig   │     - Dataclass validation
│  SamplerResult, TrainerResult   │     - Type safety
├─────────────────────────────────┤
│  Infrastructure Layer           │  ← External integrations (NEW)
│  FFmpegRunner, YOLORunner       │     - ffmpeg wrapper
│  UserConfigManager              │     - YOLO wrapper
│                                  │     - Cross-platform config
└─────────────────────────────────┘
```

**Key Benefits:**
- **Type Safety**: Catch errors at development time with comprehensive type hints
- **Testability**: Mock external dependencies for unit testing
- **Maintainability**: Clear separation makes code easier to understand and modify
- **Portability**: Cross-platform config management works when packaged
- **Scalability**: Easy to add new workflows following established patterns

### Typical Workflow

1. **Sample Frames**: Extract frames from traffic videos
   - Select input video or directory
   - Choose sampling method and rate
   - Specify output directory

2. **Define ROI**: Create regions of interest for detection zones
   - Load sampled images
   - Use the ROI tool to draw zones
   - Export ROI JSON configuration

3. **Apply Masks** (optional): Crop or mask images to ROI areas
   - Load ROI JSON
   - Configure border cropping if needed
   - Generate masked image dataset

4. **Auto-Label**: Generate initial labels using pre-trained models
   - Select image directory
   - Configure inference parameters
   - Run prediction to create YOLO labels

5. **QA/QC**: Review and clean labels
   - Manual review recommended
   - Use bulk tools to remove invalid labels
   - Filter by class, proximity, or patterns

6. **Split Dataset**: Create train/val/test splits
   - Define site regex for disjoint splits
   - Set split ratios (e.g., 70/20/10)
   - Generate organized dataset structure

7. **Train Model**: Train custom YOLO model
   - Configure data.yaml with dataset paths
   - Select base model weights
   - Adjust training hyperparameters
   - Monitor training progress

8. **Analyze Traffic**: Run trained model on new videos
   - Process videos with tracking
   - Generate movement reports (LTR analysis)
   - Export CSV results

## Dependencies

Core dependencies (see [requirements.txt](requirements.txt) for exact versions):
- **ultralytics**: YOLOv8 training and inference
- **opencv-python**: Image and video processing
- **numpy**: Numerical operations
- **pillow**: Image manipulation
- **shapely**: Geometric operations for ROI
- **lap**: Linear assignment problem solver used by the ultralytics trackers (ByteTrack / BoT-SORT)

## Project Structure

```
traffic_studio_accordion/
├── app.py                          # Main GUI application
├── example_usage.py                # NEW: Examples using refactored architecture
├── requirements.txt                # Core dependencies (no PyTorch)
├── requirements-dev.txt            # NEW: Dev dependencies with PyTorch
│
├── traffic_studio/                 # NEW: Refactored architecture (v2.0)
│   ├── domain/                     # Domain models (typed dataclasses)
│   │   ├── sampler.py             # SamplerConfig, SamplerResult
│   │   ├── autolabel.py           # AutolabelConfig, AutolabelResult
│   │   ├── trainer.py             # TrainerConfig, TrainerResult
│   │   ├── analyzer.py            # AnalyzerConfig, AnalyzerResult
│   │   └── project.py             # ProjectModel (refactored from app.py)
│   ├── services/                   # Business logic (testable services)
│   │   ├── protocols.py           # Service interfaces (ISamplerService, etc.)
│   │   ├── sampler.py             # Frame sampling service
│   │   ├── autolabel.py           # Auto-labeling service
│   │   ├── trainer.py             # Training service
│   │   ├── analyzer.py            # Traffic analysis service
│   │   └── dependency_check.py    # System dependency validation
│   ├── infrastructure/             # External integrations
│   │   ├── config.py              # UserConfigManager (cross-platform)
│   │   ├── ffmpeg.py              # FFmpegRunner (ffmpeg wrapper)
│   │   └── yolo.py                # YOLORunner (ultralytics wrapper)
│   ├── utils/                      # Utilities
│   │   ├── logging.py             # Logger factory
│   │   └── validators.py          # Field validators
│   └── container.py                # Dependency injection container
│
├── tests/                          # NEW: Test infrastructure
│   ├── conftest.py                # Pytest fixtures
│   └── unit/
│       └── test_sampler_service.py  # Example unit tests
│
├── studio/                         # Legacy modules (being phased out)
│   ├── sampler.py                 # Frame sampling logic
│   ├── roi_utils.py               # ROI processing
│   ├── autolabel.py               # Auto-labeling with YOLO
│   ├── trainer.py                 # Model training
│   ├── analyzer.py                # Traffic analysis and tracking
│   ├── devices.py                 # CUDA/CPU device resolution
│   ├── qaqc.py                    # Quality control tools
│   ├── threads.py                 # Background worker threads
│   ├── logutil.py                 # Logging utilities
│   └── theme.py                   # GUI theming
│
├── third_party/                   # External tools and scripts
│   ├── frame_sampler.py           # Sub-second frame sampling
│   ├── roi_tool_gui.py            # Interactive ROI editor
│   ├── autolabel_predict.py       # YOLO prediction wrapper
│   ├── train_yolo.py              # YOLO training wrapper
│   ├── traffic_analysis_tracking.py  # Vehicle tracking
│   └── crop_margin_dataset.py     # Dataset cropping
│
├── tools/
│   └── bulk_write_label.py        # Bulk label manipulation CLI
│
├── config/                         # Legacy config (deprecated)
│   └── bulk_presets.json          # Use UserConfigManager instead
│
└── docs/                           # NEW: Architecture documentation
    ├── ARCHITECTURE_REFACTOR.md   # Architecture design
    ├── MIGRATION_GUIDE.md         # Migration steps
    ├── PACKAGING.md               # PyTorch packaging strategy
    └── IMPLEMENTATION_SUMMARY.md  # Implementation overview
```

### Configuration Storage (New)

User configuration is now stored in platform-appropriate directories:
- **Windows**: `%APPDATA%\TrafficStudio`
- **macOS**: `~/Library/Application Support/TrafficStudio`
- **Linux**: `~/.config/traffic-studio`

This ensures configs work correctly when the application is packaged as an executable.

## Configuration

### Data Layout
The application expects a typical YOLO dataset structure:
```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── classes.txt                    # Class names (one per line)
└── data.yaml                      # YOLO dataset config
```

### Example data.yaml
```yaml
path: ./data
train: images/train
val: images/val
test: images/test

nc: 4  # number of classes
names: ['car', 'truck', 'bus', 'motorcycle']
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest --cov=traffic_studio tests/
```

Type checking with mypy:
```bash
mypy traffic_studio/
```

Code formatting with black:
```bash
black traffic_studio/ tests/
```

## Tips and Best Practices

### Workflow
1. **Sampling**: Start with lower sampling rates (e.g., 1 FPS) to keep dataset manageable
2. **ROI Setup**: Carefully define entry/exit zones for accurate traffic flow analysis
3. **Auto-Labeling**: Use high confidence threshold (0.7+) to minimize false positives
4. **QA/QC**: Always review auto-labeled data before training
5. **Training**: Start with a pre-trained YOLOv8s model for faster convergence
6. **Augmentation**: Reduce augmentation if training on consistent lighting conditions
7. **Site-Disjoint Splits**: Use site-based splitting to prevent data leakage
8. **Analysis Throughput**: Untick *Write annotated video* when you only need the CSVs, and export
   the model to TensorRT (`yolo export model=best.pt format=engine`) for 2-3x faster inference on NVIDIA GPUs

### Architecture (New)
1. **Use Container**: Always create services through the Container for proper dependency injection
2. **Validate Configs**: Call `config.validate()` before running operations
3. **Check Results**: Always check `result.success` and `result.errors` after operations
4. **Progress Callbacks**: Use progress callbacks for long-running operations
5. **Type Hints**: Leverage IDE autocomplete with the typed domain models
6. **Testing**: Write unit tests using protocol-based mocks (see [tests/unit/test_sampler_service.py](tests/unit/test_sampler_service.py))

## Troubleshooting

### ffmpeg not found
Ensure ffmpeg is installed and available on your system PATH:
```bash
# Windows: Download from ffmpeg.org and add to PATH
# Linux: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
```

### CUDA/GPU Issues
The application will use CPU if CUDA is unavailable. For GPU acceleration:
- Install CUDA toolkit matching your PyTorch version
- Verify GPU detection: `python -c "import torch; print(torch.cuda.is_available())"`
- See [PACKAGING.md](PACKAGING.md) for detailed PyTorch installation guidance

### Training Crashes
- Reduce batch size if running out of memory
- Use `cache=False` if disk cache causes issues
- Reduce number of workers if encountering data loading errors

### Config Not Saving (Windows Installer)
If you packaged the app and configs aren't saving:
- The new architecture stores configs in `%APPDATA%\TrafficStudio` (user-writable)
- Old architecture used `__file__/config` which breaks when packaged
- Migrate to new UserConfigManager (see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md))

## License

This project integrates multiple components. Please review licensing for third-party dependencies and tools.

## Acknowledgments

Built on top of:
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- ffmpeg for video processing
- Various open-source Python libraries

## Quick Reference

### Common Tasks

**Sample frames from video:**
```python
from traffic_studio.container import Container
from traffic_studio.domain.sampler import SamplerConfig
config = SamplerConfig(video_path=Path("video.mp4"), output_dir=Path("frames/"), fps=1.0)
result = Container().sampler_service().sample_frames(config)
```

**Auto-label images:**
```python
from traffic_studio.domain.autolabel import AutolabelConfig
config = AutolabelConfig(images_dir=Path("frames/"), model_path=Path("best.pt"), conf=0.70)
result = Container().autolabel_service().autolabel(config)
```

**Train a model:**
```python
from traffic_studio.domain.trainer import TrainerConfig
config = TrainerConfig(data_yaml=Path("data.yaml"), epochs=50, imgsz=640)
result = Container().trainer_service().train(config)
```

**Check system dependencies:**
```python
service = Container().dependency_check_service()
results = service.check_all()
for dep, (available, info) in results.items():
    print(f"{dep}: {'✓' if available else '✗'} {info}")
```

### Documentation Index

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | This file - overview and quick start |
| [ARCHITECTURE_REFACTOR.md](ARCHITECTURE_REFACTOR.md) | Detailed architecture design and rationale |
| [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) | Step-by-step migration from old to new architecture |
| [PACKAGING.md](PACKAGING.md) | PyTorch packaging strategies (CPU vs CUDA builds) |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Complete implementation summary |
| [example_usage.py](example_usage.py) | Working code examples for all workflows |

### API Quick Reference

**Domain Models:**
- `SamplerConfig` / `SamplerResult` - Frame sampling
- `AutolabelConfig` / `AutolabelResult` - YOLO auto-labeling
- `TrainerConfig` / `TrainerResult` - Model training
- `AnalyzerConfig` / `AnalyzerResult` - Traffic analysis

**Services:**
- `SamplerService.sample_frames()` - Extract frames from videos
- `AutolabelService.autolabel()` - Generate YOLO labels
- `TrainerService.train()` - Train YOLO models
- `AnalyzerService.analyze()` - Analyze traffic videos
- `DependencyCheckService.check_all()` - Validate system dependencies

**Infrastructure:**
- `UserConfigManager` - Cross-platform config management
- `FFmpegRunner` - ffmpeg wrapper
- `YOLORunner` - YOLO wrapper with device management
- `Container` - Dependency injection container

---

For questions, issues, or contributions, please refer to the project repository.
