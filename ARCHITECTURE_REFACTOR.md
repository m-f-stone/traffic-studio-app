# Architecture Refactoring Plan

## Overview
This document outlines the refactoring strategy to address six critical architectural issues in Traffic Studio.

## Issues Addressed

### 1. Monolithic app.py
**Problem**: app.py owns all state, validation, layout, and orchestration (~2000+ lines).

**Solution**: Implement MVP/Presenter pattern
- **Models**: Domain objects with proper typing (not StringVar)
- **Views**: Pure UI components that delegate to presenters
- **Presenters**: Coordinate between UI and services, handle validation

### 2. Stringly-Typed State
**Problem**: Everything is `tk.StringVar()` even for numbers/paths/booleans.

**Solution**: Create typed domain models
```python
@dataclass
class SamplerConfig:
    video_path: Path
    output_dir: Path
    interval: Optional[float] = None
    fps: Optional[float] = None
    method: Literal["fps", "select"] = "fps"
    tolerance: float = 0.002
```

### 3. Non-Library-Grade Core Logic
**Problem**: Modules rely on global logger, sys.argv manipulation.

**Solution**: Refactor to dependency-injectable services
```python
class SamplerService:
    def __init__(self, logger: Logger, ffmpeg_path: str):
        self.logger = logger
        self.ffmpeg = ffmpeg_path

    def sample_frames(self, config: SamplerConfig) -> SamplerResult:
        # Pure function logic
```

### 4. Non-Portable Config Storage
**Problem**: `Path(__file__).parent / "config"` won't work when packaged.

**Solution**: Use platform-appropriate user config directories
- Windows: `%APPDATA%\TrafficStudio`
- macOS: `~/Library/Application Support/TrafficStudio`
- Linux: `~/.config/traffic-studio`

### 5. Missing PyTorch/CUDA Strategy
**Problem**: requirements.txt doesn't specify torch; unclear CPU vs CUDA story.

**Solution**: Document explicit packaging strategy
- Provide CPU-only installer (smaller, universal)
- Provide CUDA 11.8/12.1 installers (GPU acceleration)
- Auto-detect at runtime and warn if mismatch
- Document manual torch installation for developers

### 6. No Test Seam
**Problem**: Hard to test without running ffmpeg/YOLO.

**Solution**: Protocol-based service interfaces with mocks
```python
class ISamplerService(Protocol):
    def sample_frames(self, config: SamplerConfig) -> SamplerResult: ...

class MockSamplerService:
    def sample_frames(self, config: SamplerConfig) -> SamplerResult:
        return SamplerResult(frames_extracted=42, output_dir=config.output_dir)
```

## New Directory Structure

```
traffic_studio_accordion/
├── app.py                          # Thin launcher
├── requirements-dev.txt            # Dev dependencies (pytest, mypy)
├── requirements.txt                # Core dependencies (NO torch)
├── pyproject.toml                  # Build configuration
├── PACKAGING.md                    # PyTorch packaging guide
│
├── traffic_studio/                 # Main package (renamed from studio)
│   ├── __init__.py
│   ├── domain/                     # Domain models (typed dataclasses)
│   │   ├── __init__.py
│   │   ├── sampler.py              # SamplerConfig, SamplerResult
│   │   ├── autolabel.py            # AutolabelConfig, AutolabelResult
│   │   ├── trainer.py              # TrainerConfig, TrainerResult
│   │   ├── analyzer.py             # AnalyzerConfig, AnalyzerResult
│   │   └── project.py              # ProjectModel (refactored from app.py)
│   │
│   ├── services/                   # Service layer (testable business logic)
│   │   ├── __init__.py
│   │   ├── protocols.py            # Service interfaces (Protocols)
│   │   ├── sampler.py              # SamplerService
│   │   ├── autolabel.py            # AutolabelService
│   │   ├── trainer.py              # TrainerService
│   │   ├── analyzer.py             # AnalyzerService
│   │   └── dependency_check.py     # DependencyCheckService
│   │
│   ├── infrastructure/             # External integrations
│   │   ├── __init__.py
│   │   ├── ffmpeg.py               # FFmpegRunner
│   │   ├── yolo.py                 # YOLORunner (wraps ultralytics)
│   │   └── config.py               # UserConfigManager
│   │
│   ├── ui/                         # UI components (presenters + views)
│   │   ├── __init__.py
│   │   ├── main_window.py          # Main application window
│   │   ├── components/             # Reusable UI widgets
│   │   │   ├── validated_entry.py
│   │   │   ├── accordion.py
│   │   │   └── progress_panel.py
│   │   ├── presenters/             # Presenters for each workflow
│   │   │   ├── sampler_presenter.py
│   │   │   ├── autolabel_presenter.py
│   │   │   ├── trainer_presenter.py
│   │   │   └── analyzer_presenter.py
│   │   └── views/                  # Pure UI layouts
│   │       ├── sampler_view.py
│   │       ├── autolabel_view.py
│   │       ├── trainer_view.py
│   │       └── analyzer_view.py
│   │
│   ├── utils/                      # Utilities
│   │   ├── logging.py              # Logger factory (no globals)
│   │   └── validators.py           # Validation functions
│   │
│   └── container.py                # Dependency injection container
│
├── third_party/                    # External scripts (minimal changes)
│   └── ...
│
├── tests/                          # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/
│   │   ├── test_sampler_service.py
│   │   ├── test_autolabel_service.py
│   │   └── ...
│   └── integration/
│       └── test_workflows.py
│
└── config/                         # Bundled defaults (read-only when packaged)
    └── default_presets.json
```

## Implementation Phases

### Phase 1: Foundation (Domain Models + Config)
1. Create `traffic_studio/domain/` with typed models
2. Create `traffic_studio/infrastructure/config.py` for user config
3. Migrate `ProjectModel` from app.py to `domain/project.py`
4. Add validators to `utils/validators.py`

### Phase 2: Service Layer
1. Define service protocols in `services/protocols.py`
2. Refactor `sampler.py` → `SamplerService`
3. Refactor `autolabel.py` → `AutolabelService`
4. Refactor `trainer.py` → `TrainerService`
5. Refactor `analyzer.py` → `AnalyzerService`
6. Create `DependencyCheckService`

### Phase 3: Infrastructure
1. Create `FFmpegRunner` wrapper
2. Create `YOLORunner` wrapper
3. Create `UserConfigManager` for cross-platform config

### Phase 4: Dependency Injection
1. Create `Container` class in `container.py`
2. Wire up service dependencies
3. Provide mock container for tests

### Phase 5: UI Refactoring
1. Extract UI components to `ui/components/`
2. Create presenters in `ui/presenters/`
3. Create views in `ui/views/`
4. Refactor `app.py` to use new architecture

### Phase 6: Testing
1. Add pytest configuration
2. Write unit tests for services (with mocks)
3. Write integration tests for workflows
4. Add CI configuration

### Phase 7: Packaging
1. Create `PACKAGING.md` with PyTorch strategy
2. Update `requirements.txt` (remove implicit torch dep)
3. Add `pyproject.toml` for build config
4. Create installer scripts

## Migration Strategy

**Incremental Migration** (no big-bang rewrite):
1. Create new structure alongside old
2. Gradually move functionality
3. Keep old code working until fully migrated
4. Use feature flags if needed for A/B testing

## Type System Example

```python
# traffic_studio/domain/sampler.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
from enum import Enum

class SamplingMethod(Enum):
    FPS = "fps"
    SELECT = "select"

@dataclass
class SamplerConfig:
    """Configuration for video frame sampling."""
    video_path: Path
    output_dir: Path
    interval: Optional[float] = None  # Seconds between frames
    fps: Optional[float] = None       # Frames per second
    method: SamplingMethod = SamplingMethod.FPS
    tolerance: float = 0.002
    file_extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi")

    def validate(self) -> list[str]:
        """Return list of validation errors."""
        errors = []
        if not self.video_path.exists():
            errors.append(f"Video path does not exist: {self.video_path}")
        if self.interval is None and self.fps is None:
            errors.append("Must specify either interval or fps")
        if self.interval is not None and self.interval <= 0:
            errors.append("Interval must be positive")
        if self.fps is not None and self.fps <= 0:
            errors.append("FPS must be positive")
        return errors

@dataclass
class SamplerResult:
    """Result of frame sampling operation."""
    frames_extracted: int
    output_dir: Path
    videos_processed: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0
```

## Service Layer Example

```python
# traffic_studio/services/sampler.py
from pathlib import Path
from logging import Logger
from typing import Optional
from ..domain.sampler import SamplerConfig, SamplerResult, SamplingMethod
from ..infrastructure.ffmpeg import FFmpegRunner

class SamplerService:
    """Service for sampling video frames using ffmpeg."""

    def __init__(self, ffmpeg: FFmpegRunner, logger: Logger):
        self.ffmpeg = ffmpeg
        self.logger = logger

    def sample_frames(
        self,
        config: SamplerConfig,
        progress_callback: Optional[callable] = None,
        cancel_check: Optional[callable] = None
    ) -> SamplerResult:
        """Sample frames from video(s) based on configuration."""
        # Validate config
        errors = config.validate()
        if errors:
            return SamplerResult(
                frames_extracted=0,
                output_dir=config.output_dir,
                errors=errors
            )

        # Collect video sources
        sources = self._collect_sources(config)

        # Process each video
        total_frames = 0
        processed = []
        errors = []

        for i, video in enumerate(sources):
            if cancel_check and cancel_check():
                self.logger.info("Sampling cancelled by user")
                break

            if progress_callback:
                progress_callback(
                    fraction=i / len(sources),
                    message=f"Processing {video.name} ({i+1}/{len(sources)})"
                )

            try:
                frames = self._sample_video(video, config)
                total_frames += frames
                processed.append(video)
            except Exception as e:
                self.logger.exception(f"Failed to sample {video}")
                errors.append(f"{video.name}: {str(e)}")

        return SamplerResult(
            frames_extracted=total_frames,
            output_dir=config.output_dir,
            videos_processed=processed,
            errors=errors
        )

    def _collect_sources(self, config: SamplerConfig) -> list[Path]:
        """Collect video files from path."""
        if config.video_path.is_file():
            return [config.video_path]

        # Collect from directory
        sources = []
        for ext in config.file_extensions:
            sources.extend(config.video_path.glob(f"*{ext}"))
        return sorted(sources)

    def _sample_video(self, video: Path, config: SamplerConfig) -> int:
        """Sample a single video using ffmpeg."""
        cmd = self._build_command(video, config)
        self.logger.info(f"Sampling {video.name}")
        self.ffmpeg.run(cmd)

        # Count extracted frames
        pattern = f"{video.stem}_*.jpg"
        frames = list(config.output_dir.glob(pattern))
        return len(frames)

    def _build_command(self, video: Path, config: SamplerConfig) -> list[str]:
        """Build ffmpeg command for sampling."""
        # Delegate to existing logic in third_party/frame_sampler.py
        from third_party.frame_sampler import build_cmd
        return build_cmd(
            self.ffmpeg.path,
            video,
            config.output_dir,
            config.interval,
            config.fps,
            config.method.value,
            config.tolerance
        )
```

## User Config Management

```python
# traffic_studio/infrastructure/config.py
from pathlib import Path
import platform
import json
from typing import Any

class UserConfigManager:
    """Manages user configuration in platform-appropriate directories."""

    def __init__(self, app_name: str = "TrafficStudio"):
        self.app_name = app_name
        self._config_dir = self._get_config_dir()
        self._ensure_config_dir()

    def _get_config_dir(self) -> Path:
        """Get platform-appropriate config directory."""
        system = platform.system()

        if system == "Windows":
            base = Path.home() / "AppData" / "Roaming"
        elif system == "Darwin":  # macOS
            base = Path.home() / "Library" / "Application Support"
        else:  # Linux and others
            base = Path.home() / ".config"

        return base / self.app_name

    def _ensure_config_dir(self):
        """Create config directory if it doesn't exist."""
        self._config_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config_dir(self) -> Path:
        """Get the config directory path."""
        return self._config_dir

    def get_config_file(self, name: str) -> Path:
        """Get path to a config file."""
        return self._config_dir / name

    def load_json(self, name: str, default: Any = None) -> Any:
        """Load JSON config file."""
        path = self.get_config_file(name)
        if not path.exists():
            return default

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default

    def save_json(self, name: str, data: Any):
        """Save JSON config file."""
        path = self.get_config_file(name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get_recent_projects(self) -> list[str]:
        """Get list of recent project paths."""
        return self.load_json("recent_projects.json", default=[])

    def save_recent_projects(self, projects: list[str]):
        """Save recent projects list."""
        self.save_json("recent_projects.json", projects[:10])

    def get_presets(self, category: str) -> dict:
        """Get presets for a category."""
        return self.load_json(f"{category}_presets.json", default={})

    def save_presets(self, category: str, presets: dict):
        """Save presets for a category."""
        self.save_json(f"{category}_presets.json", presets)
```

## PyTorch Packaging Strategy

Create `PACKAGING.md`:

```markdown
# PyTorch Packaging Strategy

## Problem
PyTorch is platform and hardware-specific. We can't include one universal wheel.

## Options

### Option 1: CPU-Only Installer (Recommended for General Distribution)
- Include `torch` and `torchvision` CPU wheels
- Smaller download (~200MB vs 2GB+)
- Works on all systems
- Slower inference/training

**Build command:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pyinstaller app.spec
```

### Option 2: CUDA 11.8 Installer (For NVIDIA GPU Users)
- Include CUDA 11.8 compatible torch
- Requires NVIDIA drivers (CUDA runtime bundled)
- Fast inference/training
- Larger download (~2GB)

**Build command:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pyinstaller app.spec
```

### Option 3: CUDA 12.1 Installer (For Latest NVIDIA GPUs)
- For newer GPUs (RTX 40-series, etc.)
- Similar to Option 2

### Option 4: Let Users Install PyTorch (Developer Workflow)
- Don't bundle torch at all
- Detect at startup and show install instructions
- Users follow: https://pytorch.org/get-started/locally/

## Recommendation

**For production installers:**
1. Build 2 installers: CPU and CUDA 11.8
2. Auto-detect GPU at runtime
3. Warn if running CPU-only on GPU system

**For development:**
1. `requirements.txt` doesn't include torch
2. `requirements-dev.txt` includes: `--extra-index-url https://download.pytorch.org/whl/cu118`
3. README.md explains manual install
```

## Benefits Summary

| Issue | Current State | Refactored State |
|-------|---------------|------------------|
| **Monolithic app.py** | 2000+ lines, all concerns mixed | Separated into domain/services/ui layers |
| **Stringly-typed** | All tk.StringVar | Typed dataclasses with validation |
| **Global dependencies** | `log = setup_logger()` everywhere | Injected via constructor |
| **sys.argv hacks** | Manipulates sys.argv to call main() | Direct function calls |
| **Config storage** | `__file__/config` (breaks when packaged) | User config dir (cross-platform) |
| **No test seam** | Must run ffmpeg/YOLO to test | Protocol-based mocks |
| **PyTorch unclear** | requirements.txt ignores it | Explicit packaging docs + detection |

## Next Steps

1. Review this plan with stakeholders
2. Begin Phase 1 implementation
3. Set up CI/CD for testing
4. Create packaging scripts
