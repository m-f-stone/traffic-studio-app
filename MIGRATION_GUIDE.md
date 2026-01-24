# Migration Guide: Refactored Architecture

This guide explains how to migrate from the old architecture to the new refactored architecture.

## Overview of Changes

The refactoring addresses six major architectural issues:

1. **Monolithic app.py** → Separated into domain/services/UI layers
2. **Stringly-typed state** → Proper typed dataclasses
3. **Global dependencies** → Dependency injection
4. **sys.argv hacks** → Direct function calls
5. **Non-portable config** → Cross-platform user config directories
6. **No test seam** → Protocol-based service interfaces with mocks

## New Structure

```
traffic_studio_accordion/
├── traffic_studio/              # NEW: Main package
│   ├── domain/                  # Domain models (typed dataclasses)
│   ├── services/                # Business logic (testable services)
│   ├── infrastructure/          # External integrations (ffmpeg, YOLO, config)
│   ├── utils/                   # Utilities (logging, validation)
│   └── container.py             # Dependency injection container
│
├── studio/                      # OLD: Legacy modules (being phased out)
│   ├── sampler.py              # → traffic_studio/services/sampler.py
│   ├── autolabel.py            # → traffic_studio/services/autolabel.py
│   ├── trainer.py              # → traffic_studio/services/trainer.py
│   ├── analyzer.py             # → traffic_studio/services/analyzer.py
│   └── logutil.py              # → traffic_studio/utils/logging.py
│
├── app.py                       # UI layer (to be refactored)
├── tests/                       # NEW: Unit and integration tests
└── config/                      # OLD: Will be migrated to user config dir
```

## Migration Path

### Phase 1: Use New Services Alongside Old Code (Current)

You can start using the new architecture immediately without breaking existing code:

```python
# Old way (still works):
from studio import sampler
result = sampler.sample_frames(video, out, fps=1.0)

# New way (recommended):
from traffic_studio.container import Container
from traffic_studio.domain.sampler import SamplerConfig

container = Container()
config = SamplerConfig(
    video_path=video,
    output_dir=out,
    fps=1.0
)
result = container.sampler_service().sample_frames(config)
```

### Phase 2: Migrate UI Code (Future)

The UI layer (app.py) will be refactored to use presenters and views:

```python
# Current: Monolithic app.py with all concerns mixed
class TrafficStudio(tk.Tk):
    def __init__(self):
        # ... 2000+ lines of mixed UI/logic/state ...

    def _do_sample(self):
        # Direct calls to studio modules
        # StringVar everywhere
        # No separation of concerns

# Future: Clean separation with presenters
class TrafficStudio(tk.Tk):
    def __init__(self, container: Container):
        self.container = container
        # Build UI from views
        self.sampler_view = SamplerView(self, SamplerPresenter(container))

class SamplerPresenter:
    def __init__(self, container: Container):
        self.service = container.sampler_service()

    def sample_frames(self, view_data: dict) -> Result:
        # Convert UI data to domain config
        config = SamplerConfig(**view_data)
        # Call service
        return self.service.sample_frames(config)
```

## Key Concepts

### 1. Domain Models (Typed Dataclasses)

**Old:**
```python
# Everything is StringVar
self.video = tk.StringVar(value="")
self.fps = tk.StringVar(value="1.0")

# Manual validation, no type safety
if not self.fps.get():
    messagebox.showerror("Error", "FPS required")
    return

fps = float(self.fps.get())  # Runtime error if invalid
```

**New:**
```python
from traffic_studio.domain.sampler import SamplerConfig

config = SamplerConfig(
    video_path=Path("/path/to/video.mp4"),
    output_dir=Path("/output"),
    fps=1.0  # Type-checked at creation
)

# Built-in validation
errors = config.validate()
if errors:
    show_errors(errors)
    return

# Type-safe access
fps: float = config.fps  # IDE knows this is float
```

### 2. Service Layer (Dependency Injection)

**Old:**
```python
# Global logger instance
from studio.logutil import setup_logger
log = setup_logger()  # Global!

def sample_frames(...):
    log.info("Sampling...")  # Uses global
    # Hard to test - must mock global
```

**New:**
```python
class SamplerService:
    def __init__(self, ffmpeg: FFmpegRunner, logger: Logger):
        self.ffmpeg = ffmpeg  # Injected dependency
        self.logger = logger  # Injected dependency

    def sample_frames(self, config: SamplerConfig) -> SamplerResult:
        self.logger.info("Sampling...")
        # Easy to test - inject mocks
```

### 3. Infrastructure Layer

**Old:**
```python
# Direct subprocess calls
import subprocess
subprocess.run(["ffmpeg", "-i", video, ...])  # No abstraction

# No error handling
# No logging
# Hard to test
```

**New:**
```python
from traffic_studio.infrastructure.ffmpeg import FFmpegRunner

ffmpeg = FFmpegRunner(logger=logger)

# Clean interface
ffmpeg.extract_frames_fps(video, output_dir, fps=1.0)

# Testable - can mock FFmpegRunner
```

### 4. Configuration Management

**Old:**
```python
# Hardcoded paths - breaks when packaged
PRESET_FILE = Path(__file__).parent / "config" / "presets.json"
# In Program Files: Read-only! 💥
```

**New:**
```python
from traffic_studio.infrastructure.config import UserConfigManager

config = UserConfigManager()
# Windows: %APPDATA%\TrafficStudio
# macOS: ~/Library/Application Support/TrafficStudio
# Linux: ~/.config/traffic-studio

presets = config.get_presets("sampler")
config.save_presets("sampler", presets)
```

### 5. Protocol-Based Testing

**Old:**
```python
# Testing requires real ffmpeg/YOLO
def test_sampling():
    result = sampler.sample_frames(...)  # Runs real ffmpeg!
    # Slow, brittle, requires ffmpeg installed
```

**New:**
```python
from unittest.mock import Mock

def test_sampling():
    # Mock the infrastructure
    mock_ffmpeg = Mock(spec=FFmpegRunner)
    mock_ffmpeg.run = Mock()

    # Inject mock into service
    service = SamplerService(ffmpeg=mock_ffmpeg, logger=logger)

    # Test business logic without running ffmpeg
    result = service.sample_frames(config)

    # Verify behavior
    assert mock_ffmpeg.run.called
```

## Common Migration Patterns

### Pattern 1: Migrating a Workflow Function

**Before:**
```python
def _do_sample(self):
    video = self.video.get()
    out = self.sample_out.get()
    fps = float(self.fps.get()) if self.fps.get() else None

    if not video:
        messagebox.showerror("Error", "Video required")
        return

    # Call old module
    from studio import sampler
    sampler.sample_frames(Path(video), Path(out), fps=fps)
```

**After:**
```python
def _do_sample(self):
    # Build domain config from UI state
    config = SamplerConfig(
        video_path=Path(self.video.get()),
        output_dir=Path(self.sample_out.get()),
        fps=float(self.fps.get()) if self.fps.get() else None
    )

    # Validate
    if not config.is_valid:
        messagebox.showerror("Validation Error", "\n".join(config.validate()))
        return

    # Call service
    result = self.container.sampler_service().sample_frames(config)

    # Handle result
    if result.success:
        messagebox.showinfo("Success", f"Extracted {result.frames_extracted} frames")
    else:
        messagebox.showerror("Error", "\n".join(result.errors))
```

### Pattern 2: Migrating Config Storage

**Before:**
```python
RECENT_PROJECTS = Path(__file__).parent / "config" / "recent_projects.json"

def load_recent_projects():
    with open(RECENT_PROJECTS, 'r') as f:
        return json.load(f)
```

**After:**
```python
def __init__(self):
    self.config = UserConfigManager()  # Injected or created

def load_recent_projects(self):
    return self.config.get_recent_projects()
```

### Pattern 3: Migrating Validation

**Before:**
```python
class FieldValidator:
    @staticmethod
    def path_exists(path_str: str) -> bool:
        # ... validation logic ...

# Used in app.py:
if not FieldValidator.path_exists(self.video.get()):
    show_error("Invalid path")
```

**After:**
```python
# Validation built into domain model
config = SamplerConfig(
    video_path=Path(self.video.get()),
    # ...
)

errors = config.validate()  # Built-in validation
if errors:
    show_errors(errors)
```

## Testing Examples

### Unit Test with Mocks

```python
def test_sampler_service(temp_dir):
    """Test sampling service with mock ffmpeg."""
    # Arrange
    mock_ffmpeg = Mock(spec=FFmpegRunner)
    mock_ffmpeg.run = Mock()
    mock_ffmpeg.path = "/usr/bin/ffmpeg"

    service = SamplerService(ffmpeg=mock_ffmpeg, logger=logger)

    video = temp_dir / "test.mp4"
    video.write_text("dummy")

    config = SamplerConfig(
        video_path=video,
        output_dir=temp_dir / "output",
        fps=1.0
    )

    # Act
    result = service.sample_frames(config)

    # Assert
    assert mock_ffmpeg.run.called
    assert result.frames_extracted >= 0
```

### Integration Test

```python
def test_full_workflow(temp_dir):
    """Test full sampling workflow with real container."""
    # Create container with real dependencies
    container = Container()

    # (Assumes ffmpeg is installed)
    config = SamplerConfig(...)
    result = container.sampler_service().sample_frames(config)

    assert result.success
    assert result.frames_extracted > 0
```

## Checklist for Migration

- [ ] Replace global logger with injected logger
- [ ] Convert StringVar state to domain models
- [ ] Use UserConfigManager for config storage
- [ ] Replace direct subprocess calls with infrastructure wrappers
- [ ] Add validation to domain models
- [ ] Write unit tests with mocks
- [ ] Remove sys.argv manipulation where possible
- [ ] Use Container for service creation
- [ ] Document PyTorch packaging strategy
- [ ] Update README with new structure

## Gradual Migration Strategy

1. **Week 1-2**: Use new services alongside old code
   - Start using `Container` in new code
   - Keep old code working

2. **Week 3-4**: Migrate one workflow at a time
   - Sample → Autolabel → Train → Analyze
   - Test each migration thoroughly

3. **Week 5-6**: Refactor UI layer
   - Extract presenters from app.py
   - Create view components
   - Wire up with dependency injection

4. **Week 7-8**: Testing and documentation
   - Write comprehensive tests
   - Update user documentation
   - Create packaging scripts

## Benefits After Migration

✅ **Testability**: Mock services for unit tests
✅ **Maintainability**: Clear separation of concerns
✅ **Type Safety**: Catch errors at development time
✅ **Portability**: Config works when packaged
✅ **Scalability**: Easy to add new features
✅ **Documentation**: Self-documenting typed interfaces

## Questions or Issues?

See [ARCHITECTURE_REFACTOR.md](ARCHITECTURE_REFACTOR.md) for detailed architecture design.
