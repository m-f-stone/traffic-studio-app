## PyTorch Packaging Strategy for Traffic Studio

### Problem Statement

PyTorch is platform-specific and hardware-specific. We cannot include one universal wheel that works across all platforms and configurations. The key challenges are:

1. **Platform Variations**: Windows, macOS, Linux require different builds
2. **Hardware Variations**: CPU-only vs CUDA 11.8 vs CUDA 12.1+ vs ROCm (AMD)
3. **Size Constraints**: CUDA-enabled PyTorch is ~2GB, CPU-only is ~200MB
4. **User Environment**: Some users have GPUs, others don't

### Recommended Approach

#### Option 1: **CPU-Only Installer** (Recommended for General Distribution)

**Pros:**
- Smaller download size (~200MB for PyTorch)
- Works on all systems (universal compatibility)
- Simpler packaging and distribution
- Users can manually upgrade to GPU version if needed

**Cons:**
- Slower inference and training (10-50x slower without GPU)
- Not optimal for production use on GPU systems

**Build Steps:**
```bash
# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Package with PyInstaller
pyinstaller app.spec
```

**When to use:** Default installer for first-time users, evaluation, systems without GPUs

---

#### Option 2: **CUDA 11.8 Installer** (For NVIDIA GPUs)

**Pros:**
- Fast inference and training on NVIDIA GPUs
- Broad GPU compatibility (RTX 20/30/40-series, GTX 16-series, etc.)
- CUDA runtime bundled (users don't need to install CUDA toolkit)

**Cons:**
- Large download (~2GB)
- Only works on systems with NVIDIA GPUs
- Requires recent NVIDIA drivers (version 450.80+)

**Build Steps:**
```bash
# Install CUDA 11.8 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Package with PyInstaller
pyinstaller app.spec
```

**When to use:** Primary installer for users with NVIDIA GPUs (RTX 20/30-series, GTX 16-series)

---

#### Option 3: **CUDA 12.1+ Installer** (For Latest NVIDIA GPUs)

**Pros:**
- Best performance on newest GPUs (RTX 40-series, H100, etc.)
- Latest CUDA features and optimizations

**Cons:**
- Requires very recent drivers (version 525.60+)
- Same large download size as CUDA 11.8

**Build Steps:**
```bash
# Install CUDA 12.1 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Package with PyInstaller
pyinstaller app.spec
```

**When to use:** For users with latest-generation NVIDIA GPUs who want maximum performance

---

#### Option 4: **User-Installed PyTorch** (Developer Workflow)

**Pros:**
- Smallest installer (doesn't bundle PyTorch at all)
- Users can choose their own PyTorch configuration
- Best for developers and advanced users

**Cons:**
- Requires manual PyTorch installation
- More complex setup process
- Risk of version incompatibility

**Implementation:**
```python
# In app startup code:
try:
    import torch
except ImportError:
    show_install_instructions()
    sys.exit(1)

def show_install_instructions():
    """Display PyTorch installation instructions."""
    message = """
    PyTorch is not installed.

    Please install PyTorch manually:

    CPU-only (Universal):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

    NVIDIA GPU (CUDA 11.8):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    NVIDIA GPU (CUDA 12.1):
    pip install torch torchvision

    For more options, visit: https://pytorch.org/get-started/locally/
    """
    messagebox.showerror("PyTorch Required", message)
```

**When to use:** For developers or when distributing as a Python package (not standalone executable)

---

### Production Recommendation

**Distribute Two Installers:**

1. **`TrafficStudio-{version}-CPU.exe`** (Windows)
   - CPU-only PyTorch
   - ~200MB download
   - Universal compatibility
   - For evaluation and systems without GPUs

2. **`TrafficStudio-{version}-CUDA118.exe`** (Windows)
   - CUDA 11.8 PyTorch
   - ~2GB download
   - For NVIDIA GPU systems
   - Best performance

**Runtime Detection:**

On first launch, detect hardware and show recommendation:

```python
def check_gpu_recommendation():
    """Check if user should be using GPU version."""
    import torch

    if torch.cuda.is_available():
        # User has GPU but running CPU-only version
        show_warning("GPU Detected",
            "An NVIDIA GPU was detected, but you're running the CPU-only version. "
            "For 10-50x faster performance, download the CUDA version from our website.")

    elif not torch.cuda.is_available() and "cuda" in torch.__file__:
        # User has CUDA version but no GPU
        show_warning("No GPU Detected",
            "You're running the CUDA version, but no NVIDIA GPU was detected. "
            "The application will run on CPU. Consider downloading the CPU-only version "
            "for a smaller installation size.")
```

---

### Development Workflow

For **development** (not packaging):

1. **requirements.txt** - Core dependencies (no PyTorch):
   ```txt
   ultralytics==8.3.178
   opencv-python==4.12.0.88
   numpy==2.2.6
   pillow==11.2.1
   shapely==2.1.1
   filterpy==1.4.5
   lap==0.5.12
   ```

2. **requirements-dev.txt** - Development dependencies:
   ```txt
   -r requirements.txt
   --extra-index-url https://download.pytorch.org/whl/cu118
   torch
   torchvision
   pytest
   mypy
   black
   ```

3. **README.md** - Installation instructions:
   ```markdown
   ## Installation for Developers

   1. Clone the repository
   2. Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```

   3. Install PyTorch (choose one):

      **CPU-only** (universal):
      ```bash
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      ```

      **NVIDIA GPU (CUDA 11.8)**:
      ```bash
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
      ```

      **NVIDIA GPU (CUDA 12.1)**:
      ```bash
      pip install torch torchvision
      ```

   4. Run the application:
      ```bash
      python app.py
      ```
   ```

---

### PyInstaller Configuration

**app.spec** (CPU version):
```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config', 'config'),
        ('third_party', 'third_party'),
    ],
    hiddenimports=[
        'torch',
        'torchvision',
        'ultralytics',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TrafficStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TrafficStudio',
)
```

**Build script** (build.py):
```python
"""Build script for creating installers."""

import subprocess
import sys
from pathlib import Path

def build_cpu():
    """Build CPU-only version."""
    print("Building CPU version...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ], check=True)

    subprocess.run(["pyinstaller", "app.spec"], check=True)

def build_cuda118():
    """Build CUDA 11.8 version."""
    print("Building CUDA 11.8 version...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ], check=True)

    subprocess.run(["pyinstaller", "app.spec"], check=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["cpu", "cuda118"], required=True)
    args = parser.parse_args()

    if args.variant == "cpu":
        build_cpu()
    elif args.variant == "cuda118":
        build_cuda118()
```

---

### User Documentation

**Quick Start Guide:**

```markdown
# Which Version Should I Download?

## Do you have an NVIDIA GPU?

**Yes** → Download **TrafficStudio-CUDA** for best performance
- 10-50x faster inference and training
- Requires NVIDIA GPU (GTX 16-series or newer, RTX 20/30/40-series)
- Requires recent drivers (version 450.80+)
- ~2GB download

**No** or **Unsure** → Download **TrafficStudio-CPU**
- Works on all systems
- Smaller download (~200MB)
- Slower performance (acceptable for evaluation)

## Already Installed?

The application will automatically detect your hardware and show a
recommendation if you're using the wrong version.
```

---

### Summary

**For Production:**
- Build 2 installers: CPU and CUDA 11.8
- Default to CPU for universal compatibility
- Detect hardware at runtime and recommend upgrade if needed

**For Development:**
- Don't include PyTorch in requirements.txt
- Document manual PyTorch installation in README
- Use requirements-dev.txt with CUDA 11.8 as default

**For Advanced Users:**
- Provide option to use their own PyTorch installation
- Document all supported configurations
