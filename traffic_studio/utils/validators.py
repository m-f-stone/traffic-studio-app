"""Field validation utilities.

Migrated from app.py to be reusable across the application.
"""

from pathlib import Path
import subprocess


class FieldValidator:
    """Validation functions for various field types."""

    @staticmethod
    def path_exists(path_str: str, must_be_dir: bool = False, must_be_file: bool = False) -> bool:
        """Check if path exists and optionally if it's a file or directory.

        Args:
            path_str: Path string to validate
            must_be_dir: If True, path must be a directory
            must_be_file: If True, path must be a file

        Returns:
            True if path is valid
        """
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
        """Check if string is a valid float within optional bounds.

        Args:
            val: String to validate
            min_val: Optional minimum value
            max_val: Optional maximum value

        Returns:
            True if valid float within bounds
        """
        if not val or not val.strip():
            return False

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
        """Check if string is a valid integer within optional bounds.

        Args:
            val: String to validate
            min_val: Optional minimum value
            max_val: Optional maximum value

        Returns:
            True if valid integer within bounds
        """
        if not val or not val.strip():
            return False

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
        """Check if ffmpeg is available on PATH.

        Returns:
            True if ffmpeg is available
        """
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def check_cuda() -> bool:
        """Check if CUDA is available via PyTorch.

        Returns:
            True if CUDA is available
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
