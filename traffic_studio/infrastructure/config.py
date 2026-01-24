"""Cross-platform user configuration management.

Handles storage of user preferences, recent projects, and presets in
platform-appropriate directories that remain writable after packaging.
"""

from pathlib import Path
import platform
import json
from typing import Any, Optional
from logging import Logger, getLogger


class UserConfigManager:
    """Manages user configuration in platform-appropriate directories.

    Storage locations:
    - Windows: %APPDATA%\\TrafficStudio
    - macOS: ~/Library/Application Support/TrafficStudio
    - Linux: ~/.config/traffic-studio

    This ensures configuration remains writable when the application is
    packaged and installed in read-only system directories.
    """

    def __init__(self, app_name: str = "TrafficStudio", logger: Optional[Logger] = None):
        """Initialize config manager.

        Args:
            app_name: Application name for config directory
            logger: Optional logger instance
        """
        self.app_name = app_name
        self.logger = logger or getLogger(__name__)
        self._config_dir = self._get_config_dir()
        self._ensure_config_dir()

    def _get_config_dir(self) -> Path:
        """Get platform-appropriate config directory.

        Returns:
            Path to config directory
        """
        system = platform.system()

        if system == "Windows":
            # %APPDATA%\TrafficStudio
            base = Path.home() / "AppData" / "Roaming"
        elif system == "Darwin":  # macOS
            # ~/Library/Application Support/TrafficStudio
            base = Path.home() / "Library" / "Application Support"
        else:  # Linux and others
            # ~/.config/traffic-studio
            base = Path.home() / ".config"

        return base / self.app_name

    def _ensure_config_dir(self):
        """Create config directory if it doesn't exist."""
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Config directory: {self._config_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create config directory: {e}")

    @property
    def config_dir(self) -> Path:
        """Get the config directory path.

        Returns:
            Path to config directory
        """
        return self._config_dir

    def get_config_file(self, name: str) -> Path:
        """Get path to a config file.

        Args:
            name: Config file name (e.g., 'recent_projects.json')

        Returns:
            Full path to config file
        """
        return self._config_dir / name

    def load_json(self, name: str, default: Any = None) -> Any:
        """Load JSON config file.

        Args:
            name: Config file name
            default: Default value if file doesn't exist or can't be loaded

        Returns:
            Loaded data or default value
        """
        path = self.get_config_file(name)

        if not path.exists():
            return default

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load {name}: {e}")
            return default

    def save_json(self, name: str, data: Any):
        """Save JSON config file.

        Args:
            name: Config file name
            data: Data to save (must be JSON-serializable)
        """
        path = self.get_config_file(name)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save {name}: {e}")

    # ========================================================================
    # Recent Projects
    # ========================================================================

    def get_recent_projects(self, max_count: int = 10) -> list[str]:
        """Get list of recent project paths.

        Args:
            max_count: Maximum number of recent projects to return

        Returns:
            List of project file paths (most recent first)
        """
        recent = self.load_json("recent_projects.json", default=[])
        # Filter out non-existent paths
        recent = [p for p in recent if Path(p).exists()]
        return recent[:max_count]

    def add_recent_project(self, path: Path, max_count: int = 10):
        """Add a project to recent projects list.

        Args:
            path: Path to project file
            max_count: Maximum number of recent projects to keep
        """
        path_str = str(path.resolve())
        recent = self.load_json("recent_projects.json", default=[])

        # Remove if already exists (to move to front)
        recent = [p for p in recent if p != path_str]

        # Add to front
        recent.insert(0, path_str)

        # Trim to max_count
        recent = recent[:max_count]

        self.save_json("recent_projects.json", recent)

    # ========================================================================
    # Presets
    # ========================================================================

    def get_presets(self, category: str) -> dict:
        """Get presets for a category.

        Args:
            category: Preset category (e.g., 'bulk_tool', 'sampler')

        Returns:
            Dictionary of presets
        """
        return self.load_json(f"{category}_presets.json", default={})

    def save_presets(self, category: str, presets: dict):
        """Save presets for a category.

        Args:
            category: Preset category
            presets: Dictionary of presets
        """
        self.save_json(f"{category}_presets.json", presets)

    def get_preset(self, category: str, name: str) -> Optional[dict]:
        """Get a specific preset by name.

        Args:
            category: Preset category
            name: Preset name

        Returns:
            Preset data or None if not found
        """
        presets = self.get_presets(category)
        return presets.get(name)

    def save_preset(self, category: str, name: str, data: dict):
        """Save a specific preset.

        Args:
            category: Preset category
            name: Preset name
            data: Preset data
        """
        presets = self.get_presets(category)
        presets[name] = data
        self.save_presets(category, presets)

    def delete_preset(self, category: str, name: str) -> bool:
        """Delete a preset.

        Args:
            category: Preset category
            name: Preset name

        Returns:
            True if preset was deleted, False if not found
        """
        presets = self.get_presets(category)
        if name in presets:
            del presets[name]
            self.save_presets(category, presets)
            return True
        return False
