"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from logging import Logger
import tempfile

from traffic_studio.container import Container
from traffic_studio.utils.logging import create_logger


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def logger():
    """Create a test logger."""
    return create_logger("test", level=10)  # DEBUG level


@pytest.fixture
def container(logger):
    """Create a test container."""
    return Container(logger=logger)
