"""Smoke tests verifying package structure, imports, and configuration loading."""
import pytest
from src import __version__
from src.config import load_config, AppConfig
from src.utils.logging import setup_logger


def test_package_metadata():
    """Verify package version is defined."""
    assert __version__ == "0.1.0"


def test_default_config_loading():
    """Verify default configuration loads with correct defaults without requiring environment variables."""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.project.name == "hiver-support-agent"
    assert config.data.selected_brand == "AmazonHelp"
    assert config.retrieval.top_k == 3
    assert config.retrieval.relevance_threshold == 0.45
    assert config.agent.temperature == 0.0


def test_logger_initialization():
    """Verify structured logger can be initialized and logs messages without errors."""
    logger = setup_logger(name="test_logger", level="DEBUG")
    assert logger is not None
    logger.info("Smoke test logger message verified.")
