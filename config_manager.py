"""
Configuration management for trading data loader using Pydantic.

This module provides a Config class that handles reading, writing, and managing
configuration settings for the trading application. It supports both trading
parameters (symbol, timeframe, etc.) and persistent state tracking with
full validation through Pydantic models.
"""

import json
import os
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator
from filename_utils import generate_filename, get_data_path


class PipelineConfig(BaseModel):
    """Configuration for data pipeline processing."""

    enabled: bool = False


class Config(BaseModel):
    """
    Manages configuration settings for trading data operations.

    This class provides a persistent configuration system that stores trading
    parameters and application state. Configuration is saved to and loaded from
    a JSON file with full Pydantic validation.
    """

    symbol: str = "BTC/USD"
    timeframe: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "3d", "1w"] = "1m"
    min_bars: int = Field(default=10_000, gt=0)
    paper_mode: bool = True
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    # Non-model fields
    config_path: str = Field(exclude=True)

    class Config:
        """Pydantic config."""

        validate_assignment = True
        extra = "forbid"

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate trading symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    def __init__(self, config_path: str, **kwargs) -> None:
        """
        Initialize configuration manager.

        Args:
            config_path: Path where configuration file should be stored/loaded
            **kwargs: Additional configuration values to override
        """
        # Load existing config or use defaults
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)

                # Merge with any provided kwargs
                config_data.update(kwargs)
        else:
            config_data = kwargs

        # Initialize with loaded/merged data
        super().__init__(config_path=config_path, **config_data)

        # Save if new file
        if not os.path.exists(config_path):
            self.save()

    def save(self) -> None:
        """
        Save current configuration to file.

        Creates the directory structure if it doesn't exist and writes the
        configuration as formatted JSON.
        """
        dir_path = os.path.dirname(self.config_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Export to dict excluding the config_path
        config_dict = self.model_dump(exclude={"config_path"})

        with open(self.config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @property
    def pipeline_enabled(self) -> bool:
        """Check if pipeline is enabled."""
        return self.pipeline.enabled

    @property
    def config(self) -> dict[str, Any]:
        """
        Get configuration as a dictionary for backward compatibility.

        Returns:
            Dictionary representation of the configuration
        """
        return self.model_dump(exclude={"config_path"})

    @staticmethod
    def create_config_file(
        symbol: str, timeframe: str, min_bars: int = 10_000, paper_mode: bool = True
    ) -> str:
        """
        Create a new configuration file for the given symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD", "AAPL")
            timeframe: Time interval (e.g., "1m", "1d")
            min_bars: Minimum number of bars to load (default: 10,000)
            paper_mode: Whether to use paper trading mode (default: True)

        Returns:
            Path to the created config file
        """
        # Generate config filename
        config_filename = generate_filename(symbol, timeframe, "config", "json")
        config_path = get_data_path(config_filename, "configs")

        # Create config instance
        config = Config(
            config_path=config_path,
            symbol=symbol,
            timeframe=timeframe,
            min_bars=min_bars,
            paper_mode=paper_mode,
        )

        # Save to file
        config.save()

        return config_path
