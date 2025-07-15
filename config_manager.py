"""
Configuration management for trading data loader.

This module provides a Config class that handles reading, writing, and managing
configuration settings for the trading application. It supports both trading
parameters (symbol, timeframe, etc.) and persistent state tracking.
"""

import json
import os
from typing import Optional, Union
from datetime import datetime
from filename_utils import generate_filename, get_data_path


ConfigValue = Union[str, int, bool, None]
ConfigState = dict[str, ConfigValue]
ConfigDict = dict[str, Union[ConfigValue, ConfigState]]


class Config:
    """
    Manages configuration settings for trading data operations.
    
    This class provides a persistent configuration system that stores trading
    parameters and application state. Configuration is saved to and loaded from
    a JSON file. All property setters automatically persist changes to disk.
    
    Attributes:
        config_path: Path to the configuration JSON file
        config: Dictionary containing all configuration values
    """
    
    def __init__(self, config_path: str) -> None:
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path where configuration file should be stored/loaded
        """
        self.config_path = config_path
        self.config: ConfigDict = {}
        self.load()

    def load(self) -> None:
        """
        Load configuration from file or create with defaults.
        
        If the config file exists, it will be loaded. Otherwise, a new config
        file is created with default values and saved to disk.
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.get_defaults()
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
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get_defaults(self) -> ConfigDict:
        """
        Get default configuration values.
        
        Returns:
            Dictionary with default settings for a new configuration
        """
        return {
            "symbol": "BTC/USD",
            "timeframe": "1m",
            "min_bars": 1000,
            "paper_mode": True,
            "state": {
                "last_update": None,
                "total_bars": 0
            }
        }

    @property
    def symbol(self) -> str:
        value = self.config.get("symbol", "BTC/USD")
        if isinstance(value, str):
            return value
        return "BTC/USD"

    @symbol.setter
    def symbol(self, value: str) -> None:
        self.config["symbol"] = value
        self.save()

    @property
    def timeframe(self) -> str:
        value = self.config.get("timeframe", "1m")
        if isinstance(value, str):
            return value
        return "1m"

    @timeframe.setter
    def timeframe(self, value: str) -> None:
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "3d", "1w"]
        if value not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {value}. Must be one of {valid_timeframes}")
        self.config["timeframe"] = value
        self.save()

    @property
    def min_bars(self) -> int:
        value = self.config.get("min_bars", 1000)
        if isinstance(value, int):
            return value
        return 1000

    @min_bars.setter
    def min_bars(self, value: int) -> None:
        self.config["min_bars"] = value
        self.save()

    @property
    def paper_mode(self) -> bool:
        value = self.config.get("paper_mode", True)
        if isinstance(value, bool):
            return value
        return True

    @paper_mode.setter
    def paper_mode(self, value: bool) -> None:
        self.config["paper_mode"] = value
        self.save()

    def get_state(self, key: str) -> Optional[ConfigValue]:
        """
        Retrieve a value from the state section of configuration.
        
        Args:
            key: The state key to retrieve
            
        Returns:
            The state value if found, None otherwise
        """
        state = self.config.get("state", {})
        if isinstance(state, dict):
            return state.get(key)
        return None

    def set_state(self, key: str, value: ConfigValue) -> None:
        """
        Set a value in the state section of configuration.
        
        State values are used to track runtime information that should persist
        between sessions (e.g., last update time, total bars processed).
        
        Args:
            key: The state key to set
            value: The value to store
        """
        if "state" not in self.config:
            self.config["state"] = {}
        state = self.config.get("state", {})
        if isinstance(state, dict):
            state[key] = value
            self.config["state"] = state
        self.save()

    def update_last_sync(self, timestamp: Optional[str] = None) -> None:
        """
        Update the last synchronization timestamp in state.
        
        Args:
            timestamp: ISO format timestamp string. If None, uses current UTC time.
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        self.set_state("last_update", timestamp)
    
    @staticmethod
    def create_config_file(symbol: str, timeframe: str, min_bars: int = 1000, paper_mode: bool = True) -> str:
        """
        Create a new configuration file for the given symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USD", "AAPL")
            timeframe: Time interval (e.g., "1m", "1d")
            min_bars: Minimum number of bars to load (default: 1000)
            paper_mode: Whether to use paper trading mode (default: True)
            
        Returns:
            Path to the created config file
        """
        # Generate config filename
        config_filename = generate_filename(symbol, timeframe, "config", "json")
        config_path = get_data_path(config_filename, "configs")
        
        # Create config data
        config_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "min_bars": min_bars,
            "paper_mode": paper_mode,
            "state": {}
        }
        
        # Write config file
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return config_path
