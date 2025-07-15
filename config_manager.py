"""
Configuration management for trading data loader using Pydantic.

This module provides a Config class that handles reading, writing, and managing
configuration settings for the trading application. It supports both trading
parameters (symbol, timeframe, etc.) and persistent state tracking with
full validation through Pydantic models.
"""

import json
import os
from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from filename_utils import generate_filename, get_data_path
from state import RuntimeState, StateValue


class DollarBarsConfig(BaseModel):
    """Configuration for dollar bar aggregation."""
    enabled: bool = False
    threshold: Optional[float] = Field(default=None, gt=0, description="Dollar volume threshold for bar generation")
    price_column: Literal["open", "high", "low", "close", "hlc3", "ohlc4"] = "hlc3"


class OptimizerConfig(BaseModel):
    """Configuration for genetic algorithm optimization."""
    population_size: int = Field(default=50, gt=0, description="Number of individuals in GA population")
    generations: int = Field(default=30, gt=0, description="Number of generations to run GA")
    n_splits: int = Field(default=3, gt=0, description="Number of walk-forward splits")
    test_ratio: Optional[float] = Field(default=0.3, ge=0.1, le=0.9, description="Test ratio as fraction (0.3 = 30% test, 70% train). None for expanding window.")


class PipelineConfig(BaseModel):
    """Configuration for data pipeline processing."""
    enabled: bool = False
    zero_volume_keep_percentage: float = Field(default=0.1, ge=0.0, le=1.0)
    dollar_bars: DollarBarsConfig = Field(default_factory=DollarBarsConfig)




class Config(BaseModel):
    """
    Manages configuration settings for trading data operations.
    
    This class provides a persistent configuration system that stores trading
    parameters and application state. Configuration is saved to and loaded from
    a JSON file with full Pydantic validation.
    """
    
    symbol: str = "BTC/USD"
    timeframe: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "3d", "1w"] = "1m"
    min_bars: int = Field(default=5000, gt=0)
    paper_mode: bool = True
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    state: RuntimeState = Field(default_factory=RuntimeState)
    
    # Non-model fields
    config_path: str = Field(exclude=True)
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "forbid"
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate trading symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()
    
    def __init__(self, config_path: str, **kwargs):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path where configuration file should be stored/loaded
            **kwargs: Additional configuration values to override
        """
        # Load existing config or use defaults
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                # Handle legacy state format (convert dict to RuntimeState fields)
                if 'state' in config_data and isinstance(config_data['state'], dict):
                    state_data = config_data['state']
                    # Only keep fields that are valid for RuntimeState
                    valid_state_fields = {
                        'last_update', 'total_bars', 'last_run_duration', 
                        'last_error', 'last_successful_run', 'custom_data'
                    }
                    # Filter out any invalid fields and ensure custom_data exists
                    filtered_state = {k: v for k, v in state_data.items() if k in valid_state_fields}
                    if 'custom_data' not in filtered_state:
                        filtered_state['custom_data'] = {}
                    config_data['state'] = filtered_state
                
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
        config_dict = self.model_dump(exclude={'config_path'})
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @property
    def pipeline_enabled(self) -> bool:
        """Check if pipeline is enabled."""
        return self.pipeline.enabled
    
    @property
    def zero_volume_keep_percentage(self) -> float:
        """Get zero volume keep percentage."""
        return self.pipeline.zero_volume_keep_percentage
    
    @property
    def dollar_bars_enabled(self) -> bool:
        """Check if dollar bars generation is enabled."""
        return self.pipeline.dollar_bars.enabled
    
    @property
    def dollar_bars_threshold(self) -> Optional[float]:
        """Get dollar bar threshold."""
        return self.pipeline.dollar_bars.threshold
    
    @property
    def dollar_bars_price_column(self) -> str:
        """Get price column for dollar bars."""
        return self.pipeline.dollar_bars.price_column
    
    def get_state(self, key: str) -> Optional[StateValue]:
        """
        Retrieve a value from the state section of configuration.
        
        Args:
            key: The state key to retrieve
            
        Returns:
            The state value if found, None otherwise
        """
        # Check if it's a predefined field
        if hasattr(self.state, key):
            return getattr(self.state, key)
        # Otherwise check custom data
        return self.state.get_custom(key)
    
    def set_state(self, key: str, value: StateValue) -> None:
        """
        Set a value in the state section of configuration.
        
        State values are used to track runtime information that should persist
        between sessions (e.g., last update time, total bars processed).
        
        Args:
            key: The state key to set
            value: The value to store (must be str, int, float, bool, or None)
        """
        # Check if it's a predefined field
        if hasattr(self.state, key) and key not in ['custom_data']:
            setattr(self.state, key, value)
        else:
            # Store in custom data
            self.state.set_custom(key, value)
        self.save()
    
    def update_last_sync(self, timestamp: Optional[str] = None) -> None:
        """
        Update the last synchronization timestamp in state.
        
        Args:
            timestamp: ISO format timestamp string. If None, uses current UTC time.
        """
        self.state.update_last_sync(timestamp)
        self.save()
    
    @property
    def config(self) -> dict:
        """
        Get configuration as a dictionary for backward compatibility.
        
        Returns:
            Dictionary representation of the configuration
        """
        return self.model_dump(exclude={'config_path'})
    
    @staticmethod
    def create_config_file(
        symbol: str, 
        timeframe: str, 
        min_bars: int = 5000, 
        paper_mode: bool = True
    ) -> str:
        """
        Create a new configuration file for the given symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USD", "AAPL")
            timeframe: Time interval (e.g., "1m", "1d")
            min_bars: Minimum number of bars to load (default: 5000)
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
            paper_mode=paper_mode
        )
        
        # Save to file
        config.save()
        
        return config_path