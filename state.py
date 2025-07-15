"""
State management for trading data loader.

This module provides state tracking functionality for the trading application,
including runtime information that should persist between sessions.
"""

from typing import Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


# Define specific types for state values
StateValue = Union[str, int, float, bool, None]


class RuntimeState(BaseModel):
    """
    Runtime state tracking with strongly typed fields.
    
    This model tracks runtime information that should persist between sessions,
    such as last update times and processing statistics.
    """
    
    last_update: Optional[str] = Field(
        default=None,
        description="ISO format timestamp of last data synchronization"
    )
    
    total_bars: int = Field(
        default=0,
        description="Total number of bars processed in last run",
        ge=0
    )
    
    last_run_duration: Optional[float] = Field(
        default=None,
        description="Duration of last run in seconds",
        ge=0
    )
    
    last_error: Optional[str] = Field(
        default=None,
        description="Last error message if any"
    )
    
    last_successful_run: Optional[str] = Field(
        default=None,
        description="ISO format timestamp of last successful run"
    )
    
    # Additional custom state can be stored here
    custom_data: Dict[str, StateValue] = Field(
        default_factory=dict,
        description="Custom state data that doesn't fit predefined fields"
    )
    
    def update_last_sync(self, timestamp: Optional[str] = None) -> None:
        """
        Update the last synchronization timestamp.
        
        Args:
            timestamp: ISO format timestamp string. If None, uses current UTC time.
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        self.last_update = timestamp
    
    def record_successful_run(self, total_bars: int, duration: float) -> None:
        """
        Record information about a successful run.
        
        Args:
            total_bars: Number of bars processed
            duration: Run duration in seconds
        """
        self.total_bars = total_bars
        self.last_run_duration = duration
        self.last_successful_run = datetime.utcnow().isoformat()
        self.last_error = None  # Clear any previous error
    
    def record_error(self, error_message: str) -> None:
        """
        Record an error that occurred during processing.
        
        Args:
            error_message: Description of the error
        """
        self.last_error = error_message
    
    def get_custom(self, key: str, default: Optional[StateValue] = None) -> Optional[StateValue]:
        """
        Get a custom state value.
        
        Args:
            key: The key to retrieve
            default: Default value if key not found
            
        Returns:
            The state value if found, default otherwise
        """
        return self.custom_data.get(key, default)
    
    def set_custom(self, key: str, value: StateValue) -> None:
        """
        Set a custom state value.
        
        Args:
            key: The key to set
            value: The value to store
        """
        if value is None and key in self.custom_data:
            # Remove key if setting to None
            del self.custom_data[key]
        else:
            self.custom_data[key] = value
    
    def clear_custom(self) -> None:
        """Clear all custom state data."""
        self.custom_data.clear()
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "forbid"