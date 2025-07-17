"""
State management for trading data loader.

This module provides state tracking functionality for the trading application,
including runtime information that should persist between sessions.
"""

from typing import Optional, Dict, Union, List
from datetime import datetime
from pydantic import BaseModel, Field
import json
import os
from filename_utils import generate_filename, get_data_path


# Define specific types for state values
StateValue = Union[str, int, float, bool, None]

# Define parameter value types
ParameterValue = Union[int, float]
Parameters = Dict[str, ParameterValue]


class HallOfFameEntry(BaseModel):
    """Single entry in the hall of fame."""

    parameters: Parameters
    fitness: float
    timestamp: str
    symbol: str


class OptimizationHistory(BaseModel):
    """Single optimization history entry."""

    timestamp: str
    parameters: Parameters
    fitness: float
    is_best: bool


class RuntimeState(BaseModel):
    """
    Runtime state tracking with strongly typed fields.

    This model tracks runtime information that should persist between sessions,
    such as last update times and processing statistics.
    """

    last_update: Optional[str] = Field(
        default=None, description="ISO format timestamp of last data synchronization"
    )

    total_bars: int = Field(
        default=0, description="Total number of bars processed in last run", ge=0
    )

    last_run_duration: Optional[float] = Field(
        default=None, description="Duration of last run in seconds", ge=0
    )

    last_error: Optional[str] = Field(
        default=None, description="Last error message if any"
    )

    last_successful_run: Optional[str] = Field(
        default=None, description="ISO format timestamp of last successful run"
    )

    # Optimization tracking
    best_params: Optional[Parameters] = Field(
        default=None, description="Current best optimization parameters"
    )

    best_fitness: float = Field(
        default=-float("inf"), description="Best fitness score achieved"
    )

    optimization_history: List[OptimizationHistory] = Field(
        default_factory=list, description="History of optimization runs (last 100)"
    )

    hall_of_fame: List[HallOfFameEntry] = Field(
        default_factory=list, description="Top performing parameter sets (max 20)"
    )

    # Additional custom state can be stored here
    custom_data: Dict[str, StateValue] = Field(
        default_factory=dict,
        description="Custom state data that doesn't fit predefined fields",
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

    def get_custom(
        self, key: str, default: Optional[StateValue] = None
    ) -> Optional[StateValue]:
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

    def update_best_params(self, params: Parameters, fitness: float) -> bool:
        """
        Update best parameters if new fitness is better.

        Args:
            params: Parameter dictionary
            fitness: Fitness score

        Returns:
            True if this was a new best
        """
        timestamp = datetime.utcnow().isoformat()

        # Never accept 0.0 fitness - always prefer any non-zero fitness over 0.0
        if self.best_fitness == 0.0 and fitness != 0.0:
            is_new_best = True
        elif fitness == 0.0:
            is_new_best = False  # Never accept 0.0 as best
        else:
            is_new_best = fitness > self.best_fitness

        if is_new_best:
            self.best_params = params
            self.best_fitness = fitness

        # Add to history
        self.optimization_history.append(
            OptimizationHistory(
                timestamp=timestamp,
                parameters=params,
                fitness=fitness,
                is_best=is_new_best,
            )
        )

        # Keep only last 100 entries
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]

        return is_new_best

    def update_hall_of_fame(
        self, params: Parameters, fitness: float, symbol: str
    ) -> None:
        """
        Update hall of fame with new parameters if they're good enough.

        Args:
            params: Parameter dictionary
            fitness: Fitness score
            symbol: Trading symbol
        """
        entry = HallOfFameEntry(
            parameters=params,
            fitness=fitness,
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
        )

        # Add to hall of fame
        self.hall_of_fame.append(entry)

        # Sort by fitness (descending)
        self.hall_of_fame.sort(key=lambda x: x.fitness, reverse=True)

        # Keep only top 20
        self.hall_of_fame = self.hall_of_fame[:20]

    @classmethod
    def load_from_file(cls, symbol: str, timeframe: str) -> "RuntimeState":
        """
        Load state from file or create new instance.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            RuntimeState instance
        """
        filename = generate_filename(symbol, timeframe, "state", "json")
        filepath = get_data_path(filename, "configs")

        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                print(f"Warning: Could not load state from {filepath}: {e}")

        return cls()

    def save_to_file(self, symbol: str, timeframe: str) -> None:
        """
        Save state to file.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        """
        filename = generate_filename(symbol, timeframe, "state", "json")
        filepath = get_data_path(filename, "configs")

        try:
            with open(filepath, "w") as f:
                json.dump(self.model_dump(), f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save state to {filepath}: {e}")

    class Config:
        """Pydantic config."""

        validate_assignment = True
        extra = "forbid"
