"""
Base classes for strategy parameters using Pydantic with Range types.

This module provides base classes and utilities for defining strongly-typed
strategy parameters that can be optimized using the genetic algorithm.
"""

import math
from typing import Dict, Tuple, Type, TypeVar, Union, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from abc import ABC, abstractmethod
import numpy as np
import random


T = TypeVar('T', bound='BaseStrategyParameters')


class BaseRange(ABC, BaseModel):
    """Abstract base class for parameter ranges."""
    
    @abstractmethod
    def sample(self) -> float:
        """Sample a random value from the range."""
        pass
    
    @abstractmethod
    def clip(self, value: float) -> float:
        """Ensure value is within valid range."""
        pass
    
    @abstractmethod
    def get_bounds(self) -> Tuple[float, float]:
        """Get min and max bounds for optimization."""
        pass


class MinMaxRange(BaseRange):
    """Continuous range between min and max values."""
    
    min_value: float = Field(description="Minimum value")
    max_value: float = Field(description="Maximum value")

    def __init__(
            self,
            min_value: float,
            max_value: float,
            **kwargs
    ):
        # Fix swapped min/max
        kwargs['min_value'] = min(min_value, max_value)
        kwargs['max_value'] = max(min_value, max_value)
        # Set the values in kwargs for parent class
        super().__init__(**kwargs)

    @field_validator('max_value')
    @classmethod
    def validate_max(cls, v: float, info) -> float:
        """Ensure max > min."""
        if 'min_value' in info.data and v <= info.data['min_value']:
            raise ValueError('max_value must be greater than min_value')
        return v
    
    def sample(self) -> float:
        """Sample uniformly between min and max."""
        return random.uniform(self.min_value, self.max_value)
    
    def clip(self, value: float) -> float:
        """Clip value to [min, max]."""
        return max(self.min_value, min(self.max_value, value))
    
    def get_bounds(self) -> Tuple[float, float]:
        """Return (min, max) tuple."""
        return (self.min_value, self.max_value)


class DiscreteRange(BaseRange):
    """Discrete set of allowed values."""
    
    values: List[float] = Field(description="List of allowed values")
    
    @field_validator('values')
    @classmethod
    def validate_values(cls, v: List[float]) -> List[float]:
        """Ensure at least one value."""
        if not v:
            raise ValueError('values must not be empty')
        return sorted(v)  # Keep sorted for consistency
    
    def sample(self) -> float:
        """Sample randomly from allowed values."""
        return random.choice(self.values)
    
    def clip(self, value: float) -> float:
        """Return nearest allowed value."""
        # Find closest value
        idx = np.searchsorted(self.values, value)
        if idx == 0:
            return self.values[0]
        elif idx == len(self.values):
            return self.values[-1]
        else:
            # Return closer of two neighbors
            if abs(value - self.values[idx-1]) < abs(value - self.values[idx]):
                return self.values[idx-1]
            else:
                return self.values[idx]
    
    def get_bounds(self) -> Tuple[float, float]:
        """Return min and max of allowed values."""
        return (min(self.values), max(self.values))


class LogRange(BaseRange):
    """Logarithmic range for parameters that vary over orders of magnitude."""
    
    min_value: float = Field(gt=0, description="Minimum value (must be positive)")
    max_value: float = Field(gt=0, description="Maximum value (must be positive)")
    base: float = Field(default=10.0, gt=1, description="Logarithm base")
    
    @field_validator('max_value')
    @classmethod
    def validate_max(cls, v: float, info) -> float:
        """Ensure max > min."""
        if 'min_value' in info.data and v <= info.data['min_value']:
            raise ValueError('max_value must be greater than min_value')
        return v
    
    def sample(self) -> float:
        """Sample uniformly in log space."""
        log_min = np.log(self.min_value) / np.log(self.base)
        log_max = np.log(self.max_value) / np.log(self.base)
        log_value = random.uniform(log_min, log_max)
        return self.base ** log_value
    
    def clip(self, value: float) -> float:
        """Clip value to [min, max]."""
        return max(self.min_value, min(self.max_value, value))
    
    def get_bounds(self) -> Tuple[float, float]:
        """Return (min, max) tuple."""
        return (self.min_value, self.max_value)


class ConstantRange(BaseRange):
    """Single constant value (no optimization)."""
    
    value: float = Field(description="Constant value")
    
    def sample(self) -> float:
        """Always return the constant value."""
        return self.value
    
    def clip(self, value: float) -> float:
        """Always return the constant value."""
        return self.value
    
    def get_bounds(self) -> Tuple[float, float]:
        """Return (value, value) tuple."""
        return (self.value, self.value)


class IntegerRange(MinMaxRange):
    """Integer range between min and max values."""
    
    # min_value: int = Field(description="Minimum value")
    # max_value: int = Field(description="Maximum value")
    
    def sample(self) -> float:
        """Sample uniformly between min and max, return as float."""
        return float(random.randint(int(self.min_value), int(self.max_value)))
    
    def clip(self, value: float) -> float:
        """Clip and round to nearest integer."""
        rounded = round(value)
        return float(max(self.min_value, min(self.max_value, rounded)))


class StepRange(BaseRange):
    """Range with fixed step size."""
    
    min_value: float = Field(description="Minimum value")
    max_value: float = Field(description="Maximum value")
    step: float = Field(gt=0, description="Step size")
    
    @field_validator('max_value')
    @classmethod
    def validate_max(cls, v: float, info) -> float:
        """Ensure max > min."""
        if 'min_value' in info.data and v <= info.data['min_value']:
            raise ValueError('max_value must be greater than min_value')
        return v
    
    def sample(self) -> float:
        """Sample from allowed steps."""
        n_steps = int((self.max_value - self.min_value) / self.step)
        step_idx = random.randint(0, n_steps)
        return self.min_value + step_idx * self.step
    
    def clip(self, value: float) -> float:
        """Clip to nearest step value."""
        if value <= self.min_value:
            return self.min_value
        elif value >= self.max_value:
            return self.max_value
        else:
            # Round to nearest step
            steps_from_min = round((value - self.min_value) / self.step)
            return self.min_value + steps_from_min * self.step
    
    def get_bounds(self) -> Tuple[float, float]:
        """Return (min, max) tuple."""
        return (self.min_value, self.max_value)


class BinaryRange(BaseRange):
    """Binary choice range (0 or 1)."""
    
    def sample(self) -> float:
        """Sample randomly 0 or 1."""
        return float(random.randint(0, 1))
    
    def clip(self, value: float) -> float:
        """Round to nearest binary value (0 or 1)."""
        return 0.0 if value < 0.5 else 1.0
    
    def get_bounds(self) -> Tuple[float, float]:
        """Return (0, 1) bounds."""
        return (0.0, 1.0)


# Type alias for any range type
RangeType = Union[MinMaxRange, DiscreteRange, LogRange, ConstantRange, IntegerRange, StepRange, BinaryRange]


class BaseStrategyParameters(BaseModel):
    """
    Base class for strategy parameters with ranges.
    
    Subclasses should define parameters with their range types.
    
    Example:
        class MAStrategyParams(BaseStrategyParameters):
            fast_ma: IntegerRange = IntegerRange(min_value=5, max_value=50)
            slow_ma: IntegerRange = IntegerRange(min_value=20, max_value=200)
            threshold: MinMaxRange = MinMaxRange(min_value=0.0, max_value=1.0)
    """
    
    @classmethod
    def get_param_ranges(cls) -> Dict[str, BaseRange]:
        """
        Get all parameter ranges.
        
        Returns:
            Dictionary mapping parameter names to Range objects
        """
        ranges = {}
        
        # Get default values from class definition
        for field_name, field_info in cls.model_fields.items():
            default_value = field_info.default
            
            # Check if it's a Range type
            if isinstance(default_value, BaseRange):
                ranges[field_name] = default_value
            # Check for factory function
            elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                instance = field_info.default_factory()
                if isinstance(instance, BaseRange):
                    ranges[field_name] = instance
                    
        return ranges
    
    @classmethod
    def sample_random(cls: Type[T]) -> T:
        """
        Create instance with random parameter values.
        
        Returns:
            Instance with randomly sampled parameters
        """
        ranges = cls.get_param_ranges()
        params = {}
        
        for field_name, range_obj in ranges.items():
            params[field_name] = range_obj.sample()
            
        return cls.from_dict(params)
    
    @classmethod
    def from_dict(cls: Type[T], params: Dict[str, float]) -> T:
        """
        Create instance from dictionary of parameter values.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            Instance with parameter values as ConstantRange objects
        """
        # Create a new instance with default ranges first
        instance = cls()
        
        # Then replace each range with a ConstantRange containing the specific value
        for field_name, value in params.items():
            if hasattr(instance, field_name):
                setattr(instance, field_name, ConstantRange(value=value))
                
        return instance
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert parameters to dictionary of values.
        
        Returns:
            Dictionary of parameter values
        """
        values = {}
        for field_name, field_value in self:
            if isinstance(field_value, BaseRange):
                # For ConstantRange or sampled values
                if isinstance(field_value, ConstantRange):
                    values[field_name] = field_value.value
                else:
                    # Sample a value for non-constant ranges
                    values[field_name] = field_value.sample()
                    
        return values
    
    def get_optimization_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get optimization bounds for all parameters.
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        bounds = {}
        for field_name, field_value in self:
            if isinstance(field_value, BaseRange):
                bounds[field_name] = field_value.get_bounds()
                
        return bounds
    
    def validate_constraints(self) -> bool:
        """
        Validate any custom constraints between parameters.
        
        Override this method to add cross-parameter validation.
        
        Returns:
            True if all constraints are satisfied
        """
        return True


# Example parameter classes

class MAStrategyParameters(BaseStrategyParameters):
    """Moving Average strategy parameters."""
    
    fast_ma: IntegerRange = Field(
        default_factory=lambda: IntegerRange(min_value=5, max_value=50),
        description="Fast moving average period"
    )
    slow_ma: IntegerRange = Field(
        default_factory=lambda: IntegerRange(min_value=20, max_value=200),
        description="Slow moving average period"
    )
    
    def validate_constraints(self) -> bool:
        """Ensure fast MA is shorter than slow MA."""
        # Get actual values
        fast_val = self.fast_ma.sample() if isinstance(self.fast_ma, BaseRange) else self.fast_ma
        slow_val = self.slow_ma.sample() if isinstance(self.slow_ma, BaseRange) else self.slow_ma
        return fast_val < slow_val


class RSIStrategyParameters(BaseStrategyParameters):
    """RSI strategy parameters."""
    
    rsi_period: IntegerRange = Field(
        default_factory=lambda: IntegerRange(min_value=5, max_value=50),
        description="RSI calculation period"
    )
    rsi_oversold: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=10.0, max_value=40.0),
        description="Oversold threshold"
    )
    rsi_overbought: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=60.0, max_value=90.0),
        description="Overbought threshold"
    )
    
    def validate_constraints(self) -> bool:
        """Ensure oversold < overbought."""
        oversold_val = self.rsi_oversold.sample() if isinstance(self.rsi_oversold, BaseRange) else self.rsi_oversold
        overbought_val = self.rsi_overbought.sample() if isinstance(self.rsi_overbought, BaseRange) else self.rsi_overbought
        return oversold_val < overbought_val


class BollingerBandsParameters(BaseStrategyParameters):
    """Bollinger Bands strategy parameters."""
    
    bb_period: IntegerRange = Field(
        default_factory=lambda: IntegerRange(min_value=10, max_value=50),
        description="Bollinger Bands period"
    )
    bb_std_dev: StepRange = Field(
        default_factory=lambda: StepRange(min_value=1.0, max_value=3.0, step=0.5),
        description="Standard deviation multiplier"
    )
    entry_threshold: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=0.8, max_value=1.0),
        description="Entry threshold"
    )
    exit_threshold: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=0.0, max_value=0.8),
        description="Exit threshold"
    )


class AdaptiveParameters(BaseStrategyParameters):
    """Example with different range types."""
    
    lookback: DiscreteRange = Field(
        default_factory=lambda: DiscreteRange(values=[10, 20, 30, 50, 100]),
        description="Lookback period options"
    )
    learning_rate: LogRange = Field(
        default_factory=lambda: LogRange(min_value=0.0001, max_value=0.1, base=10),
        description="Learning rate in log scale"
    )
    momentum: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=0.0, max_value=0.99),
        description="Momentum factor"
    )
    update_freq: ConstantRange = Field(
        default_factory=lambda: ConstantRange(value=60.0),
        description="Update frequency (fixed)"
    )