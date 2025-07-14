"""
Parameter range classes for genetic algorithm optimization
Provides different types of parameter ranges for various optimization needs
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any


class ParameterRange(ABC):
    """Abstract base for parameter ranges with different sampling strategies"""
    
    @abstractmethod
    def sample(self) -> float:
        """Sample a random value from the range"""
        pass
    
    @abstractmethod
    def mutate(self, current_value: float) -> float:
        """Mutate current value within the range"""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        pass
    
    @abstractmethod
    def get_range(self) -> float:
        """Get the range span"""
        pass
    
    @abstractmethod
    def clip(self, value: float) -> float:
        """Clip value to valid range"""
        pass


@dataclass
class MinMaxRange(ParameterRange):
    """Traditional continuous min-max range"""
    min_val: float
    max_val: float
    
    def sample(self) -> float:
        """Sample uniformly from min to max"""
        return random.uniform(self.min_val, self.max_val)
    
    def mutate(self, current_value: float) -> float:
        """Gaussian mutation within bounds"""
        noise = random.gauss(0, (self.max_val - self.min_val) * 0.1)
        return max(self.min_val, min(self.max_val, current_value + noise))
    
    def get_range(self) -> float:
        """Get the range span"""
        return self.max_val - self.min_val
    
    def clip(self, value: float) -> float:
        """Clip value to valid range"""
        return max(self.min_val, min(self.max_val, value))
    
    def to_dict(self) -> Dict:
        return {"type": "min_max", "min": self.min_val, "max": self.max_val}


@dataclass
class LogRange(ParameterRange):
    """Discrete logarithmic-spaced values"""
    values: List[float]
    
    def sample(self) -> float:
        """Sample from discrete values"""
        return random.choice(self.values)
    
    def mutate(self, current_value: float) -> float:
        """Mutate to nearby discrete value"""
        try:
            current_idx = self.values.index(current_value)
            # Pick adjacent value (within 1 step)
            if current_idx == 0:
                return self.values[1]  # Move up
            elif current_idx == len(self.values) - 1:
                return self.values[-2]  # Move down
            else:
                # Randomly pick adjacent
                return random.choice([self.values[current_idx-1], self.values[current_idx+1]])
        except ValueError:
            # If current value not in discrete set, pick random
            return random.choice(self.values)
    
    def get_range(self) -> float:
        """Get the range span"""
        return max(self.values) - min(self.values) if self.values else 0.0
    
    def clip(self, value: float) -> float:
        """Clip value to nearest valid discrete value"""
        if not self.values:
            return value
        # Find closest value
        return min(self.values, key=lambda x: abs(x - value))
    
    def to_dict(self) -> Dict:
        return {"type": "log_range", "values": self.values}


@dataclass
class DiscreteRange(ParameterRange):
    """Discrete values with equal probability"""
    values: List[float]
    
    def sample(self) -> float:
        """Sample uniformly from discrete values"""
        return random.choice(self.values)
    
    def mutate(self, current_value: float) -> float:
        """Mutate to a different discrete value"""
        # Get all values except current
        other_values = [v for v in self.values if v != current_value]
        if other_values:
            return random.choice(other_values)
        return current_value  # No other values available
    
    def get_range(self) -> float:
        """Get the range span"""
        return max(self.values) - min(self.values) if self.values else 0.0
    
    def clip(self, value: float) -> float:
        """Clip value to nearest valid discrete value"""
        if not self.values:
            return value
        # Find closest value
        return min(self.values, key=lambda x: abs(x - value))
    
    def to_dict(self) -> Dict:
        return {"type": "discrete", "values": self.values}


def create_log_range(start: float, end: float, num_points: int = 4) -> LogRange:
    """
    Create logarithmic range with fine spacing at start, larger gaps at end
    
    Examples:
    - create_log_range(28, 42, 4) -> [28.0, 32.0, 37.0, 42.0]
    - create_log_range(1.5, 4.5, 4) -> [1.5, 2.2, 3.2, 4.5]
    """
    if num_points <= 1:
        return LogRange([start])
    
    log_factor = 2.5  # Controls curve steepness
    
    # Generate logarithmic points
    points = []
    for i in range(num_points):
        # Normalized position (0 to 1)
        t = i / (num_points - 1)
        
        # Apply log transformation
        log_t = (pow(log_factor, t) - 1) / (log_factor - 1)
        
        # Map to value range
        value = start + (end - start) * log_t
        points.append(round(value, 1))
    
    return LogRange(points)


def create_discrete_range(values: List[float]) -> DiscreteRange:
    """Create a discrete range from a list of values"""
    return DiscreteRange(values)