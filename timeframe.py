"""
Universal timeframe representation for trading data.

This module provides exchange-agnostic timeframe classes that can be converted
to exchange-specific formats as needed. This allows the base data loader to work
with a consistent timeframe format regardless of the data source.
"""

from enum import Enum
from typing import Tuple


class TimeFrameUnit(Enum):
    """
    Enumeration of supported time units.
    
    These units can be combined with amounts to create specific timeframes
    (e.g., 5 minutes, 1 hour, 1 day).
    """
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"


class TimeFrame:
    """
    Universal timeframe representation for market data intervals.
    
    This class provides a consistent way to represent timeframes across different
    exchanges and data sources. It can be converted to exchange-specific formats
    as needed.
    
    Attributes:
        amount: The numeric amount (e.g., 5 for "5 minutes")
        unit: The time unit from TimeFrameUnit enum
    """

    def __init__(self, amount: int, unit: TimeFrameUnit) -> None:
        """
        Initialize a timeframe.
        
        Args:
            amount: The numeric amount for the timeframe
            unit: The unit of time (minute, hour, day, week)
        """
        self.amount = amount
        self.unit = unit

    @classmethod
    def from_string(cls, timeframe_str: str) -> "TimeFrame":
        """
        Create TimeFrame from a string representation.
        
        Args:
            timeframe_str: String like '1m', '5m', '1h', '1d', etc.
            
        Returns:
            TimeFrame instance
            
        Raises:
            ValueError: If the timeframe string is not supported
            
        Examples:
            >>> tf = TimeFrame.from_string("5m")
            >>> tf.amount
            5
            >>> tf.unit
            TimeFrameUnit.MINUTE
        """
        mapping: dict[str, Tuple[int, TimeFrameUnit]] = {
            "1m": (1, TimeFrameUnit.MINUTE),
            "5m": (5, TimeFrameUnit.MINUTE),
            "15m": (15, TimeFrameUnit.MINUTE),
            "30m": (30, TimeFrameUnit.MINUTE),
            "1h": (1, TimeFrameUnit.HOUR),
            "4h": (4, TimeFrameUnit.HOUR),
            "1d": (1, TimeFrameUnit.DAY),
            "3d": (3, TimeFrameUnit.DAY),
            "1w": (1, TimeFrameUnit.WEEK),
        }

        if timeframe_str not in mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe_str}")

        amount, unit = mapping[timeframe_str]
        return cls(amount, unit)

    def to_minutes(self) -> int:
        """
        Convert the timeframe to total minutes.
        
        Returns:
            Total number of minutes represented by this timeframe
            
        Examples:
            >>> TimeFrame(5, TimeFrameUnit.MINUTE).to_minutes()
            5
            >>> TimeFrame(1, TimeFrameUnit.HOUR).to_minutes()
            60
            >>> TimeFrame(1, TimeFrameUnit.DAY).to_minutes()
            1440
        """
        unit_minutes = {
            TimeFrameUnit.MINUTE: 1,
            TimeFrameUnit.HOUR: 60,
            TimeFrameUnit.DAY: 1440,
            TimeFrameUnit.WEEK: 10080,
        }
        return self.amount * unit_minutes[self.unit]

    def __str__(self) -> str:
        """
        Get string representation of the timeframe.
        
        Returns:
            String like "5m", "1h", "1d", etc.
        """
        unit_str = {
            TimeFrameUnit.MINUTE: "m",
            TimeFrameUnit.HOUR: "h",
            TimeFrameUnit.DAY: "d",
            TimeFrameUnit.WEEK: "w",
        }
        return f"{self.amount}{unit_str[self.unit]}"

    def __repr__(self) -> str:
        """Get detailed string representation for debugging."""
        return f"TimeFrame({self.amount}, {self.unit.value})"
