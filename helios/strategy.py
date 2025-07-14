"""
Trading strategy implementation for Helios Trader
Includes action matrix, position sizing, and trade execution logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """Trading actions"""

    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class Position:
    """Position information"""

    shares: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None

    @property
    def is_open(self) -> bool:
        return self.shares != 0.0


@dataclass
class Trade:
    """Trade record"""

    timestamp: pd.Timestamp
    action: str
    shares: float
    price: float
    position_before: float
    position_after: float
    cash_before: float
    cash_after: float
    portfolio_value: float

    def __repr__(self):
        return (
            f"Trade({self.timestamp}, {self.action}, "
            f"shares={self.shares:.0f}, price=${self.price:.2f})"
        )
