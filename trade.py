import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """Trading actions"""

    STRONG_LONG = "Strong Long"
    LONG = "Long"
    FLAT = "Flat"
    SHORT = "Short"
    STRONG_SHORT = "Strong Short"


@dataclass
class Position:
    """Enhanced position information"""

    units: float = 0.0  # Number of units (shares)
    cost_basis: float = 0.0  # Total cost basis
    entry_price: float = 0.0  # Average entry price
    entry_time: Optional[pd.Timestamp] = None
    stop_loss: float = 0.0
    peak_price: float = -float("inf")  # For trailing stop

    @property
    def is_open(self) -> bool:
        return abs(self.units) > 1e-9

    @property
    def avg_price(self) -> float:
        if abs(self.units) > 1e-9:
            return abs(self.cost_basis / self.units)
        return 0.0


@dataclass
class Trade:
    """Enhanced trade record"""

    timestamp: pd.Timestamp
    action: str
    units: float  # Units traded (positive for buy, negative for sell)
    price: float
    units_before: float
    units_after: float
    cost_basis_before: float
    cost_basis_after: float
    pnl: float  # Realized P&L for this trade
    cash_before: float
    cash_after: float
    portfolio_value: float
    stop_loss: float
    reason: str  # Entry, Exit, Stop Loss, etc.
