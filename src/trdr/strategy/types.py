"""Shared trading types."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SignalAction(Enum):
    """Trading signal actions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class Signal:
    """Trading signal from strategy."""

    action: SignalAction
    price: float
    confidence: float  # 0.0 to 1.0
    reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    timestamp: str = ""
    position_size_pct: float = 1.0  # 0.0-1.0, proportion of max position
    # Paper exchange extensions:
    trailing_stop: float | None = None  # Trail % (0.02 = 2%) or $ amount
    quantity: float | None = None  # Explicit quantity (overrides position_size_pct)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class Position:
    """Current position state."""

    symbol: str
    side: str  # "long" or "short" or "none"
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float | None
