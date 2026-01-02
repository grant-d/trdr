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
class VolumeProfile:
    """Calculated volume profile with key levels."""

    poc: float  # Point of Control (highest volume price)
    vah: float  # Value Area High
    val: float  # Value Area Low
    hvns: list[float]  # High Volume Nodes
    lvns: list[float]  # Low Volume Nodes
    price_levels: list[float]  # All price level midpoints
    volumes: list[float]  # Volume at each level
    total_volume: float


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
    position_size_ratio: float = 1.0  # 0.0-1.0, proportion of max position

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
