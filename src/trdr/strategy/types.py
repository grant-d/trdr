"""Shared trading types."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.duration import Duration
    from ..core.timeframe import Timeframe


@dataclass
class DataRequirement:
    """Specification for a data feed.

    Args:
        symbol: Trading symbol (e.g., "crypto:ETH/USD", "crypto:BTC/USD")
        timeframe: Timeframe
        lookback: Duration
        role: "primary" (trading feed) or "informative" (reference data)
    """

    symbol: str
    timeframe: "Timeframe"
    lookback: "Duration"
    role: str = "informative"

    @property
    def key(self) -> str:
        """Unique key for this data feed."""
        return f"{self.symbol}:{self.timeframe}"

    @property
    def lookback_bars(self) -> int:
        """Lookback as bar count."""
        return self.lookback.to_bars(self.timeframe, self.symbol)

    def __post_init__(self) -> None:
        if self.role not in ("primary", "informative"):
            raise ValueError(
                f"role must be 'primary' or 'informative', got '{self.role}'"
            )


class SignalAction(Enum):
    """Trading signal actions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class Signal:
    """Trading signal from strategy.

    Args:
        action: BUY, SELL, CLOSE, or HOLD
        price: Current market price (informational, not order price)
        confidence: Signal strength 0.0-1.0
        reason: Entry/exit reason
        stop_loss: Stop loss trigger price (exit order)
        take_profit: Take profit trigger price (exit order)
        stop_price: Stop trigger for stop-limit entry. When price crosses this,
            a limit order at limit_price is placed. Requires limit_price.
        limit_price: Limit order entry price (fills at this price, no slippage).
            If None, entry uses market order at current price.
        timestamp: Signal generation time
        position_size_pct: Size as proportion of max position (0.0-1.0)
        trailing_stop: Trail % (<1) or $ amount
        quantity: Explicit quantity (overrides position_size_pct)
    """

    action: SignalAction
    price: float  # Current market price (informational)
    confidence: float  # 0.0 to 1.0
    reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    stop_price: float | None = None  # Stop trigger for stop-limit entry
    limit_price: float | None = None  # Limit entry price (None = market order)
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
