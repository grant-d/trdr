"""Market data feed identifier (symbol + timeframe)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .symbol import Symbol
    from .timeframe import Timeframe


@dataclass(frozen=True)
class Feed:
    """Market data feed identifier.

    Represents a specific data feed identified by symbol and timeframe.
    Example: crypto:BTC/USD at 15m intervals.

    Args:
        symbol: Trading symbol
        timeframe: Bar timeframe

    Examples:
        feed = Feed(Symbol.parse("crypto:BTC/USD"), Timeframe.parse("15m"))
        feed = Feed.parse("crypto:BTC/USD:15m")
        str(feed)  # "crypto:BTC/USD:15m"
    """

    symbol: "Symbol"
    timeframe: "Timeframe"

    def __str__(self) -> str:
        """String representation in 'symbol:timeframe' format."""
        return f"{self.symbol}:{self.timeframe}"

    @classmethod
    def parse(cls, s: str) -> "Feed":
        """Parse from 'crypto:BTC/USD:15m' format.

        Args:
            s: Feed string (e.g., "crypto:BTC/USD:15m")

        Returns:
            Feed instance

        Raises:
            ValueError: If format is invalid

        Examples:
            Feed.parse("crypto:BTC/USD:15m")
            Feed.parse("stock:AAPL:1d")
        """
        from .symbol import Symbol
        from .timeframe import Timeframe

        # Split on last ":" to separate symbol from timeframe
        # "crypto:BTC/USD:15m" -> "crypto:BTC/USD" and "15m"
        parts = s.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid feed format: {s}. Expected 'symbol:timeframe'")

        # Parse components
        symbol_str, tf_str = parts
        symbol = Symbol.parse(symbol_str)
        timeframe = Timeframe.parse(tf_str)
        return cls(symbol, timeframe)
