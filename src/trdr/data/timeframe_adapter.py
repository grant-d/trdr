"""Alpaca-specific timeframe translation and aggregation."""

from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame
from alpaca.data.timeframe import TimeFrameUnit

from ..core import Symbol, Timeframe


class TimeframeAdapter:
    """Translates Timeframe to Alpaca API constraints.

    Alpaca limits: minutes 1-59, hours 1-23, days 1, week 1, month 1/2/3/6/12.
    Non-native timeframes require fetching base bars and aggregating.
    """

    def __init__(self, timeframe: Timeframe, symbol: Symbol) -> None:
        self.timeframe = timeframe
        self._symbol = symbol

    @property
    def needs_aggregation(self) -> bool:
        """True if this timeframe exceeds Alpaca's native support."""
        c = self.timeframe.canonical
        if c.unit == "m":
            return c.amount > 59
        if c.unit == "h":
            return c.amount > 23
        if c.unit == "d":
            return c.amount > 1
        if c.unit == "w":
            return c.amount > 1
        if c.unit == "mo":
            return c.amount not in (1, 2, 3, 6, 12)
        return False

    @property
    def base_timeframe(self) -> Timeframe:
        """Base timeframe for aggregation (or self if native)."""
        if not self.needs_aggregation:
            return self.timeframe
        c = self.timeframe.canonical
        if c.unit == "m":
            return Timeframe(1, "m")
        if c.unit == "h":
            return Timeframe(1, "h")
        if c.unit in ("d", "w", "mo"):
            return Timeframe(1, "d")
        return self.timeframe

    @property
    def aggregation_factor(self) -> int:
        """Number of base bars to aggregate (1 if native)."""
        if not self.needs_aggregation:
            return 1
        c = self.timeframe.canonical
        if c.unit in ("m", "h", "d"):
            return c.amount

        # Crypto trades 24/7, stocks trade 5 days/week
        if c.unit == "w":
            days_per_week = 7 if self._symbol.is_crypto else 5
            return c.amount * days_per_week
        if c.unit == "mo":
            days_per_month = 30 if self._symbol.is_crypto else 21
            return c.amount * days_per_month
        return 1

    def to_alpaca(self) -> AlpacaTimeFrame:
        """Convert to Alpaca TimeFrame.

        Uses canonical form for native (60m→1h) or base for aggregation (90m→1m).
        """
        if self.needs_aggregation:
            tf = self.base_timeframe
        else:
            tf = self.timeframe.canonical
        unit_map = {
            "m": TimeFrameUnit.Minute,
            "h": TimeFrameUnit.Hour,
            "d": TimeFrameUnit.Day,
            "w": TimeFrameUnit.Week,
            "mo": TimeFrameUnit.Month,
        }
        return AlpacaTimeFrame(tf.amount, unit_map[tf.unit])
