"""Backtest utilities for timeframe parsing and data alignment."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame, TimeFrameUnit

if TYPE_CHECKING:
    from ..data.market import Bar


@dataclass(frozen=True)
class Timeframe:
    """Our timeframe abstraction - accepts any valid timeframe.

    Handles translation to Alpaca's constrained TimeFrame internally.
    Alpaca limits: minutes 1-59, hours 1-23, days 1, week 1, month 1/2/3/6/12.
    """

    amount: int
    unit: str  # Normalized: "m", "h", "d", "w", "mo"

    @property
    def needs_aggregation(self) -> bool:
        """True if this timeframe exceeds Alpaca's native support.

        Checks canonical form first (60m→1h is native, no aggregation).
        """
        # Check canonical form - 60m=1h, 120m=2h, 24h=1d are all native
        c = self.canonical
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
    def base_timeframe(self) -> "Timeframe":
        """Return base timeframe for aggregation (or self if native).

        Uses canonical form: 48h→2d uses 1d base, not 1h.
        """
        if not self.needs_aggregation:
            return self
        c = self.canonical
        if c.unit == "m":
            return Timeframe(1, "m")
        if c.unit == "h":
            return Timeframe(1, "h")
        if c.unit in ("d", "w", "mo"):
            return Timeframe(1, "d")
        return self

    @property
    def aggregation_factor(self) -> int:
        """Number of base bars to aggregate (1 if native).

        Uses canonical form: 48h→2d has factor 2, not 48.
        """
        if not self.needs_aggregation:
            return 1
        c = self.canonical
        if c.unit in ("m", "h", "d"):
            return c.amount
        if c.unit == "w":
            return c.amount * 5  # 5 trading days per week
        if c.unit == "mo":
            return c.amount * 21  # ~21 trading days per month
        return 1

    @property
    def alpaca_timeframe(self) -> AlpacaTimeFrame:
        """Convert to Alpaca TimeFrame.

        Uses canonical form for native (60m→1h) or base for aggregation (90m→1m).
        """
        if self.needs_aggregation:
            tf = self.base_timeframe
        else:
            tf = self.canonical  # 60m→1h for native Alpaca call
        unit_map = {
            "m": TimeFrameUnit.Minute,
            "h": TimeFrameUnit.Hour,
            "d": TimeFrameUnit.Day,
            "w": TimeFrameUnit.Week,
            "mo": TimeFrameUnit.Month,
        }
        return AlpacaTimeFrame(tf.amount, unit_map[tf.unit])

    @property
    def seconds(self) -> int:
        """Duration in seconds."""
        if self.unit == "m":
            return self.amount * 60
        if self.unit == "h":
            return self.amount * 3600
        if self.unit == "d":
            return self.amount * 86400
        if self.unit == "w":
            return self.amount * 7 * 86400
        if self.unit == "mo":
            return self.amount * 30 * 86400
        return 0

    @property
    def canonical(self) -> "Timeframe":
        """Return canonical form (60m→1h, 24h→1d, etc.)."""
        # Minutes to hours (60m→1h, 120m→2h)
        if self.unit == "m" and self.amount >= 60 and self.amount % 60 == 0:
            return Timeframe(self.amount // 60, "h").canonical
        # Hours to days (24h→1d, 48h→2d)
        if self.unit == "h" and self.amount >= 24 and self.amount % 24 == 0:
            return Timeframe(self.amount // 24, "d").canonical
        # Don't convert days→weeks (5d != 1w semantically in trading)
        return self

    def __str__(self) -> str:
        """Return canonical string representation."""
        c = self.canonical
        return f"{c.amount}{c.unit}"


def parse_timeframe(tf_str: str) -> Timeframe:
    """Parse any timeframe string into our Timeframe abstraction.

    Accepts arbitrary timeframes: 15m, 60m, 4h, 24h, 1d, 3d, 2w, etc.
    Alpaca constraints are handled internally via aggregation.

    Args:
        tf_str: Timeframe string (e.g., "15m", "60m", "3d", "2w")

    Returns:
        Timeframe object

    Raises:
        ValueError: If timeframe format is invalid
    """
    tf = tf_str.lower().strip()
    match = re.match(r"^(\d+)([a-z]+)$", tf)

    if match:
        amount = int(match.group(1))
        unit_str = match.group(2)

        if amount <= 0:
            raise ValueError(f"Timeframe amount must be positive: {tf_str}")

        # Normalize unit
        if unit_str in ("m", "min", "minute"):
            return Timeframe(amount, "m")
        if unit_str in ("h", "hour"):
            return Timeframe(amount, "h")
        if unit_str in ("d", "day"):
            return Timeframe(amount, "d")
        if unit_str in ("w", "week"):
            return Timeframe(amount, "w")
        if unit_str in ("mo", "month"):
            return Timeframe(amount, "mo")

    # Fallback for bare unit names
    if tf in ("m", "min", "minute"):
        return Timeframe(1, "m")
    if tf in ("h", "hour"):
        return Timeframe(1, "h")
    if tf in ("d", "day"):
        return Timeframe(1, "d")

    raise ValueError(f"Invalid timeframe format: {tf_str}")


# Legacy compatibility - will be removed
@dataclass
class AggregationConfig:
    """Deprecated: Use Timeframe.needs_aggregation instead."""

    base_timeframe: str
    factor: int


def get_aggregation_config(tf: str) -> AggregationConfig | None:
    """Deprecated: Use parse_timeframe(tf).needs_aggregation instead."""
    try:
        parsed = parse_timeframe(tf)
    except ValueError:
        return None
    if parsed.needs_aggregation:
        return AggregationConfig(str(parsed.base_timeframe), parsed.aggregation_factor)
    return None


def get_interval_seconds(tf: str) -> int:
    """Get interval duration in seconds for a timeframe string.

    Args:
        tf: Timeframe string (e.g., "15m", "1h", "4h", "1d", "3d")

    Returns:
        Interval duration in seconds
    """
    return parse_timeframe(tf).seconds


def align_feeds(
    primary_bars: list["Bar"],
    informative_bars: list["Bar"],
) -> list["Bar | None"]:
    """Align informative feed to primary feed timestamps via forward-fill.

    For each primary bar, finds the most recent informative bar that
    precedes it. Works for both multi-timeframe (same symbol) and
    multi-symbol scenarios.

    Args:
        primary_bars: Primary feed bars (determines output length)
        informative_bars: Informative feed bars to align

    Returns:
        List of aligned bars (same length as primary_bars), with None
        where no informative bar is available yet
    """
    if not informative_bars:
        return [None] * len(primary_bars)

    aligned: list["Bar | None"] = []
    info_idx = 0

    for bar in primary_bars:
        # Advance to latest informative bar that precedes this bar
        while (
            info_idx < len(informative_bars) - 1
            and informative_bars[info_idx + 1].timestamp <= bar.timestamp
        ):
            info_idx += 1

        # Only include if informative bar timestamp <= primary bar timestamp
        if informative_bars[info_idx].timestamp <= bar.timestamp:
            aligned.append(informative_bars[info_idx])
        else:
            aligned.append(None)

    return aligned
