"""Backtest utilities for timeframe parsing and data alignment."""

import re
from typing import TYPE_CHECKING

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

if TYPE_CHECKING:
    from ..data.market import Bar


def get_interval_seconds(tf: str) -> int:
    """Get interval duration in seconds for a timeframe string.

    Args:
        tf: Timeframe string (e.g., "15m", "1h", "4h", "1d")

    Returns:
        Interval duration in seconds
    """
    tf_lower = tf.lower().strip()
    match = re.match(r"^(\d+)([a-z]+)$", tf_lower)

    if match:
        amount, unit = int(match.group(1)), match.group(2)
        if unit in ("m", "min", "minute"):
            return amount * 60
        elif unit in ("h", "hour"):
            return amount * 3600
        elif unit in ("d", "day"):
            return amount * 86400

    # Fallback for simple names
    if tf_lower in ("m", "min", "minute"):
        return 60
    elif tf_lower in ("h", "hour"):
        return 3600
    elif tf_lower in ("d", "day"):
        return 86400

    raise ValueError(f"Unsupported timeframe: {tf}")


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


def parse_timeframe(tf_str: str) -> TimeFrame:
    """Parse timeframe string to Alpaca TimeFrame.

    Accepts arbitrary int+unit format: 23m, 5h, 1d, etc.
    Alpaca constraints: minutes 1-59, hours 1-23, days 1 only.

    Args:
        tf_str: Timeframe string (e.g., "23m", "5h", "1d")

    Returns:
        Alpaca TimeFrame object

    Raises:
        ValueError: If timeframe format invalid or out of Alpaca range
    """
    unit_map = {
        "m": TimeFrameUnit.Minute,
        "min": TimeFrameUnit.Minute,
        "minute": TimeFrameUnit.Minute,
        "h": TimeFrameUnit.Hour,
        "hour": TimeFrameUnit.Hour,
        "d": TimeFrameUnit.Day,
        "day": TimeFrameUnit.Day,
    }

    tf = tf_str.lower().strip()
    match = re.match(r"^(\d+)([a-z]+)$", tf)

    if match:
        amount, unit_str = int(match.group(1)), match.group(2)
        unit = unit_map.get(unit_str)
        if unit:
            # Validate Alpaca constraints
            if unit == TimeFrameUnit.Minute and not (1 <= amount <= 59):
                raise ValueError(f"Minutes must be 1-59, got {amount}")
            if unit == TimeFrameUnit.Hour and not (1 <= amount <= 23):
                raise ValueError(f"Hours must be 1-23, got {amount}")
            if unit == TimeFrameUnit.Day and amount != 1:
                raise ValueError("Days must be 1 (Alpaca constraint)")
            return TimeFrame(amount, unit)

    # Fallback to simple names
    unit = unit_map.get(tf)
    if unit:
        return TimeFrame(1, unit)

    raise ValueError(f"Invalid timeframe format: {tf_str}")
