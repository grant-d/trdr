"""Backtest utilities for timeframe parsing."""

import re

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def parse_timeframe(tf_str: str) -> TimeFrame:
    """Parse timeframe string to Alpaca TimeFrame.

    Supports: "1h", "4h", "15m", "1d", "hour", "minute", "day"
    Note: Day only supports amount=1 (Alpaca constraint).

    Args:
        tf_str: Timeframe string (e.g., "1h", "4h", "15m", "1d")

    Returns:
        Alpaca TimeFrame object
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
            # Alpaca: Day only allows amount=1
            if unit == TimeFrameUnit.Day:
                amount = 1
            return TimeFrame(amount, unit)

    # Fallback to simple names or default hour
    unit = unit_map.get(tf)
    return TimeFrame(1, unit) if unit else TimeFrame.Hour
