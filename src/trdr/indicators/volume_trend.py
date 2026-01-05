"""Volume trend analysis indicator."""

import numpy as np

from ..data import Bar


def volume_trend(bars: list[Bar], lookback: int = 5) -> str:
    """Analyze recent volume trend.

    Args:
        bars: List of OHLCV bars
        lookback: Number of recent bars to analyze

    Returns:
        "increasing", "declining", or "neutral"
    """
    if len(bars) < lookback:
        return "neutral"

    recent_volumes = [b.volume for b in bars[-lookback:]]
    earlier_volumes = [b.volume for b in bars[-lookback * 2 : -lookback]]

    if not earlier_volumes:
        return "neutral"

    recent_avg = np.mean(recent_volumes)
    earlier_avg = np.mean(earlier_volumes)

    if recent_avg > earlier_avg * 1.2:
        return "increasing"
    elif recent_avg < earlier_avg * 0.8:
        return "declining"
    return "neutral"
