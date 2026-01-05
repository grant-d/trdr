"""Market Structure Score for regime detection."""

import numpy as np

from ..data import Bar
from .atr import atr


def mss(bars: list[Bar], lookback: int = 20) -> float:
    """Calculate Market Structure Score for regime detection.

    Args:
        bars: List of OHLCV bars
        lookback: Period for calculations

    Returns:
        MSS value (-100 to +100). >30 = bullish, <-30 = bearish, -30 to 30 = neutral
    """
    if len(bars) < lookback:
        return 0.0

    recent_bars = bars[-lookback:]
    closes = [b.close for b in recent_bars]

    # Trend: simple linear regression slope
    x = np.arange(lookback)
    y = np.array(closes)
    slope = np.polyfit(x, y, 1)[0]
    trend_pct = (slope / closes[-1] * 100) if closes[-1] != 0 else 0

    # Volatility: ATR as percentage (inverted: higher vol = lower score)
    atr_val = atr(bars, lookback)
    # Normalize by a reasonable ATR level
    volatility_pct = max(0, 80 - (atr_val / closes[-1] * 100 * 3)) if closes[-1] != 0 else 40

    # Exhaustion: price deviation from recent high/low
    recent_high = max(b.high for b in recent_bars)
    recent_low = min(b.low for b in recent_bars)
    recent_range = recent_high - recent_low
    if recent_range > 0:
        exhaustion = ((closes[-1] - recent_low) / recent_range * 100) - 50  # -50 to +50
    else:
        exhaustion = 0

    # Combine with weights favoring trend and exhaustion
    mss_val = (trend_pct * 0.5) + (volatility_pct * 0.2) + (exhaustion * 0.3)
    return float(np.clip(mss_val, -100, 100))
