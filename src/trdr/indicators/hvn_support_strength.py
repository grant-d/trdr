"""HVN support strength indicator for liquidity analysis."""

from ..data import Bar


def hvn_support_strength(bars: list[Bar], val_level: float, lookback: int = 30) -> float:
    """Detect strength of support at HVN levels using historical touches.

    Counts how many times price bounced from or consolidated at VAL level,
    indicating accumulated liquidity and institutional interest.

    Args:
        bars: List of OHLCV bars
        val_level: VAL price level to analyze
        lookback: Historical bars to check

    Returns:
        Support strength score (0.0 to 1.0)
    """
    if len(bars) < lookback:
        return 0.0

    recent_bars = bars[-lookback:]
    touches = 0
    bounces = 0

    for i in range(1, len(recent_bars)):
        low = recent_bars[i].low
        high = recent_bars[i].high

        # Bar touched the VAL level
        if low <= val_level <= high:
            touches += 1
            # Bar bounced from VAL (closed above after touching)
            if recent_bars[i].close > val_level:
                bounces += 1

    if touches == 0:
        return 0.0

    # Support strength = historical bounce rate at this level
    bounce_rate = bounces / touches if touches > 0 else 0
    # Also factor in frequency of touches (more touches = more tested level)
    touch_frequency = touches / lookback

    return float(min(bounce_rate * 0.6 + touch_frequency * 0.4, 1.0))
