"""Heikin-Ashi smoothed candlestick transformation."""

from ..data import Bar


def heikin_ashi(bars: list[Bar]) -> list[dict]:
    """Transform bars into Heikin-Ashi representation.

    HA smooths price action by using:
    - HA Close = (O + H + L + C) / 4 (average of all prices)
    - HA Open = (prior HA Open + prior HA Close) / 2
    - HA High = max(H, HA Open, HA Close)
    - HA Low = min(L, HA Open, HA Close)

    Args:
        bars: List of original OHLCV bars

    Returns:
        List of dicts with Heikin-Ashi values
    """
    if len(bars) < 1:
        return []

    ha_bars = []

    for i, bar in enumerate(bars):
        # HA Close is average of all prices
        ha_close = (bar.open + bar.high + bar.low + bar.close) / 4.0

        # HA Open: average of prior HA bar's open/close
        if i == 0:
            ha_open = (bar.open + bar.close) / 2.0
        else:
            prev_bar = ha_bars[-1]
            ha_open = (prev_bar["open"] + prev_bar["close"]) / 2.0

        # HA High/Low: covers all prices
        ha_high = max(bar.high, ha_open, ha_close)
        ha_low = min(bar.low, ha_open, ha_close)

        ha_bar = {
            "open": ha_open,
            "high": ha_high,
            "low": ha_low,
            "close": ha_close,
            "volume": bar.volume,
            "timestamp": getattr(bar, "timestamp", None),
        }
        ha_bars.append(ha_bar)

    return ha_bars
