"""Average True Range indicator."""

import numpy as np

from ..data import Bar


def atr(bars: list[Bar], period: int = 14) -> float:
    """Calculate Average True Range.

    Args:
        bars: List of OHLCV bars
        period: ATR period

    Returns:
        Current ATR value
    """
    if len(bars) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(bars)):
        high = bars[i].high
        low = bars[i].low
        prev_close = bars[i - 1].close

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    # Wilder's smoothed ATR
    atr_val = np.mean(true_ranges[-period:])
    return float(atr_val)


class AtrIndicator:
    """Streaming ATR calculator."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._bars: list[Bar] = []

    def update(self, bar: Bar) -> float:
        self._bars.append(bar)
        return atr(self._bars, self.period)

    @property
    def value(self) -> float:
        return atr(self._bars, self.period) if self._bars else 0.0
