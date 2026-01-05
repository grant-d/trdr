"""Hull Moving Average indicator for trend confirmation."""

import numpy as np

from ..data import Bar
from .wma import _wma_values


def hma(bars: list[Bar], period: int = 9) -> float:
    """Calculate Hull Moving Average for trend confirmation.

    Args:
        bars: List of OHLCV bars
        period: HMA period

    Returns:
        Current HMA value
    """
    if not bars:
        return 0.0
    if period <= 1:
        return bars[-1].close
    if len(bars) < period:
        return 0.0

    closes = [b.close for b in bars]
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))

    wma_half = _wma_values(closes, half_period)
    wma_full = _wma_values(closes, period)
    diff = [2 * h - f for h, f in zip(wma_half, wma_full)]
    hma_series = _wma_values(diff, sqrt_period)

    return float(hma_series[-1])


class HmaIndicator:
    """Streaming HMA calculator."""

    def __init__(self, period: int = 9) -> None:
        self.period = period
        self._bars: list[Bar] = []

    def update(self, bar: Bar) -> float:
        self._bars.append(bar)
        return hma(self._bars, self.period)

    @property
    def value(self) -> float:
        return hma(self._bars, self.period) if self._bars else 0.0


def hma_slope(bars: list[Bar], period: int = 9, lookback: int = 3) -> float:
    """Calculate HMA slope over lookback period.

    Args:
        bars: List of OHLCV bars
        period: HMA period
        lookback: Number of bars to measure slope

    Returns:
        HMA slope (positive = uptrend, negative = downtrend)
    """
    if len(bars) < period + lookback:
        return 0.0

    hma_current = hma(bars, period)
    hma_prev = hma(bars[:-lookback], period)

    return hma_current - hma_prev


class HmaSlopeIndicator:
    """Streaming HMA slope calculator."""

    def __init__(self, period: int = 9, lookback: int = 3) -> None:
        self.period = period
        self.lookback = lookback
        self._bars: list[Bar] = []

    def update(self, bar: Bar) -> float:
        self._bars.append(bar)
        return hma_slope(self._bars, self.period, self.lookback)

    @property
    def value(self) -> float:
        return hma_slope(self._bars, self.period, self.lookback) if self._bars else 0.0
