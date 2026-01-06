"""Hull Moving Average indicator for trend confirmation."""

import numpy as np

from ..data import Bar
from .wma import _wma_values


def _hma_calculate(bars: list[Bar], period: int = 9) -> float:
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

    @staticmethod
    def calculate(bars: list[Bar], period: int = 9) -> float:
        return _hma_calculate(bars, period)

    def update(self, bar: Bar) -> float:
        self._bars.append(bar)
        return self.calculate(self._bars, self.period)

    @property
    def value(self) -> float:
        return self.calculate(self._bars, self.period) if self._bars else 0.0


def _hma_slope_calculate(bars: list[Bar], period: int = 9, lookback: int = 3) -> float:
    if len(bars) < period + lookback:
        return 0.0

    hma_current = _hma_calculate(bars, period)
    hma_prev = _hma_calculate(bars[:-lookback], period)

    return hma_current - hma_prev


class HmaSlopeIndicator:
    """Streaming HMA slope calculator."""

    def __init__(self, period: int = 9, lookback: int = 3) -> None:
        self.period = period
        self.lookback = lookback
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(bars: list[Bar], period: int = 9, lookback: int = 3) -> float:
        return _hma_slope_calculate(bars, period, lookback)

    def update(self, bar: Bar) -> float:
        self._bars.append(bar)
        return self.calculate(self._bars, self.period, self.lookback)

    @property
    def value(self) -> float:
        return self.calculate(self._bars, self.period, self.lookback) if self._bars else 0.0
