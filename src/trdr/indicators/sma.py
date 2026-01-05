"""Simple Moving Average indicator."""

import numpy as np

from ..data import Bar


def sma(bars: list[Bar], period: int) -> float:
    """Calculate Simple Moving Average.

    Args:
        bars: List of OHLCV bars
        period: SMA period

    Returns:
        Current SMA value
    """
    if len(bars) < period:
        return bars[-1].close if bars else 0.0
    closes = [b.close for b in bars[-period:]]
    return float(np.mean(closes))


class SmaIndicator:
    """Streaming SMA calculator."""

    def __init__(self, period: int) -> None:
        self.period = max(1, period)
        self._values: list[float] = []

    def update(self, bar: Bar) -> float:
        self._values.append(bar.close)
        if len(self._values) < self.period:
            return self._values[-1]
        return float(np.mean(self._values[-self.period:]))

    @property
    def value(self) -> float:
        if not self._values:
            return 0.0
        if len(self._values) < self.period:
            return self._values[-1]
        return float(np.mean(self._values[-self.period:]))


def sma_series(bars: list[Bar], period: int) -> list[float]:
    """Calculate SMA series for all bars.

    Args:
        bars: List of OHLCV bars
        period: SMA period

    Returns:
        List of SMA values (0s for insufficient data)
    """
    if len(bars) < period:
        return [0.0] * len(bars)

    closes = [b.close for b in bars]
    result = [0.0] * len(closes)

    for i in range(period - 1, len(closes)):
        result[i] = float(np.mean(closes[i - period + 1 : i + 1]))

    return result


class SmaSeriesIndicator:
    """Stateful SMA series generator."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._bars: list[Bar] = []

    def update(self, bar: Bar) -> list[float]:
        self._bars.append(bar)
        return sma_series(self._bars, self.period)

    @property
    def series(self) -> list[float]:
        return sma_series(self._bars, self.period)
