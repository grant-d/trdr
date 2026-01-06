"""Simple Moving Average indicator."""

import numpy as np

from ..data import Bar


class SmaIndicator:
    """Streaming SMA calculator."""

    def __init__(self, period: int) -> None:
        self.period = max(1, period)
        self._values: list[float] = []

    def update(self, bar: Bar) -> float:
        self._values.append(bar.close)
        if len(self._values) < self.period:
            return self._values[-1]
        window = self._values[-self.period :]
        return float(np.mean(window))

    @property
    def value(self) -> float:
        if not self._values:
            return 0.0
        if len(self._values) < self.period:
            return self._values[-1]
        window = self._values[-self.period :]
        return float(np.mean(window))

    @staticmethod
    def calculate(bars: list[Bar], period: int) -> float:
        if not bars:
            return 0.0
        period = max(1, period)
        closes = [b.close for b in bars]
        if len(closes) < period:
            return closes[-1]
        window = closes[-period:]
        return float(np.mean(window))


def sma_series(values: list[float], period: int) -> list[float]:
    """Calculate SMA series for values."""
    if not values:
        return []
    period = max(1, period)
    if len(values) < period:
        return list(values)

    result: list[float] = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(float(values[i]))
            continue
        window = values[i - period + 1 : i + 1]
        result.append(float(np.mean(window)))

    return result


class SmaSeriesIndicator:
    """Stateful SMA series generator."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._values: list[float] = []

    def update(self, bar: Bar) -> list[float]:
        self._values.append(bar.close)
        return sma_series(self._values, self.period)

    @property
    def value(self) -> list[float]:
        return sma_series(self._values, self.period)

    @staticmethod
    def calculate(values: list[float], period: int) -> list[float]:
        return sma_series(values, period)
