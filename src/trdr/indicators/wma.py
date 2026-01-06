"""Weighted Moving Average indicator."""

import numpy as np

from ..data import Bar


def _wma_values(values: list[float], period: int) -> list[float]:
    """Calculate WMA series for raw values."""
    if period <= 0:
        return [0.0] * len(values)
    if len(values) < period:
        return [0.0] * len(values)

    weights = np.arange(1, period + 1)
    weight_sum = np.sum(weights)
    result = [0.0] * len(values)

    for i in range(period - 1, len(values)):
        window = np.array(values[i - period + 1 : i + 1])
        result[i] = float(np.sum(window * weights) / weight_sum)

    return result


class WmaIndicator:
    """Streaming WMA calculator."""

    def __init__(self, period: int) -> None:
        self.period = max(1, period)
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(bars: list[Bar], period: int) -> float:
        if len(bars) < period:
            return bars[-1].close if bars else 0.0

        closes = np.array([b.close for b in bars[-period:]])
        weights = np.arange(1, period + 1)
        return float(np.sum(closes * weights) / np.sum(weights))

    def update(self, bar: Bar) -> float:
        self._bars.append(bar)
        return self.calculate(self._bars, self.period)

    @property
    def value(self) -> float:
        return self.calculate(self._bars, self.period) if self._bars else 0.0
