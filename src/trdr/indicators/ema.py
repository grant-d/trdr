"""Exponential Moving Average indicator."""

import numpy as np

from ..data import Bar


def ema_series(values: list[float], period: int) -> list[float]:
    """Calculate EMA series from raw values.

    Args:
        values: List of price values
        period: EMA period

    Returns:
        List of EMA values (0s for insufficient data)
    """
    if len(values) < period:
        return [0.0] * len(values)

    alpha = 2 / (period + 1)
    ema_values = [0.0] * len(values)

    # Start with SMA for first period
    ema_values[period - 1] = np.mean(values[:period])

    # Calculate EMA for rest
    for i in range(period, len(values)):
        ema_values[i] = alpha * values[i] + (1 - alpha) * ema_values[i - 1]

    return ema_values


class EmaIndicator:
    """Streaming EMA calculator."""

    def __init__(self, period: int) -> None:
        self.period = max(1, period)
        self.alpha = 2 / (self.period + 1)
        self._value: float | None = None

    def update(self, bar: Bar) -> float:
        close = bar.close
        if self._value is None:
            self._value = float(close)
            return self._value
        self._value = self.alpha * close + (1 - self.alpha) * self._value
        return float(self._value)

    @staticmethod
    def calculate(bars: list[Bar], period: int) -> float:
        if not bars:
            return 0.0
        if period <= 1:
            return bars[-1].close
        if len(bars) < period:
            return bars[-1].close
        calc = EmaIndicator(period)
        for bar in bars:
            calc.update(bar)
        return calc.value

    @property
    def value(self) -> float:
        if self._value is not None:
            return float(self._value)
        return 0.0
