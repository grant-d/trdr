"""Bollinger Bands volatility indicator."""

import numpy as np

from ..data import Bar


class BollingerBandsIndicator:
    """Streaming Bollinger Bands calculator."""

    def __init__(self, period: int = 20, std_mult: float = 2.0) -> None:
        self.period = max(1, period)
        self.std_mult = std_mult
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(
        bars: list[Bar], period: int = 20, std_mult: float = 2.0
    ) -> tuple[float, float, float]:
        if len(bars) < period:
            price = bars[-1].close if bars else 0
            return (price, price, price)

        closes = np.array([b.close for b in bars[-period:]])
        middle = float(np.mean(closes))
        std = float(np.std(closes))

        upper = middle + std_mult * std
        lower = middle - std_mult * std

        return (upper, middle, lower)

    def update(self, bar: Bar) -> tuple[float, float, float]:
        self._bars.append(bar)
        return self.calculate(self._bars, self.period, self.std_mult)

    @property
    def value(self) -> tuple[float, float, float]:
        return self.calculate(self._bars, self.period, self.std_mult)
