"""Bollinger Bands volatility indicator."""

import numpy as np

from ..data import Bar


def bollinger_bands(
    bars: list[Bar], period: int = 20, std_mult: float = 2.0
) -> tuple[float, float, float]:
    """Calculate Bollinger Bands.

    Args:
        bars: List of OHLCV bars
        period: SMA period
        std_mult: Standard deviation multiplier

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if len(bars) < period:
        price = bars[-1].close if bars else 0
        return (price, price, price)

    closes = np.array([b.close for b in bars[-period:]])
    middle = float(np.mean(closes))
    std = float(np.std(closes))

    upper = middle + std_mult * std
    lower = middle - std_mult * std

    return (upper, middle, lower)


class BollingerBandsIndicator:
    """Streaming Bollinger Bands calculator."""

    def __init__(self, period: int = 20, std_mult: float = 2.0) -> None:
        self.period = period
        self.std_mult = std_mult
        self._bars: list[Bar] = []

    def update(self, bar: Bar) -> tuple[float, float, float]:
        self._bars.append(bar)
        return bollinger_bands(self._bars, self.period, self.std_mult)

    @property
    def value(self) -> tuple[float, float, float]:
        return bollinger_bands(self._bars, self.period, self.std_mult)
