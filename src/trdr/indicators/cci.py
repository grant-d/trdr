"""Commodity Channel Index indicator."""

import numpy as np

from ..data import Bar


def cci(bars: list[Bar], period: int = 20) -> float:
    """Calculate Commodity Channel Index.

    Args:
        bars: List of OHLCV bars
        period: CCI period

    Returns:
        CCI value (typically -100 to +100, can exceed)
    """
    if len(bars) < period:
        return 0.0

    # Typical price = (high + low + close) / 3
    typical_prices = [(b.high + b.low + b.close) / 3 for b in bars[-period:]]
    sma_tp = np.mean(typical_prices)
    mean_deviation = np.mean([abs(tp - sma_tp) for tp in typical_prices])

    if mean_deviation == 0:
        return 0.0

    cci_val = (typical_prices[-1] - sma_tp) / (0.015 * mean_deviation)
    return float(cci_val)


class CciIndicator:
    """Streaming CCI calculator."""

    def __init__(self, period: int = 20) -> None:
        self.period = period
        self._bars: list[Bar] = []

    def update(self, bar: Bar) -> float:
        self._bars.append(bar)
        return cci(self._bars, self.period)

    @property
    def value(self) -> float:
        return cci(self._bars, self.period) if self._bars else 0.0
