"""Market Structure Score for regime detection."""

import numpy as np

from ..data import Bar
from .atr import AtrIndicator


class MssIndicator:
    """Streaming MSS calculator."""

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = max(1, lookback)
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(bars: list[Bar], lookback: int = 20) -> float:
        if len(bars) < lookback:
            return 0.0

        recent_bars = bars[-lookback:]
        closes = [b.close for b in recent_bars]

        x = np.arange(lookback)
        y = np.array(closes)
        slope = np.polyfit(x, y, 1)[0]
        trend_pct = (slope / closes[-1] * 100) if closes[-1] != 0 else 0

        atr_val = AtrIndicator.calculate(bars, lookback)
        volatility_pct = max(0, 80 - (atr_val / closes[-1] * 100 * 3)) if closes[-1] != 0 else 40

        recent_high = max(b.high for b in recent_bars)
        recent_low = min(b.low for b in recent_bars)
        recent_range = recent_high - recent_low
        if recent_range > 0:
            exhaustion = ((closes[-1] - recent_low) / recent_range * 100) - 50
        else:
            exhaustion = 0

        mss_val = (trend_pct * 0.5) + (volatility_pct * 0.2) + (exhaustion * 0.3)
        return float(np.clip(mss_val, -100, 100))

    def update(self, bar: Bar) -> float:
        self._bars.append(bar)
        return self.calculate(self._bars, lookback=self.lookback)

    @property
    def value(self) -> float:
        if not self._bars:
            return 0.0
        return self.calculate(self._bars, lookback=self.lookback)
