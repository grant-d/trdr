"""Volume trend analysis indicator."""

import numpy as np

from ..data import Bar


class VolumeTrendIndicator:
    """Streaming volume trend detector."""

    def __init__(self, lookback: int = 5) -> None:
        self.lookback = max(1, lookback)
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(bars: list[Bar], lookback: int = 5) -> str:
        if len(bars) < lookback:
            return "neutral"

        recent_volumes = [b.volume for b in bars[-lookback:]]
        earlier_volumes = [b.volume for b in bars[-lookback * 2 : -lookback]]

        if not earlier_volumes:
            return "neutral"

        recent_avg = np.mean(recent_volumes)
        earlier_avg = np.mean(earlier_volumes)

        if recent_avg > earlier_avg * 1.2:
            return "increasing"
        if recent_avg < earlier_avg * 0.8:
            return "declining"
        return "neutral"

    def update(self, bar: Bar) -> str:
        self._bars.append(bar)
        return self.calculate(self._bars, lookback=self.lookback)

    @property
    def value(self) -> str:
        if not self._bars:
            return "neutral"
        return self.calculate(self._bars, lookback=self.lookback)
