"""Stochastic Momentum Index indicator."""

from ..data import Bar


class SmiIndicator:
    """Streaming SMI calculator."""

    def __init__(self, k: int = 10, d: int = 3) -> None:
        self.k = max(1, k)
        self.d = max(1, d)
        self._bars: list[Bar] = []
        self._count = 0

    @staticmethod
    def calculate(bars: list[Bar], k: int = 10, d: int = 3) -> float:
        if len(bars) < k + d:
            return 0.0

        closes = [b.close for b in bars[-k:]]
        highs = [b.high for b in bars[-k:]]
        lows = [b.low for b in bars[-k:]]

        hh = max(highs)
        ll = min(lows)

        if hh == ll:
            return 0.0

        # Distance from midpoint
        mid = (hh + ll) / 2.0
        distance = closes[-1] - mid

        # Simplified: use price distance as numerator, range as denominator
        num = distance
        den = (hh - ll) / 2.0

        if den == 0:
            return 0.0

        smi_val = 100 * (num / den)
        return float(smi_val)

    def update(self, bar: Bar) -> float:
        self._bars.append(bar)
        self._count += 1
        if self._count < self.k + self.d:
            return 0.0
        return self.calculate(self._bars, k=self.k, d=self.d)

    @property
    def value(self) -> float:
        return self.calculate(self._bars, k=self.k, d=self.d) if self._bars else 0.0
