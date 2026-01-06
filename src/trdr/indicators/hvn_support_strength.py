"""HVN support strength indicator for liquidity analysis."""

from ..data import Bar


class HvnSupportStrengthIndicator:
    """Streaming HVN support strength calculator."""

    def __init__(self, val_level: float, lookback: int = 30) -> None:
        self.val_level = val_level
        self.lookback = max(1, lookback)
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(bars: list[Bar], val_level: float, lookback: int = 30) -> float:
        if len(bars) < lookback:
            return 0.0

        recent_bars = bars[-lookback:]
        touches = 0
        bounces = 0

        for i in range(1, len(recent_bars)):
            low = recent_bars[i].low
            high = recent_bars[i].high

            if low <= val_level <= high:
                touches += 1
                if recent_bars[i].close > val_level:
                    bounces += 1

        if touches == 0:
            return 0.0

        bounce_rate = bounces / touches if touches > 0 else 0
        touch_frequency = touches / lookback

        return float(min(bounce_rate * 0.6 + touch_frequency * 0.4, 1.0))

    def update(self, bar: Bar) -> float:
        self._bars.append(bar)
        return self.calculate(self._bars, val_level=self.val_level, lookback=self.lookback)

    @property
    def value(self) -> float:
        if not self._bars:
            return 0.0
        return self.calculate(self._bars, val_level=self.val_level, lookback=self.lookback)
