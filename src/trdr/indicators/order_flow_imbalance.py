"""Order Flow Imbalance indicator for buy/sell pressure analysis."""

from ..data import Bar


class OrderFlowImbalanceIndicator:
    """Streaming order flow imbalance calculator."""

    def __init__(self, lookback: int = 5) -> None:
        self.lookback = max(1, lookback)
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(bars: list[Bar], lookback: int = 5) -> float:
        if len(bars) < lookback + 1:
            return 0.0

        recent_bars = bars[-lookback:]
        buy_volume = 0.0
        sell_volume = 0.0

        for i in range(len(recent_bars)):
            if i == 0:
                continue
            if recent_bars[i].close > recent_bars[i - 1].close:
                buy_volume += recent_bars[i].volume
            elif recent_bars[i].close < recent_bars[i - 1].close:
                sell_volume += recent_bars[i].volume
            else:
                buy_volume += recent_bars[i].volume * 0.5
                sell_volume += recent_bars[i].volume * 0.5

        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0

        ofi = (buy_volume / total_volume) - 0.5
        return float(ofi)

    def update(self, bar: Bar) -> float:
        self._bars.append(bar)
        return self.calculate(self._bars, lookback=self.lookback)

    @property
    def value(self) -> float:
        if not self._bars:
            return 0.0
        return self.calculate(self._bars, lookback=self.lookback)
