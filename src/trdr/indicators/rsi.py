"""Relative Strength Index indicator."""

from collections import deque

from ..data import Bar


class RsiIndicator:
    """Streaming RSI with rolling average of gains/losses."""

    def __init__(self, period: int = 14) -> None:
        self.period = max(1, period)
        self._prev_close: float | None = None
        self._gains: deque[float] = deque(maxlen=self.period)
        self._losses: deque[float] = deque(maxlen=self.period)
        self._sum_gain = 0.0
        self._sum_loss = 0.0
        self._value = 50.0

    def update(self, bar: Bar) -> float:
        close = bar.close
        if self._prev_close is None:
            self._prev_close = close
            return self._value

        change = close - self._prev_close
        gain = max(change, 0.0)
        loss = abs(min(change, 0.0))

        if len(self._gains) == self.period:
            self._sum_gain -= self._gains[0]
            self._sum_loss -= self._losses[0]

        self._gains.append(gain)
        self._losses.append(loss)
        self._sum_gain += gain
        self._sum_loss += loss

        if len(self._gains) < self.period:
            self._value = 50.0
        else:
            avg_gain = self._sum_gain / self.period
            avg_loss = self._sum_loss / self.period
            if avg_gain == 0 and avg_loss == 0:
                self._value = 50.0
            elif avg_loss == 0:
                self._value = 100.0
            else:
                rs = avg_gain / avg_loss
                self._value = float(100 - (100 / (1 + rs)))

        self._prev_close = close
        return self._value

    @staticmethod
    def calculate(bars: list[Bar], period: int = 14) -> float:
        if not bars:
            return 50.0
        calc = RsiIndicator(period)
        for bar in bars:
            calc.update(bar)
        return calc.value

    @property
    def value(self) -> float:
        return float(self._value)
