"""Average True Range indicator."""

import numpy as np

from ..data import Bar


class AtrIndicator:
    """Streaming ATR calculator."""

    def __init__(self, period: int = 14) -> None:
        self.period = max(1, period)
        self._prev_close: float | None = None
        self._tr_values: list[float] = []
        self._atr: float | None = None

    def update(self, bar: Bar) -> float:
        if self._prev_close is None:
            tr = bar.high - bar.low
            self._tr_values.append(tr)
            self._prev_close = bar.close
            self._atr = tr
            return float(self._atr)

        tr = max(
            bar.high - bar.low,
            abs(bar.high - self._prev_close),
            abs(bar.low - self._prev_close),
        )

        if self._atr is None:
            self._tr_values.append(tr)
            if len(self._tr_values) < self.period:
                self._atr = float(np.mean(self._tr_values))
            else:
                self._atr = float(np.mean(self._tr_values[-self.period :]))
        else:
            self._atr = (self._atr * (self.period - 1) + tr) / self.period

        self._prev_close = bar.close
        return float(self._atr)

    @staticmethod
    def calculate(bars: list[Bar], period: int = 14) -> float:
        if not bars:
            return 0.0
        if len(bars) == 1:
            return bars[0].close
        calc = AtrIndicator(period)
        for bar in bars:
            calc.update(bar)
        return calc.value

    @property
    def value(self) -> float:
        return float(self._atr) if self._atr is not None else 0.0
