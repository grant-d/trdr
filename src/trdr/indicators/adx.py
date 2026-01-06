"""Average Directional Index indicator."""

from collections import deque

import numpy as np

from ..data import Bar


class AdxIndicator:
    """Streaming ADX using Wilder smoothing."""

    def __init__(self, period: int = 14) -> None:
        self.period = max(1, period)
        self._prev_bar: Bar | None = None
        self._tr: deque[float] = deque(maxlen=self.period)
        self._plus_dm: deque[float] = deque(maxlen=self.period)
        self._minus_dm: deque[float] = deque(maxlen=self.period)
        self._atr: float | None = None
        self._plus_dm_s: float | None = None
        self._minus_dm_s: float | None = None
        self._dx: deque[float] = deque(maxlen=self.period)
        self._adx: float | None = None

    def update(self, bar: Bar) -> float:
        if self._prev_bar is None:
            self._prev_bar = bar
            return 0.0

        high = bar.high
        low = bar.low
        prev_close = self._prev_bar.close
        high_diff = high - self._prev_bar.high
        low_diff = self._prev_bar.low - low

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0.0
        minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0.0

        self._tr.append(tr)
        self._plus_dm.append(plus_dm)
        self._minus_dm.append(minus_dm)

        if self._atr is None:
            if len(self._tr) < self.period:
                self._prev_bar = bar
                return 0.0
            self._atr = float(np.mean(self._tr))
            self._plus_dm_s = float(np.mean(self._plus_dm))
            self._minus_dm_s = float(np.mean(self._minus_dm))
        else:
            self._atr = (self._atr * (self.period - 1) + tr) / self.period
            self._plus_dm_s = (self._plus_dm_s * (self.period - 1) + plus_dm) / self.period
            self._minus_dm_s = (self._minus_dm_s * (self.period - 1) + minus_dm) / self.period

        if self._atr == 0:
            dx = 0.0
        else:
            plus_di = (self._plus_dm_s / self._atr) * 100
            minus_di = (self._minus_dm_s / self._atr) * 100
            denom = plus_di + minus_di
            dx = abs(plus_di - minus_di) / denom * 100 if denom != 0 else 0.0

        self._dx.append(dx)
        if self._adx is None:
            if len(self._dx) == self.period:
                self._adx = float(np.mean(self._dx))
        else:
            self._adx = (self._adx * (self.period - 1) + dx) / self.period

        self._prev_bar = bar
        return float(self._adx) if self._adx is not None else 0.0

    @staticmethod
    def calculate(bars: list[Bar], period: int = 14) -> float:
        if not bars:
            return 0.0
        calc = AdxIndicator(period)
        for bar in bars:
            calc.update(bar)
        return calc.value

    @property
    def value(self) -> float:
        return float(self._adx) if self._adx is not None else 0.0
