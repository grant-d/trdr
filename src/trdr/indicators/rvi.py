"""Relative Volatility Index indicator."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..data import Bar

_RviMode = str


def rvi(bars: list[Bar], period: int = 10, mode: _RviMode = "ema") -> float:
    """Calculate Relative Volatility Index.

    Args:
        bars: List of OHLCV bars
        period: RVI period
        mode: "ema" (TradingView-style) or "sma" smoothing

    Returns:
        RVI value (0-100)
    """
    if not bars:
        return 50.0
    calc = RviIndicator(period, mode=mode)
    for bar in bars:
        calc.update(bar)
    return calc.value


class _EmaValue:
    """EMA helper for scalar values."""

    def __init__(self, period: int) -> None:
        self.period = max(1, period)
        self.alpha = 2 / (self.period + 1)
        self._seed: list[float] = []
        self._value: float | None = None

    def update(self, value: float) -> float:
        if self.period <= 1:
            self._value = float(value)
            return self._value

        if self._value is None:
            self._seed.append(float(value))
            if len(self._seed) < self.period:
                return self._seed[-1]
            self._value = sum(self._seed) / self.period
            return self._value

        self._value = self.alpha * float(value) + (1 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        if self._value is not None:
            return float(self._value)
        return float(self._seed[-1]) if self._seed else 0.0


class RviIndicator:
    """Streaming RVI calculator."""

    def __init__(self, period: int = 10, mode: _RviMode = "ema") -> None:
        self.period = max(1, period)
        if mode not in {"ema", "sma"}:
            raise ValueError(f"Unsupported RVI mode: {mode}")
        self.mode = mode
        self._prev_close: float | None = None
        self._close_window: deque[float] = deque(maxlen=self.period)
        self._value = 50.0

        self._std_up: deque[float] = deque(maxlen=self.period)
        self._std_down: deque[float] = deque(maxlen=self.period)
        self._sum_up = 0.0
        self._sum_down = 0.0

        self._ema_up = _EmaValue(self.period)
        self._ema_down = _EmaValue(self.period)

    @staticmethod
    def calculate(bars: list[Bar], period: int = 10, mode: _RviMode = "ema") -> float:
        return rvi(bars, period=period, mode=mode)

    def _update_sma(self, up_val: float, down_val: float) -> tuple[float, float]:
        if len(self._std_up) == self.period:
            self._sum_up -= self._std_up[0]
            self._sum_down -= self._std_down[0]

        self._std_up.append(up_val)
        self._std_down.append(down_val)
        self._sum_up += up_val
        self._sum_down += down_val

        upper = self._sum_up / len(self._std_up)
        lower = self._sum_down / len(self._std_down)
        return upper, lower

    def _update_ema(self, up_val: float, down_val: float) -> tuple[float, float]:
        upper = self._ema_up.update(up_val)
        lower = self._ema_down.update(down_val)
        return upper, lower

    def update(self, bar: Bar) -> float:
        close = bar.close
        if self._prev_close is None:
            self._prev_close = close
            self._close_window.append(close)
            return self._value

        change = close - self._prev_close
        self._close_window.append(close)

        if len(self._close_window) < self.period:
            self._prev_close = close
            return self._value

        std = float(np.std(self._close_window))
        if change > 0:
            up_val = std
            down_val = 0.0
        elif change < 0:
            up_val = 0.0
            down_val = std
        else:
            up_val = 0.0
            down_val = 0.0

        if self.mode == "ema":
            upper, lower = self._update_ema(up_val, down_val)
        else:
            upper, lower = self._update_sma(up_val, down_val)

        if upper + lower == 0:
            self._value = 50.0
        else:
            self._value = float((upper / (upper + lower)) * 100)

        self._prev_close = close
        return self._value

    @property
    def value(self) -> float:
        return float(self._value)
