"""Volatility regime classification using realized volatility."""

from collections import deque

import numpy as np

from ..data import Bar


class VolatilityRegimeIndicator:
    """Streaming volatility regime classifier."""

    def __init__(self, lookback: int = 50) -> None:
        self.lookback = max(1, lookback)
        self._rv_values: deque[float] = deque(maxlen=lookback)
        self._prev_close: float | None = None
        self._regime = "medium"

    @staticmethod
    def calculate(bars: list[Bar], lookback: int = 50) -> str:
        if len(bars) < lookback + 1:
            return "medium"

        recent_bars = bars[-lookback:]
        close_prices = [b.close for b in recent_bars]

        rv_values = []
        for i in range(1, len(close_prices)):
            prev_close = close_prices[i - 1]
            if prev_close == 0:
                continue
            ret = (close_prices[i] - prev_close) / prev_close
            rv_values.append(abs(ret))

        if not rv_values:
            return "medium"

        current_rv = np.mean(rv_values[-20:]) if len(rv_values) >= 20 else np.mean(rv_values)
        hist_rv = np.mean(rv_values)
        hist_std = np.std(rv_values) if len(rv_values) > 1 else 0.01

        low_threshold = hist_rv - hist_std
        high_threshold = hist_rv + hist_std

        if current_rv < low_threshold:
            return "low"
        if current_rv > high_threshold:
            return "high"
        return "medium"

    def update(self, bar: Bar) -> str:
        # Calculate return from previous close
        if self._prev_close is not None and self._prev_close != 0:
            ret = (bar.close - self._prev_close) / self._prev_close
            self._rv_values.append(abs(ret))

        self._prev_close = bar.close

        # Need enough data to classify (match calculate() warmup)
        if len(self._rv_values) < self.lookback:
            return "medium"

        # Calculate current and historical RV
        rv_array = np.array(self._rv_values)
        current_rv = np.mean(rv_array[-20:]) if len(rv_array) >= 20 else np.mean(rv_array)
        hist_rv = np.mean(rv_array)
        hist_std = np.std(rv_array) if len(rv_array) > 1 else 0.01

        low_threshold = hist_rv - hist_std
        high_threshold = hist_rv + hist_std

        if current_rv < low_threshold:
            self._regime = "low"
        elif current_rv > high_threshold:
            self._regime = "high"
        else:
            self._regime = "medium"

        return self._regime

    @property
    def value(self) -> str:
        return self._regime
