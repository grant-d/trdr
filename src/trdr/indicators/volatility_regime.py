"""Volatility regime classification using realized volatility."""

import numpy as np

from ..data import Bar


def _true_ranges(bars: list[Bar]) -> list[float]:
    """Compute true range series."""
    trs = [0.0] * len(bars)
    for i in range(1, len(bars)):
        high = bars[i].high
        low = bars[i].low
        prev_close = bars[i - 1].close
        trs[i] = max(high - low, abs(high - prev_close), abs(low - prev_close))
    return trs


def _atr_series(bars: list[Bar], period: int) -> list[float]:
    """Compute Wilder-smoothed ATR series."""
    if len(bars) < period + 1 or period <= 0:
        return [0.0] * len(bars)

    tr = _true_ranges(bars)
    series = [0.0] * len(bars)
    atr_val = float(np.mean(tr[1 : period + 1]))
    series[period] = atr_val

    for i in range(period + 1, len(bars)):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        series[i] = atr_val

    return series


def _sma_series(values: list[float], period: int) -> list[float]:
    """Compute simple moving average series."""
    if period <= 0 or len(values) < period:
        return [0.0] * len(values)
    series = [0.0] * len(values)
    window_sum = float(np.sum(values[:period]))
    series[period - 1] = window_sum / period
    for i in range(period, len(values)):
        window_sum += values[i] - values[i - period]
        series[i] = window_sum / period
    return series


class VolatilityRegimeIndicator:
    """Streaming volatility regime classifier."""

    def __init__(self, lookback: int = 50) -> None:
        self.lookback = max(1, lookback)
        self._bars: list[Bar] = []

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
        self._bars.append(bar)
        return self.calculate(self._bars, lookback=self.lookback)

    @property
    def value(self) -> str:
        if not self._bars:
            return "medium"
        return self.calculate(self._bars, lookback=self.lookback)
