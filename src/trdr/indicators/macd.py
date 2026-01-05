"""MACD (Moving Average Convergence Divergence) indicator."""

from ..data import Bar
from .ema import ema_series


def macd(
    bars: list[Bar], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float, float, float]:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        bars: List of OHLCV bars
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    return MacdIndicator.calculate(bars, fast=fast, slow=slow, signal=signal)


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


class MacdIndicator:
    """Streaming MACD calculator."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = max(1, fast)
        self.slow = max(1, slow)
        self.signal = max(1, signal)
        self._fast_ema = _EmaValue(self.fast)
        self._slow_ema = _EmaValue(self.slow)
        self._signal_ema = _EmaValue(self.signal)
        self._value = (0.0, 0.0, 0.0)
        self._count = 0

    @staticmethod
    def calculate(
        bars: list[Bar],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[float, float, float]:
        min_bars = slow + signal + 1
        if len(bars) < min_bars:
            return (0.0, 0.0, 0.0)

        closes = [b.close for b in bars]

        fast_ema = ema_series(closes, fast)
        slow_ema = ema_series(closes, slow)

        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
        signal_line = ema_series(macd_line, signal)

        macd_current = macd_line[-1]
        signal_current = signal_line[-1]
        histogram = macd_current - signal_current

        return (macd_current, signal_current, histogram)

    def update(self, bar: Bar) -> tuple[float, float, float]:
        close = bar.close
        self._count += 1
        fast_val = self._fast_ema.update(close)
        slow_val = self._slow_ema.update(close)
        macd_line = fast_val - slow_val
        signal_val = self._signal_ema.update(macd_line)
        histogram = macd_line - signal_val
        if self._count < self.slow + self.signal + 1:
            self._value = (0.0, 0.0, 0.0)
        else:
            self._value = (float(macd_line), float(signal_val), float(histogram))
        return self._value

    @property
    def value(self) -> tuple[float, float, float]:
        return self._value
