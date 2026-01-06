"""Tests for RVI (Relative Volatility Index) indicator."""

import math

from trdr.data import Bar
from trdr.indicators.rvi import RviIndicator


def make_bars(closes: list[float], volume: float = 1000.0) -> list[Bar]:
    """Create bars from close prices."""
    return [
        Bar(
            timestamp="2024-01-01T00:00:00Z",
            open=c,
            high=c * 1.01,
            low=c * 0.99,
            close=c,
            volume=volume,
        )
        for c in closes
    ]


class TestRvi:
    """Tests for RVI calculation."""

    def test_rvi_ema_matches_streaming(self) -> None:
        bars = make_bars(list(range(100, 130)))
        calc = RviIndicator(10, mode="ema")
        for bar in bars:
            calc.update(bar)
        expected = RviIndicator.calculate(bars, period=10, mode="ema")
        assert math.isclose(calc.value, expected, rel_tol=1e-6)

    def test_rvi_ema_uptrend_bias(self) -> None:
        bars = make_bars(list(range(1, 80)))
        value = RviIndicator.calculate(bars, period=10, mode="ema")
        assert value > 60.0


class TestRviIndicator:
    """Tests for RviIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = RviIndicator(period=10, mode="ema")
        for bar in make_bars(list(range(100, 130))):
            value = ind.update(bar)
        assert isinstance(value, float)
        assert isinstance(ind.value, float)
