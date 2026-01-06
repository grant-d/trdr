"""Tests for Laguerre RSI indicator."""

from trdr.data import Bar
from trdr.indicators.laguerre_rsi import LaguerreRsiIndicator


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


class TestLaguerreRsi:
    """Tests for Laguerre RSI calculation."""

    def test_uptrend_bias(self) -> None:
        bars = make_bars(list(range(50, 80)))
        result = LaguerreRsiIndicator.calculate(bars, alpha=0.2)
        assert result > 50.0

    def test_insufficient_data(self) -> None:
        result = LaguerreRsiIndicator.calculate([])
        assert result == 50.0


class TestLaguerreRsiIndicator:
    """Tests for LaguerreRsiIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = LaguerreRsiIndicator(alpha=0.2)
        for bar in make_bars(list(range(50, 80))):
            value = ind.update(bar)
        assert 0.0 <= value <= 100.0
        assert 0.0 <= ind.value <= 100.0
