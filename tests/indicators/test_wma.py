"""Tests for WMA (Weighted Moving Average) indicator."""

from trdr.data import Bar
from trdr.indicators.wma import WmaIndicator


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


class TestWma:
    """Tests for WMA calculation."""

    def test_basic(self) -> None:
        bars = make_bars([10, 20, 30])
        result = WmaIndicator.calculate(bars, 3)
        assert abs(result - 23.33) < 0.1

    def test_period_1(self) -> None:
        bars = make_bars([10, 20, 30])
        assert WmaIndicator.calculate(bars, 1) == 30.0

    def test_insufficient_data(self) -> None:
        bars = make_bars([10, 20])
        result = WmaIndicator.calculate(bars, 5)
        assert result == 20.0


class TestWmaIndicator:
    """Tests for WmaIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = WmaIndicator(period=3)
        for bar in make_bars([10, 20, 30]):
            value = ind.update(bar)
        assert abs(value - 23.33) < 0.1
        assert abs(ind.value - 23.33) < 0.1

    def test_period_zero_clamped(self) -> None:
        """Period of 0 clamped to 1."""
        ind = WmaIndicator(period=0)
        result = ind.update(make_bars([100.0])[0])
        assert result == 100.0

    def test_negative_period_clamped(self) -> None:
        """Negative period clamped to 1."""
        ind = WmaIndicator(period=-5)
        result = ind.update(make_bars([100.0])[0])
        assert result == 100.0

    def test_very_large_period(self) -> None:
        """Very large period handled."""
        ind = WmaIndicator(period=1000000)
        result = ind.update(make_bars([100.0])[0])
        assert result == 100.0

    def test_zero_price(self) -> None:
        """Zero prices handled."""
        ind = WmaIndicator(period=3)
        for bar in make_bars([0.0, 0.0, 0.0]):
            result = ind.update(bar)
        assert result == 0.0
