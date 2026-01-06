"""Tests for HMA (Hull Moving Average) indicator."""

from trdr.data import Bar
from trdr.indicators.hma import HmaIndicator, HmaSlopeIndicator


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


class TestHma:
    """Tests for HMA."""

    def test_basic(self) -> None:
        bars = make_bars([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        result = HmaIndicator.calculate(bars, 9)
        assert result > 0

    def test_insufficient_data(self) -> None:
        bars = make_bars([10, 20, 30])
        result = HmaIndicator.calculate(bars, 9)
        assert result == 0.0

    def test_period_1(self) -> None:
        bars = make_bars([10, 20, 30])
        result = HmaIndicator.calculate(bars, 1)
        assert result == 30


class TestHmaSlope:
    """Tests for HMA slope."""

    def test_uptrend_positive_slope(self) -> None:
        bars = make_bars(list(range(10, 100, 5)))
        result = HmaSlopeIndicator.calculate(bars, 9, 3)
        assert result > 0

    def test_downtrend_negative_slope(self) -> None:
        bars = make_bars(list(range(100, 10, -5)))
        result = HmaSlopeIndicator.calculate(bars, 9, 3)
        assert result < 0


class TestHmaIndicator:
    """Tests for HmaIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = HmaIndicator(period=9)
        for bar in make_bars([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
            value = ind.update(bar)
        assert value > 0
        assert ind.value > 0


class TestHmaSlopeIndicator:
    """Tests for HmaSlopeIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = HmaSlopeIndicator(period=9, lookback=3)
        for bar in make_bars(list(range(10, 100, 5))):
            value = ind.update(bar)
        assert value > 0
        assert isinstance(ind.value, float)
