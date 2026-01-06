"""Tests for ADX (Average Directional Index) indicator."""

from trdr.data import Bar
from trdr.indicators.adx import AdxIndicator


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


class TestAdx:
    """Tests for ADX calculation."""

    def test_trending_market(self) -> None:
        bars = make_bars(list(range(50, 80)))
        result = AdxIndicator.calculate(bars, period=14)
        assert result >= 0.0

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 101])
        result = AdxIndicator.calculate(bars, period=14)
        assert result == 0.0


class TestAdxIndicator:
    """Tests for AdxIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = AdxIndicator(period=14)
        for bar in make_bars(list(range(50, 80))):
            value = ind.update(bar)
        assert value >= 0.0
        assert ind.value >= 0.0
