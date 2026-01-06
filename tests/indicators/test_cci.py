"""Tests for CCI (Commodity Channel Index) indicator."""

from trdr.data import Bar
from trdr.indicators.cci import CciIndicator


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


class TestCci:
    """Tests for CCI calculation."""

    def test_flat_market(self) -> None:
        bars = make_bars([100] * 30)
        result = CciIndicator.calculate(bars, period=20)
        assert result == 0.0

    def test_trending_market(self) -> None:
        bars = make_bars(list(range(80, 120)))
        result = CciIndicator.calculate(bars, period=20)
        assert result != 0.0

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 101])
        result = CciIndicator.calculate(bars, period=20)
        assert result == 0.0


class TestCciIndicator:
    """Tests for CciIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = CciIndicator(period=20)
        for bar in make_bars(list(range(80, 120))):
            value = ind.update(bar)
        assert isinstance(value, float)
        assert isinstance(ind.value, float)
