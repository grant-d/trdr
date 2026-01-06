"""Tests for multi-timeframe POC indicator."""

from trdr.data import Bar
from trdr.indicators.multi_timeframe_poc import MultiTimeframePocIndicator


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


class TestMultiTimeframePoc:
    """Tests for multi-timeframe PoC calculation."""

    def test_returns_three_values(self) -> None:
        bars = make_bars([100, 102, 101, 103, 100] * 5)
        poc1, poc2, poc3 = MultiTimeframePocIndicator.calculate(bars)
        assert isinstance(poc1, float)
        assert isinstance(poc2, float)
        assert isinstance(poc3, float)

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 102])
        poc1, poc2, poc3 = MultiTimeframePocIndicator.calculate(bars)
        assert poc1 == poc2 == poc3


class TestMultiTimeframePocIndicator:
    """Tests for MultiTimeframePocIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = MultiTimeframePocIndicator()
        for bar in make_bars([100, 102, 101, 103, 100] * 5):
            value = ind.update(bar)
        assert isinstance(value, tuple)
        assert isinstance(ind.value, tuple)
