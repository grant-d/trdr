"""Tests for SMI (Stochastic Momentum Index) indicator."""

from trdr.data import Bar
from trdr.indicators.smi import SmiIndicator


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


class TestSmi:
    """Tests for SMI calculation."""

    def test_basic(self) -> None:
        bars = make_bars(list(range(50, 80)))
        result = SmiIndicator.calculate(bars, k=10, d=3)
        assert isinstance(result, float)

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 101])
        result = SmiIndicator.calculate(bars, k=10, d=3)
        assert result == 0.0


class TestSmiIndicator:
    """Tests for SmiIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = SmiIndicator(k=10, d=3)
        for bar in make_bars(list(range(50, 80))):
            value = ind.update(bar)
        assert isinstance(value, float)
        assert isinstance(ind.value, float)
