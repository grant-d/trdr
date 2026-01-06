"""Tests for HVN support strength indicator."""

from trdr.data import Bar
from trdr.indicators.hvn_support_strength import HvnSupportStrengthIndicator


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


class TestHvnSupportStrength:
    """Tests for HVN support strength calculation."""

    def test_strong_support(self) -> None:
        bars = [
            Bar(timestamp="2024-01-01", open=100, high=105, low=100, close=103, volume=1000)
            for _ in range(10)
        ]
        result = HvnSupportStrengthIndicator.calculate(bars, val_level=100, lookback=10)
        assert result > 0.5

    def test_no_touches(self) -> None:
        bars = make_bars([120, 121, 122, 123, 124] * 6)
        result = HvnSupportStrengthIndicator.calculate(bars, val_level=100, lookback=30)
        assert result == 0.0

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 102])
        result = HvnSupportStrengthIndicator.calculate(bars, val_level=100, lookback=30)
        assert result == 0.0


class TestHvnSupportStrengthIndicator:
    """Tests for HvnSupportStrengthIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = HvnSupportStrengthIndicator(val_level=100, lookback=10)
        for bar in make_bars([100, 101, 100, 102, 100, 103, 100, 104, 100, 105]):
            value = ind.update(bar)
        assert isinstance(value, float)
        assert isinstance(ind.value, float)
