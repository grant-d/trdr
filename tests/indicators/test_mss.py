"""Tests for Market Structure Score indicator."""

from trdr.data import Bar
from trdr.indicators.mss import MssIndicator


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


class TestMss:
    """Tests for MSS calculation."""

    def test_bullish_trend(self) -> None:
        bars = make_bars(list(range(50, 80)))
        result = MssIndicator.calculate(bars, 20)
        assert result > 0

    def test_bearish_trend(self) -> None:
        bars = make_bars(list(range(80, 50, -1)))
        result = MssIndicator.calculate(bars, 20)
        assert result < 0

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 102])
        result = MssIndicator.calculate(bars, 20)
        assert result == 0.0


class TestMssIndicator:
    """Tests for MssIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = MssIndicator(lookback=20)
        for bar in make_bars(list(range(50, 80))):
            value = ind.update(bar)
        assert isinstance(value, float)
        assert isinstance(ind.value, float)
