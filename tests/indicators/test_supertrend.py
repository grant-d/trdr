"""Tests for SuperTrend indicator."""

from trdr.data import Bar
from trdr.indicators.supertrend import SupertrendIndicator


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


class TestSupertrend:
    """Tests for SuperTrend calculation."""

    def test_basic(self) -> None:
        bars = make_bars(list(range(100, 140)))
        st_value, direction = SupertrendIndicator.calculate(bars, period=10, multiplier=3.0)
        assert isinstance(st_value, float)
        assert direction in (1, -1)

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 101])
        st_value, direction = SupertrendIndicator.calculate(bars, period=10, multiplier=3.0)
        assert direction == 1
        assert st_value == bars[-1].close


class TestSupertrendIndicator:
    """Tests for SupertrendIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = SupertrendIndicator(period=10, multiplier=3.0)
        for bar in make_bars(list(range(100, 140))):
            st_value, direction = ind.update(bar)
        assert isinstance(st_value, float)
        assert direction in (1, -1)
        assert isinstance(ind.value, tuple)
