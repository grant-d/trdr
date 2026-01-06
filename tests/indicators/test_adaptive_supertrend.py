"""Tests for adaptive SuperTrend indicator."""

from trdr.data import Bar
from trdr.indicators.adaptive_supertrend import AdaptiveSupertrendIndicator


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


class TestAdaptiveSupertrend:
    """Tests for adaptive SuperTrend calculation."""

    def test_basic(self) -> None:
        bars = make_bars([100 + i * 0.5 for i in range(120)])
        st_value, direction, level, clusters = AdaptiveSupertrendIndicator.calculate(
            bars,
            atr_period=10,
            training_period=100,
        )
        assert isinstance(st_value, float)
        assert direction in (1, -1)
        assert level in {"high", "medium", "low"}
        assert len(clusters) == 3

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 101, 102])
        st_value, direction, level, clusters = AdaptiveSupertrendIndicator.calculate(
            bars, training_period=100
        )
        assert direction == 1
        assert clusters == []


class TestAdaptiveSupertrendIndicator:
    """Tests for AdaptiveSupertrendIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = AdaptiveSupertrendIndicator(atr_period=10, training_period=100)
        for bar in make_bars([100 + i * 0.5 for i in range(120)]):
            st_value, direction, level, clusters = ind.update(bar)
        assert isinstance(st_value, float)
        assert direction in (1, -1)
        assert level in {"high", "medium", "low"}
        assert isinstance(ind.value, tuple)
