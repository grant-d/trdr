"""Tests for volume trend indicator."""

from trdr.data import Bar
from trdr.indicators.volume_trend import VolumeTrendIndicator


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


class TestVolumeTrend:
    """Tests for volume trend calculation."""

    def test_increasing(self) -> None:
        bars = [
            Bar(
                timestamp="2024-01-01",
                open=100,
                high=101,
                low=99,
                close=100,
                volume=v,
            )
            for v in [100, 110, 120, 130, 140, 200, 250, 300, 350, 400]
        ]
        result = VolumeTrendIndicator.calculate(bars, 5)
        assert result == "increasing"

    def test_declining(self) -> None:
        bars = [
            Bar(
                timestamp="2024-01-01",
                open=100,
                high=101,
                low=99,
                close=100,
                volume=v,
            )
            for v in [400, 350, 300, 250, 200, 100, 90, 80, 70, 60]
        ]
        result = VolumeTrendIndicator.calculate(bars, 5)
        assert result == "declining"

    def test_neutral_on_insufficient(self) -> None:
        bars = make_bars([100, 102])
        result = VolumeTrendIndicator.calculate(bars, 5)
        assert result == "neutral"


class TestVolumeTrendIndicator:
    """Tests for VolumeTrendIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = VolumeTrendIndicator(lookback=5)
        for bar in make_bars([100, 102, 101, 103, 100] * 3):
            value = ind.update(bar)
        assert value in {"increasing", "declining", "neutral"}
        assert ind.value in {"increasing", "declining", "neutral"}
