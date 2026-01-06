"""Tests for Kalman Filter indicator."""

from trdr.data import Bar
from trdr.indicators.kalman import KalmanIndicator, kalman_series


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


class TestKalman:
    """Tests for Kalman filter calculation."""

    def test_basic(self) -> None:
        bars = make_bars([100, 102, 104, 106, 108])
        result = KalmanIndicator.calculate(bars)
        assert result > 100

    def test_single_bar(self) -> None:
        bars = make_bars([100])
        result = KalmanIndicator.calculate(bars)
        assert result == 100.0

    def test_empty_bars(self) -> None:
        result = KalmanIndicator.calculate([])
        assert result == 0.0

    def test_series(self) -> None:
        bars = make_bars([100, 102, 104])
        result = kalman_series(bars)
        assert len(result) == 3
        assert result[0] == 100.0


class TestKalmanIndicator:
    """Tests for KalmanIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = KalmanIndicator()
        for bar in make_bars([100, 102, 104, 106]):
            value = ind.update(bar)
        assert isinstance(value, float)
        assert isinstance(ind.value, float)
