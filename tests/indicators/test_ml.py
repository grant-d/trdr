"""Tests for ML indicators."""


from trdr.data import Bar
from trdr.indicators import kalman, kalman_series


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
    """Tests for Kalman Filter."""

    def test_basic(self):
        bars = make_bars([100, 102, 104, 106, 108])
        result = kalman(bars)
        assert result > 100  # Should track price trend

    def test_single_bar(self):
        bars = make_bars([100])
        result = kalman(bars)
        assert result == 100.0

    def test_empty_bars(self):
        result = kalman([])
        assert result == 0.0

    def test_smoothing(self):
        # Noisy data should be smoothed
        bars = make_bars([100, 105, 95, 110, 90, 115])
        result = kalman(bars, measurement_noise=1.0)
        # Result should be less volatile than raw price
        assert 90 < result < 115


class TestKalmanSeries:
    """Tests for Kalman Filter series."""

    def test_returns_list(self):
        bars = make_bars([100, 102, 104, 106, 108])
        result = kalman_series(bars)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_empty_bars(self):
        result = kalman_series([])
        assert result == []

    def test_first_value_preserved(self):
        bars = make_bars([100, 102, 104])
        result = kalman_series(bars)
        assert result[0] == 100.0
