"""Tests for volume indicators."""

import pytest

from trdr.data import Bar
from trdr.indicators import VolumeProfile, order_flow_imbalance, volume_profile, volume_trend


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


class TestVolumeProfile:
    """Tests for Volume Profile."""

    def test_basic(self):
        bars = make_bars([100, 102, 101, 103, 100])
        result = volume_profile(bars)
        assert isinstance(result, VolumeProfile)
        assert result.poc > 0
        assert result.vah >= result.val

    def test_flat_market(self):
        # Note: make_bars adds 1% high/low variance, so not truly flat
        bars = make_bars([100] * 10)
        result = volume_profile(bars)
        # PoC should be near 100 (within the high/low range)
        assert 99 < result.poc < 101
        assert result.vah >= result.val

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            volume_profile([])


class TestVolumeTrend:
    """Tests for volume trend detection."""

    def test_increasing(self):
        # Create bars with increasing volume
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
        result = volume_trend(bars, 5)
        assert result == "increasing"

    def test_declining(self):
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
        result = volume_trend(bars, 5)
        assert result == "declining"

    def test_neutral_on_insufficient(self):
        bars = make_bars([100, 102])
        result = volume_trend(bars, 5)
        assert result == "neutral"


class TestOrderFlowImbalance:
    """Tests for order flow imbalance."""

    def test_buying_pressure(self):
        # Steadily rising prices = buying pressure
        bars = make_bars(list(range(100, 110)))
        result = order_flow_imbalance(bars, lookback=5)
        assert result > 0  # Positive = buying

    def test_selling_pressure(self):
        # Falling prices = selling pressure
        bars = make_bars(list(range(110, 100, -1)))
        result = order_flow_imbalance(bars, lookback=5)
        assert result < 0  # Negative = selling

    def test_insufficient_data(self):
        bars = make_bars([100])
        result = order_flow_imbalance(bars, lookback=5)
        assert result == 0.0

    def test_range_bound(self):
        result = order_flow_imbalance(make_bars([100, 102, 101, 100, 102]), lookback=5)
        assert -1.0 <= result <= 1.0
