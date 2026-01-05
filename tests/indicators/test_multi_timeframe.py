"""Tests for multi-timeframe indicators."""


from trdr.data import Bar
from trdr.indicators import hvn_support_strength, multi_timeframe_poc


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


class TestMultiTimeframePOC:
    """Tests for multi-timeframe Point of Control."""

    def test_returns_three_values(self):
        bars = make_bars([100, 102, 101, 103, 100] * 5)
        poc1, poc2, poc3 = multi_timeframe_poc(bars)
        assert isinstance(poc1, float)
        assert isinstance(poc2, float)
        assert isinstance(poc3, float)

    def test_insufficient_data(self):
        bars = make_bars([100, 102])
        poc1, poc2, poc3 = multi_timeframe_poc(bars)
        # All should return same value when insufficient data
        assert poc1 == poc2 == poc3


class TestHVNSupportStrength:
    """Tests for HVN support strength."""

    def test_strong_support(self):
        # Price repeatedly bounces from 100
        bars = [
            Bar(timestamp="2024-01-01", open=100, high=105, low=100, close=103, volume=1000)
            for _ in range(10)
        ]
        result = hvn_support_strength(bars, val_level=100, lookback=10)
        assert result > 0.5  # Strong support

    def test_no_touches(self):
        bars = make_bars([120, 121, 122, 123, 124] * 6)
        result = hvn_support_strength(bars, val_level=100, lookback=30)
        assert result == 0.0  # No support at that level

    def test_insufficient_data(self):
        bars = make_bars([100, 102])
        result = hvn_support_strength(bars, val_level=100, lookback=30)
        assert result == 0.0
