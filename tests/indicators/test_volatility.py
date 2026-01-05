"""Tests for volatility indicators."""


from trdr.data import Bar
from trdr.indicators import volatility_regime


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


class TestVolatilityRegime:
    """Tests for volatility regime detection."""

    def test_returns_valid_regime(self):
        bars = make_bars([100, 102, 101, 103, 100] * 10)
        result = volatility_regime(bars, lookback=50)
        assert result in ["low", "medium", "high"]

    def test_insufficient_data(self):
        bars = make_bars([100, 102])
        result = volatility_regime(bars, lookback=50)
        assert result == "medium"

    def test_high_volatility(self):
        # Very volatile prices
        bars = make_bars([100, 120, 80, 130, 70, 140] * 10)
        result = volatility_regime(bars, lookback=50)
        assert result in ["medium", "high"]
