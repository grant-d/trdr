"""Tests for volatility regime indicator."""

from trdr.data import Bar
from trdr.indicators.volatility_regime import VolatilityRegimeIndicator


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
    """Tests for volatility regime calculation."""

    def test_returns_valid_regime(self) -> None:
        bars = make_bars([100, 102, 101, 103, 100] * 10)
        result = VolatilityRegimeIndicator.calculate(bars, lookback=50)
        assert result in ["low", "medium", "high"]

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 102])
        result = VolatilityRegimeIndicator.calculate(bars, lookback=50)
        assert result == "medium"


class TestVolatilityRegimeIndicator:
    """Tests for VolatilityRegimeIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = VolatilityRegimeIndicator(lookback=50)
        for bar in make_bars([100, 102, 101, 103, 100] * 10):
            value = ind.update(bar)
        assert value in ["low", "medium", "high"]
        assert ind.value in ["low", "medium", "high"]
