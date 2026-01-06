"""Tests for Bollinger Bands indicator."""

from trdr.data import Bar
from trdr.indicators.bollinger_bands import BollingerBandsIndicator


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


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_basic(self) -> None:
        bars = make_bars([100] * 20)
        upper, middle, lower = BollingerBandsIndicator.calculate(bars, 20)
        assert upper == middle == lower == 100

    def test_with_variance(self) -> None:
        bars = make_bars([90, 95, 100, 105, 110] * 4)
        upper, middle, lower = BollingerBandsIndicator.calculate(bars, 20)
        assert upper > middle > lower

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 102])
        upper, middle, lower = BollingerBandsIndicator.calculate(bars, 20)
        assert upper == middle == lower == 102


class TestBollingerBandsIndicator:
    """Tests for BollingerBandsIndicator (stateful variant)."""

    def test_update_flat_prices(self) -> None:
        ind = BollingerBandsIndicator(period=20, std_mult=2.0)
        for bar in make_bars([100] * 20):
            upper, middle, lower = ind.update(bar)
        assert upper == middle == lower == 100

    def test_update_with_variance(self) -> None:
        ind = BollingerBandsIndicator(period=20, std_mult=2.0)
        for bar in make_bars([90, 95, 100, 105, 110] * 4):
            upper, middle, lower = ind.update(bar)
        assert upper > middle > lower

    def test_period_zero_clamped(self) -> None:
        """Period of 0 clamped to 1."""
        ind = BollingerBandsIndicator(period=0, std_mult=2.0)
        assert ind.period == 1
        upper, middle, lower = ind.update(make_bars([100.0])[0])
        assert upper == middle == lower == 100.0

    def test_negative_period_clamped(self) -> None:
        """Negative period clamped to 1."""
        ind = BollingerBandsIndicator(period=-5, std_mult=2.0)
        assert ind.period == 1

    def test_very_large_period(self) -> None:
        """Very large period handled."""
        ind = BollingerBandsIndicator(period=1000000, std_mult=2.0)
        upper, middle, lower = ind.update(make_bars([100.0])[0])
        assert upper == middle == lower == 100.0

    def test_zero_prices(self) -> None:
        """Zero prices handled."""
        ind = BollingerBandsIndicator(period=3, std_mult=2.0)
        for bar in make_bars([0.0, 0.0, 0.0]):
            upper, middle, lower = ind.update(bar)
        assert upper == middle == lower == 0.0
