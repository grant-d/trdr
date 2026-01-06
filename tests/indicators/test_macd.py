"""Tests for MACD indicator."""

from trdr.data import Bar
from trdr.indicators.macd import MacdIndicator


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


class TestMacd:
    """Tests for MACD calculation."""

    def test_basic(self) -> None:
        bars = make_bars(list(range(50, 100)))
        macd_line, signal_line, histogram = MacdIndicator.calculate(bars, 12, 26, 9)
        assert macd_line != 0
        assert signal_line != 0
        assert isinstance(histogram, float)

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 102, 98])
        macd_line, signal_line, histogram = MacdIndicator.calculate(bars, 12, 26, 9)
        assert macd_line == signal_line == histogram == 0.0


class TestMacdIndicator:
    """Tests for MacdIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = MacdIndicator(fast=12, slow=26, signal=9)
        for bar in make_bars(list(range(50, 100))):
            macd_line, signal_line, _ = ind.update(bar)
        assert macd_line != 0
        assert signal_line != 0
        assert isinstance(ind.value, tuple)

    def test_period_zero_clamped(self) -> None:
        """Period of 0 clamped to 1."""
        ind = MacdIndicator(fast=0, slow=0, signal=0)

        assert ind.fast == 1
        assert ind.slow == 1
        assert ind.signal == 1

    def test_very_large_periods(self) -> None:
        """Very large periods handled."""
        ind = MacdIndicator(fast=1000000, slow=2000000, signal=1000000)

        macd, signal, hist = ind.update(make_bars([100.0])[0])
        assert macd == 0.0
        assert signal == 0.0
        assert hist == 0.0

    def test_zero_prices(self) -> None:
        """Zero prices handled."""
        ind = MacdIndicator(fast=12, slow=26, signal=9)

        for bar in make_bars([0.0] * 50):
            ind.update(bar)

        macd, signal, hist = ind.value
        assert macd == 0.0
        assert signal == 0.0
        assert hist == 0.0

    def test_same_price_no_signal(self) -> None:
        """Same price repeatedly gives no MACD signal."""
        ind = MacdIndicator(fast=12, slow=26, signal=9)

        for bar in make_bars([100.0] * 50):
            ind.update(bar)

        macd, signal, hist = ind.value
        assert abs(macd) < 0.01  # Should be near zero
        assert abs(signal) < 0.01
        assert abs(hist) < 0.01
