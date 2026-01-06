"""Tests for EMA (Exponential Moving Average) indicator."""

from trdr.data import Bar
from trdr.indicators.ema import EmaIndicator


def make_bar(close: float, timestamp: str = "2024-01-01T00:00:00Z") -> Bar:
    """Create test bar with given close price."""
    return Bar(
        timestamp=timestamp,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1000,
    )


class TestEmaIndicator:
    """Tests for EMA indicator (stateful variant)."""

    def test_update_calculates_ema(self) -> None:
        """Update calculates EMA correctly."""
        ind = EmaIndicator(period=3)

        # First value is the close
        assert ind.update(make_bar(10.0)) == 10.0

        # Subsequent values use EMA formula
        val2 = ind.update(make_bar(20.0))
        assert val2 > 10.0 and val2 < 20.0

        val3 = ind.update(make_bar(30.0))
        assert val3 > val2 and val3 < 30.0

    def test_value_property(self) -> None:
        """Value property returns current EMA."""
        ind = EmaIndicator(period=3)

        ind.update(make_bar(10.0))
        ind.update(make_bar(20.0))
        ind.update(make_bar(30.0))

        assert ind.value > 0.0

    def test_period_one(self) -> None:
        """Period of 1 returns close price."""
        ind = EmaIndicator(period=1)

        assert ind.update(make_bar(123.45)) == 123.45
        assert ind.update(make_bar(456.78)) == 456.78

    def test_negative_period_clamped(self) -> None:
        """Negative period clamped to 1."""
        ind = EmaIndicator(period=-5)

        assert ind.period == 1
        assert ind.update(make_bar(100.0)) == 100.0

    def test_period_zero_clamped(self) -> None:
        """Period of 0 clamped to 1."""
        ind = EmaIndicator(period=0)

        assert ind.period == 1
        assert ind.update(make_bar(100.0)) == 100.0

    def test_very_large_period(self) -> None:
        """Very large period handled."""
        ind = EmaIndicator(period=1000000)

        result = ind.update(make_bar(100.0))
        assert result == 100.0

    def test_zero_price(self) -> None:
        """Zero prices handled."""
        ind = EmaIndicator(period=3)

        assert ind.update(make_bar(0.0)) == 0.0
        assert ind.update(make_bar(0.0)) == 0.0

    def test_ema_converges_to_price(self) -> None:
        """EMA converges toward consistent price."""
        ind = EmaIndicator(period=5)

        # Feed same price repeatedly
        for _ in range(20):
            val = ind.update(make_bar(100.0))

        # Should converge very close to 100.0
        assert abs(val - 100.0) < 0.01


class TestEmaCalculate:
    """Tests for EMA static calculate method (legacy variant)."""

    def test_calculate_ema(self) -> None:
        """Calculate returns correct EMA."""
        bars = [make_bar(10.0), make_bar(20.0), make_bar(30.0)]

        result = EmaIndicator.calculate(bars, period=3)

        assert result > 10.0 and result < 30.0

    def test_calculate_single_bar(self) -> None:
        """Calculate returns close for single bar."""
        bars = [make_bar(100.0)]

        result = EmaIndicator.calculate(bars, period=5)

        assert result == 100.0

    def test_calculate_empty_bars(self) -> None:
        """Calculate returns 0.0 for empty bars."""
        result = EmaIndicator.calculate([], period=3)

        assert result == 0.0

    def test_calculate_consistent_price(self) -> None:
        """Calculate with consistent price."""
        bars = [make_bar(50.0) for _ in range(20)]

        result = EmaIndicator.calculate(bars, period=5)

        # Should be very close to 50.0
        assert abs(result - 50.0) < 0.01

    def test_calculate_matches_stateful(self) -> None:
        """Calculate matches stateful update."""
        bars = [make_bar(10.0), make_bar(20.0), make_bar(30.0), make_bar(40.0)]

        static_result = EmaIndicator.calculate(bars, period=3)

        ind = EmaIndicator(period=3)
        for bar in bars:
            stateful_result = ind.update(bar)

        assert abs(static_result - stateful_result) < 0.0001
