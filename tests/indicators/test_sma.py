"""Tests for SMA (Simple Moving Average) indicator."""

from trdr.data import Bar
from trdr.indicators.sma import SmaIndicator, SmaSeriesIndicator


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


class TestSmaIndicator:
    """Tests for SMA indicator (stateful variant)."""

    def test_update_calculates_sma(self) -> None:
        """Update calculates SMA correctly."""
        ind = SmaIndicator(period=3)

        assert ind.update(make_bar(10.0)) == 10.0
        assert ind.update(make_bar(20.0)) == 20.0
        assert ind.update(make_bar(30.0)) == 20.0
        assert ind.update(make_bar(40.0)) == 30.0

    def test_value_property(self) -> None:
        """Value property returns current SMA."""
        ind = SmaIndicator(period=3)

        ind.update(make_bar(10.0))
        ind.update(make_bar(20.0))
        ind.update(make_bar(30.0))

        assert ind.value == 20.0

    def test_insufficient_data(self) -> None:
        """Returns close when insufficient data."""
        ind = SmaIndicator(period=5)

        assert ind.update(make_bar(100.0)) == 100.0
        assert ind.update(make_bar(110.0)) == 110.0

    def test_period_one(self) -> None:
        """Period of 1 returns close price."""
        ind = SmaIndicator(period=1)

        assert ind.update(make_bar(123.45)) == 123.45

    def test_negative_period_clamped(self) -> None:
        """Negative period clamped to 1."""
        ind = SmaIndicator(period=-5)

        assert ind.period == 1
        assert ind.update(make_bar(100.0)) == 100.0

    def test_period_zero_clamped(self) -> None:
        """Period of 0 clamped to 1."""
        ind = SmaIndicator(period=0)

        assert ind.period == 1
        assert ind.update(make_bar(100.0)) == 100.0

    def test_very_large_period(self) -> None:
        """Very large period handled."""
        ind = SmaIndicator(period=1000000)

        assert ind.update(make_bar(100.0)) == 100.0
        assert ind.update(make_bar(200.0)) == 200.0

    def test_zero_price(self) -> None:
        """Zero prices handled."""
        ind = SmaIndicator(period=3)

        assert ind.update(make_bar(0.0)) == 0.0
        assert ind.update(make_bar(0.0)) == 0.0
        assert ind.update(make_bar(0.0)) == 0.0

    def test_negative_price(self) -> None:
        """Negative prices handled."""
        ind = SmaIndicator(period=2)

        assert ind.update(make_bar(-10.0)) == -10.0
        assert ind.update(make_bar(-20.0)) == -15.0


class TestSmaCalculate:
    """Tests for SMA static calculate method (legacy variant)."""

    def test_calculate_sma(self) -> None:
        """Calculate returns correct SMA."""
        bars = [make_bar(10.0), make_bar(20.0), make_bar(30.0)]

        result = SmaIndicator.calculate(bars, period=3)

        assert result == 20.0

    def test_calculate_sliding_window(self) -> None:
        """Calculate uses sliding window."""
        bars = [
            make_bar(10.0),
            make_bar(20.0),
            make_bar(30.0),
            make_bar(40.0),
        ]

        result = SmaIndicator.calculate(bars, period=3)

        assert result == 30.0

    def test_calculate_insufficient_data(self) -> None:
        """Calculate returns close when insufficient data."""
        bars = [make_bar(100.0)]

        result = SmaIndicator.calculate(bars, period=5)

        assert result == 100.0

    def test_calculate_empty_bars(self) -> None:
        """Calculate returns 0.0 for empty bars."""
        result = SmaIndicator.calculate([], period=3)

        assert result == 0.0


class TestSmaSeriesIndicator:
    """Tests for SMA series indicator (stateful variant)."""

    def test_update_returns_series(self) -> None:
        """Update returns full SMA series."""
        ind = SmaSeriesIndicator(period=2)

        series1 = ind.update(make_bar(10.0))
        assert series1 == [10.0]

        series2 = ind.update(make_bar(20.0))
        assert series2 == [10.0, 15.0]

        series3 = ind.update(make_bar(30.0))
        assert series3 == [10.0, 15.0, 25.0]

    def test_value_property(self) -> None:
        """Value property returns current series."""
        ind = SmaSeriesIndicator(period=2)

        ind.update(make_bar(10.0))
        ind.update(make_bar(20.0))

        assert ind.value == [10.0, 15.0]


class TestSmaSeriesCalculate:
    """Tests for SMA series static calculate method (legacy variant)."""

    def test_calculate_series(self) -> None:
        """Calculate returns SMA series."""
        values = [10.0, 20.0, 30.0, 40.0]

        result = SmaSeriesIndicator.calculate(values, period=2)

        assert result == [10.0, 15.0, 25.0, 35.0]

    def test_calculate_empty_values(self) -> None:
        """Calculate returns empty list for empty values."""
        result = SmaSeriesIndicator.calculate([], period=2)

        assert result == []

    def test_calculate_insufficient_period(self) -> None:
        """Calculate returns input values when period too large."""
        values = [10.0, 20.0]

        result = SmaSeriesIndicator.calculate(values, period=5)

        assert result == [10.0, 20.0]
