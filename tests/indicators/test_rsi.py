"""Tests for RSI (Relative Strength Index) indicator."""

from trdr.data import Bar
from trdr.indicators.rsi import RsiIndicator


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


class TestRsiIndicator:
    """Tests for RSI indicator (stateful variant)."""

    def test_update_calculates_rsi(self) -> None:
        """Update calculates RSI correctly."""
        ind = RsiIndicator(period=14)

        # RSI should be 50.0 initially (neutral)
        assert ind.update(make_bar(100.0)) == 50.0

        # Add upward movement
        for i in range(1, 15):
            ind.update(make_bar(100.0 + i))

        # RSI should be high after consistent gains
        assert ind.value > 70.0

    def test_value_property(self) -> None:
        """Value property returns current RSI."""
        ind = RsiIndicator(period=14)

        for i in range(20):
            ind.update(make_bar(100.0 + i))

        rsi = ind.value
        assert 0.0 <= rsi <= 100.0

    def test_rsi_downtrend(self) -> None:
        """RSI decreases in downtrend."""
        ind = RsiIndicator(period=14)

        # Initialize
        ind.update(make_bar(100.0))

        # Add downward movement
        for i in range(1, 15):
            ind.update(make_bar(100.0 - i))

        # RSI should be low after consistent losses
        assert ind.value < 30.0

    def test_rsi_neutral_movement(self) -> None:
        """RSI stays neutral with alternating movement."""
        ind = RsiIndicator(period=14)

        # Alternate up and down
        for i in range(30):
            price = 100.0 + (i % 2)
            ind.update(make_bar(price))

        # Should be near neutral
        assert 40.0 < ind.value < 60.0

    def test_period_bounds(self) -> None:
        """Period is properly bounded."""
        ind = RsiIndicator(period=-5)
        assert ind.period == 1

        ind2 = RsiIndicator(period=100)
        assert ind2.period == 100

    def test_period_zero_clamped(self) -> None:
        """Period of 0 clamped to 1."""
        ind = RsiIndicator(period=0)
        assert ind.period == 1

    def test_very_large_period(self) -> None:
        """Very large period handled."""
        ind = RsiIndicator(period=1000000)

        result = ind.update(make_bar(100.0))
        assert result == 50.0  # Default RSI when insufficient data

    def test_zero_price(self) -> None:
        """Zero prices handled."""
        ind = RsiIndicator(period=14)

        for _ in range(20):
            ind.update(make_bar(0.0))

        assert ind.value == 50.0  # No change = neutral RSI

    def test_all_same_price(self) -> None:
        """Same price repeatedly gives neutral RSI."""
        ind = RsiIndicator(period=14)

        for _ in range(20):
            ind.update(make_bar(100.0))

        assert ind.value == 50.0  # No gain/loss = neutral

    def test_slight_upward_bias(self) -> None:
        """Slight upward movement gives RSI slightly above 50."""
        ind = RsiIndicator(period=14)

        # Alternate with slight upward bias: +1, -0.8, +1, -0.8...
        price = 100.0
        for i in range(30):
            if i % 2 == 0:
                price += 1.0
            else:
                price -= 0.8
            ind.update(make_bar(price))

        # Should be above 50 due to upward bias
        assert 50.0 < ind.value < 60.0

    def test_slight_downward_bias(self) -> None:
        """Slight downward movement gives RSI slightly below 50."""
        ind = RsiIndicator(period=14)

        # Alternate with slight downward bias: +0.8, -1, +0.8, -1...
        price = 100.0
        for i in range(30):
            if i % 2 == 0:
                price += 0.8
            else:
                price -= 1.0
            ind.update(make_bar(price))

        # Should be below 50 due to downward bias
        assert 40.0 < ind.value < 50.0


class TestRsiCalculate:
    """Tests for RSI static calculate method (legacy variant)."""

    def test_calculate_rsi(self) -> None:
        """Calculate returns correct RSI."""
        bars = [make_bar(100.0 + i) for i in range(20)]

        result = RsiIndicator.calculate(bars, period=14)

        assert result > 50.0
        assert result <= 100.0

    def test_calculate_insufficient_data(self) -> None:
        """Calculate returns 50.0 for insufficient data."""
        bars = [make_bar(100.0)]

        result = RsiIndicator.calculate(bars, period=14)

        assert result == 50.0

    def test_calculate_empty_bars(self) -> None:
        """Calculate returns 50.0 for empty bars."""
        result = RsiIndicator.calculate([], period=14)

        assert result == 50.0

    def test_calculate_all_gains(self) -> None:
        """Calculate with all gains approaches 100."""
        bars = [make_bar(100.0 + i * 2) for i in range(30)]

        result = RsiIndicator.calculate(bars, period=14)

        assert result > 80.0

    def test_calculate_all_losses(self) -> None:
        """Calculate with all losses approaches 0."""
        bars = [make_bar(100.0 - i * 2) for i in range(30)]

        result = RsiIndicator.calculate(bars, period=14)

        assert result < 20.0

    def test_calculate_matches_stateful(self) -> None:
        """Calculate matches stateful update."""
        bars = [make_bar(100.0 + i * 0.5) for i in range(25)]

        static_result = RsiIndicator.calculate(bars, period=14)

        ind = RsiIndicator(period=14)
        for bar in bars:
            stateful_result = ind.update(bar)

        assert abs(static_result - stateful_result) < 0.01
