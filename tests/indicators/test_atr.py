"""Tests for ATR (Average True Range) indicator."""

from trdr.data import Bar
from trdr.indicators.atr import AtrIndicator


def make_bar(
    close: float,
    high: float | None = None,
    low: float | None = None,
    timestamp: str = "2024-01-01T00:00:00Z",
) -> Bar:
    """Create test bar with given prices."""
    if high is None:
        high = close + 1
    if low is None:
        low = close - 1

    return Bar(
        timestamp=timestamp,
        open=close,
        high=high,
        low=low,
        close=close,
        volume=1000,
    )


class TestAtrIndicator:
    """Tests for ATR indicator (stateful variant)."""

    def test_update_calculates_atr(self) -> None:
        """Update calculates ATR correctly."""
        ind = AtrIndicator(period=14)

        # First bar
        val1 = ind.update(make_bar(close=100.0, high=102.0, low=98.0))
        assert val1 > 0.0

        # Subsequent bars
        val2 = ind.update(make_bar(close=101.0, high=103.0, low=99.0))
        assert val2 > 0.0

    def test_value_property(self) -> None:
        """Value property returns current ATR."""
        ind = AtrIndicator(period=14)

        for i in range(20):
            ind.update(make_bar(close=100.0 + i, high=102.0 + i, low=98.0 + i))

        assert ind.value > 0.0

    def test_increasing_volatility(self) -> None:
        """ATR increases with higher volatility."""
        ind = AtrIndicator(period=5)

        # Low volatility
        for _ in range(10):
            ind.update(make_bar(close=100.0, high=100.5, low=99.5))

        low_vol_atr = ind.value

        # High volatility
        for _ in range(10):
            ind.update(make_bar(close=100.0, high=105.0, low=95.0))

        high_vol_atr = ind.value

        assert high_vol_atr > low_vol_atr

    def test_gap_movement(self) -> None:
        """ATR captures gap movements."""
        ind = AtrIndicator(period=5)

        ind.update(make_bar(close=100.0, high=101.0, low=99.0))

        # Gap up
        gap_bar = make_bar(close=110.0, high=111.0, low=109.0)
        atr_after_gap = ind.update(gap_bar)

        # ATR should increase due to gap
        assert atr_after_gap > 2.0

    def test_period_bounds(self) -> None:
        """Period is properly bounded."""
        ind = AtrIndicator(period=-5)
        assert ind.period == 1

        ind2 = AtrIndicator(period=100)
        assert ind2.period == 100

    def test_period_zero_clamped(self) -> None:
        """Period of 0 clamped to 1."""
        ind = AtrIndicator(period=0)
        assert ind.period == 1

    def test_very_large_period(self) -> None:
        """Very large period handled."""
        ind = AtrIndicator(period=1000000)

        bar = make_bar(100.0, 102.0, 98.0)
        result = ind.update(bar)
        assert result >= 0.0

    def test_zero_range_bars(self) -> None:
        """Zero range bars (high=low) handled."""
        ind = AtrIndicator(period=3)

        # Bars with no range
        bar = Bar(
            timestamp="2024-01-01T00:00:00Z",
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1000,
        )

        result = ind.update(bar)
        assert result == 0.0


class TestAtrCalculate:
    """Tests for ATR static calculate method (legacy variant)."""

    def test_calculate_atr(self) -> None:
        """Calculate returns correct ATR."""
        bars = [make_bar(close=100.0 + i, high=102.0 + i, low=98.0 + i) for i in range(20)]

        result = AtrIndicator.calculate(bars, period=14)

        assert result > 0.0

    def test_calculate_insufficient_data(self) -> None:
        """Calculate returns close for insufficient data."""
        bars = [make_bar(close=100.0, high=102.0, low=98.0)]

        result = AtrIndicator.calculate(bars, period=14)

        assert result == 100.0

    def test_calculate_empty_bars(self) -> None:
        """Calculate returns 0.0 for empty bars."""
        result = AtrIndicator.calculate([], period=14)

        assert result == 0.0

    def test_calculate_stable_range(self) -> None:
        """Calculate with stable range."""
        bars = [make_bar(close=100.0, high=101.0, low=99.0) for _ in range(20)]

        result = AtrIndicator.calculate(bars, period=14)

        # ATR should converge to the range (2.0)
        assert 1.8 < result < 2.2

    def test_calculate_matches_stateful(self) -> None:
        """Calculate matches stateful update."""
        bars = [
            make_bar(close=100.0 + i * 0.5, high=102.0 + i * 0.5, low=98.0 + i * 0.5)
            for i in range(25)
        ]

        static_result = AtrIndicator.calculate(bars, period=14)

        ind = AtrIndicator(period=14)
        for bar in bars:
            stateful_result = ind.update(bar)

        assert abs(static_result - stateful_result) < 0.01
