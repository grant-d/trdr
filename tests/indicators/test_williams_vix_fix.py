"""Tests for Williams Vix Fix indicator."""

from trdr.data import Bar
from trdr.indicators.williams_vix_fix import WilliamsVixFixIndicator


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


class TestWilliamsVixFix:
    """Tests for Williams Vix Fix calculation."""

    def test_basic(self) -> None:
        bars = make_bars([100 + i * 0.5 for i in range(80)])
        value, state = WilliamsVixFixIndicator.calculate(bars)
        assert isinstance(value, float)
        assert state in {"high", "normal"}

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 101, 102])
        value, state = WilliamsVixFixIndicator.calculate(bars, pd=22, bbl=20, lb=50)
        assert value == 0.0
        assert state == "normal"


class TestWilliamsVixFixIndicator:
    """Tests for WilliamsVixFixIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = WilliamsVixFixIndicator()
        for bar in make_bars([100 + i * 0.5 for i in range(80)]):
            value, state = ind.update(bar)
        assert isinstance(value, float)
        assert state in {"high", "normal"}
        assert isinstance(ind.value, tuple)
