"""Tests for order flow imbalance indicator."""

from trdr.data import Bar
from trdr.indicators.order_flow_imbalance import OrderFlowImbalanceIndicator


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


class TestOrderFlowImbalance:
    """Tests for order flow imbalance calculation."""

    def test_buying_pressure(self) -> None:
        bars = make_bars(list(range(100, 110)))
        result = OrderFlowImbalanceIndicator.calculate(bars, lookback=5)
        assert result > 0

    def test_selling_pressure(self) -> None:
        bars = make_bars(list(range(110, 100, -1)))
        result = OrderFlowImbalanceIndicator.calculate(bars, lookback=5)
        assert result < 0

    def test_insufficient_data(self) -> None:
        bars = make_bars([100])
        result = OrderFlowImbalanceIndicator.calculate(bars, lookback=5)
        assert result == 0.0

    def test_range_bound(self) -> None:
        result = OrderFlowImbalanceIndicator.calculate(
            make_bars([100, 102, 101, 100, 102]), lookback=5
        )
        assert -1.0 <= result <= 1.0


class TestOrderFlowImbalanceIndicator:
    """Tests for OrderFlowImbalanceIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = OrderFlowImbalanceIndicator(lookback=5)
        for bar in make_bars(list(range(100, 110))):
            value = ind.update(bar)
        assert isinstance(value, float)
        assert isinstance(ind.value, float)
