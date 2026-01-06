"""Tests for Heikin-Ashi indicator."""

from trdr.data import Bar
from trdr.indicators.heikin_ashi import HeikinAshiIndicator


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


class TestHeikinAshi:
    """Tests for Heikin-Ashi calculation."""

    def test_returns_list_of_dicts(self) -> None:
        bars = make_bars([100, 102, 104, 106, 108])
        result = HeikinAshiIndicator.calculate(bars)
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(bar, dict) for bar in result)

    def test_has_required_keys(self) -> None:
        bars = make_bars([100, 102, 104])
        result = HeikinAshiIndicator.calculate(bars)
        for ha_bar in result:
            assert "open" in ha_bar
            assert "high" in ha_bar
            assert "low" in ha_bar
            assert "close" in ha_bar
            assert "volume" in ha_bar

    def test_empty_bars(self) -> None:
        result = HeikinAshiIndicator.calculate([])
        assert result == []


class TestHeikinAshiIndicator:
    """Tests for HeikinAshiIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = HeikinAshiIndicator()
        for bar in make_bars([100, 102, 104, 106, 108]):
            value = ind.update(bar)
        assert isinstance(value, dict)
        assert isinstance(ind.value, dict)
