"""Tests for Lorentzian classifier indicator."""

from trdr.data import Bar
from trdr.indicators.lorentzian_classifier import LorentzianClassifierIndicator


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


class TestLorentzianClassifier:
    """Tests for Lorentzian classifier calculation."""

    def test_prediction_bounds(self) -> None:
        bars = make_bars(list(range(100, 220)))
        result = LorentzianClassifierIndicator.calculate(bars, neighbors=5, training_window=4)
        assert result in (-1, 0, 1)

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 101])
        result = LorentzianClassifierIndicator.calculate(bars, neighbors=5, training_window=4)
        assert result == 0


class TestLorentzianClassifierIndicator:
    """Tests for LorentzianClassifierIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = LorentzianClassifierIndicator(neighbors=5, training_window=4)
        for bar in make_bars(list(range(100, 220))):
            value = ind.update(bar)
        assert value in (-1, 0, 1)
        assert ind.value in (-1, 0, 1)
