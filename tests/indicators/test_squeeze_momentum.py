"""Tests for Squeeze Momentum indicator."""

from trdr.data import Bar
from trdr.indicators.squeeze_momentum import SqueezeMomentumIndicator


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


class TestSqueezeMomentum:
    """Tests for squeeze momentum calculation."""

    def test_basic(self) -> None:
        bars = make_bars(list(range(100, 140)))
        momentum, state = SqueezeMomentumIndicator.calculate(bars, bb_length=20, kc_length=20)
        assert isinstance(momentum, float)
        assert state in {"squeeze_on", "squeeze_off", "no_squeeze"}

    def test_insufficient_data(self) -> None:
        bars = make_bars([100, 101])
        momentum, state = SqueezeMomentumIndicator.calculate(bars, bb_length=20, kc_length=20)
        assert momentum == 0.0
        assert state == "no_squeeze"


class TestSqueezeMomentumIndicator:
    """Tests for SqueezeMomentumIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = SqueezeMomentumIndicator(bb_length=20, kc_length=20)
        for bar in make_bars(list(range(100, 140))):
            momentum, state = ind.update(bar)
        assert isinstance(momentum, float)
        assert state in {"squeeze_on", "squeeze_off", "no_squeeze"}
        value = ind.value
        assert isinstance(value, tuple)
