"""Tests for volume profile indicator."""

import pytest

from trdr.data import Bar
from trdr.indicators.volume_profile import VolumeProfile, VolumeProfileIndicator


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


class TestVolumeProfile:
    """Tests for volume profile calculation."""

    def test_basic(self) -> None:
        bars = make_bars([100, 102, 101, 103, 100])
        result = VolumeProfileIndicator.calculate(bars)
        assert isinstance(result, VolumeProfile)
        assert result.poc > 0
        assert result.vah >= result.val

    def test_flat_market(self) -> None:
        bars = make_bars([100] * 10)
        result = VolumeProfileIndicator.calculate(bars)
        assert 99 < result.poc < 101
        assert result.vah >= result.val

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            VolumeProfileIndicator.calculate([])


class TestVolumeProfileIndicator:
    """Tests for VolumeProfileIndicator (stateful variant)."""

    def test_update_and_value(self) -> None:
        ind = VolumeProfileIndicator()
        for bar in make_bars([100, 102, 101, 103, 100]):
            value = ind.update(bar)
        assert isinstance(value, VolumeProfile)
        assert isinstance(ind.value, VolumeProfile)
