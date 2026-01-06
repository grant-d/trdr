"""Multi-timeframe Point of Control indicator."""

from typing import NamedTuple

from ..data import Bar
from .volume_profile import VolumeProfileIndicator


class _AggBar(NamedTuple):
    """Aggregated bar for multi-timeframe calculations."""

    high: float
    low: float
    close: float
    volume: float


class MultiTimeframePocIndicator:
    """Streaming multi-timeframe PoC calculator."""

    def __init__(self) -> None:
        self._bars: list[Bar] = []

    @staticmethod
    def calculate(bars: list[Bar]) -> tuple[float, float, float]:
        if len(bars) < 12:
            current_poc = VolumeProfileIndicator.calculate(bars[-min(20, len(bars)) :]).poc
            return current_poc, current_poc, current_poc

        poc_tf1 = VolumeProfileIndicator.calculate(bars[-20:]).poc

        agg4_bars = []
        for i in range(0, min(20, len(bars)), 4):
            chunk = bars[-20 + i : -20 + i + 4]
            if chunk:
                agg4_bars.append(
                    _AggBar(
                        high=max(b.high for b in chunk),
                        low=min(b.low for b in chunk),
                        close=chunk[-1].close,
                        volume=sum(b.volume for b in chunk),
                    )
                )

        poc_tf2 = VolumeProfileIndicator.calculate(agg4_bars).poc if agg4_bars else poc_tf1

        agg12_bars = []
        for i in range(0, min(20, len(bars)), 12):
            chunk = bars[-20 + i : -20 + i + 12]
            if chunk:
                agg12_bars.append(
                    _AggBar(
                        high=max(b.high for b in chunk),
                        low=min(b.low for b in chunk),
                        close=chunk[-1].close,
                        volume=sum(b.volume for b in chunk),
                    )
                )

        poc_tf3 = VolumeProfileIndicator.calculate(agg12_bars).poc if agg12_bars else poc_tf1

        return float(poc_tf1), float(poc_tf2), float(poc_tf3)

    def update(self, bar: Bar) -> tuple[float, float, float]:
        self._bars.append(bar)
        return self.calculate(self._bars)

    @property
    def value(self) -> tuple[float, float, float]:
        if not self._bars:
            return (0.0, 0.0, 0.0)
        return self.calculate(self._bars)
