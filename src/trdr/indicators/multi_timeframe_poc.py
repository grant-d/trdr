"""Multi-timeframe Point of Control indicator."""

from typing import NamedTuple

from ..data import Bar
from .volume_profile import volume_profile


class _AggBar(NamedTuple):
    """Aggregated bar for multi-timeframe calculations."""

    high: float
    low: float
    close: float
    volume: float


def multi_timeframe_poc(bars: list[Bar]) -> tuple[float, float, float]:
    """Calculate PoC at multiple aggregation levels.

    Simulates different timeframes by aggregating bars:
    - TF1: Current bars (native resolution)
    - TF2: 4-bar aggregation (4x longer timeframe)
    - TF3: 12-bar aggregation (12x longer timeframe)

    Returns:
        Tuple of (poc_tf1, poc_tf2, poc_tf3)
    """
    if len(bars) < 12:
        current_poc = volume_profile(bars[-min(20, len(bars)) :]).poc
        return current_poc, current_poc, current_poc

    # TF1: Current resolution
    poc_tf1 = volume_profile(bars[-20:]).poc

    # TF2: 4-bar aggregation (aggregate last 20 bars into 5 "super bars")
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

    poc_tf2 = volume_profile(agg4_bars).poc if agg4_bars else poc_tf1

    # TF3: 12-bar aggregation
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

    poc_tf3 = volume_profile(agg12_bars).poc if agg12_bars else poc_tf1

    return float(poc_tf1), float(poc_tf2), float(poc_tf3)
