"""Bar aggregation for arbitrary timeframes."""

from dataclasses import dataclass
from typing import Iterator

from .bar import Bar


@dataclass
class BarAggregator:
    """Aggregate OHLCV bars to larger timeframes.

    Combines n consecutive bars into single bars following standard OHLCV rules:
    - Open: first bar's open
    - High: max of all highs
    - Low: min of all lows
    - Close: last bar's close
    - Volume: sum of all volumes
    - Timestamp: last bar's timestamp
    """

    def aggregate(
        self, bars: list[Bar], n: int, drop_incomplete: bool = True
    ) -> list[Bar]:
        """Aggregate n consecutive bars into single bars.

        Args:
            bars: Source bars (must be sorted by timestamp ascending)
            n: Number of bars to combine (e.g., 3 for 3-day)
            drop_incomplete: If True, drop incomplete first period

        Returns:
            Aggregated bars
        """
        if n <= 1:
            return bars

        result = []
        for group in self._group_bars(bars, n, drop_incomplete):
            result.append(self._aggregate_group(group))
        return result

    def _group_bars(
        self, bars: list[Bar], n: int, drop_incomplete: bool
    ) -> Iterator[list[Bar]]:
        """Yield groups of n bars for aggregation.

        Groups from end to ensure last period is complete.
        """
        # Start from end to ensure last bar is in a complete period
        groups = []
        for i in range(len(bars), 0, -n):
            start = max(0, i - n)
            group = bars[start:i]
            if len(group) == n or not drop_incomplete:
                groups.append(group)
        # Reverse to maintain chronological order
        return reversed(groups)

    def _aggregate_group(self, group: list[Bar]) -> Bar:
        """Aggregate a group of bars into a single bar."""
        return Bar(
            timestamp=group[-1].timestamp,  # Last bar's timestamp
            open=group[0].open,  # First bar's open
            high=max(b.high for b in group),
            low=min(b.low for b in group),
            close=group[-1].close,  # Last bar's close
            volume=sum(b.volume for b in group),
        )
