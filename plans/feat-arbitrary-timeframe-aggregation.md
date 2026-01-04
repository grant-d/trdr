# feat: Arbitrary Timeframe Aggregation

## Overview

Allow strategies to specify **any** timeframe (e.g., "60m", "2h", "3d", "2w") when the underlying Alpaca API has constraints. The system transparently fetches the base unit and aggregates to the requested timeframe.

**User Experience**: Strategy writes `DataRequirement(symbol, "3d", lookback=100)` and receives 3-day OHLCV bars - no knowledge of underlying aggregation required.

## Problem Statement / Motivation

**Alpaca SDK Constraints** (from `alpaca/data/timeframe.py:72-100`):

| Unit | Valid Range | Rejected Examples |
| --- | --- | --- |
| Minute | 1-59 | 60m, 90m, 120m |
| Hour | 1-23 | 24h, 48h |
| Day | 1 only | 2d, 3d, 5d |
| Week | 1 only | 2w, 4w |
| Month | 1, 2, 3, 6, 12 | 4M, 5M, 7M |

**Current trdr Limitation**:
- `parse_timeframe()` in `timeframe.py:117-123` enforces these constraints
- Strategies wanting 60m, 2h, 3d, etc. cannot be expressed

**Use Cases**:

1. **60-minute bars** - Common trading timeframe, requires aggregating 60 x 1m bars
2. **2-hour bars** - Standard swing trading, aggregate from 1h or 120 x 1m
3. **3-day bars** - Swing trading signals with reduced noise
4. **2-week bars** - Longer-term regime detection

**Industry Standard**: FreqTrade, Backtrader, and VectorBT all support arbitrary timeframe resampling.

## Proposed Solution

### High-Level Approach

```text
Strategy requests "3d" bars
         │
         ▼
┌─────────────────────────────────┐
│ MarketDataClient.get_bars()     │
│ - Detect "3d" > Alpaca limit    │
│ - Calculate: need 3x 1d bars    │
│ - Fetch 1d bars from Alpaca     │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│ BarAggregator.aggregate()       │
│ - Group 1d bars into 3-day      │
│ - Apply OHLCV aggregation rules │
│ - Handle incomplete periods     │
└─────────────┬───────────────────┘
              ▼
     Strategy receives "3d" bars
```

### OHLCV Aggregation Rules

| Field | Rule | Example (3 bars) |
| --- | --- | --- |
| Open | First bar's open | `bars[0].open` |
| High | Max of all highs | `max(b.high for b in bars)` |
| Low | Min of all lows | `min(b.low for b in bars)` |
| Close | Last bar's close | `bars[-1].close` |
| Volume | Sum of all volumes | `sum(b.volume for b in bars)` |
| Timestamp | Last bar's timestamp | `bars[-1].timestamp` |

**Interpretation**: "3d" means 3 **trading days**, not calendar days. Weekends/holidays are skipped.

## Technical Approach

### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ src/trdr/data/aggregator.py (NEW)                           │
│                                                             │
│ class BarAggregator:                                        │
│   - aggregate(bars: list[Bar], n: int) -> list[Bar]         │
│   - _group_bars(bars, n) -> Iterator[list[Bar]]             │
│   - _aggregate_group(group: list[Bar]) -> Bar               │
│   - Handles incomplete first/last periods                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ src/trdr/data/market.py (MODIFIED)                          │
│                                                             │
│ MarketDataClient.get_bars():                                │
│   - Detect multi-day request ("3d", "5d", etc.)             │
│   - Calculate 1d bars needed: n * lookback * 1.1            │
│   - Fetch 1d bars from Alpaca                               │
│   - Call BarAggregator.aggregate()                          │
│   - Return aggregated bars to caller                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ src/trdr/backtest/timeframe.py (MODIFIED)                   │
│                                                             │
│ parse_timeframe():                                          │
│   - Remove "days must be 1" constraint                      │
│   - Return TimeFrame(1, Day) for multi-day (aggregation)    │
│   - Add get_aggregation_factor(tf: str) -> int | None       │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: Core Aggregation

| File | Changes |
| --- | --- |
| `src/trdr/data/aggregator.py` | NEW - BarAggregator class |
| `src/trdr/backtest/timeframe.py` | Remove multi-day rejection, add `get_aggregation_factor()` |

```python
# src/trdr/data/aggregator.py
from dataclasses import dataclass
from ..data.market import Bar

@dataclass
class BarAggregator:
    """Aggregate OHLCV bars to larger timeframes."""

    def aggregate(self, bars: list[Bar], n: int, drop_incomplete: bool = True) -> list[Bar]:
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

    def _group_bars(self, bars: list[Bar], n: int, drop_incomplete: bool):
        """Yield groups of n bars for aggregation."""
        # Start from end to ensure last bar is complete
        # Group backwards, then reverse
        groups = []
        for i in range(len(bars), 0, -n):
            start = max(0, i - n)
            group = bars[start:i]
            if len(group) == n or not drop_incomplete:
                groups.append(group)
        return reversed(groups)

    def _aggregate_group(self, group: list[Bar]) -> Bar:
        """Aggregate a group of bars into a single bar."""
        return Bar(
            timestamp=group[-1].timestamp,  # Last bar's timestamp
            open=group[0].open,             # First bar's open
            high=max(b.high for b in group),
            low=min(b.low for b in group),
            close=group[-1].close,          # Last bar's close
            volume=sum(b.volume for b in group),
        )
```

```python
# src/trdr/backtest/timeframe.py additions

def get_aggregation_factor(tf: str) -> int | None:
    """Return aggregation factor for multi-day timeframes.

    Args:
        tf: Timeframe string (e.g., "3d", "5d", "1w")

    Returns:
        Number of 1d bars to aggregate, or None if no aggregation needed
    """
    tf_lower = tf.lower()
    match = re.match(r'^(\d+)([dw])$', tf_lower)
    if not match:
        return None

    amount = int(match.group(1))
    unit = match.group(2)

    if unit == 'd' and amount > 1:
        return amount
    if unit == 'w':
        return amount * 5  # 5 trading days per week
    return None
```

#### Phase 2: MarketDataClient Integration

| File | Changes |
| --- | --- |
| `src/trdr/data/market.py` | Detect multi-day, fetch 1d, aggregate |

```python
# src/trdr/data/market.py modifications

async def get_bars(
    self,
    symbol: str,
    lookback: int,
    timeframe: TimeFrame | str,  # Accept string for convenience
) -> list[Bar]:
    """Fetch bars, aggregating multi-day if needed."""

    # Handle string timeframe
    if isinstance(timeframe, str):
        tf_str = timeframe
        agg_factor = get_aggregation_factor(timeframe)

        if agg_factor:
            # Multi-day: fetch 1d bars and aggregate
            bars_needed = lookback * agg_factor + agg_factor  # Extra for first period
            raw_bars = await self._fetch_bars(symbol, TimeFrame.Day, bars_needed)

            aggregator = BarAggregator()
            return aggregator.aggregate(raw_bars, agg_factor)[-lookback:]
        else:
            timeframe = parse_timeframe(tf_str)

    # Standard fetch for non-multi-day
    return await self._fetch_bars(symbol, timeframe, lookback)
```

#### Phase 3: Week/Month Handling

| Decision | Recommendation |
| --- | --- |
| "1w" | Use Alpaca native `TimeFrame.Week` (no aggregation) |
| "2w" | Aggregate 10 trading days from 1d bars |
| "1M" | Use Alpaca native `TimeFrame.Month` |

```python
def get_aggregation_factor(tf: str) -> int | None:
    """Return aggregation factor, handling week/month."""
    tf_lower = tf.lower()

    # Native Alpaca support - no aggregation
    if tf_lower in ('1w', '1m', '1month'):
        return None

    match = re.match(r'^(\d+)([dwm])$', tf_lower)
    if not match:
        return None

    amount = int(match.group(1))
    unit = match.group(2)

    if unit == 'd' and amount > 1:
        return amount
    if unit == 'w' and amount > 1:
        return amount * 5  # Multi-week
    if unit == 'm' and amount not in (1, 2, 3, 6, 12):
        # Alpaca only supports 1, 2, 3, 6, 12 month
        return amount * 21  # ~21 trading days per month
    return None
```

#### Phase 4: Testing & Validation

| Test File | Coverage |
| --- | --- |
| `tests/test_aggregator.py` | Unit tests for BarAggregator |
| `tests/test_timeframe.py` | Update for multi-day parsing |
| `tests/test_market.py` | Integration tests with mock Alpaca |

## Acceptance Criteria

### Functional Requirements

- [x] "3d" timeframe returns 3-trading-day aggregated bars
- [ ] "5d" timeframe returns 5-trading-day aggregated bars
- [ ] "2w" timeframe returns 10-trading-day aggregated bars
- [ ] "1w" uses Alpaca native Week (no aggregation overhead)
- [ ] BACKTEST_TIMEFRAME="3d" works as override
- [ ] Multi-feed alignment works: 15m primary + 3d informative
- [ ] Incomplete first period is dropped (no partial bars)
- [ ] Strategies receive bars transparently (no API change)

### Non-Functional Requirements

- [ ] Aggregation adds < 100ms latency for 1000 bars
- [ ] Memory usage < 2x baseline for aggregated data
- [ ] Existing "1d" strategies produce identical results (regression)

### Quality Gates

- [ ] Unit tests for `BarAggregator` with edge cases
- [ ] Integration test: 15m primary + 3d informative alignment
- [ ] Regression test: existing "1d" strategy unchanged
- [ ] Validation: compare aggregated "1w" to Alpaca native Week

## Dependencies & Prerequisites

1. **None blocking** - All dependencies are internal
2. **Existing infrastructure**: `MarketDataClient`, `align_feeds()`, `DataRequirement`
3. **pandas not required** - Pure Python aggregation is sufficient

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Look-ahead bias in aggregation | Medium | Critical | Group from end, drop incomplete first |
| Incorrect OHLCV formula | Low | Critical | Unit tests with known-good examples |
| Cross-TF alignment bugs | Medium | High | Reuse existing `align_feeds()` logic |
| Performance for large datasets | Low | Medium | Lazy aggregation, no caching initially |

## References & Research

### Internal References

- `parse_timeframe()`: `src/trdr/backtest/timeframe.py:85-131`
- Multi-day rejection: `src/trdr/backtest/timeframe.py:117-123`
- `MarketDataClient.get_bars()`: `src/trdr/data/market.py:130-227`
- `align_feeds()`: `src/trdr/backtest/timeframe.py:44-82`
- Existing aggregation POC: `src/trdr/indicators/indicators.py:587-637`

### External References

- Alpaca TimeFrame limits: <https://forum.alpaca.markets/t/multi-bar-day-limit-says-maximum-period-is-1/8808>
- pandas OHLCV resampling: <https://github.com/matplotlib/mplfinance/wiki/Resampling-Time-Series-Data-with-Pandas>
- Backtrader resampling: <https://www.backtrader.com/docu/data-resampling/data-resampling/>

## Key Design Decisions

1. **Trading days, not calendar days** - "3d" = 3 trading days, weekends/holidays skipped
2. **Drop incomplete first period** - Ensures all bars are complete, no partial data
3. **Timestamp = last bar** - Aggregated bar represents when period completed
4. **No caching initially** - Keep simple, add caching if performance requires
5. **Prefer Alpaca native for 1w/1M** - More accurate, less processing
6. **Pure Python aggregation** - Avoid pandas dependency for simple use case
