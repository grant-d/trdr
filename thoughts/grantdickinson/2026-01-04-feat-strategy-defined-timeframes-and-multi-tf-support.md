# feat: Strategy-Defined Timeframes and Multi-Timeframe/Multi-Symbol Support

## Status: ✅ IMPLEMENTED

| Phase | Status | Notes |
| --- | --- | --- |
| Phase 1: Core API | ✅ Done | `DataRequirement`, `get_data_requirements()`, new `generate_signal()` signature |
| Phase 2: Data Fetching | ✅ Done | `get_bars_multi()`, `align_feeds()`, `get_interval_seconds()` |
| Phase 3: SICA Runner | ✅ Done | Multi-feed fetch, `BACKTEST_TIMEFRAME` override |
| Phase 4: PaperExchange | ✅ Done | `primary_feed` config, dict-based bar handling |
| Phase 5: Strategy Updates | ✅ Done | VAB, MeanReversion, MACD (with MTF example) |
| Phase 6: SICA Config | ✅ N/A | No changes needed - existing configs work |
| Quality Gates | ✅ Done | 276 tests pass, all acceptance criteria met |
| **Enhancement: Arbitrary TF** | ✅ Done | `BarAggregator`, `get_aggregation_config()` - supports 60m, 3d, 2w, etc. |
| **Enhancement: Timeframe Class** | ✅ Done | `Timeframe` dataclass with canonical normalization (60m→1h, 48h→2d), `parse_timeframe()` returns our abstraction, Alpaca constraints handled internally |

## Overview

Allow strategies to define their own data requirements: timeframe, lookback, and even reference data from other symbols. Currently, infrastructure dictates timeframe (15m) and length (3000 bars) externally. This change inverts control: strategies declare what data they need, infrastructure provides it.

**v0 Breaking Change** - No backward compatibility required. All existing strategies will be updated.

## Problem Statement / Motivation

**Current Limitations:**

1. **External timeframe control** - `BACKTEST_TIMEFRAME` and `BACKTEST_LOOKBACK` env vars force all strategies to use the same settings
2. **Single timeframe** - Strategies cannot request multiple timeframes (e.g., 15m for signals + 4h for trend context)
3. **Single symbol** - Cannot reference other assets (e.g., use BTC/USD 1h as market regime indicator while trading ETH/USD)
4. **Workaround exists** - `multi_timeframe_poc()` aggregates bars internally to simulate higher TFs, but this is imprecise

**Industry Standard:**
FreqTrade's `informative_pairs()` supports multi-symbol + multi-timeframe. This is the pattern we'll adopt.

## Proposed Solution

### Core API: DataRequirement

```python
@dataclass
class DataRequirement:
    """Specification for a data feed."""
    symbol: Symbol       # e.g., "crypto:ETH/USD", "crypto:BTC/USD"
    timeframe: str       # e.g., "15m", "1h", "4h"
    lookback: int        # Number of bars to fetch
    role: str = "informative"  # "primary" or "informative"

    @property
    def key(self) -> str:
        """Unique key for this data feed."""
        return f"{self.symbol}:{self.timeframe}"
```

### Strategy Declaration

```python
class BaseStrategy(ABC):
    @abstractmethod
    def get_data_requirements(self) -> list[DataRequirement]:
        """Declare all data feeds this strategy needs.

        Returns:
            List of DataRequirement. Exactly one must have role="primary".
        """
        ...
```

### Multi-TF + Multi-Symbol Example

```python
class MTFVolumeStrategy(BaseStrategy):
    def get_data_requirements(self) -> list[DataRequirement]:
        return [
            # Primary: what we're trading
            DataRequirement("crypto:ETH/USD", "15m", 3000, role="primary"),
            # Same symbol, higher TFs for context
            DataRequirement("crypto:ETH/USD", "1h", 500),
            DataRequirement("crypto:ETH/USD", "4h", 125),
            # Different symbol for market regime
            DataRequirement("crypto:BTC/USD", "1h", 500),
        ]

    def generate_signal(
        self,
        bars: dict[str, list[Bar]],  # Keyed by "symbol:timeframe"
        position: Position | None
    ) -> Signal:
        eth_15m = bars["crypto:ETH/USD:15m"]
        eth_1h = bars["crypto:ETH/USD:1h"]
        btc_1h = bars["crypto:BTC/USD:1h"]
        # Multi-TF + multi-symbol logic
```

### Timeframe Override

`BACKTEST_TIMEFRAME` env var (if set) overrides the strategy's primary timeframe:

```bash
# Use strategy's declared timeframe
python sica_bench.py

# Override primary timeframe (existing SICA configs continue to work!)
BACKTEST_TIMEFRAME=1h python sica_bench.py
```

This means **no changes needed to existing SICA configs** - they already set `BACKTEST_TIMEFRAME`, which now acts as an override.

## Technical Approach

### Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│ SICA Config / CLI                                               │
│ BACKTEST_TIMEFRAME=1h (optional, overrides strategy's primary)  │
└─────────────────────┬───────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ sica_runner.py                                                  │
│ 1. Load strategy class                                          │
│ 2. Call get_data_requirements()                                 │
│ 3. Apply BACKTEST_TIMEFRAME if set (overrides primary TF)       │
│ 4. Fetch bars for each requirement via AlpacaDataClient         │
│ 5. Align informative feeds to primary feed timestamps           │
│ 6. Pass dict[str, list[Bar]] to PaperExchange                   │
└─────────────────────┬───────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ PaperExchange                                                   │
│ 1. Receive bars: dict[str, list[Bar]] keyed by "symbol:tf"      │
│ 2. Iterate over primary feed bars                               │
│ 3. Slice all feeds to visible window (no lookahead)             │
│ 4. Pass to strategy.generate_signal()                           │
└─────────────────────┬───────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Strategy.generate_signal()                                      │
│ - Receives dict[str, list[Bar]] keyed by "symbol:timeframe"     │
│ - Primary feed: bars["crypto:ETH/USD:15m"]                      │
│ - Informative: bars["crypto:BTC/USD:1h"]                        │
└─────────────────────────────────────────────────────────────────┘
```

### Data Alignment Algorithm

Forward-fill higher timeframe bars to primary timeframe timestamps using interval containment:

```python
def align_timeframes(
    primary_bars: list[Bar],
    higher_tf_bars: list[Bar],
    primary_tf: str,
    higher_tf: str,
) -> list[Bar]:
    """Align higher TF bars to primary TF timestamps.

    For each primary bar, find the most recent higher TF bar
    whose interval contains or precedes the primary bar's timestamp.
    """
    aligned = []
    htf_idx = 0
    htf_interval = get_interval_seconds(higher_tf)

    for bar in primary_bars:
        # Find latest higher TF bar that precedes this primary bar
        while (htf_idx < len(higher_tf_bars) - 1 and
               higher_tf_bars[htf_idx + 1].timestamp <= bar.timestamp):
            htf_idx += 1

        if htf_idx < len(higher_tf_bars):
            aligned.append(higher_tf_bars[htf_idx])
        else:
            aligned.append(None)  # No higher TF data yet

    return aligned
```

**Example:**

- Primary: 15m bars at [10:00, 10:15, 10:30, 10:45, 11:00]
- Higher TF: 1h bars at [10:00, 11:00]
- Aligned: 1h bar at 10:00 is used for all 15m bars from 10:00-10:45; 1h bar at 11:00 used for 11:00+

### Implementation Phases

#### Phase 1: Core API

| File | Changes |
| --- | --- |
| `src/trdr/strategy/types.py` | Add `DataRequirement` dataclass |
| `src/trdr/strategy/base_strategy.py` | Add abstract `get_data_requirements()`, change `generate_signal()` signature |

```python
# types.py additions
@dataclass
class DataRequirement:
    """Specification for a data feed."""
    symbol: Symbol
    timeframe: str
    lookback: int
    role: str = "informative"  # "primary" or "informative"

    @property
    def key(self) -> str:
        """Unique key for this data feed."""
        return f"{self.symbol}:{self.timeframe}"

    def __post_init__(self):
        if self.role not in ("primary", "informative"):
            raise ValueError(f"role must be 'primary' or 'informative', got '{self.role}'")
```

```python
# base_strategy.py changes
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    @abstractmethod
    def get_data_requirements(self) -> list[DataRequirement]:
        """Declare all data feeds this strategy needs.

        Returns:
            List of DataRequirement. Exactly one must have role="primary".
            The primary feed determines bar iteration and trading symbol.
        """
        ...

    @abstractmethod
    def generate_signal(
        self,
        bars: dict[str, list[Bar]],  # Keyed by "symbol:timeframe"
        position: Position | None,
    ) -> Signal:
        """Generate trading signal from multi-feed data."""
        ...
```

#### Phase 2: Data Fetching

| File | Changes |
| --- | --- |
| `src/trdr/data/market.py` | Add `get_bars_multi(requirements: list[DataRequirement])` |
| `src/trdr/backtest/utils.py` | Add `align_feeds()` function |
| `src/trdr/backtest/utils.py` | Add `get_interval_seconds(tf: str)` helper |

```python
# market.py additions
async def get_bars_multi(
    self,
    requirements: list[DataRequirement],
) -> dict[str, list[Bar]]:
    """Fetch bars for multiple symbol/timeframe combinations.

    Args:
        requirements: List of DataRequirement specifying each feed

    Returns:
        Dict mapping "symbol:timeframe" to list of bars
    """
    result = {}
    for req in requirements:
        bars = await self.get_bars(req.symbol, req.lookback, parse_timeframe(req.timeframe))
        result[req.key] = bars
    return result
```

```python
# utils.py additions
def get_interval_seconds(tf: str) -> int:
    """Get interval duration in seconds for a timeframe string."""
    tf_lower = tf.lower()
    if "m" in tf_lower:
        minutes = int(tf_lower.replace("m", "").replace("in", ""))
        return minutes * 60
    elif "h" in tf_lower:
        hours = int(tf_lower.replace("h", "").replace("our", ""))
        return hours * 3600
    elif "d" in tf_lower:
        return 86400
    raise ValueError(f"Unsupported timeframe: {tf}")

def align_feeds(
    primary_bars: list[Bar],
    informative_bars: list[Bar],
) -> list[Bar | None]:
    """Align informative feed to primary feed timestamps via forward-fill.

    Works for both multi-timeframe (same symbol) and multi-symbol scenarios.
    """
    if not informative_bars:
        return [None] * len(primary_bars)

    aligned = []
    info_idx = 0

    for bar in primary_bars:
        # Advance to latest informative bar that precedes this bar
        while (info_idx < len(informative_bars) - 1 and
               informative_bars[info_idx + 1].timestamp <= bar.timestamp):
            info_idx += 1

        # Only include if informative bar timestamp <= primary bar timestamp
        if informative_bars[info_idx].timestamp <= bar.timestamp:
            aligned.append(informative_bars[info_idx])
        else:
            aligned.append(None)

    return aligned
```

#### Phase 3: SICA Runner Integration

| File | Changes |
| --- | --- |
| `src/trdr/strategy/sica_runner.py` | Fetch multi-feed data, apply override, align feeds |

```python
# sica_runner.py modifications

def get_primary_requirement(requirements: list[DataRequirement]) -> DataRequirement:
    """Find the primary requirement. Raises if not exactly one."""
    primaries = [r for r in requirements if r.role == "primary"]
    if len(primaries) != 1:
        raise ValueError(f"Expected exactly 1 primary requirement, got {len(primaries)}")
    return primaries[0]

async def _get_bars(
    client: AlpacaDataClient,
    strategy: BaseStrategy,
) -> tuple[dict[str, list[Bar]], DataRequirement]:
    """Fetch bars based on strategy requirements.

    Args:
        client: Market data client
        strategy: Strategy instance

    Returns:
        Tuple of (bars dict keyed by "symbol:tf", primary requirement)
    """
    requirements = strategy.get_data_requirements()
    primary = get_primary_requirement(requirements)

    # Apply BACKTEST_TIMEFRAME override if specified
    timeframe_override = os.environ.get("BACKTEST_TIMEFRAME")
    if timeframe_override:
        # Replace primary's timeframe
        requirements = [
            DataRequirement(r.symbol, timeframe_override, r.lookback, r.role)
            if r.role == "primary" else r
            for r in requirements
        ]
        primary = get_primary_requirement(requirements)

    # Fetch all feeds
    bars_dict = await client.get_bars_multi(requirements)
    primary_bars = bars_dict[primary.key]

    # Align informative feeds to primary
    aligned = {primary.key: primary_bars}
    for req in requirements:
        if req.role != "primary":
            aligned[req.key] = align_feeds(primary_bars, bars_dict[req.key])

    return aligned, primary
```

```python
# sica_runner.py - run_sica_benchmark updates

async def run_sica_benchmark(strategy: BaseStrategy) -> dict:
    bars, primary = await _get_bars(client, strategy)
    # BACKTEST_TIMEFRAME override is applied inside _get_bars()

    config = PaperExchangeConfig(
        symbol=primary.symbol,  # Trade the primary symbol
        bars=bars,
        primary_key=primary.key,
        # ... other config
    )
    # ...
```

#### Phase 4: PaperExchange Updates

| File | Changes |
| --- | --- |
| `src/trdr/backtest/paper_exchange.py` | Accept multi-feed bars, slice per-feed windows |

```python
# paper_exchange.py modifications

@dataclass
class PaperExchangeConfig:
    symbol: Symbol  # Primary trading symbol
    bars: dict[str, list[Bar]]  # Keyed by "symbol:timeframe"
    primary_key: str  # Key for primary feed (e.g., "crypto:ETH/USD:15m")
    # ... other existing fields ...

    @property
    def primary_bars(self) -> list[Bar]:
        return self.bars[self.primary_key]

# In PaperExchange.run():
def _get_visible_bars(self, index: int) -> dict[str, list[Bar]]:
    """Get bars visible up to index (no lookahead)."""
    return {
        key: bars[:index + 1]
        for key, bars in self.config.bars.items()
    }
```

#### Phase 5: Strategy Updates (All Existing Strategies)

| File | Changes |
| --- | --- |
| `src/trdr/strategy/volume_area_breakout/strategy.py` | Update to new API |
| `src/trdr/strategy/mean_reversion/strategy.py` | Update to new API |
| `src/trdr/strategy/macd_template/strategy.py` | Update to new API |

```python
# volume_area_breakout/strategy.py - full update

class VolumeAreaBreakoutStrategy(BaseStrategy):
    def get_data_requirements(self) -> list[DataRequirement]:
        """Declare data feeds for this strategy."""
        symbol = self.config.symbol
        primary_tf = self.config.timeframe or "15m"
        return [
            DataRequirement(symbol, primary_tf, 3000, role="primary"),
            DataRequirement(symbol, "1h", 500),   # Trend context
            DataRequirement(symbol, "4h", 125),   # Regime detection
        ]

    def generate_signal(
        self,
        bars: dict[str, list[Bar]],
        position: Position | None,
    ) -> Signal:
        # Get primary bars
        primary_key = f"{self.config.symbol}:{self.config.timeframe or '15m'}"
        primary_bars = bars[primary_key]

        # Get informative feeds
        htf_1h = bars.get(f"{self.config.symbol}:1h", [])
        htf_4h = bars.get(f"{self.config.symbol}:4h", [])

        # Existing logic on primary_bars, enhanced with MTF context
        # ...
```

**Multi-Symbol Example (new capability):**

```python
class BTCCorrelationStrategy(BaseStrategy):
    """Trade ETH based on BTC correlation signals."""

    def get_data_requirements(self) -> list[DataRequirement]:
        return [
            DataRequirement("crypto:ETH/USD", "15m", 3000, role="primary"),
            DataRequirement("crypto:ETH/USD", "1h", 500),
            DataRequirement("crypto:BTC/USD", "1h", 500),  # Cross-asset reference
            DataRequirement("crypto:BTC/USD", "4h", 125),
        ]

    def generate_signal(
        self,
        bars: dict[str, list[Bar]],
        position: Position | None,
    ) -> Signal:
        eth_15m = bars["crypto:ETH/USD:15m"]
        eth_1h = bars["crypto:ETH/USD:1h"]
        btc_1h = bars["crypto:BTC/USD:1h"]
        btc_4h = bars["crypto:BTC/USD:4h"]

        # Use BTC trend as regime filter
        btc_trend = self._calc_trend(btc_4h)
        if btc_trend < 0:
            return Signal(action=SignalAction.HOLD, reason="BTC bearish regime")

        # Trade ETH based on its own signals + BTC confirmation
        # ...
```

#### Phase 6: SICA Config Compatibility (No Migration Needed!)

Existing SICA configs use `BACKTEST_TIMEFRAME` and `BACKTEST_LOOKBACK`. **No changes required.**

**How it works:**

1. `BACKTEST_TIMEFRAME` env var now acts as an override (same name, new semantics)
2. Strategy defines its requirements via `get_data_requirements()`
3. If `BACKTEST_TIMEFRAME` is set, it overrides the strategy's primary timeframe
4. Existing SICA configs continue to work unchanged

**Config files (no changes needed):**

- `.sica/configs/btcusd-4h/config.json`
- `.sica/configs/ethusd-4h/config.json`
- `.sica/configs/solusd-4h/config.json`
- `.sica/configs/btcusd-1d-vab/config.json`
- `.sica/configs/btcusd-1d-mr/config.json`
- `.sica/configs/aapl-1d-vab/config.json`
- `.sica/configs/aapl-1d-mr/config.json`
- `.sica/configs/btcusd-15m/config.json`
- `.sica/configs/ethusd-15m/config.json`

**Example (unchanged):**

```json
{
  "benchmark_cmd": "BACKTEST_SYMBOL={symbol} BACKTEST_TIMEFRAME={timeframe} BACKTEST_LOOKBACK={lookback} python ...",
  "params": {
    "symbol": "crypto:ETH/USD",
    "timeframe": "15m",
    "lookback": "3000"
  }
}
```

When SICA runs this, `BACKTEST_TIMEFRAME=15m` overrides whatever the strategy declares as its primary timeframe.

## Acceptance Criteria

### Functional Requirements

- [x] `DataRequirement` dataclass with symbol, timeframe, lookback, role
- [x] `BaseStrategy.get_data_requirements()` is abstract (required for all strategies)
- [x] `generate_signal()` receives `dict[str, list[Bar]]` keyed by "symbol:timeframe"
- [x] Exactly one requirement must have `role="primary"`
- [x] Informative feeds are forward-filled and aligned to primary feed
- [x] `BACKTEST_TIMEFRAME` env var overrides strategy's primary timeframe (existing configs work)
- [x] Multi-symbol support: strategy can reference bars from different symbols
- [x] SICA runner fetches and aligns all required feeds

### Non-Functional Requirements

- [x] Memory usage stays reasonable (< 2x current for 3-feed strategy)
- [x] API calls minimized (fetch each feed once, cache appropriately)
- [x] No SICA config changes required (strategy declares its own requirements)

### Quality Gates

- [x] Unit tests for `align_feeds()` with edge cases
- [x] Unit tests for `get_primary_requirement()` validation
- [x] Integration test with 3-feed strategy (2 TFs + 1 cross-symbol)
- [x] All existing strategies updated to new API
- [x] Coverage maintained at 80%+ (276 tests pass)

## Dependencies & Prerequisites

1. **None blocking** - All dependencies are internal to this repo
2. **AlpacaDataClient** - Already supports single-TF fetching, just needs multi-TF wrapper
3. **Cache** - Existing CSV cache works per-TF, no changes needed

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Data alignment bugs | Medium | High | Comprehensive unit tests for `align_feeds()` |
| Memory spike with many feeds | Low | Medium | Document recommended limits (4-5 feeds max) |
| API rate limits | Low | Low | Existing caching handles this |
| Warmup period confusion | Medium | Medium | Document warmup applies to primary feed |
| Multi-symbol timezone misalignment | Low | Medium | All bars use UTC timestamps |

## References & Research

### Internal References

- Base Strategy: `src/trdr/strategy/base_strategy.py:14-52`
- SICA Runner: `src/trdr/strategy/sica_runner.py:79-100`
- AlpacaDataClient: `src/trdr/data/market.py:130-227`
- PaperExchange: `src/trdr/backtest/paper_exchange.py:475-652`
- Multi-TF POC (workaround): `src/trdr/indicators/indicators.py:587-638`
- SICA config: `.sica/configs/ethusd-15m/config.json`
- Volume Area Breakout: `src/trdr/strategy/volume_area_breakout/strategy.py`

### External References

- FreqTrade informative_pairs: <https://www.freqtrade.io/en/stable/strategy-customization/>
- Backtrader Multi-TF: <https://www.backtrader.com/docu/data-multitimeframe/data-multitimeframe/>
- QuantConnect Consolidators: <https://www.quantconnect.com/docs/v2/writing-algorithms/consolidating-data/>

### Industry Pattern Summary

| Framework | Declaration Method | Our Equivalent |
| --- | --- | --- |
| FreqTrade | `informative_pairs()` | `get_data_requirements()` |
| Backtrader | `adddata()` + `resampledata()` | Automatic in runner |
| QuantConnect | `AddEquity(resolution)` + Consolidators | `get_data_requirements()` |

## Key Design Decisions

1. **Method vs Class Attribute**: Use method `get_data_requirements()` not class attribute - allows dynamic requirements based on config
2. **List of DataRequirement**: Return `list[DataRequirement]` not `dict` - allows multi-symbol + per-feed lookback
3. **Explicit Primary Role**: Use `role="primary"` not positional - explicit is better than implicit
4. **Compound Key**: Use `"symbol:timeframe"` as dict key - unambiguous for multi-symbol scenarios
5. **Forward-Fill Alignment**: Use forward-fill (last known value) not interpolation - prevents lookahead
6. **Abstract Method**: `get_data_requirements()` is required (abstract) - no backward compat needed for v0
7. **Timeframe Override**: `BACKTEST_TIMEFRAME_OVERRIDE` env var - allows experimentation without code changes
