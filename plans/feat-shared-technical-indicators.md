# Feature: Shared Technical Indicators Module

**Status: COMPLETE**

Create a centralized indicators module for reusable technical analysis functions.

## Problem Statement

Indicator functions are duplicated across strategy files:

- `macd_template/strategy.py` has `_ema()` (private)
- `volume_area_breakout/strategy.py` has 15+ indicators defined inline

New strategies must copy-paste indicators or reimplement them. No shared location exists.

## Proposed Solution

Create `src/trdr/indicators/` package with common technical indicators.

### Location Decision

`src/trdr/indicators/` as standalone package because:

1. Indicators are general-purpose utilities usable beyond strategies
2. Keeps related code together (VolumeProfile dataclass with volume_profile function)
3. Clean separation of concerns

## Acceptance Criteria

- [x] Create `src/trdr/indicators/__init__.py` with common indicators
- [x] Export from `src/trdr/strategy/__init__.py`
- [x] Update `volume_area_breakout/strategy.py` to import from shared module
- [x] Update `macd_template/strategy.py` to import from shared module
- [x] All existing tests pass (177 tests)
- [x] Add unit tests for indicators (42 tests including edge cases like period=1)

## Implementation

### indicators/__init__.py

```python
# src/trdr/indicators/__init__.py
"""Technical indicators for trading strategies.

All functions accept bars: list[Bar] as first parameter.
Returns float for single values, tuple for multiple.
"""

from ..data.market import Bar

# Moving Averages
def sma(bars: list[Bar], period: int) -> float: ...
def ema(bars: list[Bar], period: int) -> float: ...
def wma(bars: list[Bar], period: int) -> float: ...
def hma(bars: list[Bar], period: int) -> float: ...

# Volatility
def atr(bars: list[Bar], period: int = 14) -> float: ...
def bollinger_bands(bars: list[Bar], period: int = 20, std_mult: float = 2.0) -> tuple[float, float, float]: ...

# Momentum
def rsi(bars: list[Bar], period: int = 14) -> float: ...
def macd(bars: list[Bar], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]: ...

# Trend
def hma_slope(bars: list[Bar], period: int = 9, lookback: int = 3) -> float: ...

# Volume
def volume_profile(bars: list[Bar], num_levels: int = 40, value_area_pct: float = 0.70) -> VolumeProfile: ...
def volume_trend(bars: list[Bar], lookback: int = 5) -> str: ...
def order_flow_imbalance(bars: list[Bar], lookback: int = 5) -> float: ...

# Market Structure
def mss(bars: list[Bar], lookback: int = 20) -> float: ...
def volatility_regime(bars: list[Bar], lookback: int = 50) -> str: ...

# Pattern Recognition
def sax_pattern(bars: list[Bar], window: int = 20, segments: int = 5) -> str: ...
def sax_bullish_reversal(pattern: str) -> bool: ...
def heikin_ashi(bars: list[Bar]) -> list[dict]: ...
```

### Naming Convention

| Current | Shared Module |
| --- | --- |
| `calculate_atr()` | `atr()` |
| `calculate_rsi()` | `rsi()` |
| `calculate_bollinger_bands()` | `bollinger_bands()` |
| `calculate_hma()` | `hma()` |
| `calculate_volume_profile()` | `volume_profile()` |
| `compute_sax_pattern()` | `sax_pattern()` |
| `detect_sax_bullish_reversal()` | `sax_bullish_reversal()` |

Drop `calculate_`/`compute_`/`detect_` prefixes - they're redundant in an indicators module.

### Edge Cases

All indicators handle:

1. **Insufficient data**: Return neutral defaults (0.0 for float, 50.0 for RSI, "neutral" for strings)
2. **Empty bars**: Return neutral or raise if truly invalid
3. **Division by zero**: Guard all denominators

### Exports

```python
# src/trdr/strategy/__init__.py
from ..indicators import (
    sma, ema, wma, hma, hma_slope,
    atr, bollinger_bands,
    rsi, macd,
    volume_profile, volume_trend, order_flow_imbalance,
    mss, volatility_regime,
    sax_pattern, sax_bullish_reversal, heikin_ashi,
)
```

## Migration

### volume_area_breakout/strategy.py

```python
# Before (inline)
def calculate_atr(bars: list[Bar], period: int = 14) -> float:
    ...

class VolumeAreaBreakoutStrategy(BaseStrategy):
    def generate_signal(self, bars, position):
        atr_val = calculate_atr(bars)

# After (import)
from ..indicators import atr, rsi, bollinger_bands, volume_profile

class VolumeAreaBreakoutStrategy(BaseStrategy):
    def generate_signal(self, bars, position):
        atr_val = atr(bars)
```

### macd_template/strategy.py

```python
# Before (inline private)
def _ema(values: list[float], period: int) -> list[float]:
    ...

# After (import)
from ..indicators import ema
```

## Testing

### tests/test_indicators.py

| Test | Coverage |
| --- | --- |
| `test_sma_basic` | SMA with sufficient data |
| `test_ema_basic` | EMA with sufficient data |
| `test_atr_basic` | ATR calculation |
| `test_atr_insufficient_data` | Returns 0.0 |
| `test_rsi_overbought` | RSI > 70 |
| `test_rsi_oversold` | RSI < 30 |
| `test_rsi_neutral` | RSI ~50 on insufficient data |
| `test_bollinger_bands` | Upper, middle, lower bands |
| `test_hma_basic` | HMA calculation |
| `test_volume_profile` | PoC, VAH, VAL |
| `test_macd_crossover` | MACD line, signal, histogram |

## File Changes

| File | Change |
| --- | --- |
| `src/trdr/indicators/__init__.py` | New - all indicator functions + VolumeProfile |
| `src/trdr/strategy/__init__.py` | Import/export from indicators package |
| `src/trdr/strategy/types.py` | Removed VolumeProfile (moved to indicators) |
| `src/trdr/strategy/volume_area_breakout/strategy.py` | Import from `...indicators` |
| `src/trdr/strategy/macd_template/strategy.py` | Import from `...indicators`, removed `_ema` |
| `tests/test_indicators.py` | New - 42 indicator tests |

## Dependencies

- numpy (already used)
- No new dependencies required
