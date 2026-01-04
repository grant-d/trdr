# Indicators API

Technical indicators for strategy development. All accept `bars: list[Bar]` first.

## Quick Start

```python
from trdr.indicators import atr, rsi, volume_profile, ema_series

atr_val = atr(bars, period=14)
rsi_val = rsi(bars, period=14)
profile = volume_profile(bars)
```

## Moving Averages

| Function | Signature | Returns |
| --- | --- | --- |
| `sma` | `(bars, period)` | Current SMA |
| `ema` | `(bars, period)` | Current EMA |
| `ema_series` | `(values: list[float], period)` | Full EMA list |
| `wma` | `(bars, period)` | Weighted MA |
| `hma` | `(bars, period=9)` | Hull MA |
| `hma_slope` | `(bars, period=9, lookback=3)` | HMA slope |

## Volatility

| Function | Signature | Returns |
| --- | --- | --- |
| `atr` | `(bars, period=14)` | Average True Range |
| `bollinger_bands` | `(bars, period=20, std_mult=2.0)` | `(upper, middle, lower)` |
| `volatility_regime` | `(bars, lookback=50)` | `"low"`, `"medium"`, `"high"` |

## Momentum

| Function | Signature | Returns |
| --- | --- | --- |
| `rsi` | `(bars, period=14)` | 0-100 |
| `macd` | `(bars, fast=12, slow=26, signal=9)` | `(macd_line, signal_line, histogram)` |
| `mss` | `(bars, lookback=20)` | -100 to +100 (Market Structure Score) |

## Volume

| Function | Signature | Returns |
| --- | --- | --- |
| `volume_profile` | `(bars, num_levels=40, value_area_pct=0.70)` | `VolumeProfile` |
| `volume_trend` | `(bars, lookback=5)` | `"increasing"`, `"declining"`, `"neutral"` |
| `order_flow_imbalance` | `(bars, lookback=5)` | -0.5 to +0.5 |
| `multi_timeframe_poc` | `(bars)` | `(poc_tf1, poc_tf2, poc_tf3)` |
| `hvn_support_strength` | `(bars, val_level, lookback=30)` | 0.0-1.0 |

## Pattern Recognition

| Function | Signature | Returns |
| --- | --- | --- |
| `sax_pattern` | `(bars, window=20, segments=5)` | Pattern string `"aabcd"` |
| `sax_bullish_reversal` | `(pattern: str)` | `bool` |
| `heikin_ashi` | `(bars)` | List of HA dicts |

## VolumeProfile

```python
@dataclass
class VolumeProfile:
    poc: float           # Point of Control
    vah: float           # Value Area High
    val: float           # Value Area Low
    hvns: list[float]    # High Volume Nodes
    lvns: list[float]    # Low Volume Nodes
    price_levels: list[float]
    volumes: list[float]
    total_volume: float
```

## Edge Cases

All indicators return neutral defaults on insufficient data:

| Type | Default |
| --- | --- |
| `float` | `0.0` or last close |
| RSI | `50.0` |
| Regime strings | `"neutral"` / `"medium"` |
| Tuples | `(0.0, 0.0, 0.0)` |

## Usage in Strategy

```python
from ...indicators import atr, rsi, volume_profile

class MyStrategy(BaseStrategy):
    def generate_signal(self, bars, position):
        atr_val = atr(bars, 14)
        rsi_val = rsi(bars, 14)
        profile = volume_profile(bars[-50:])

        near_val = abs(bars[-1].close - profile.val) < atr_val * 0.5
        oversold = rsi_val < 30

        if near_val and oversold:
            return Signal(action=SignalAction.BUY, ...)
```
