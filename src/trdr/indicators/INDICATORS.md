# Indicators API

Technical indicators for strategy development. All accept `bars: list[Bar]` first.

## Quick Start

**Recommended: Stateful Indicators** (for performance)

```python
from trdr.indicators import AtrIndicator, RsiIndicator

# Initialize once
atr_ind = AtrIndicator(period=14)
rsi_ind = RsiIndicator(period=14)

# Update with each bar
for bar in bars:
    atr_val = atr_ind.update(bar)
    rsi_val = rsi_ind.update(bar)
```

**Stateless: `calculate`** (convenience, slower)

```python
from trdr.indicators import AtrIndicator, RsiIndicator, VolumeProfileIndicator

atr_val = AtrIndicator.calculate(bars, period=14)
rsi_val = RsiIndicator.calculate(bars, period=14)
profile = VolumeProfileIndicator.calculate(bars)
```

## Moving Averages

| Indicator Class | Static Method | Signature | Returns |
| --- | --- | --- | --- |
| `SmaIndicator(period)` | `SmaIndicator.calculate` | `(bars, period)` | Current SMA |
| `EmaIndicator(period)` | `EmaIndicator.calculate` | `(bars, period)` | Current EMA |
| - | `ema_series` | `(values: list[float], period)` | Full EMA list |
| `WmaIndicator(period)` | `WmaIndicator.calculate` | `(bars, period)` | Weighted MA |
| `HmaIndicator(period=9)` | `HmaIndicator.calculate` | `(bars, period=9)` | Hull MA |
| `HmaSlopeIndicator(period=9, lookback=3)` | `HmaSlopeIndicator.calculate` | `(bars, period=9, lookback=3)` | HMA slope |

## Volatility

| Indicator Class | Static Method | Signature | Returns |
| --- | --- | --- | --- |
| `AtrIndicator(period=14)` | `AtrIndicator.calculate` | `(bars, period=14)` | Average True Range |
| `BollingerBandsIndicator(period=20, std_mult=2.0)` | `BollingerBandsIndicator.calculate` | `(bars, period=20, std_mult=2.0)` | `(upper, middle, lower)` |
| `VolatilityRegimeIndicator(lookback=50)` | `VolatilityRegimeIndicator.calculate` | `(bars, lookback=50)` | `"low"`, `"medium"`, `"high"` |
| `SupertrendIndicator(period=10, multiplier=3.0)` | `SupertrendIndicator.calculate` | `(bars, period=10, multiplier=3.0)` | `(value, direction)` |
| `AdaptiveSupertrendIndicator(...)` | `AdaptiveSupertrendIndicator.calculate` | `(bars, ...)` | `(value, cluster)` |
| `RviIndicator(period=10, mode="ema")` | `RviIndicator.calculate` | `(bars, period=10, mode="ema")` | Relative Volatility Index |
| `WilliamsVixFixIndicator(...)` | `WilliamsVixFixIndicator.calculate` | `(bars, ...)` | `(wvf_value, alert_state)` |

## Momentum

| Indicator Class | Static Method | Signature | Returns |
| --- | --- | --- | --- |
| `RsiIndicator(period=14)` | `RsiIndicator.calculate` | `(bars, period=14)` | 0-100 |
| `MacdIndicator(fast=12, slow=26, signal=9)` | `MacdIndicator.calculate` | `(bars, fast=12, slow=26, signal=9)` | `(macd_line, signal_line, histogram)` |
| `MssIndicator(lookback=20)` | `MssIndicator.calculate` | `(bars, lookback=20)` | -100 to +100 (Market Structure Score) |
| `CciIndicator(period=20)` | `CciIndicator.calculate` | `(bars, period=20)` | Commodity Channel Index |
| `AdxIndicator(period=14)` | `AdxIndicator.calculate` | `(bars, period=14)` | Average Directional Index |
| `LaguerreRsiIndicator(gamma=0.5)` | `LaguerreRsiIndicator.calculate` | `(bars, gamma=0.5)` | Laguerre RSI |
| `SmiIndicator(k=10, d=3)` | `SmiIndicator.calculate` | `(bars, k=10, d=3)` | Stochastic Momentum Index |

## Volume

| Indicator Class | Static Method | Signature | Returns |
| --- | --- | --- | --- |
| `VolumeProfile(num_levels=40, value_area_pct=0.70)` | `VolumeProfileIndicator.calculate` | `(bars, num_levels=40, value_area_pct=0.70)` | `VolumeProfile` |
| `VolumeTrendIndicator(lookback=5)` | `VolumeTrendIndicator.calculate` | `(bars, lookback=5)` | `"increasing"`, `"declining"`, `"neutral"` |
| `OrderFlowImbalanceIndicator(lookback=5)` | `OrderFlowImbalanceIndicator.calculate` | `(bars, lookback=5)` | -0.5 to +0.5 |
| `MultiTimeframePocIndicator()` | `MultiTimeframePocIndicator.calculate` | `(bars)` | `(poc_tf1, poc_tf2, poc_tf3)` |
| `HvnSupportStrengthIndicator(lookback=30)` | `HvnSupportStrengthIndicator.calculate` | `(bars, val_level, lookback=30)` | 0.0-1.0 |

## Pattern Recognition

| Indicator Class | Static Method | Signature | Returns |
| --- | --- | --- | --- |
| `SaxPatternIndicator(window=20, segments=5)` | `SaxPatternIndicator.calculate` | `(bars, window=20, segments=5)` | Pattern string `"aabcd"` |
| - | `sax_bullish_reversal` | `(pattern: str)` | `bool` |
| `HeikinAshiIndicator()` | `HeikinAshiIndicator.calculate` | `(bars)` | List of HA dicts |
| `MssIndicator(lookback=20)` | `MssIndicator.calculate` | `(bars, lookback=20)` | Market Structure Score |

## Machine Learning

| Indicator Class | Static Method | Signature | Returns |
| --- | --- | --- | --- |
| `KalmanIndicator()` | `KalmanIndicator.calculate` | `(bars)` | Kalman filtered price |
| - | `kalman_series` | `(bars)` | Full Kalman series |
| `LorentzianClassifierIndicator(neighbors=8, max_bars_back=2000)` | `LorentzianClassifierIndicator.calculate` | `(bars, ...)` | 1 (long), -1 (short), 0 (neutral) |
| `SqueezeMomentumIndicator(...)` | `SqueezeMomentumIndicator.calculate` | `(bars, ...)` | `(momentum, squeeze_state)` |

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

**Recommended: Stateful Indicators** (initialize in `__init__`, update in `generate_signal`)

```python
from ...indicators import AtrIndicator, RsiIndicator, VolumeProfileIndicator

class MyStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.atr_ind = AtrIndicator(period=14)
        self.rsi_ind = RsiIndicator(period=14)
        self.vol_profile = VolumeProfileIndicator()

    def generate_signal(self, bars, position):
        # Update indicators with latest bar
        atr_val = self.atr_ind.update(bars[-1])
        rsi_val = self.rsi_ind.update(bars[-1])
        for bar in bars[-50:]:
            self.vol_profile.update(bar)
        profile = self.vol_profile.value

        near_val = abs(bars[-1].close - profile.val) < atr_val * 0.5
        oversold = rsi_val < 30

        if near_val and oversold:
            return Signal(action=SignalAction.BUY, ...)
```

**Stateless: `calculate`** (simpler but recomputes on every call)

```python
from ...indicators import AtrIndicator, RsiIndicator, VolumeProfileIndicator

class MyStrategy(BaseStrategy):
    def generate_signal(self, bars, position):
        atr_val = AtrIndicator.calculate(bars, 14)
        rsi_val = RsiIndicator.calculate(bars, 14)
        profile = VolumeProfileIndicator.calculate(bars[-50:])

        near_val = abs(bars[-1].close - profile.val) < atr_val * 0.5
        oversold = rsi_val < 30

        if near_val and oversold:
            return Signal(action=SignalAction.BUY, ...)
```
