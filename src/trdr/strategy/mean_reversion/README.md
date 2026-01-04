# MeanReversion Strategy

Adaptive regime strategy that switches between momentum and mean reversion based on market conditions.

## Key Insight

Mean reversion fails in trending markets. This strategy:

1. Detects market regime (trending vs ranging) using ADX and slope
2. Trends: Follow momentum (buy strength, not weakness)
3. Ranges: Mean revert (buy oversold, sell overbought)

## Entry Paths

### Breakout Entry

- Price breaks above N-day high (breakout_period=10)
- Volume surge >= 1.6x average
- 30-day trend gain > 7% (bullish confirmation)
- Stop: VAL - 0.05 ATR
- Target: VAH + 20 ATR (let winners run)

### VAH Breakout

- Price crosses above VAH with volume surge
- Stop: VAH - 0.3 ATR
- Target: Entry + 4 ATR

### POC Mean Reversion

- Price oversold (2.0 ATR below VAL)
- Declining volume (< 0.8x average)
- Stop: Entry - 1.0 ATR
- Target: POC + 20 ATR

## Exit Rules

- Trailing stop: 3.0 ATR (protects profits)
- Max holding period: 60 days
- Time stop exits at neutral

## Configuration

```python
MeanReversionConfig(
    symbol="crypto:BTC/USD",
    timeframe="1d",
    lookback=1000,  # bars (or use Duration: "3y")

    # Breakout settings
    breakout_period=10,
    volume_multiplier=1.6,

    # Risk management
    stop_loss_atr_mult=2.0,
    trailing_stop_atr_mult=3.0,
    max_holding_days=60,

    # Position sizing
    base_position_pct=0.5,
)
```

## Research Basis

- BTC daily returns show AR(1) coefficient of -0.1203 (reversal tendency)
- Mean reversion Sharpe ~2.3 post-2021 in ranging markets
- Trend-following essential during strong trends
- Turn-of-month calendar effects documented in equity markets

## Files

- `strategy.py` - Strategy implementation
- `sica_bench.py` - SICA benchmark runner
