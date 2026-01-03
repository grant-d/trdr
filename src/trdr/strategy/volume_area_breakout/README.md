# VolumeAreaBreakout Strategy

Timeframe-aware Volume Profile strategy with VAH breakout entries.
Optimized via SICA across multiple symbol/timeframe combinations.

## Entry Paths

### Daily Timeframe (1d)

**VAH Breakout:**
- Price breaks above Value Area High (VAH)
- HMA slope positive (trend confirmation)
- MSS regime > 0 (bullish)
- Base confidence: 0.65

**VAH Pullback:**
- Price near VAH after prior breakout
- HMA bullish and trending up
- MSS > 5
- Base confidence: 0.60

### Intraday Timeframes (1h, 4h)

**VAH Breakout:**
- Price breaks above VAH
- Volume >= 1.0x average
- MSS > 5, price > HMA
- Base confidence: 0.65

**VAL Bounce:**
- Price bounces from Value Area Low (VAL)
- Volume >= 1.0x average
- MSS > 0
- Base confidence: 0.70
- Declining volume bonus: +0.25

## Exit Rules

**Daily:**
- Target: VAH + 10x ATR (trend-following)
- Stop: VAH - 1x ATR

**Intraday:**
- Target: VAH + 3x ATR (breakout) or POC (bounce)
- Stop: VAH - 0.3x ATR (breakout) or entry - 2x ATR (bounce)

## Indicators

- **Volume Profile**: VAH, VAL, POC from 40-bucket distribution
- **HMA**: Hull Moving Average (period 9) with slope detection
- **MSS**: Market Structure Score (-100 to +100) for regime
- **ATR**: 14-period Average True Range
- **Multi-TF POC**: Confluence across 1x, 4x, 12x aggregations
- **HVN Strength**: Historical support validation at VAL

## Configuration

```python
VolumeAreaBreakoutConfig(
    symbol="stock:AAPL",  # or "crypto:BTC/USD"
    timeframe="1d",       # 1d, 4h, 1h
    atr_threshold=2.0,
    stop_loss_multiplier=1.75,
)
```

## Performance (AAPL 1d, SICA Run 4)

| Metric | Value |
| --- | --- |
| SICA Score | 0.73 |
| Trades | 20 |
| Win Rate | 50% |
| Profit Factor | 7.86 |
| Max Drawdown | 6.4% |
| Sortino | 1498 |
| P&L | +$15,280 |
| Alpha | 1.40x buy-hold |

## SICA Optimization

100+ iterations across 4 runs. Structural ceiling at 73% due to:
- 10 VAH breakouts in AAPL daily data (limited opportunities)
- Strategy already captures all high-quality setups

Tested variations (all regressed or neutral):
- Volume spike entries
- Volatility regime filters
- Order flow imbalance
- SAX pattern trading
- Multi-timeframe POC confluence
- Heikin-Ashi smoothing

## Files

- `strategy.py` - Strategy implementation
- `sica_bench.py` - SICA benchmark runner
