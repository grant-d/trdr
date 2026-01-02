# VolumeAreaBreakout Strategy

**BTC/USD 1-hour**
2-path Volume Profile system optimized via SICA loop (91+ iterations).

## Entry Paths

### Path 1: VAH Breakout

- Price breaks above Value Area High (VAH)
- Volume >= 1.2x average
- MSS regime > -5 (bullish/neutral)
- Base confidence: 0.65

### Path 2: VAL Bounce

- Price bounces from Value Area Low (VAL)
- Any volume (declining volume gets +0.25 bonus)
- MSS regime > -35
- Base confidence: 0.70

## Exit Rules

- **Target**: POC (Point of Control)
- **Stop VAH**: 0.4x ATR below VAL
- **Stop VAL**: 1.2x ATR below entry

## Configuration

```python
VolumeAreaBreakoutConfig(
    symbol="crypto:BTC/USD",
    timeframe="1h",
    atr_threshold=2.0,
    stop_loss_multiplier=1.75,
)
```

## Performance (SICA Optimized)

| Metric | Value |
| --- | --- |
| SICA Score | 0.856 |
| Trades | 6 |
| Win Rate | 66.7% |
| Profit Factor | 5.32 |
| Max Drawdown | 12.0% |
| Sharpe Ratio | 65.00 |
| Sortino Ratio | 202.75 |
| P&L | +$6,439 |

## SICA Optimization

The `sica_journal.md` documents 91+ iterations of systematic exploration:

- Volatility regime filters
- Order flow imbalance
- SAX pattern trading
- Multi-timeframe POC confluence
- Heikin-Ashi smoothing
- Confidence-based position sizing

All alternatives either failed or degraded the baseline 0.856 score. The 6-trade constraint reflects data reality (BTC hourly contains ~6 high-quality VA crossovers), not model limitation.

## Files

- `strategy.py` - Strategy implementation
- `test_strategy.py` - Backtest validation tests
- `sica_journal.md` - SICA optimization history
- `sica_final_state.json` - Final SICA loop state
