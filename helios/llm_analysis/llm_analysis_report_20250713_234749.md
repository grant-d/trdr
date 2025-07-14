# Helios Trading Strategy - LLM Performance Analysis Report
Generated: 2025-07-13 23:47:49

## EXECUTIVE SUMMARY
============================================================
HELIOS TRADER PERFORMANCE REPORT
============================================================

OVERALL PERFORMANCE
------------------------------
Initial Capital:        $100,000.00
Final Portfolio Value:  $132,467.44
Total P&L:             $32,467.44
Total Return:          32.47%
Annualized Return:     1.23%
Geometric Ann. Return: 1.23%

RISK METRICS
------------------------------
Volatility (Annual):   5.81%
Maximum Drawdown:      -34.45%
Sharpe Ratio:          -0.11
Sortino Ratio:         -0.13
Calmar Ratio:          0.04

RETURN STATISTICS
------------------------------
Arithmetic Mean:       0.0055%
Geometric Mean:        0.0048%

TRADING STATISTICS
------------------------------
Total Trades:          1280
Position Cycles:       0
Winning Trades:        631
Losing Trades:         95
Win Rate:              86.91%
Average Win:           6.94%
Average Loss:          -1.83%
Profit Factor:         1.17

============================================================

## STRATEGY CONFIGURATION
### Data Source
- **File**: /Users/grantdickinson/repos/trdr/data/MSFT.csv
- **Initial Capital**: $100,000.00
- **Dollar Bars**: Enabled
- **Dollar Threshold**: $137,367,246

### Optimized Parameters
```json
{
  "weight_trend": 0.5561157695847588,
  "weight_volatility": 0.4585909389448469,
  "weight_exhaustion": 0.9500934401734018,
  "lookback_int": 25.0,
  "stop_loss_multiplier_strong": 2.7,
  "stop_loss_multiplier_weak": 0.8,
  "entry_step_size": 0.7,
  "max_position_pct": 0.8,
  "strong_bull_threshold": 11.4,
  "weak_bull_threshold": 10.0,
  "neutral_threshold_upper": 3.0,
  "neutral_threshold_lower": -6.0,
  "weak_bear_threshold": -9.1,
  "strong_bear_threshold": -20.0
}
```

### Market Regime Thresholds
- **Strong Bull**: MSS ≥ 11.4
- **Weak Bull**: 10.0 ≤ MSS < 11.4
- **Neutral Upper**: 3.0 ≤ MSS < 10.0
- **Neutral Lower**: -6.0 ≤ MSS < 3.0
- **Weak Bear**: -9.1 ≤ MSS < -6.0
- **Strong Bear**: MSS < -20.0

## REGIME-BASED PERFORMANCE ANALYSIS

### Trading Performance by Market Regime

#### Neutral Lower
- **Total Trades**: 1280
- **Win Rate**: 49.3%
- **Total P&L**: $15,401.45
- **Average P&L per Trade**: $12.03
- **Wins**: 631, **Losses**: 649


## DETAILED TRADE LOG ANALYSIS

### Sample Recent Trades (Last 20)
| Timestamp | Action | Units | Price | P&L | Reason | Portfolio Value |
|-----------|--------|-------|-------|-----|--------|----------------|
| 2023-02-16 00:00 | Sell | -207.93 | $262.15 | $887.18 | Stop Loss | $131,842.02 |
| 2023-02-16 00:00 | Buy | 213.62 | $262.15 | N/A | Weak Bull Signal | $131,842.02 |
| 2023-02-17 00:00 | Buy | 3.39 | $258.06 | N/A | Weak Bull Signal | $130,968.32 |
| 2023-02-21 00:00 | Buy | 4.63 | $252.67 | N/A | Weak Bull Signal | $129,798.67 |
| 2023-02-22 00:00 | Buy | 1.02 | $251.51 | N/A | Weak Bull Signal | $129,541.58 |
| 2023-02-23 00:00 | Sell | -2.85 | $254.77 | $-20.15 | Weak Bull Signal | $130,247.29 |
| 2023-02-24 00:00 | Sell | -219.81 | $249.22 | $-2774.36 | Neutral Signal | $126,253.00 |
| 2023-03-03 00:00 | Buy | 219.36 | $255.29 | N/A | Weak Bull Signal | $126,253.00 |
| 2023-03-06 00:00 | Sell | -1.35 | $256.87 | $2.13 | Weak Bull Signal | $126,601.72 |
| 2023-03-07 00:00 | Sell | -218.01 | $254.15 | $-248.53 | Neutral Signal | $125,760.21 |
| 2023-03-14 00:00 | Buy | 214.73 | $260.79 | N/A | Weak Bull Signal | $125,760.21 |
| 2023-03-15 00:00 | Sell | -3.76 | $265.44 | $17.49 | Weak Bull Signal | $126,776.20 |
| 2023-03-16 00:00 | Sell | -8.22 | $276.20 | $126.65 | Weak Bull Signal | $129,172.90 |
| 2023-03-17 00:00 | Sell | -2.34 | $279.43 | $43.69 | Weak Bull Signal | $129,871.47 |
| 2023-03-20 00:00 | Sell | -200.41 | $272.23 | $2292.67 | Stop Loss | $130,721.20 |
| 2023-03-20 00:00 | Buy | 205.71 | $272.23 | N/A | Weak Bull Signal | $130,721.20 |
| 2023-03-21 00:00 | Sell | -1.16 | $273.78 | $1.81 | Weak Bull Signal | $131,041.85 |
| 2023-03-22 00:00 | Buy | 1.12 | $272.29 | N/A | Weak Bull Signal | $130,737.08 |
| 2023-03-23 00:00 | Sell | -3.98 | $277.66 | $21.60 | Weak Bull Signal | $131,863.09 |
| 2023-03-24 00:00 | Sell | -2.09 | $280.57 | $17.45 | Weak Bull Signal | $132,467.44 |


## MARKET CONDITIONS ANALYSIS

### Volatility Analysis
- **High Volatility Threshold**: 8.7590
- **Low Volatility Threshold**: 5.9046
- **Current Volatility Regime**: Medium

### Maximum Drawdown Event
- **Date**: 5773
- **Drawdown**: -34.45%
- **Portfolio Value at Drawdown**: $121,288.22

## KEY PERFORMANCE INDICATORS

### Return Metrics
- **Total Return**: 32.47%
- **Annualized Return**: 1.22%
- **Volatility (Annualized)**: 5.81%

### Risk Metrics
- **Maximum Drawdown**: -34.45%
- **Sharpe Ratio**: 0.24
- **Sortino Ratio**: 0.12

## PROMPTS FOR LLM ANALYSIS

### Performance Review Questions
1. **Regime Analysis**: Which market regime shows the worst performance? What specific pattern in the trade log suggests the strategy is struggling in that regime?

2. **Volatility Impact**: How does the strategy perform during high vs. low volatility periods? Are there specific volatility conditions where the strategy consistently loses money?

3. **Drawdown Analysis**: What market conditions led to the maximum drawdown period? What regime and volatility characteristics were present?

4. **Trade Timing**: Looking at the recent trades, are there patterns in entry/exit timing that suggest systematic issues?

5. **Parameter Optimization**: Based on the regime-specific performance, which parameters (thresholds, stop losses, position sizing) appear most problematic?

### Weakness Identification
Please analyze the above data and identify:
- **Single Biggest Weakness**: The most significant performance issue (e.g., "losing money in high-volatility chop", "poor exits in bear markets")
- **Root Cause Hypothesis**: What specific aspect of the strategy logic causes this weakness?
- **Proposed Rule Change**: A concrete, implementable modification to address the weakness

### Market State Analysis
Focus on these specific scenarios:
- High volatility + neutral/choppy markets
- Regime transitions (bull to bear, bear to bull)
- Extended periods in single regimes
- Volatility breakouts and breakdowns

---

*This report contains all necessary data for comprehensive strategy analysis. Copy the relevant sections to your LLM for detailed performance review and weakness identification.*
