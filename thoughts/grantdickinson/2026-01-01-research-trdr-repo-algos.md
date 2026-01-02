---
date: 2026-01-01
author: grantdickinson
repo: grant-d/trdr
branch: feat/helios-3
git_commit: f9ba9f1bb91ab213ecc07465db00a5f95d4d98ea
repo_url: https://github.com/grantdickinson/trdr
topic: "Trading Algorithm Branch Analysis"
tags: [research, trading, algorithms, genetic-algorithm, options, technical-analysis, backtesting]
---

# Research: Trading Algorithms in `grant-d/trdr` Repository

## Research Question

Analyze all trading algorithm branches in the repository. Document core algorithms, approaches, patterns, architecture, viability, and testing for each branch. Focus on pure code algorithms (exclude LLM-based approaches). Provide sufficient detail for algorithm recreation.

## Summary

The repository contains **6 distinct algorithmic trading approaches** across two main technology stacks (Python and TypeScript). The branches represent an evolution of trading system design, from regime-based equity/crypto strategies to options strategies and data pipeline infrastructure.

| Branch | Stack | Focus | Core Algorithm | Viability |
| --- | --- | --- | --- | --- |
| helios-1 | Python | Regime Trading | MSS + GA Optimization | High |
| helios-2 | Python | WFO Framework | Regime Strategy + DEAP GA | High |
| helios-3 | Python | Multi-Strategy | MA/MSS/RSI + Portfolio Engine | High |
| project-1 | TypeScript | Agent Architecture | Multi-Agent Technical Analysis | Medium |
| project-2 | TypeScript | Data Pipeline | ETL + Feature Engineering | High (Infrastructure) |
| aapl-backtest | Python | Options | HMA Bull Put Spreads | Medium |

---

## Detailed Findings

### Branch 1: feat/helios-1 (Python)

**Purpose:** Adaptive quantitative trading with regime-based position management and genetic optimization.

#### Core Algorithm: Market State Score (MSS)

The MSS combines three normalized factors (-100 to +100 scale) into a single market state indicator:

```text
MSS = w_trend × Trend + w_volatility × Volatility_norm + w_exhaustion × Exhaustion

Where:
  Trend = (linear_regression_slope / price) × 100
  Volatility = ATR / price × 100 → normalized to 100 - (vol × 2)
  Exhaustion = clip((price - SMA) / ATR × 10, -100, 100)
```

**Regime Classification:**

- Strong Bull (MSS > 50): Full long position, 2x ATR stop
- Weak Bull (20-50): Hold longs only, 1x ATR stop
- Neutral (-20 to 20): Exit all positions
- Weak Bear (-50 to -20): Hold shorts only, 1x ATR stop
- Strong Bear (< -50): Full short position, 2x ATR stop

**Position Sizing:** Gradual entry/exit limited by `entry_step_size × max_position_pct` per bar.

#### Dollar Bars

Aggregates tick data by dollar volume (not time) for consistent information content per bar:

```text
accumulate dollar_volume until >= threshold → emit bar
```

#### Genetic Algorithm Optimization

**Parameters (14 total):**

- `w_trend`, `w_volatility`, `w_exhaustion`: 0.0-1.0 (normalized sum = 1.0)
- `trend_lookback`: 10-200 bars
- `volatility_lookback`: 5-100 bars
- `exhaustion_lookback`: 5-100 bars
- `strong_bull_threshold`: 30-80
- `weak_bull_threshold`: 10-50
- `weak_bear_threshold`: -50 to -10
- `strong_bear_threshold`: -80 to -30
- `stop_multiplier_strong`: 1.0-5.0 (ATR multiplier)
- `stop_multiplier_weak`: 0.5-3.0 (ATR multiplier)
- `entry_step_size`: 0.05-0.5 (5-50% of max position per bar)
- `max_position_pct`: 0.5-2.0 (50-200% of capital)

**Fitness Function:**

- Primary metric: Sortino or Calmar ratio
- Win rate penalty: fitness × (win_rate / 0.4) if win_rate < 40%
- Drawdown penalty: fitness × (0.25 / max_dd) if max_dd > 25%

**Genetic Operators:**

- Crossover: Single-point, then enforce threshold ordering and renormalize weights
- Mutation: Gaussian perturbation (10% of param range), clip to bounds, retry if constraints violated
- Walk-Forward: 70/30 train/test split, non-overlapping windows, aggregate results

**Dollar Bars:**

- Accumulate tick dollar volume until threshold reached
- Threshold = median_daily_dollar_volume / target_bars_per_day
- Emit OHLCV bar on threshold, reset accumulator

**Gradual Position Sizing:**

- Max change per bar = step_size × max_position_pct
- Ramp into/out of positions over multiple bars
- Example: 20% step → 5 bars to reach full position

**Key Files:** `helios/factors.py`, `helios/strategy_enhanced.py`, `helios/generic_algorithm.py`

---

### Branch 2: feat/helios-2 (Python)

**Purpose:** Refined regime trading with improved genetic algorithm using DEAP framework.

#### Two-Pass Dynamic MSS

1. **Pass 1:** Calculate static MSS with default weights → classify regime
2. **Pass 2:** Apply regime-specific weight multipliers → recalculate MSS

```text
Regime Weight Multipliers:
  Bull:    trend=1.2, vol=0.8, exhaust=0.8
  Neutral: trend=0.9, vol=1.1, exhaust=1.1
  Bear:    trend=0.7, vol=1.2, exhaust=1.3
```

#### DEAP Genetic Algorithm

- Population: 50 individuals, 100 generations
- Selection: Tournament (size=3)
- Crossover: Uniform (50% gene swap probability)
- Mutation: Gaussian (10% of parameter range)
- Constraint handling: Smart generation + adaptive mutation

#### Multi-Objective Fitness

```text
fitness = 0.35×Sharpe + 0.25×Sortino + 0.20×Calmar
        + 0.10×(transaction_penalty×10) + 0.10×(holding_bonus×10)
```

#### Walk-Forward Optimization (Hybrid Sliding Windows)

- Training windows overlap for continuous refinement
- Test windows non-overlapping to prevent data leakage
- Consensus parameters: median across splits (robust to outliers)

#### Trading Filters (Overtrading Prevention)

1. Minimum position change ≥ 2%
2. Trade cooldown ≥ 2 bars
3. MSS change requirement (regime-specific thresholds)

#### DEAP Parameter Details

**Individual (20 parameters):**

- 3 lookbacks: trend (10-200), volatility (5-100), exhaustion (5-100)
- 3 weights: normalized to sum = 1.0
- 6 thresholds: ordered constraints (strong_bull > weak_bull > neutral > weak_bear > strong_bear)
- 4 position params: stop multipliers, step size, max position
- 4 filter params: min change, cooldown, regime weight multipliers

**GA Configuration:**

- Population: 50, Generations: 100
- Selection: Tournament (size 3)
- Crossover: Uniform (50% swap prob), prob 0.7
- Mutation: Gaussian (10% range), prob 0.3
- Constraint enforcement after each operation

#### WFO Window Strategy

**Window Strategy:**

- Train: 2 years (504 bars), Test: 3 months (63 bars)
- Step: 3 months (sliding)
- Training windows overlap, test windows non-overlapping (no lookahead)

**Consensus Calculation:**

- Run GA on each training window
- Take median of each parameter across all windows
- Re-normalize weights after median

#### Dynamic MSS Recalculation

1. **Pass 1:** Calculate static MSS → classify regime (bull/bear/neutral)
2. **Pass 2:** Apply regime-specific weight multipliers:
   - Bull: trend ×1.2, vol ×0.8, exhaust ×0.8
   - Bear: trend ×0.7, vol ×1.2, exhaust ×1.3
   - Neutral: trend ×0.9, vol ×1.1, exhaust ×1.1
3. Renormalize weights, recalculate MSS

**Key Files:** `genetic_algorithm.py`, `walk_forward_optimizer.py`, `regime_strategy.py`, `indicators.py`

---

### Branch 3: feat/helios-3 (Python)

**Purpose:** Multi-strategy framework with comprehensive portfolio management.

#### Strategy 1: Moving Average Crossover

```text
Signal Generation:
  MA_fast = SMA(close, fast_period)  # 5-50 bars
  MA_slow = SMA(close, slow_period)  # 20-200 bars

  BUY when MA_fast crosses above MA_slow
  SELL when MA_fast crosses below MA_slow
```

#### Strategy 2: Market Structure Shift (MSS)

Enhanced version with 14 optimizable parameters:

- 3 lookback periods (trend, volatility, exhaustion)
- 3 weights (normalized to 1.0)
- 6 regime thresholds
- 2 ATR stop multipliers

#### Strategy 3: RSI Reversal

```text
RSI = 100 - (100 / (1 + avg_gains/avg_losses))

BUY when RSI < oversold_threshold (20-40)
SELL when RSI > overbought_threshold (60-80)
```

#### Portfolio Engine

Full position tracking with:

- Order types: MARKET, LIMIT, STOP, STOP_LIMIT, TRAILING_STOP
- Order sides: BUY, SELL, SHORT, COVER
- Commission/slippage modeling
- Margin requirement tracking
- Mark-to-market updates per bar

#### Data Pipeline

1. Validate structure (required OHLCV columns)
2. Handle missing values (forward fill)
3. Outlier detection (Z-score for prices, IQR for volume)
4. OHLCV integrity (high ≥ low, clamp open/close)
5. Fractional differentiation for stationarity

#### Order Execution Details

**Order Types:**

- MARKET: Fill at open of next bar
- LIMIT: Fill if price reaches limit (better or equal)
- STOP: Trigger at stop price, fill at market
- STOP_LIMIT: Two-stage (stop triggers limit order)
- TRAILING_STOP: Dynamic stop follows favorable price movement

**Order Sides:** BUY, SELL, SHORT, COVER

**Execution Model:**

- Commission: percentage of trade value
- Slippage: percentage of trade value
- Fill simulation: use bar OHLC to determine realistic fills
- Validate cash/margin before accepting orders

**Mark-to-Market:**

- Update unrealized P&L each bar using close price
- Track peak equity for drawdown calculation
- Realized P&L on position close

#### Data Cleaning

**OHLCV Integrity Rules:**

1. High ≥ Low (swap if reversed)
2. High ≥ max(Open, Close)
3. Low ≤ min(Open, Close)
4. No zero/negative prices (forward fill)
5. Volume ≥ 0

**Outlier Detection:**

- Prices: Z-score > 3.0 → replace with median
- Volume: IQR method (1.5× multiplier) → replace with median

**Fractional Differentiation:**

- Order d ∈ (0, 1): balance stationarity vs memory
- Typical d = 0.3-0.5 for financial series
- Weight formula: w[k] = -w[k-1] × (d - k + 1) / k
- Apply weighted sum over lookback window
- Validate with ADF test (p < 0.05 = stationary)

**Key Files:** `ma_strategy.py`, `mss_strategy.py`, `rsi_strategy.py`, `portfolio_engine.py`, `base_data_loader.py`

---

### Branch 4: feat/project-1 (TypeScript)

**Purpose:** Pluggable multi-agent trading system with adaptive parameter adjustment.

#### Agent Architecture

```text
MarketContext → AdaptiveBaseAgent.analyze() → AgentSignal

AdaptiveBaseAgent provides:
- Market regime detection
- Parameter adaptation based on regime
- Confidence adjustment for regime alignment
- Performance tracking
```

#### Market Regime Detection (5 Dimensions)

1. **Trend:** 3-MA comparison (10/20/50 periods) → bullish/bearish/neutral
2. **Volatility:** StdDev + ATR → high/normal/low
3. **Momentum:** ROC + consecutive moves → strong/moderate/weak
4. **Volume:** Recent vs older comparison → increasing/decreasing/stable
5. **Classification:** Trending/Breakout/Reversal/Ranging

#### Bollinger Bands Agent

```text
Bandwidth = (Upper - Lower) / Middle
%B = (Price - Lower) / (Upper - Lower)

Signals (priority order):
1. Squeeze Release: Bandwidth expanding + direction confirmation
2. Band Touches: %B ≤ 0 (buy) or %B ≥ 1 (sell)
3. Squeeze Preparation: Bandwidth in lowest 20%
4. Mean Reversion: %B crosses 0.2 or 0.8
```

Adaptive parameters by regime:

- High volatility: Period × 0.8
- Trending: StdDev × 1.2
- Ranging: StdDev × 0.9

#### MACD Agent

```text
MACD = EMA(12) - EMA(26)
Signal = EMA(MACD, 9)
Histogram = MACD - Signal

Signals (priority order):
1. Divergence: Price vs MACD divergence
2. Zero-Line Crossover: MACD crosses 0
3. Signal-Line Crossover: Histogram sign change
4. Momentum Continuation: Histogram expanding
```

#### RSI Agent

```text
Signals (priority order):
1. Divergence: Price vs RSI divergence
2. Oversold/Overbought: RSI < 30 or > 70
3. Momentum: RSI in 45-55 with direction

Adaptive levels by trend:
- Bullish: Oversold=35, Overbought=75
- Bearish: Oversold=25, Overbought=65
```

#### Momentum Agent (Composite)

Combines RSI + MACD with confluence scoring:

- Counts bullish vs bearish signals
- Weights by divergence detection
- Confluence > 75% = strong signal

#### Volume Profile Agent

```text
Build profile: Price buckets × volume accumulation
POC = Highest volume price level
Support/Resistance = Levels with volume > 70% of POC

Signals:
1. Volume Spikes: Current > 2x average
2. S/R Proximity: Within 0.3% of level
3. Profile Patterns: Accumulation/distribution
```

#### No-Shorting Enforcement

All agents prevent sell signals when no position exists.

#### Regime Detection Detailed Rules

**Trend Classification:**

- Compare MA(10), MA(20), MA(50)
- Bullish: MA(10) > MA(20) > MA(50) AND all slopes positive
- Bearish: MA(10) < MA(20) < MA(50) AND all slopes negative
- Neutral: Mixed ordering or flat slopes

**Volatility Classification:**

- Calculate rolling StdDev(20) and ATR(14)
- High: StdDev > 75th percentile OR ATR > 1.5x median
- Low: StdDev < 25th percentile AND ATR < 0.5x median
- Normal: Otherwise

**Momentum Classification:**

- ROC = (price - price[n_periods_ago]) / price[n_periods_ago] × 100
- Count consecutive up/down bars
- Strong: |ROC| > 5% AND consecutive >= 3
- Moderate: |ROC| > 2% OR consecutive >= 2
- Weak: Otherwise

**Volume Trend:**

- Compare recent volume MA(10) vs older MA(50)
- Increasing: Recent > Older × 1.2
- Decreasing: Recent < Older × 0.8
- Stable: Between thresholds

**Market Classification Logic:**

- Trending: Strong momentum + clear trend direction
- Breakout: High volatility + increasing volume + momentum
- Reversal: Divergence detected OR overbought/oversold
- Ranging: Low volatility + weak momentum + neutral trend

#### Agent Signal Priority and Confidence

**Bollinger Bands Priority Flow:**

1. Check squeeze release (highest priority): Bandwidth expanding after compression
2. Check band touches: Price at extreme bands
3. Check squeeze preparation: Bandwidth contracting
4. Check mean reversion: %B crossing thresholds

**Confidence Calculation:**

- Base confidence from signal strength (0.0-1.0)
- Regime alignment bonus: +0.2 if signal matches regime
- Volume confirmation: +0.1 if volume supports signal
- Divergence detection: +0.15 if present
- Final confidence clamped to [0.0, 1.0]

**Adaptive Parameter Examples:**

- Bollinger period in high volatility: 20 × 0.8 = 16
- Bollinger stdDev in trending: 2.0 × 1.2 = 2.4
- RSI overbought in bullish trend: 70 + 5 = 75

**Key Files:** `packages/cli/src/agents/*.ts`, `market-regime-detector.ts`, `agent-price-utils.ts`

---

### Branch 5: feat/project-2 (TypeScript)

**Purpose:** Data pipeline infrastructure for market data ingestion and feature engineering.

#### Pipeline Architecture

```text
┌────────────────┐     ┌───────────────┐     ┌────────────────┐
│   Providers    │ ──▶ │   Transforms  │ ──▶ │  Repositories  │
│ Alpaca/Coinbase│     │ Indicators/   │     │  CSV/JSONL     │
│ CSV/JSONL      │     │ Normalizers   │     │  Output        │
└────────────────┘     └───────────────┘     └────────────────┘
```

**Execution Modes:**

- Pipeline: Single execution, config-driven
- Server: Continuous with backfill + realtime streaming
- Interactive: REPL (scaffolding)

#### Data Providers

1. **Alpaca:** Stocks + crypto, REST + WebSocket, 200 req/sec
2. **Coinbase:** Crypto only, REST, 10 req/sec with rate limiting
3. **File:** CSV/JSONL with column mapping

#### Transform Pipeline

**Bar Generators:**

- Time/Tick/Volume/Dollar bars (standard)
- **Lorentzian Distance Bars:** Relativistic geometry in price-time-volume space

  ```text
  d = √(c²Δt² - Δp² - Δv²)  # where c is scaling factor
  ```

- Shannon Information Bars: Information-theory boundaries
- Statistical Regime Bars: Change-point detection

**Normalizers:**

- Log Returns: `ln(p[t]/p[t-1])`
- Z-Score: `(x - mean) / std`
- Min-Max: `(x - min) / (max - min)`
- **Fractional Differentiation:** Stationarity while preserving memory

  ```text
  d = 0..1 (fractional order)
  Uses binomial series for weight calculation
  ```

**Technical Indicators:**
SMA, EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, Heikin-Ashi

#### Advanced Bar Types

**Lorentzian Distance Bars:**

- Model price-time-volume as spacetime geometry
- Distance = √(c²Δt² - Δp² - Δv²) where c is scaling factor
- Emit bar when accumulated distance ≥ threshold
- Adapts to volatility and volume; time-symmetric

**Shannon Information Bars:**

- Track entropy of price move distribution
- Information gain = |current_entropy - prev_entropy|
- Emit bar when accumulated info ≥ threshold
- More bars during unpredictable price action

**Statistical Regime Bars (CUSUM):**

- Track cumulative deviations from running mean
- cusum_pos = max(0, cusum + deviation - drift)
- Emit bar when cusum exceeds threshold (regime change)
- Bars align with statistical shifts

#### Fractional Differentiation

- Order d ∈ (0.3, 0.7): balance stationarity vs memory
- Weights: w[k] = -w[k-1] × (d - k + 1) / k
- Apply weighted sum over lookback
- Validate: ADF test p < 0.05

#### Configuration System

JSON-based with Zod validation:

```json
{
  "input": {"type": "provider", "provider": "alpaca", "symbols": ["BTC-USD"]},
  "output": {"path": "./output/btc.csv", "format": "csv"},
  "transformations": [
    {"type": "fractionalDiff", "params": {"d": 0.3}},
    {"type": "rsi", "params": {"window": 14}}
  ]
}
```

**Key Files:** `src/pipeline/buffer-pipeline.ts`, `src/providers/alpaca/`, `src/transforms/`

---

### Branch 6: claude/aapl-backtest-strategy (Python)

**Purpose:** Options income strategy using Hull Moving Average for trend confirmation.

#### Hull Moving Average (HMA)

```text
HMA(n) = WMA(2 × WMA(n/2) - WMA(n), √n)

Trend Signal:
  HMA_UP = current_HMA > HMA[3_periods_ago]
```

Entry requires BOTH HMA-50 AND HMA-200 trending up.

#### Bull Put Spread Strategy

```text
Components:
  Short Put: -0.30 delta (30% OTM), higher strike
  Long Put: $5 below short strike (protection)

Net Credit = Short premium - Long premium
Max Profit = Net credit × 100
Max Loss = (Spread width × 100) - Max profit
```

#### Black-Scholes Implementation

```text
d1 = [ln(S/K) + (r + σ²/2)×T] / (σ×√T)
d2 = d1 - σ×√T

Put Price = K×e^(-rT)×N(-d2) - S×N(-d1)
Put Delta = N(d1) - 1
```

#### Strike Selection (Binary Search)

```text
Target: -0.30 delta put
Search range: 70% to 99% of spot price
Tolerance: 0.01 delta
Return: Strike matching target delta
```

#### Entry/Exit Rules

**Entry Conditions (ALL required):**

- HMA_50_UP = True
- HMA_200_UP = True
- Open positions < 5
- Net credit > 0

**Exit Conditions (ANY triggers):**

- Profit Target: P&L ≥ 50% of max profit
- Stop Loss: P&L ≤ -200% of net credit
- Trend Reversal: Either HMA turns down
- Expiration: 30 DTE reached

#### Risk Management

- Max 5 concurrent positions
- $5 spread width caps loss at ~$350/spread
- Stop loss at 2x credit limits realized loss
- Trend filter exits on momentum breakdown

#### WMA and HMA Formulas

**WMA:** weights[i] = (n - i), WMA = Σ(price × weight) / Σ(weights)

**HMA:** Reduces lag while maintaining smoothness

1. wma_half = WMA(n/2)
2. wma_full = WMA(n)
3. raw = 2 × wma_half - wma_full
4. HMA = WMA(raw, √n)

**Trend:** Compare current HMA to HMA[3 bars ago]

#### Black-Scholes

**Formulas:**

- d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
- d2 = d1 - σ√T
- Put = Ke^(-rT)N(-d2) - SN(-d1)
- Delta = N(d1) - 1

**Inputs:** S=spot, K=strike, r=risk-free (0.02), σ=volatility, T=time in years

#### Strike Selection

Binary search for target delta (-0.30):

- Search range: 70-99% of spot
- Tolerance: ±0.01 delta
- Result: ~10% OTM put for 30-delta

#### Spread Construction

**Bull Put Spread:**

- Short: 30-delta put (higher strike)
- Long: $5 below short strike
- Net credit = short premium - long premium
- Max profit = credit × 100
- Max loss = (spread width - credit) × 100

#### Strategy Logic

**Entry:** HMA-50 up AND HMA-200 up AND positions < 5 AND credit > 0

**Exit (any triggers):**

1. Profit target: P&L ≥ 50% max profit
2. Stop loss: P&L ≤ -200% credit
3. Trend reversal: Either HMA turns down
4. Expiration: DTE ≤ 30

**Key Files:** `aapl_hma_backtest/strategy/hma_bull_put_strategy.py`, `aapl_hma_backtest/indicators/hma.py`, `aapl_hma_backtest/options/black_scholes.py`

---

## Architecture Documentation

### Common Patterns Across Branches

1. **Factor-Based Regime Detection:** helios-1/2/3 all use weighted factor combinations
2. **Genetic Optimization:** helios-1/2 use GA with walk-forward validation
3. **Adaptive Parameters:** project-1 agents adjust based on detected market regime
4. **Data Pipeline Separation:** project-2 isolates ETL from strategy logic
5. **Position Management:** All branches track positions with stop-loss management

### Technology Stack Comparison

| Aspect | Python (helios/aapl) | TypeScript (project) |
| --- | --- | --- |
| Optimization | DEAP GA, scipy | N/A (agents are rule-based) |
| Data Source | Alpaca, CSV | Alpaca, Coinbase, CSV |
| Backtesting | Custom portfolio engine | PortfolioEngine class |
| Configuration | JSON/Pydantic | JSON/Zod |
| Indicators | NumPy/Pandas | Custom implementations |

---

## Viability Assessment

### High Viability

**helios-1/2/3:**

- Sound theoretical basis (regime detection, factor models)
- Robust optimization (walk-forward prevents overfitting)
- Comprehensive risk management
- Well-tested components

**project-2:**

- Production-grade data infrastructure
- Flexible provider/transform architecture
- Proper error handling and retries

### Medium Viability

**project-1:**

- Good agent abstraction but no optimization layer
- Regime detection adds value but parameters are static defaults
- Would benefit from backtesting integration

**aapl-backtest:**

- Options-specific (limited to equities with options)
- Relies on accurate volatility estimation
- Single-asset focus limits diversification

---

## Algorithm Recreation Guide

To recreate any algorithm, implement in order:

### For Regime-Based Equity/Crypto (helios)

**Phase 1: Data Infrastructure**

1. OHLCV data ingestion with validation
2. Dollar bar aggregation (tick → dollar volume bars)
3. Data cleaning (missing values, outliers, integrity checks)
4. Fractional differentiation for stationarity (optional)

**Phase 2: Factor System**
5. Linear regression trend calculation

- Fit regression over lookback window
- Normalize slope by price
- Scale to [-100, 100]

1. Volatility factor
   - ATR calculation with lookback
   - Normalize by price
   - Invert and scale to [0, 100]

2. Exhaustion factor
   - Price deviation from SMA
   - Normalize by ATR
   - Clip to [-100, 100]

**Phase 3: MSS and Regime Detection**
8. Weighted MSS combination with normalization constraint
9. Regime classification with threshold ordering
10. Two-pass dynamic MSS (helios-2 only)

**Phase 4: Position Management**
11. Gradual position sizing with step limits
12. ATR-based stop loss with regime-specific multipliers
13. Entry/exit filters (min change, cooldown, MSS delta)

**Phase 5: Optimization**
14. DEAP genetic algorithm setup

- Individual representation with constraints
- Smart initialization honoring dependencies
- Tournament selection (size 3)
- Uniform crossover with weight renormalization
- Gaussian mutation with constraint enforcement

1. Walk-forward optimization

- Sliding window generation (overlapping train, non-overlapping test)
- Consensus parameter calculation (median aggregation)
- Out-of-sample validation

**Phase 6: Backtesting**
16. Portfolio simulation with realistic fills
17. Commission and slippage modeling
18. Performance metrics (Sharpe, Sortino, Calmar, win rate, drawdown)
19. Equity curve and trade journal

### For Multi-Agent System (project-1)

**Phase 1: Indicators**

1. SMA, EMA calculations (exponential smoothing)
2. Bollinger Bands (middle, upper, lower, bandwidth, %B)
3. MACD (fast EMA, slow EMA, signal line, histogram)
4. RSI (gain/loss averaging, 0-100 scale)
5. ATR (true range averaging)
6. Volume profile (price level bucketing, POC)

**Phase 2: Regime Detection**
7. Trend classification (3-MA comparison with slope)
8. Volatility regime (StdDev + ATR percentiles)
9. Momentum strength (ROC + consecutive bars)
10. Volume trend (recent vs older MA comparison)
11. Market classification (trending/breakout/reversal/ranging)

**Phase 3: Base Agent Framework**
12. Agent interface: analyze() → AgentSignal
13. Adaptive parameter system (regime-based multipliers)
14. Confidence scoring with bonuses
15. Performance tracking per agent

**Phase 4: Specialized Agents**
16. Bollinger Bands agent (squeeze, touches, reversions)
17. MACD agent (divergence, crossovers, momentum)
18. RSI agent (divergence, extremes, momentum)
19. Momentum composite (RSI + MACD confluence)
20. Volume profile agent (spikes, S/R, patterns)

**Phase 5: Signal Aggregation**
21. Priority-based signal selection
22. Regime alignment filtering
23. Position sizing (Kelly-inspired)
24. No-shorting enforcement

### For Options Strategy (aapl-backtest)

**Phase 1: Indicators**

1. WMA calculation (weighted by recency)
2. HMA calculation (nested WMA with sqrt smoothing)
3. Trend detection (slope over lookback)

**Phase 2: Options Pricing**
4. Black-Scholes formula implementation

- d1/d2 calculation
- Normal CDF (cumulative distribution)
- Put price and delta

1. Implied volatility estimation
   - Historical volatility (30-day rolling StdDev)
   - Annualization factor (√252)

**Phase 3: Strike Selection**
6. Binary search for target delta

- Initialize bounds (70%-99% of spot)
- Iterative refinement (tolerance 0.01 delta)
- Return closest strike

**Phase 4: Spread Construction**
7. 30-delta short put selection
8. Long put $5 below short strike
9. Spread metrics (credit, max profit, max loss, breakeven)
10. Risk validation (min credit, max risk/reward ratio)

**Phase 5: Strategy Logic**
11. Entry conditions (dual HMA + position limit + credit check)
12. Position tracking (DTE countdown, mark-to-market)
13. Exit conditions (profit target OR stop loss OR trend reversal OR expiration)
14. Portfolio management (max 5 concurrent positions)

**Phase 6: Backtesting**
15. Daily option pricing updates
16. P&L calculation per spread
17. Trade journal (entries, exits, reasons)
18. Performance metrics (win rate, average P&L, max drawdown)

---

## Performance Metrics Reference

### Risk-Adjusted Returns

| Metric | Formula | Good | Excellent |
| --- | --- | --- | --- |
| Sharpe | (Return - Rf) / StdDev | > 1.0 | > 2.0 |
| Sortino | (Return - Rf) / DownsideStdDev | > 1.5 | > 2.0 |
| Calmar | AnnualReturn / MaxDrawdown | > 1.0 | > 3.0 |

### Trade Statistics

| Metric | Formula | Typical Range |
| --- | --- | --- |
| Win Rate | Wins / Total Trades | 35-65% |
| Profit Factor | Gross Profits / Gross Losses | > 1.5 |
| Expectancy | (WinRate × AvgWin) - (LossRate × AvgLoss) | > 0 |

### Drawdown Thresholds

- Conservative: < 15%
- Moderate: 15-25%
- Aggressive: 25-40%
- High risk: > 40%

### Risk Metrics

- **VaR (95%):** 5th percentile of daily returns
- **CVaR:** Mean of returns below VaR (tail risk)

### Transaction Costs

- Commission drag should be < 5-10% of P&L
- Slippage models: fixed % (0.05-0.10%), volatility-based (0.5×ATR), or volume-based

### helios Multi-Objective Fitness

fitness = 0.35×Sharpe + 0.25×Sortino + 0.20×Calmar + penalties

Penalties: win rate < 40%, max drawdown > 25%

### Options Metrics

- Premium capture rate: > 50%
- Assignment risk: < 5%
- Theta contribution: > 60% of profits

---

## Open Questions

1. **Live Trading:** None of the branches show production live trading integration
2. **Multi-Asset:** helios-1 PRD mentions multi-instrument but implementation is single-asset
3. **Transaction Costs:** Varying accuracy across branches (some use fixed rates)
4. **Execution Slippage:** Minimal modeling in most branches
5. **Market Impact:** Not addressed in any branch

---

## Testing and Validation

### Unit Testing

**Indicators:**

- Known input/output pairs (verify against TA-Lib)
- Edge cases: empty, single value, NaN, lookback > series length
- Example SMA test: [10,20,30,40,50] period=3 → [NaN,NaN,20,30,40]

**Factors (helios):**

- Test MSS components independently
- Verify weight sum = 1.0 after all operations
- Integration test with fixed params against hand calculation

**Genetic Algorithm:**

- Constraint preservation after crossover/mutation
- Weight normalization maintained
- Selection pressure: tournament selects higher fitness
- Population diversity tracked

**Options Pricing:**

- Compare Black-Scholes against QuantLib/vollib
- Verify put-call parity: C - P = S - Ke^(-rT)
- Binary search converges in < 20 iterations

### Integration Testing

**Backtest Scenarios:**

1. Trending market → regime detection captures trend
2. Ranging market → neutral regime classification
3. High volatility → position sizing adjusts
4. Drawdown → stop losses trigger correctly

**Walk-Forward Checks:**

- Train/test windows don't overlap in test periods
- Parameters stable across windows
- Red flags: in-sample Sharpe > 3 but out-of-sample < 1, parameter swings

### Statistical Validation

**Stationarity:** ADF p < 0.05, KPSS p > 0.05

**Overfitting Indicators:**

- In-sample Sharpe > 3.0, out-of-sample < 1.0
- Win rate drops > 20% from train to test
- > 20 optimizable parameters
- Mitigation: regularization, fewer params, cross-validation

**Regime Detection:** Target > 70% accuracy vs manual labels

### Benchmarks

Compare against:

1. Buy-and-hold (baseline)
2. 60/40 stock/bond portfolio
3. Simple MA crossover
4. Random entry/exit

Requirements: Sharpe > benchmark, max drawdown < benchmark, Calmar > 1.0

### Robustness

**Parameter Sensitivity:**

- Vary each parameter ±20%
- Fragile if > 50% performance degradation

**Symbol Robustness:** Same params, similar performance across assets

**Time Period Robustness:** Test across 2008 crisis, 2020 COVID, 2022 rates

### Agent Testing (project-1)

**Signal Quality:**

- Precision: % buy signals → profitable trades
- Recall: % profitable moves caught
- F1 Score target: > 0.5 per agent

**Confluence:** Win rate increases with multi-agent agreement

- 1 agent: ~50%, 2 agents: ~60%, 3+ agents: ~70%

### Data Quality

**OHLCV Integrity:**

- High ≥ Low (100% of bars)
- High ≥ max(Open, Close)
- Low ≤ min(Open, Close)
- Volume ≥ 0, no zero/negative prices

**Suspicious Patterns:** Identical OHLC, volume > 10× avg, gaps > 20%

**Missing Data:** Forward fill short gaps, interpolate medium, drop/warn long

---

## References

### Branch Files

- helios-1: `helios/*.py`
- helios-2: `*.py` (root level)
- helios-3: `*.py` (root level) + `tests/`
- project-1: `packages/cli/src/agents/*.ts`
- project-2: `src/**/*.ts`
- aapl-backtest: `aapl_hma_backtest/**/*.py`

### External References

- DEAP Documentation: <https://deap.readthedocs.io>
- Black-Scholes: Hull, Options, Futures, and Other Derivatives
- Fractional Differentiation: de Prado, Advances in Financial Machine Learning
