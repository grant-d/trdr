---
date: 2026-01-01
author: grantdickinson
git_commit: d36e895490ad612c1cc4b7a5f2c6e3b8c73930b2
branch: main
repository: git@github.com:grant-d/trdr.git
topic: "Combined Trading Algorithm Analysis: grant-d/trading + grant-d/trdr Repositories"
tags: [research, trading, algorithms, combined-analysis, cross-repository, genetic-algorithm, options, sax-patterns, regime-detection, volume-profile, microstructure, futures, order-flow, self-improving-agents]
source_documents:
  - 2026-01-01-research-trading-repo-algos.md (grant-d/trading)
  - 2026-01-01-research-trdr-repo-algos.md (grant-d/trdr)
external_research:
  - Autonomous Volume Profile Trading System: <https://www.perplexity.ai/search/how-to-trade-volume-profile-au-gTioI42TRJusvTwF3LJYTA#0>
  - Innovative Systematic Long-Only Trading Frameworks for Genuine Alpha: <https://www.perplexity.ai/search/you-are-an-expert-technical-tr-LyEiNQVWSTO4oQ7Ppd0IEA#12>
  - Claude Conversation (additional research): <https://claude.ai/share/4ff02848-8563-4d5d-ab75-4a9798d1fb40>
  - "SICA: Self-Improving Coding Agent" (arXiv): <https://arxiv.org/html/2504.15228v2>
  - "Darwin Gödel Machine" (arXiv): <https://arxiv.org/html/2505.22954v2>
---

# Research: Combined Trading Algorithm Analysis Across Two Repositories

## Research Question

Comprehensive cross-repository analysis of all trading algorithms across `grant-d/trading` (4 branches) and `grant-d/trdr` (6 branches). Document core algorithms, approaches, patterns, architecture, viability, and testing for each branch with clear pedigree tracking.

## System Requirements

Target: Build a production trading system with the following characteristics:

- **Timeframe:** Swing trading (hours to days, not minutes)
- **Architecture:** Robust, maintainable, clean code patterns
- **Pluggable:** Easy to swap algorithms in/out
- **Adaptive:** Handles regime change (pause or adapt)
- **Testable:** Backtesting with walk-forward epochs
- **Technical:** No fundamentals data dependency
- **Multi-Asset:** Stocks, options, crypto with different providers
- **Order Book:** WebSocket depth data support
- **Always-On:** 24/7 bot, market-hours aware
- **Stateful:** Restart recovery via local database
- **Controllable:** Interactive CLI for operation (Ink/Yoga like ClaudeCode CLI)
- **Profitable:** Alpha generation, good Sortino, controlled drawdown
- **Observable:** Capital, drawdown, trades monitoring
- **Paper Trade:** Emulated exchange if broker lacks support
- **Efficient:** Disk caching for historical data
- **Multi-Timeframe:** Analyze multiple timeframes for signals
- **Realistic:** Sound principles (PoCs, arXiv research, not wet-dream abstraction)
- **Modern:** Beyond legacy indicators (no MACD)
- **Simple:** Runs on local Mac, Python/Node
- **Latest Tools:** uv, bun, modern dev tooling
- **Libraries:** Use maintained packages, avoid reinventing

## Summary

Analysis spans **10 distinct algorithmic trading approaches** across two repositories and two technology stacks:

### grant-d/trading Repository (4 branches)

**Source:** `2026-01-01-research-trading-repo-algos.md`

| Branch | Core Algorithm | Status | Key Metric |
| --- | --- | --- | --- |
| MAIN | Adaptive Martingale + Multi-TF | Production | Multi-layer risk controls |
| RSKR | SAX Pattern + ML | Production | 65.62% return (1-year) |
| feat/ellipse | SAX + Ellipse Geometry | Production + Research | 1737% return (backtested) |
| feat/perplexity-ideas | SAX + ML Ensemble | Production + Ideation | 1737% return + 18 concepts |

### grant-d/trdr Repository (6 branches)

**Source:** `2026-01-01-research-trdr-repo-algos.md`

| Branch | Stack | Core Algorithm | Viability |
| --- | --- | --- | --- |
| helios-1 | Python | MSS + GA Optimization | High |
| helios-2 | Python | MSS + DEAP GA + WFO | High |
| helios-3 | Python | Multi-Strategy Portfolio | High |
| project-1 | TypeScript | Multi-Agent Technical Analysis | Medium |
| project-2 | TypeScript | Data Pipeline (ETL) | High (Infrastructure) |
| aapl-backtest | Python | HMA Bull Put Spreads | Medium |

---

## Cross-Repository Algorithm Taxonomy

### 1. Regime-Based Trading Systems

**Commonality:** Both repositories implement market regime detection with adaptive position management.

#### 1.1 MSS-Based (trdr Repository)

**Source:** helios-1, helios-2, helios-3 branches from `2026-01-01-research-trdr-repo-algos.md`

**Core Algorithm:**

```text
MSS = w_trend × Trend + w_volatility × Volatility_norm + w_exhaustion × Exhaustion
```

**Regime Classification (5 states):**

- Strong Bull/Bear: Full positions, 2x ATR stops
- Weak Bull/Bear: Hold only, 1x ATR stops
- Neutral: Exit all positions

**Optimization:** Genetic algorithm with 14-20 parameters optimized via DEAP

**Key Innovation (helios-2):** Two-pass dynamic MSS with regime-specific weight multipliers

**Files:** `helios/factors.py`, `genetic_algorithm.py`, `regime_strategy.py`

#### 1.2 Multi-Timeframe (trading Repository)

**Source:** MAIN branch from `2026-01-01-research-trading-repo-algos.md`

**Core Algorithm:**

```text
composite_score = (micro*0.3 + trend*0.4 + sr*0.3)
+ alignment_bonus (20%) + confluence_bonus (15%)
```

**Timeframe Integration:**

| Timeframe | Purpose | Weight |
| --- | --- | --- |
| 5h | Micro-structure patterns | 30% |
| 15h | Primary trend/momentum | 40% |
| 1h | Support/resistance | 30% |

**Key Innovation:** Risk-adaptive parameters (0-10 scale) with smart DCA conditions

**Files:** `main:martingale-bot/multi_timeframe_signal_scorer.py`, `adaptive_risk_seeker.py`

### 2. Pattern-Based Trading Systems

#### 2.1 SAX Pattern Trading (trading Repository)

**Source:** RSKR, feat/ellipse, feat/perplexity-ideas branches from `2026-01-01-research-trading-repo-algos.md`

**Algorithm:** Symbolic Aggregate Approximation converts time-series to symbolic patterns

**Process:**

1. Z-normalize: (price - mean) / std, clip outliers
2. PAA: Compress to segments (30 bars → 5 segments)
3. Discretize: Map to alphabet {a,b,c,d}
4. Enhance: Add momentum (↑/↓/→), range (F/W/N), extremes

**Pattern Discovery:**

- Min occurrences: 10-20
- Min win rate: 55-65%
- Cross-symbol validation (2+ assets)
- Optimal hold: 6 bars (24h on 4h timeframe)

**Performance (feat/ellipse):**

- Return: 1737%
- Win Rate: 90.7%
- Sharpe: 6.38
- Note: Zero-commission requirement (Alpaca stocks)

**Files:** `rskr:bot/sax/sax_pattern_discovery.py`, `feat/ellipse:bot/sax/sax_trader.py`

#### 2.2 Multi-Agent Pattern Recognition (trdr Repository)

**Source:** project-1 branch from `2026-01-01-research-trdr-repo-algos.md`

**Architecture:** 5 specialized agents with adaptive parameters

**Agents:**

1. Bollinger Bands: Squeeze detection, band touches, reversions
2. MACD: Divergence, crossovers, momentum continuation
3. RSI: Divergence, extremes, momentum zones
4. Momentum Composite: RSI + MACD confluence
5. Volume Profile: POC, S/R levels, spikes

**Regime Adaptation:**

- Bollinger period: ×0.8 in high volatility
- RSI levels: +5 in bullish trends
- Confidence bonuses: +0.2 for regime alignment

**Files:** `packages/cli/src/agents/*.ts`, `market-regime-detector.ts`

### 3. Machine Learning Integration

#### 3.1 ML-Enhanced SAX (trading Repository)

**Source:** RSKR, feat/perplexity-ideas branches from `2026-01-01-research-trading-repo-algos.md`

**Classifier Stack:**

- Primary: XGBoost with class weighting
- Secondary: Random Forest
- Tertiary: Neural Network (MLP)
- Ensemble: Voting classifier

**Feature Engineering:**

- SAX pattern features: length, entropy, transitions
- Market context: volatility, trend, RSI, MACD, Bollinger Bands
- Advanced: GARCH, S/R distance, volume, Williams %R, CCI, MFI, ADX

**Training:** Time-series split, class balancing, calibrated probabilities

**Files:** `rskr:bot/sax/ml_strategy.py`, `feat/perplexity-ideas:bot/sax/ml_module.py`

#### 3.2 Genetic Algorithm Optimization (trdr Repository)

**Source:** helios-1, helios-2 branches from `2026-01-01-research-trdr-repo-algos.md`

**DEAP Implementation (helios-2):**

- Population: 50, Generations: 100
- Selection: Tournament (size 3)
- Crossover: Uniform (50% swap), prob 0.7
- Mutation: Gaussian (10% range), prob 0.3

**Multi-Objective Fitness:**

```text
fitness = 0.35×Sharpe + 0.25×Sortino + 0.20×Calmar
        + 0.10×transaction_penalty + 0.10×holding_bonus
```

**Walk-Forward Optimization:**

- Train: 2 years (504 bars), Test: 3 months (63 bars)
- Overlapping train windows, non-overlapping test windows
- Consensus: median parameters across splits

**Files:** `genetic_algorithm.py`, `walk_forward_optimizer.py`

### 4. Options Strategies

#### 4.1 HMA Bull Put Spreads (trdr Repository)

**Source:** aapl-backtest branch from `2026-01-01-research-trdr-repo-algos.md`

**Algorithm:** Hull Moving Average trend filter + Black-Scholes pricing

**HMA Formula:**

```text
HMA(n) = WMA(2 × WMA(n/2) - WMA(n), √n)
Trend: current_HMA > HMA[3_periods_ago]
```

**Spread Construction:**

- Short Put: -0.30 delta (~10% OTM)
- Long Put: $5 below short strike
- Net Credit: Short premium - long premium

**Entry:** HMA-50 up AND HMA-200 up AND positions < 5

**Exit:** 50% profit OR -200% stop OR trend reversal OR 30 DTE

**Files:** `aapl_hma_backtest/strategy/hma_bull_put_strategy.py`, `options/black_scholes.py`

### 5. Data Infrastructure

#### 5.1 Advanced Bar Types (trdr Repository)

**Source:** project-2 branch from `2026-01-01-research-trdr-repo-algos.md`

**Dollar Bars (both repos):**

```text
accumulate dollar_volume until >= threshold → emit bar
```

**Lorentzian Distance Bars (project-2 only):**

```text
d = √(c²Δt² - Δp² - Δv²)  # Relativistic geometry
```

**Shannon Information Bars (project-2 only):**

- Track entropy of price distribution
- Emit bar when accumulated info ≥ threshold
- More bars during unpredictable action

**Fractional Differentiation (both repos):**

- Order d ∈ (0.3, 0.7): balance stationarity vs memory
- Weight formula: w[k] = -w[k-1] × (d - k + 1) / k

**Files (trdr):** `src/transforms/`, `src/pipeline/buffer-pipeline.ts`
**Files (trading):** `helios-1:helios/dollar_bars.py`

#### 5.2 Data Pipeline Architecture (trdr Repository)

**Source:** project-2 branch from `2026-01-01-research-trdr-repo-algos.md`

**Providers:** Alpaca, Coinbase, CSV/JSONL with rate limiting

**Transforms:** Bar generators, normalizers, technical indicators

**Repository:** CSV/JSONL output with configurable formats

**Configuration:** JSON with Zod validation, pluggable architecture

**Files:** `src/providers/alpaca/`, `src/transforms/`, `src/pipeline/`

---

## Cross-Repository Architecture Comparison

### Position Management

| Repository | Branch | Sizing Method | Stop Loss | DCA/Scaling |
| --- | --- | --- | --- | --- |
| trading | MAIN | Risk-adaptive (0-10 scale) | Dynamic trailing, ATR-based | Smart DCA (5-6 levels) |
| trading | RSKR/feat | Signal strength-based | ATR trailing, per-TF | No DCA |
| trdr | helios-1/2/3 | Gradual step sizing | Regime-specific ATR | No DCA |
| trdr | aapl-backtest | Fixed per spread | Profit target + stop loss | Max 5 positions |

### Optimization Frameworks

| Repository | Branch | Method | Parameters | Validation |
| --- | --- | --- | --- | --- |
| trading | MAIN | None (manual tuning) | N/A | Real-time tracker |
| trading | RSKR | Walk-forward | Safety validator limits | WFO + paper trading |
| trdr | helios-1 | Custom GA | 14 params | 70/30 train/test |
| trdr | helios-2 | DEAP GA | 20 params | Hybrid sliding WFO |

### Technology Stack

| Aspect | trading (Python) | trdr (Python) | trdr (TypeScript) |
| --- | --- | --- | --- |
| Optimization | Custom GA, XGBoost | DEAP, scipy | N/A |
| Data Source | Alpaca | Alpaca, CSV | Alpaca, Coinbase, CSV |
| Testing | pytest + custom backtest | pytest + portfolio engine | Jest (implied) |
| Configuration | Pydantic + YAML | JSON/Pydantic | Zod + JSON |
| Architecture | Service-oriented (RSKR) | Factor-based | Agent-based |

---

## Cross-Repository Algorithm Recreation

### Core Building Blocks

#### Block 1: Market Regime Detection

**Used by:** trading:MAIN, trdr:helios-1/2/3, trdr:project-1

**MSS Approach (trdr):**

```text
1. Calculate factors:
   Trend = (linear_regression_slope / price) × 100
   Volatility = 100 - (ATR / price × 100 × 2)
   Exhaustion = clip((price - SMA) / ATR × 10, -100, 100)

2. Combine with weights (sum = 1.0):
   MSS = w_trend × Trend + w_vol × Volatility + w_exhaust × Exhaustion

3. Classify regime:
   Strong Bull (MSS > 50), Weak Bull (20-50), Neutral (-20 to 20),
   Weak Bear (-50 to -20), Strong Bear (< -50)
```

**Multi-Timeframe Approach (trading):**

```text
1. Analyze per timeframe:
   Micro (5h): Candlestick patterns, micro-trends
   Trend (15h): MA, MACD, RSI with divergence
   S/R (1h): Pivot levels, Fibonacci, volume POC

2. Aggregate signals:
   composite = micro×0.3 + trend×0.4 + sr×0.3
   Add bonuses for alignment and confluence

3. Entry when composite ≥ 0.6 + volume confirmation
```

**Agent Approach (trdr:project-1):**

```text
1. Detect 5 dimensions:
   - Trend: 3-MA comparison
   - Volatility: StdDev + ATR percentiles
   - Momentum: ROC + consecutive moves
   - Volume: Recent vs older MA
   - Classification: Trending/Breakout/Reversal/Ranging

2. Adapt parameters by regime:
   - Bollinger period: ×0.8 in high vol
   - RSI levels: +5 in bullish trend

3. Confidence scoring:
   Base + regime_alignment_bonus + volume_confirmation
```

#### Block 2: Pattern Discovery (SAX)

**Used by:** trading:RSKR/feat branches

```text
# Training Phase
For each symbol in pool:
  For window in historical_data:
    1. normalized = z_normalize(window, clip_outliers=3σ)
    2. paa_segments = piecewise_aggregate(normalized, n_segments=5)
    3. pattern = discretize(paa_segments, alphabet_size=4)  # a,b,c,d
    4. enhanced = add_momentum_range_extremes(pattern)
    5. Store pattern occurrence

# Filter patterns
For pattern in all_patterns:
  If occurrences >= 10 AND win_rate >= 0.65:
    Test hold periods [3,4,5,6,7,8]
    Select hold maximizing: mean_return × win_rate
    Store as GoldenPattern

# Live Trading
For each bar:
  current_pattern = generate_sax(bars[-30:])
  If current_pattern in golden_patterns:
    Enter position with optimal hold period
    Set ATR-based trailing stop
```

**Source files:**

- `rskr:bot/sax/sax_pattern_discovery.py`
- `feat/ellipse:bot/sax/sax_backtester.py`

#### Block 3: Genetic Algorithm Optimization

**Used by:** trdr:helios-1/2

**DEAP Implementation (helios-2):**

```text
# Individual: 20 parameters with constraints
Parameters:
  - 3 lookbacks: trend (10-200), vol (5-100), exhaust (5-100)
  - 3 weights: normalized sum = 1.0
  - 6 thresholds: ordered (strong_bull > weak_bull > neutral...)
  - 4 position params: stops, step size, max position
  - 4 filter params: min change, cooldown, regime multipliers

# GA Configuration
Population: 50, Generations: 100
Selection: Tournament(size=3)
Crossover: Uniform(swap_prob=0.5), prob=0.7
Mutation: Gaussian(sigma=10%_of_range), prob=0.3

# Constraint Enforcement
After each operation:
  - Renormalize weights to sum = 1.0
  - Enforce threshold ordering
  - Clip parameters to bounds

# Fitness Function
fitness = 0.35×Sharpe + 0.25×Sortino + 0.20×Calmar
        + 0.10×transaction_penalty + 0.10×holding_bonus

Penalties:
  - Win rate < 40%: multiply by (win_rate / 0.4)
  - Max DD > 25%: multiply by (0.25 / max_dd)
```

**Walk-Forward Optimization:**

```text
# Window Strategy
Train window: 2 years (504 bars)
Test window: 3 months (63 bars)
Step: 3 months (sliding)

For each window split:
  1. Run GA on train window → best_params
  2. Test on out-of-sample test window → metrics
  3. Store params and metrics

# Consensus Parameters
For each parameter:
  consensus_value = median(all_window_values)
Renormalize weights after median calculation

# Validation
Check stability: parameter swings, performance degradation
Red flags: in-sample Sharpe > 3 but OOS < 1
```

**Source files:**

- `genetic_algorithm.py`
- `walk_forward_optimizer.py`

#### Block 4: Multi-Agent Signal Aggregation

**Used by:** trdr:project-1

```text
# Agent Interface
analyze(market_context) → AgentSignal {
  action: BUY | SELL | HOLD
  confidence: 0.0 - 1.0
  reason: string
  indicators: object
}

# Signal Priority (Bollinger example)
1. Squeeze Release (highest): Bandwidth expanding after compression
2. Band Touches: Price at extreme %B ≤ 0 or ≥ 1
3. Squeeze Prep: Bandwidth contracting
4. Mean Reversion: %B crossing 0.2 or 0.8

# Confidence Calculation
base_confidence = signal_strength  # 0.0-1.0
If signal_direction matches regime_trend:
  confidence += 0.2
If volume confirms signal:
  confidence += 0.1
If divergence detected:
  confidence += 0.15
Final = clamp(confidence, 0.0, 1.0)

# Aggregation (Momentum Composite example)
rsi_signals = RSI_agent.analyze()
macd_signals = MACD_agent.analyze()

bullish_count = count_bullish(rsi_signals, macd_signals)
bearish_count = count_bearish(rsi_signals, macd_signals)

confluence = max(bullish_count, bearish_count) / total_signals
If confluence > 0.75:
  Generate STRONG signal in dominant direction
```

**Source files:**

- `packages/cli/src/agents/bollinger-bands-agent.ts`
- `packages/cli/src/agents/momentum-composite-agent.ts`

#### Block 5: Options Pricing and Strategy

**Used by:** trdr:aapl-backtest

**Black-Scholes Implementation:**

```text
# Put Pricing
d1 = [ln(S/K) + (r + σ²/2)×T] / (σ×√T)
d2 = d1 - σ×√T
Put = K×e^(-rT)×N(-d2) - S×N(-d1)
Delta = N(d1) - 1

Where:
  S = spot price
  K = strike price
  r = risk-free rate (0.02)
  σ = implied volatility
  T = time to expiration (years)
  N = cumulative normal distribution

# Strike Selection (Binary Search)
Target: -0.30 delta
Search range: 70% to 99% of spot
Tolerance: ±0.01 delta

low, high = 0.70×spot, 0.99×spot
While high - low > tolerance:
  mid = (low + high) / 2
  delta = black_scholes_delta(mid)
  If delta < -0.30:
    high = mid
  Else:
    low = mid
Return mid

# Bull Put Spread Construction
1. Find 30-delta put → short_strike
2. long_strike = short_strike - $5
3. short_premium = bs_put_price(short_strike)
4. long_premium = bs_put_price(long_strike)
5. net_credit = short_premium - long_premium
6. max_profit = net_credit × 100
7. max_loss = (5 - net_credit) × 100

# Strategy Logic
Entry:
  IF HMA(50) trending up
  AND HMA(200) trending up
  AND positions < 5
  AND net_credit > 0
  THEN enter spread

Exit (any triggers):
  1. P&L ≥ 50% × max_profit (profit target)
  2. P&L ≤ -200% × net_credit (stop loss)
  3. HMA(50) turns down OR HMA(200) turns down (trend reversal)
  4. DTE ≤ 30 (expiration)
```

**Source files:**

- `aapl_hma_backtest/options/black_scholes.py`
- `aapl_hma_backtest/strategy/hma_bull_put_strategy.py`

---

## Viability Assessment Cross-Comparison

### Production-Ready Strategies

#### High Viability - trading:MAIN

**Source:** `2026-01-01-research-trading-repo-algos.md` MAIN branch

**Strengths:**

- Production-grade risk controls (daily limits, circuit breaker)
- Multi-timeframe validation reduces false signals
- Smart DCA prevents runaway losses
- Extensive testing infrastructure

**Weaknesses:**

- Martingale risk in trending markets
- Requires significant capital for DCA scaling
- Complex parameter tuning

**Capital Requirement:** Medium-High (support 5-6 DCA levels)

#### High Viability - trading:feat/ellipse SAX

**Source:** `2026-01-01-research-trading-repo-algos.md` feat/ellipse branch

**Strengths:**

- Exceptional backtest: 1737% return, 90.7% win rate
- Out-of-sample validation (2024-2025)
- Symbol-specific patterns prevent contamination

**Weaknesses:**

- Zero-commission dependency (Alpaca stocks only)
- Pattern discovery computationally intensive
- Needs periodic retraining

**Capital Requirement:** Low-Medium

#### High Viability - trdr:helios-2

**Source:** `2026-01-01-research-trdr-repo-algos.md` helios-2 branch

**Strengths:**

- DEAP genetic algorithm with robust constraints
- Walk-forward optimization prevents overfitting
- Two-pass dynamic MSS adapts to regime
- Multi-objective fitness function

**Weaknesses:**

- Requires significant historical data for WFO
- 20 parameters may overfit without careful validation
- Regime transitions may cause whipsaw

**Capital Requirement:** Medium

### Medium Viability Strategies

#### Medium - trdr:project-1 Agents

**Source:** `2026-01-01-research-trdr-repo-algos.md` project-1 branch

**Strengths:**

- Good agent abstraction and modularity
- Regime adaptation adds value
- Confluence scoring from multiple agents

**Weaknesses:**

- No optimization layer (static default parameters)
- Lacks backtesting integration
- Needs parameter tuning per asset

**Recommendation:** Integrate with trdr:helios-2 GA optimization

#### Medium - trdr:aapl-backtest

**Source:** `2026-01-01-research-trdr-repo-algos.md` aapl-backtest branch

**Strengths:**

- Options-specific risk management
- HMA trend filter reduces false entries
- Defined risk (max loss per spread)

**Weaknesses:**

- Single-asset focus (AAPL only)
- Requires accurate volatility estimation
- Options-specific (limited to equities with liquid options)

**Recommendation:** Expand to multi-asset options portfolio

### Infrastructure (High Viability)

#### trdr:project-2 Data Pipeline

**Source:** `2026-01-01-research-trdr-repo-algos.md` project-2 branch

**Strengths:**

- Production-grade ETL architecture
- Multiple providers with rate limiting
- Advanced bar types (Lorentzian, Shannon)
- Pluggable transforms

**Use Case:** Foundation for any strategy in either repository

---

## Research Ideation (Advanced Concepts)

### From trading:feat/perplexity-ideas

**Source:** `2026-01-01-research-trading-repo-algos.md` feat/perplexity-ideas branch

18 advanced trading concepts documented in `PERPLEXITY-IDEAS.md`:

**Quantitative Concepts (Pure Code):**

1. Quantum Price Formation Theory
2. Fractal Market Hypothesis (Hurst exponent, MFDFA)
3. Lévy Flight Recognition
4. Wavelet Decomposition
5. Multi-EMA Quantum Synthesis
6. Attention Matrix Microstructure
7. Latent Space Arbitrage
8. Frequency-Domain Attention
9. Cross-Attention Market Coupling
10. Stochastic Embedding Transitions

**LLM-Augmented (Ideation, not implemented per instructions):**

1. Chain-of-Alpha
2. Adaptive Kalman Filters
3. Transformer-Enhanced HMM
4. Multi-Asset Embedding Topology
5. X-Trend Few-Shot
6. Attention-Weighted Portfolio
7. Multi-Timeframe Order Flow
8. Granger Causality Attention

**Status:** Research framework for future experimentation

### From trading:feat/ellipse

**Source:** `2026-01-01-research-trading-repo-algos.md` feat/ellipse branch

**Ellipse/Perspective Algorithm (PoC):**

**Hypothesis:** Price charts are 2D projections of higher-dimensional manifolds

**Process:**

1. Detect ellipses in price windows (48-192 bars)
2. Conic fitting: Ax²+Bxy+Cy²+Dx+Ey+F=0
3. Extract parameters: center, axes, rotation, eccentricity
4. Solve for viewing angle and projection matrix
5. Extract curvature features

**Results:**

- SPY 4h: Curvature_delta showed promise (Sharpe 1.9)
- No incremental lift in linear fusion models
- Needs non-linear fusion research

**Status:** Research stage, not production-ready

**Files:** `feat/ellipse:poc/src/ellipse_detection.py`, `perspective_solver.py`

---

## Performance Metrics Unified Reference

### Risk-Adjusted Returns (Both Repos)

| Metric | Formula | Good | Excellent | Source |
| --- | --- | --- | --- | --- |
| Sharpe | (Return - Rf) / StdDev | > 1.0 | > 2.0 | Both |
| Sortino | (Return - Rf) / DownsideStdDev | > 1.5 | > 2.0 | Both |
| Calmar | AnnualReturn / MaxDrawdown | > 1.0 | > 3.0 | Both |

### Benchmark Performance

| Strategy | Repository | Return | Win Rate | Sharpe | Max DD | Source |
| --- | --- | --- | --- | --- | --- | --- |
| SAX (feat/ellipse) | trading | 1737% | 90.7% | 6.38 | <10% | trading doc |
| SAX (RSKR) | trading | 65.62% | 60.5% | 1.23 | -12.5% | trading doc |
| MSS + GA | trdr | N/A | N/A | N/A | N/A | trdr doc |

### Multi-Objective Fitness (trdr:helios-2)

**Source:** `2026-01-01-research-trdr-repo-algos.md` helios-2 branch

```text
fitness = 0.35×Sharpe + 0.25×Sortino + 0.20×Calmar
        + 0.10×transaction_penalty + 0.10×holding_bonus
```

Penalties:

- Win rate < 40%: multiply by (win_rate / 0.4)
- Max DD > 25%: multiply by (0.25 / max_dd)

---

## Testing and Validation Unified Guide

### Unit Testing Standards (Both Repos)

**Indicators (Common):**

- Verify against TA-Lib/known values
- Edge cases: empty, single value, NaN, lookback > series
- Example SMA: [10,20,30,40,50] period=3 → [NaN,NaN,20,30,40]

**SAX Pattern Generation (trading):**

**Source:** `2026-01-01-research-trading-repo-algos.md`

- Test z-normalization with outlier clipping
- Verify PAA compression preserves trend shape
- Check discretization breakpoints (Gaussian)
- Validate enhanced features (momentum, range, extremes)

**MSS Factor Calculation (trdr):**

**Source:** `2026-01-01-research-trdr-repo-algos.md`

- Test each factor independently against hand calculation
- Verify weight sum = 1.0 after all operations
- Check regime classification boundary conditions
- Integration test with fixed params

**Genetic Algorithm (trdr):**

**Source:** `2026-01-01-research-trdr-repo-algos.md`

- Constraint preservation after crossover/mutation
- Weight normalization maintained
- Threshold ordering preserved
- Selection pressure: tournament selects higher fitness
- Population diversity tracked

**Black-Scholes (trdr):**

**Source:** `2026-01-01-research-trdr-repo-algos.md`

- Compare against QuantLib/vollib
- Verify put-call parity: C - P = S - Ke^(-rT)
- Binary search converges in < 20 iterations
- Delta calculation accuracy

### Integration Testing

**Walk-Forward Validation (Both):**

**Sources:** Both documents

- Train/test windows don't overlap in test periods
- Parameters stable across windows (no wild swings)
- Red flags: in-sample Sharpe > 3 but OOS < 1
- Parameter sensitivity: ±20% variation check

**Backtest Scenarios (Both):**

1. Trending market → regime detection captures trend
2. Ranging market → neutral classification or pattern discovery
3. High volatility → position sizing adjusts
4. Drawdown → stops trigger correctly
5. Whipsaw → filters prevent overtrading

**Multi-Agent Confluence (trdr:project-1):**

**Source:** `2026-01-01-research-trdr-repo-algos.md`

- Win rate increases with multi-agent agreement
- Target: 1 agent ~50%, 2 agents ~60%, 3+ agents ~70%
- Signal quality: Precision, Recall, F1 > 0.5 per agent

### Statistical Validation

**Stationarity (Both):**

- ADF test: p < 0.05 (reject null of unit root)
- KPSS test: p > 0.05 (fail to reject null of stationarity)
- Fractional differentiation validation

**Overfitting Indicators (Both):**

- In-sample Sharpe > 3.0, OOS < 1.0 → overfitting
- Win rate drops > 20% from train to test
- > 20 optimizable parameters → reduce or regularize

**Regime Detection Accuracy (Both):**

- Target > 70% accuracy vs manual labels
- Check false regime transitions (noise vs signal)

---

## Cross-Repository Integration Opportunities

### 1. Hybrid SAX + MSS System

**Concept:** Combine trading:SAX pattern discovery with trdr:MSS regime filtering

**Architecture:**

```text
Data → MSS Regime Detection → Filter SAX patterns by regime
     → Execute patterns matching current regime only
```

**Benefits:**

- Reduce SAX false signals in unfavorable regimes
- Preserve SAX high win rate, add MSS trend filter

**Implementation:**

- Use trdr:helios-2 MSS calculation
- Apply to trading:RSKR SAX pattern execution
- Add regime as feature to trading:ML classifier

### 2. Multi-Agent + Genetic Optimization

**Concept:** Optimize trdr:project-1 agent parameters using trdr:helios-2 DEAP GA

**Architecture:**

```text
Agents (Bollinger, MACD, RSI, etc.) with parameterized thresholds
     → DEAP GA optimizes per-agent parameters
     → Walk-forward validation
```

**Parameters to Optimize (per agent):**

- Bollinger: period, stdDev, regime multipliers
- RSI: oversold/overbought levels, period
- MACD: fast/slow/signal periods
- Confluence: weight per agent

**Benefits:**

- Replace static defaults with optimized parameters
- Adaptive per asset/timeframe

### 3. Advanced Data Pipeline for All Strategies

**Concept:** Use trdr:project-2 pipeline as foundation for both repos

**Architecture:**

```text
trdr:project-2 ETL → Lorentzian/Shannon/Dollar bars
                   → Fractional differentiation
                   → Feed to any strategy (SAX, MSS, Agents, Options)
```

**Benefits:**

- Unified data infrastructure
- Advanced bar types may improve all strategies
- Normalize data quality across strategies

### 4. Options + Equity Multi-Asset Portfolio

**Concept:** Combine trdr:aapl-backtest options with trading:MAIN equity strategies

**Architecture:**

```text
Portfolio Manager
├── Equity Strategies (trading:MAIN, RSKR SAX)
│   └── Long/short positions
└── Options Strategies (trdr:aapl-backtest)
    └── Bull put spreads for income
```

**Benefits:**

- Diversification across asset types
- Options provide income during equity ranging markets
- Equity strategies capture trends

### 5. Ensemble Meta-Strategy

**Concept:** Meta-learner combines predictions from multiple base strategies

**Base Strategies:**

- trading:MAIN multi-timeframe
- trading:RSKR SAX patterns
- trdr:helios-2 MSS regime
- trdr:project-1 multi-agent

**Meta-Learner:**

- Stacking: Train on base strategy predictions
- Weight by recent performance (sliding window)
- Execute when consensus threshold met

**Benefits:**

- Robustness through diversification
- Capture different market conditions

---

## Volume Profile Trading Strategies

**Source:** Perplexity research - Autonomous Volume Profile Trading System

### Core Components

**High Volume Nodes (HVNs):** Price levels with maximum trading activity. Act as magnetic S/R zones where institutional orders concentrate.

**Low Volume Nodes (LVNs):** Price rejection areas with minimal liquidity. Lead to rapid price movement during breakouts.

**Point of Control (POC):** Single price level with highest volume. Market's "fair value" anchor for mean reversion.

**Value Area (VA):** Contains 70% of total trading volume. Bounded by VAH (high) and VAL (low). Defines consensus price range.

### Strategy 1: POC Mean Reversion

**Entry Rules:**

1. Price moves outside Value Area by > 2 ATR
2. Volume during move is declining (weak conviction)
3. Price begins returning toward POC
4. Enter when price crosses back into VA boundary (VAL for longs, VAH for shorts)

**Exit Rules:**

- Primary target: POC level
- Stop loss: Beyond LVN that price broke through (1.5-2x VA width)
- Profit optimization: Exit 50% at POC, trail remaining 50%

**Position Sizing:** Risk 1-2% per trade. Size = Account Risk / Stop Distance

### Strategy 2: LVN Breakout Continuation

**Entry Rules:**

1. Identify LVN between two HVNs on current session profile
2. Price enters LVN with volume > 150% of 20-period average
3. Direction aligns with higher-timeframe trend (check daily profile)
4. Enter on first close beyond LVN boundary

**Exit Rules:**

- Target 1: Next HVN or POC (take 60%)
- Target 2: Opposite VA boundary (take 40%)
- Stop: Inside LVN or beyond entry HVN

**False Breakout Filter:** Volume must surge 150%+ during breakout. Price must close decisively through LVN (not just wick).

### Strategy 3: Value Area Breakout (Trend Following)

**Entry Rules:**

1. Price closes above VAH (long) or below VAL (short) for 2+ consecutive periods
2. Volume profile shape shows directional bias: P-shaped (bullish) or b-shaped (bearish)
3. Volume during breakout > 130% of average session volume
4. VWAP alignment: Price and VWAP both above VAH for longs

**Exit Rules:**

- Trail stop using prior session's POC as dynamic S/R
- Exit if price re-enters and closes inside prior Value Area (failed auction)
- Partial profits at each subsequent HVN

**Trend Filter:** Only take trades aligned with higher TF bias. Check weekly/daily profile.

### Configuration Parameters

**Timeframes:**

- Primary execution: 30min/1h for swing, 5-15min for day trades
- Profile anchor: Daily session for intraday, weekly composite for swing
- Trend confirmation: 3x higher than execution TF

**Volume Profile Settings:**

- Number of rows: 30-50 for optimal granularity
- Value Area percentage: 70%
- Profile type: Session-based (day trading), fixed range/composite (swing)

**Position Management:**

- Max open risk: 6-8% of account
- Scale by volatility: Low ATR → +20% size, High ATR → -30% size
- Never add to losing positions outside original stop

### Supporting Indicators

**VWAP Confluence:** When POC aligns within 0.5% of VWAP, significance increases dramatically.

**Delta Volume:** Monitor cumulative delta for order flow direction. For POC reversals, look for delta divergence. Only take breakouts when delta confirms with 60%+ imbalance.

**Multi-Timeframe Profiles:** Stack daily, weekly, monthly profiles. HVNs from multiple TFs within 1-2% = institutional magnets with higher win rates.

### Expected Performance

| Strategy | Win Rate Target | R-Multiple |
| --- | --- | --- |
| POC Mean Reversion | 55-65% | 1.5R min |
| LVN Breakout | 45-55% | 2.0R min |
| VA Breakout | 45-55% | 2.0R min |

---

## Critical Assessment: OHLCV Limitations

**Source:** Perplexity research - Innovative Systematic Long-Only Trading Frameworks

### Why Common Approaches Have Zero Alpha

1 - **Frequency-Domain Attention Overfits on OHLCV**

Wavelet transforms and multi-head attention require tick-by-tick LOB data, not daily bars. With OHLC only, models memorize in-sample regime artifacts rather than discovering genuine cyclical structure.

2 - **Kalman Momentum Fails in Trend Regimes**

Mean reversion in equities is regime-dependent. Stocks entering strong directional moves exhibit order flow toxicity—informed flow dominates. Mean-reversion strategies produce negative Sharpe ratios during momentum regimes, which now dominate 60%+ of trading days.

3 - **Sparse Factor Discovery Mines Noise**
L1-penalized regression on 50-100 OHLCV-derived factors is data snooping:

- Markets adapt: Patterns disappear within 6-12 months
- Transaction costs destroy edges: 0.1-0.2% target evaporates with realistic slippage (0.05-0.15%) + commissions
- Survivorship bias: Backtests on liquid names ignore delisted losers

4 - **OHLCV Lacks Microstructure Information**

Real alpha comes from:

- Order flow imbalance
- Toxicity metrics (VPIN)
- Limit order book depth
- Aggressive buy vs sell volume
- Bid-ask stacking and spoofing activity

OHLC bars aggregate away directional information embedded in these signals.

### Implications for Repository Strategies

| Strategy | OHLCV Limitation | Mitigation |
| --- | --- | --- |
| SAX Patterns | Patterns may disappear 6-12 months | Periodic retraining, regime filter |
| MSS Regime | Works but limited alpha | Add microstructure signals |
| Multi-Agent | OHLCV indicators only | Add order flow agents |
| Options HMA | Less affected (trend filter) | Still needs vol estimation |

---

## Futures vs Stocks Analysis

**Source:** Perplexity research - Innovative Systematic Long-Only Trading Frameworks

### Comparison Table

| Dimension | Stocks | Futures |
| --- | --- | --- |
| Leverage | 2:1 (4:1 intraday), capital-intensive | 10:1-50:1, <5% margin |
| Trading Costs | 0.05-0.15% slippage + SEC fees | 0.5-1.5 ticks all-in, 60% cheaper |
| Liquidity | Fragmented across venues, toxic flow | Centralized, transparent order book |
| Hours | 9:30-16:00 ET (6.5h) | Nearly 24/5, capture overnight gaps |
| Short Selling | Locate fees, uptick rules | Unrestricted, symmetric long/short |
| Tax Treatment | Short-term cap gains (37%) | 60/40 treatment (lower effective rate) |
| Structural Edge | None | Contango/backwardation roll yield |

### Structural Alpha in Futures

**Roll Yield Harvesting:** ES and NQ futures in backwardation 70% of time. Front-month trades 0.2-1.5% above deferred months.

**Execution:**

1. Hold continuous long exposure in front-month
2. Roll 2 days before expiration
3. Capture positive roll yield (sell expensive front, buy cheaper next)

**Expected edge:** 0.5-1.2% per quarter (4-8 rolls/year) with zero directional risk. Combined with mild trend filter (50-day MA), achieves 1.5+ Sharpe.

### Recommendation

Migrate to futures for algo trading:

- Order book transparency reduces adverse selection
- Roll yield provides non-zero-sum edge
- Leverage + low costs let small edges compound
- 24/5 hours support always-on bot architecture

---

## Microstructure-Driven Strategies

**Source:** Perplexity research - Innovative Systematic Long-Only Trading Frameworks

### Strategy 1: Order Flow Imbalance (OFI) on ES/NQ

**Concept:** When aggressive buy orders (lifting offers) exceed aggressive sells (hitting bids) across top 5 price levels, short-term momentum persists for 5-20 bars.

**Implementation:**

```text
def compute_mlofi(bid_vol, ask_vol, levels=5):
    ofi = 0
    for L in range(1, levels+1):
        ofi += (ask_vol[level_L] - bid_vol[level_L])
    return ofi / levels

# Signal: Z-score of MLOFI
z = (mlofi - rolling_mean(mlofi, 50)) / rolling_std(mlofi, 50)
signal = +1 if z > 1.5 else (-1 if z < -1.5 else 0)
```

**Edge source:** HFTs cannot react fast enough when order book imbalance propagates across 3+ levels simultaneously.

**Expected performance:** 0.3-0.8% per trade, 65-75% win rate. Works on ES, NQ, CL futures.

### Strategy 2: Regime-Filtered Mean Reversion

**Concept:** Mean reversion only works in low-volatility, range-bound regimes.

1 - **Volatility Regime via Clustering**

```text
# Compute realized volatility (5-minute returns)
rv = returns.rolling(78).std()  # ~6.5 hours

# Fit 3-regime GMM model
gmm = GaussianMixture(n_components=3)
regimes = gmm.fit_predict(rv)
# 0=low-vol, 1=med-vol, 2=high-vol
```

2 - **Trade Mean Reversion ONLY in Regime 0**

```text
z = (close - rolling_mean(close, 20)) / rolling_std(close, 20)
signal = +1 if (regimes == 0) and (z < -2) else 0  # Buy oversold in calm
signal = -1 if (regimes == 0) and (z > 2) else signal  # Sell overbought
```

**Key insight:** Regime-filtered mean reversion doubles Sharpe vs unconditional strategies.

### Strategy 3: VPIN Toxicity Filter

**Concept:** Avoid trading when order flow is toxic.

```text
def compute_vpin(volume, price, n_buckets=50):
    bucket_vol = sum(volume) / n_buckets
    buy_vol, sell_vol = [], []

    for i in range(len(volume)):
        if price[i] > price[i-1]:
            buy_vol.append(volume[i])
        else:
            sell_vol.append(volume[i])

    vpin = abs(sum(buy_vol) - sum(sell_vol)) / sum(volume)
    return vpin

# Only trade when VPIN < 0.3 (low toxicity)
trade_allowed = vpin < 0.3
```

**Research finding:** VPIN spikes precede flash crashes and adverse selection. Filtering out high-VPIN periods reduces max drawdown by 40-60%.

### Realistic Performance Expectations

Combining these strategies on ES/NQ futures with proper risk controls:

| Metric | Target | Note |
| --- | --- | --- |
| Per-trade edge | 0.15-0.40% | After all costs |
| Win rate | 60-72% | Not 70%+ (unrealistic) |
| Sharpe ratio | 1.8-2.4 OOS | Not 2.5+ (overfitting) |
| Max drawdown | 2-4% | Not sub-1% (unrealistic) |
| Trade frequency | 8-15/week | Per contract |
| Capacity | $500K-$2M | Before market impact |

---

## Innovative Bot Strategies for Futures/Options

**Source:** Perplexity research - Innovative Systematic Long-Only Trading Frameworks

### 1. Adaptive Volatility Grid

Dynamic grid spacing and sizing based on real-time regime.

**Futures Logic:**

- Grid spacing expands/contracts with ATR or realized volatility
- Position size adjusts based on signal confidence
- Low vol → tight grid (scalp minor moves)
- High vol → wide grid, smaller size (defend equity)

### 2. Order Flow Imbalance Bots

Real-time detection of buy/sell pressure at DOM level.

**Futures:** Buy NQ if cumulative bid volume outpaces offers by X contracts on 3+ levels, stop below previous imbalance zone.

**Options Extension:** On bullish imbalance + IV spike, fire short-term OTM put credit spread, capturing both price thrust and IV mean reversion.

### 3. Regime-Filtered Premium Harvesting

Systematic options selling only when strong regime confirmation exists.

**Process:**

1. Trending up (dual MA confirmation, positive factor ensemble)
2. IV Rank > 60%
3. Realized volatility low
4. Sell far OTM put spreads or iron condors
5. Manage actively as regime weakens

### 4. Cross-Asset Pairs Mean Reversion

Track spreads between correlated futures (ES-NQ, CL-RB, ZN-ZF).

**Bot Logic:**

- Buy undervalued spread leg
- Simultaneously sell OTM options on overvalued leg
- Capture convergence and decay in one trade

### 5. Event-Driven Volatility Harvesting

**Futures:** Monitor news, macro calendars for volatility surge signals.

**Options:** Before scheduled event, sell high-IV iron condor or straddle. Exit majority of risk immediately after IV collapses ("IV crush"), or if price stays inside calculated range.

### 6. Dynamic Roll Yield/Calendar Algo

Exploit term structure systematically:

- If nearby in backwardation vs next → buy near/far spread
- If in contango → short spread
- Overlay with options (sell OTM calls into backwardation spikes)

---

## LLM Integration (Non-Sentiment)

**Source:** Perplexity research - Innovative Systematic Long-Only Trading Frameworks

### Legitimate LLM Use Cases (Not Sentiment)

1. **Structured Event and Regime Extraction**

- Feed economic calendar, earnings transcripts, news headlines
- Classify pre-defined event types impacting asset class
- Output: Structured event codes, triggers for schedule-driven bots

1 - **Regime Narrative Parsing**

- Parse market commentary or analyst notes
- Extract regime context ("high-volatility chop due to Fed uncertainty")
- Drive automated switching of trading systems

2 - **Real-Time Parameter Recommendation**

```text
Prompt: "Given ES futures have realized 5-minute volatility of X,
spread of Y, and depth profile Z, recommend safe grid spacing
and max lot size."
```

Use outputs as guardrails for adaptive bots.

3 - **Feature Engineering and Signal Design**

- Refactor/optimize indicator formulas
- Design new microstructure features from bid/ask, price, volume
- "Suggest three new features from order book imbalance data"

4 - **Error/Anomaly Detection from Logs**

- Stream bot logs and trade footprints
- Quick root-cause analysis ("Why was slippage high at 13:05?")
- Removing need for media/sentiment data

5 - **Order Book State Classification**

- Classify live order book states (sweep, fade, spoof, iceberg)
- From raw numerical snapshots or footprints
- Trigger microstructure-driven entries/exits

### Integration Notes

- Use in-memory message bus (Redis, Kafka) between trading system and LLM
- Keep prompts/data numeric and structured
- Use fast LLMs (Llama 3 8B, Claude Haiku) on local hardware for millisecond response
- Never send public sentiment or headlines for these use cases

---

## Platform Recommendations

**Source:** Perplexity research - Innovative Systematic Long-Only Trading Frameworks

### US Platforms with Paper Trading

| Platform | Asset Classes | API Support | Paper Trading | Best For |
| --- | --- | --- | --- | --- |
| Interactive Brokers | All | REST, Python, C++, Java | Robust (TWS/API) | Overall best |
| NinjaTrader | Futures, Stocks | C#, .NET, Python | Built-in | Futures focus |
| Alpaca | Stocks, ETFs | REST, WebSocket | Full-featured | Free, easy setup |
| QuantConnect | All + FX | C#, Python, F# | Cloud-based | Backtesting |
| TradeStation | All | REST, EasyLanguage | Simulator | Automation |
| TD Ameritrade | Stocks, Options | REST | paperMoney | User-friendly |

### Platform Selection by Use Case

| Use Case | Recommended Platform |
| --- | --- |
| Futures with microstructure | NinjaTrader, Interactive Brokers |
| Stocks with zero commission | Alpaca |
| Options strategies | Interactive Brokers, TradeStation |
| Cloud backtesting | QuantConnect |
| Multi-asset portfolio | Interactive Brokers |

### Key Features to Verify

- WebSocket support for real-time order book
- Historical tick data access
- Paper trading API parity with live
- Rate limiting and error handling
- Commission/slippage modeling in backtest

---

## Bot Architecture Implementation

**Source:** Claude Conversation - Trading Bot Development

### 7-Milestone Development Plan

1. **Foundation Setup:** Project structure, dependencies (uv, DuckDB), Alpaca SDK integration, basic CLI scaffold
2. **Data Layer:** Historical data fetch, OHLCV storage in DuckDB, caching, basic indicators
3. **Strategy Framework:** Pluggable strategy interface, MSS regime detection, simple RSI strategy for validation
4. **Paper Trading:** Local paper trading engine, order simulation, position tracking, P&L calculation
5. **Live Trading:** Alpaca order execution, position synchronization, risk controls
6. **Monitoring:** Real-time dashboard, trade logging, performance metrics, alerts
7. **Optimization:** Walk-forward backtesting, parameter tuning, regime adaptation

### Three Trading Modes

| Mode | Description | Order Execution | Position State |
| --- | --- | --- | --- |
| **live** | Real money, real exchange | Alpaca live API | Exchange is source of truth |
| **server-paper** | Alpaca's native paper trading | Alpaca paper API | Exchange paper account |
| **local-paper** | Client-side simulation | Local engine | Local DuckDB state |

**Implementation Notes:**

- `live` and `server-paper` share identical code paths (different API endpoints)
- `local-paper` requires full order matching simulation: limit orders, fills, partial fills, slippage
- All modes use identical strategy logic—only execution layer differs
- Paper modes enable testing without API rate limits

### Exchange Reconciliation

Sync local state with exchange reality. Critical for crash recovery and drift prevention.

**Three Reconciliation Types:**

1 - **Order Reconciliation**

```text
For each open order in local state:
  Query exchange for order status
  If filled: Update position, record trade, remove from open orders
  If cancelled: Remove from open orders
  If partially filled: Update fill quantity, keep order open

For each order on exchange not in local state:
  Log warning (orphan order)
  Option: Cancel or adopt into local state
```

2 - **Position Reconciliation**

```text
exchange_positions = fetch_all_positions()
local_positions = load_from_db()

For each symbol:
  If exchange_qty != local_qty:
    Log discrepancy
    Option: Trust exchange (overwrite local) or flag for manual review
  Update local state with exchange values (avg_price, market_value)
```

3 - **Cash Reconciliation**

```text
exchange_cash = fetch_account_cash()
local_cash = calculate_from_trades()

If abs(exchange_cash - local_cash) > threshold:
  Log discrepancy
  Recalculate local from trade history
  If still differs: Trust exchange, log adjustment
```

**Frequency:** Run full reconciliation every 30 minutes, light sync every 5 minutes.

### Main Loop Architecture

Multiple concurrent loops with different frequencies:

| Loop | Frequency | Purpose |
| --- | --- | --- |
| **Price Loop** | 1 second | Fetch latest prices, update quotes |
| **Trading Loop** | 5 seconds | Run strategy, generate signals, submit orders |
| **Order Loop** | 0.5 seconds | Check order fills, update positions |
| **Display Loop** | 0.5 seconds | Refresh CLI dashboard |
| **Optimization Loop** | 1 hour | Re-run parameter optimization if enabled |
| **Reconciliation Loop** | 30 minutes | Full exchange state sync |

**Implementation Pattern:**

```text
async def main():
    async with TaskGroup() as tg:
        tg.create_task(price_loop(interval=1.0))
        tg.create_task(trading_loop(interval=5.0))
        tg.create_task(order_loop(interval=0.5))
        tg.create_task(display_loop(interval=0.5))
        tg.create_task(optimization_loop(interval=3600))
        tg.create_task(reconciliation_loop(interval=1800))
```

**Graceful Shutdown:** Handle SIGINT/SIGTERM, complete pending orders, save state to DuckDB.

---

## RSI Strategy Debugging

**Source:** Claude Conversation - Trading Bot Development

### Problem: Sortino Ratio = 0

During backtesting, RSI strategy produced Sortino ratio of exactly 0, indicating zero variance in returns.

**Root Causes Identified:**

1 - **Confidence Threshold Too High**

```text
# Original: Required 70% confidence
if signal.confidence >= 0.70:
    execute_trade()

# Problem: RSI rarely produces >70% confidence
# All signals filtered out → no trades → no returns → Sortino = 0
```

2 - **Trend Filter Too Restrictive**

```text
# Original: Required 20-period trend alignment
if rsi_signal == BUY and trend_20 == UP:
    confidence += 0.2

# Problem: By the time 20-period trend confirms, move is over
# Short-term RSI signals conflict with long-term trend filter
```

3 - **Position Sizing Edge Case**

```text
# Original: Size based on confidence
size = base_size * confidence

# Problem: Low confidence → tiny positions → negligible P&L
# Returns so small they round to zero variance
```

### Fixes Applied

1 - **Lower Confidence Threshold**

```text
# Fixed: Accept 50%+ confidence
if signal.confidence >= 0.50:
    execute_trade()
```

2 - **Shorter Trend Filter**

```text
# Fixed: Use 5-period trend for RSI alignment
if rsi_signal == BUY and trend_5 == UP:
    confidence += 0.15
```

3 - **Minimum Position Size**

```text
# Fixed: Enforce minimum size
size = max(min_size, base_size * confidence)
```

4 - **Debug Logging**

```text
# Added: Log every signal for diagnosis
logger.debug(f"Signal: {signal.action} conf={signal.confidence:.2f} "
             f"trend={trend} filtered={filtered}")
```

### Lessons Learned

- Always verify trades are actually executing before analyzing metrics
- Conservative filters compound: 3 mild filters can block 99% of signals
- Log signal generation and filtering separately
- Start with permissive thresholds, tighten after validating signal flow

---

## Self-Improving Agent Architectures

**Source:** arXiv papers - SICA and Darwin Gödel Machine

### SICA: Self-Improving Coding Agent

**Core Concept:** Self-referential loop where an LLM-based agent autonomously edits its own codebase to improve performance. No distinction between meta-agent and target agent—the same system improves itself.

**Self-Improvement Cycle:**

```text
1. Evaluate current agent on benchmarks
2. Select best-performing agent as meta-agent
3. Instruct it to identify and implement improvements
4. Test the updated agent
5. Repeat
```

**Key Components:**

1 - **Archive Analysis**

- Reviews performance data from previous iterations
- Identifies failure modes and successful patterns
- Informs substantial modifications through detailed inspection

2 - **Code Modification**

- Uses basic tools (file open/close, edits, shell execution)
- Directly modifies its own implementation
- Non-gradient-based: updates through prompt optimization and code changes

3 - **Oversight Mechanism**

- Asynchronous overseer LLM running in parallel
- Monitors execution traces every 30 seconds
- Detects pathological behaviors, loops, deviations
- Can intervene or cancel runs

**Results:** On SWE-Bench, improved from 17% to 53% (188% relative improvement).

### Darwin Gödel Machine

**Core Concept:** Self-referential, self-improving system using Darwinian evolution principles. Replaces formal proofs with empirical validation—modifications tested against benchmarks.

**Population-Based Approach:**

- Maintains archive of ALL generated agents (not just the best)
- Agents serve as "stepping stones for future generations"
- Parent selection weighted by performance and evolutionary lineage
- Intermediate solutions enable later breakthroughs

**Self-Modification Process:**

```text
1. Selection: Choose parents probabilistically from archive
2. Self-modification: Parents analyze failure modes, propose improvements
3. Evaluation: Test on benchmarks (2-stage: 10-task verify, 50-task expand)
4. Archival: Retain successful modifications
```

**Key Insight:** Only top performers evaluated on full benchmarks—addresses LLM stochasticity through subset evaluation.

**Results:** SWE-bench improved from 20% to 50% (150% improvement). Polyglot improved from 14.2% to 30.7% (116% improvement). Transfers across different LLM models and held-out benchmarks.

### Application to Self-Improving Trading Agents

| Coding Agent Concept | Trading Agent Adaptation |
| --- | --- |
| Benchmark evaluation | Backtest on historical data (Sharpe, Sortino, drawdown) |
| Code modification | Strategy logic, risk parameters, order execution code |
| Archive of agents | Population of strategy variants |
| Stepping stones | Intermediate strategies enable later discoveries |
| Oversight mechanism | Real-time monitoring prevents runaway strategies |
| Utility function | Balance profit, volatility, transaction costs |

**Concrete Implementation Ideas:**

1 - **Self-Diagnosing Strategy Improvement**

```text
For each backtest run:
  Analyze losing trades → identify pattern
  Propose parameter adjustment or logic change
  Implement and re-test
  Archive if improved, discard if worse
```

2 - **Population-Based Strategy Evolution**

```text
Maintain archive of strategy variants
Each generation:
  Select parents based on Sharpe + diversity
  Generate mutations (parameter tweaks, logic changes)
  Backtest offspring
  Archive successful variants
```

3 - **Empirical Validation over Theoretical**

- Don't require mathematical proof of strategy edge
- Let backtest results determine viability
- Walk-forward optimization as fitness function
- Out-of-sample testing as selection pressure

4 - **Safety Constraints for Trading**

- Maximum position size limits (hard-coded, not modifiable)
- Drawdown circuit breakers
- Transaction cost penalties in fitness function
- Sandbox paper trading before live deployment
- Traceable modification lineage for audit

**Key Differences from Traditional Optimization:**

| Traditional GA/WFO | Self-Improving Agent |
| --- | --- |
| Optimizes parameters only | Modifies strategy logic itself |
| Fixed strategy structure | Can add/remove indicators, change rules |
| Human designs fitness function | Agent can propose new metrics |
| Single best solution | Archive of diverse strategies |
| Optimization runs offline | Continuous improvement loop |

**Potential Risks:**

- Overfitting to historical patterns
- Strategy drift without human oversight
- Modifications optimized for backtest may fail live
- Requires robust sandboxing and position limits

---

## Open Questions and Research Directions

### Cross-Repository Questions

1. **Strategy Selection:** Which algorithm performs best in specific market conditions?

   - Trending: trading:MAIN vs trdr:MSS?
   - Ranging: trading:SAX vs trdr:Agents?
   - High volatility: Which risk controls most effective?

2. **Parameter Stability:** Do optimized parameters (GA, WFO) generalize across:

   - Different assets (stocks, crypto, forex)?
   - Different timeframes (1h, 4h, daily)?
   - Different market regimes (2008 crisis, 2020 COVID, 2022 rates)?

3. **Zero-Commission Dependency:** How sensitive is trading:SAX to commission changes?

   - Max commission rate preserving edge?
   - Impact on win rate and avg return?

4. **Capital Requirements:** Minimum capital for viability per strategy?

   - trading:MAIN with DCA: Estimate based on max position sizes
   - trading:SAX: Depends on max concurrent positions
   - trdr:MSS: Gradual sizing requirements
   - trdr:Options: Spread margins + max positions

5. **Live Trading Integration:** None of the branches show production deployment

   - Order routing and fills
   - Risk monitoring and alerts
   - Performance tracking dashboards
   - What infrastructure needed?

### Repository-Specific Questions

#### From trading Repository

**Source:** `2026-01-01-research-trading-repo-algos.md`

1. **Ellipse Research:** Continue geometric approach with non-linear fusion?

   - Test neural networks for curvature feature fusion
   - Expand curve families (parabola, hyperbola, catenary)
   - Multi-asset ensemble

2. **ML Model Drift:** Retraining frequency for trading:ML strategies?

   - Monitor prediction accuracy over time
   - Trigger retraining on degradation threshold

3. **Perplexity Ideas Implementation:** Which of 18 concepts most promising?
   - Prioritize quantitative concepts (no LLM dependency)
   - Fractal Market Hypothesis, Wavelet Decomposition first?

#### From trdr Repository

**Source:** `2026-01-01-research-trdr-repo-algos.md`

1. **Agent Parameter Optimization:** Integrate project-1 agents with helios-2 GA?

   - How many parameters total? (risk of overfitting)
   - Use walk-forward to validate

2. **Two-Pass MSS Generalization:** Does helios-2 dynamic MSS improve performance?

    - Backtest comparison: static vs two-pass MSS
    - Parameter sensitivity analysis

3. **Advanced Bar Types:** Do Lorentzian/Shannon bars improve strategies?

    - Test project-2 bars with helios-2 MSS
    - Compare vs standard time/dollar bars

4. **Options Multi-Asset:** Expand aapl-backtest to portfolio of stocks?
    - Correlation analysis
    - Portfolio-level risk management

---

## References

### Source Documents

- **trading Repository Research:** `thoughts/grantdickinson/2026-01-01-research-trading-repo-algos.md`
  - Repository: <git@github.com>:grant-d/trading.git
  - Branches: main, rskr, feat/ellipse, feat/perplexity-ideas
  - Commit: 0fe8f3d4e9d90ab390752bfd67ac8f159786376b

- **trdr Repository Research:** `thoughts/grantdickinson/2026-01-01-research-trdr-repo-algos.md`
  - Repository: <git@github.com>:grant-d/trdr.git
  - Branches: helios-1, helios-2, helios-3, project-1, project-2, aapl-backtest
  - Commit: f9ba9f1bb91ab213ecc07465db00a5f95d4d98ea

### Key File References by Repository

#### trading Repository

**Source:** `2026-01-01-research-trading-repo-algos.md`

**MAIN Branch:**

- `main:martingale-bot/adaptive_risk_seeker.py:934-2202` - Main trading loop
- `main:martingale-bot/risk_manager.py:150+` - Risk scoring (0-10 scale)
- `main:martingale-bot/multi_timeframe_signal_scorer.py:103-150` - Signal aggregation
- `main:martingale-bot/adaptive_trail_manager.py:75+` - Dynamic trailing stops

**RSKR/feat Branches:**

- `rskr:bot/sax/sax_pattern_discovery.py:100-400` - Pattern discovery
- `rskr:bot/sax/ml_strategy.py:50-300` - ML ensemble
- `feat/ellipse:bot/sax/sax_trader.py:600+` - Live trading
- `feat/ellipse:poc/src/ellipse_detection.py` - Ellipse detection
- `feat/perplexity-ideas:PERPLEXITY-IDEAS.md` - 18 advanced concepts

#### trdr Repository

**Source:** `2026-01-01-research-trdr-repo-algos.md`

**helios Branches:**

- `helios/factors.py` - MSS factor calculation
- `genetic_algorithm.py` - DEAP GA implementation
- `walk_forward_optimizer.py` - WFO with consensus
- `regime_strategy.py` - Two-pass dynamic MSS

**project Branches:**

- `packages/cli/src/agents/*.ts` - Multi-agent system
- `src/pipeline/buffer-pipeline.ts` - ETL architecture
- `src/transforms/` - Advanced bar types

**aapl-backtest:**

- `aapl_hma_backtest/strategy/hma_bull_put_strategy.py` - Options strategy
- `aapl_hma_backtest/options/black_scholes.py` - Pricing model

### External References

- DEAP Documentation: <https://deap.readthedocs.io>
- Black-Scholes: Hull, Options, Futures, and Other Derivatives
- Fractional Differentiation: de Prado, Advances in Financial Machine Learning
- SAX Representation: Lin et al., "Experiencing SAX: a novel symbolic representation of time series"
- SICA: <https://arxiv.org/html/2504.15228v2>
- Godel: <https://arxiv.org/html/2505.22954v2>

---

## Appendix: Pedigree Index

Quick reference for finding original code by section:

| Algorithm/Concept | Repository | Branch | Source Document | Key Files |
| --- | --- | --- | --- | --- |
| MSS Regime Detection | trdr | helios-1/2/3 | research-trdr-repo-algos.md | helios/factors.py |
| Multi-Timeframe Scoring | trading | MAIN | research-trading-repo-algos.md | multi_timeframe_signal_scorer.py |
| SAX Pattern Trading | trading | RSKR/feat | research-trading-repo-algos.md | sax_pattern_discovery.py |
| ML Ensemble | trading | RSKR/feat/perplexity | research-trading-repo-algos.md | ml_strategy.py |
| DEAP Genetic Algorithm | trdr | helios-2 | research-trdr-repo-algos.md | genetic_algorithm.py |
| Walk-Forward Optimization | trdr | helios-2 | research-trdr-repo-algos.md | walk_forward_optimizer.py |
| Multi-Agent System | trdr | project-1 | research-trdr-repo-algos.md | packages/cli/src/agents/*.ts |
| Advanced Bar Types | trdr | project-2 | research-trdr-repo-algos.md | src/transforms/ |
| HMA Bull Put Spreads | trdr | aapl-backtest | research-trdr-repo-algos.md | hma_bull_put_strategy.py |
| Black-Scholes Pricing | trdr | aapl-backtest | research-trdr-repo-algos.md | black_scholes.py |
| Ellipse Detection | trading | feat/ellipse | research-trading-repo-algos.md | ellipse_detection.py |
| Perplexity Ideas | trading | feat/perplexity-ideas | research-trading-repo-algos.md | PERPLEXITY-IDEAS.md |
| Adaptive Risk Management | trading | MAIN | research-trading-repo-algos.md | risk_manager.py |
| Dollar Bars | Both | helios-1, project-2 | Both docs | dollar_bars.py, transforms/ |
| Fractional Differentiation | Both | helios-3, project-2 | Both docs | base_data_loader.py, transforms/ |
