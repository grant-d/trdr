---
date: 2026-01-01
author: grantdickinson
repo: grant-d/trading
branch: main
git_commit: 0fe8f3d4e9d90ab390752bfd67ac8f159786376b
topic: "Trading Algorithm Analysis Across All Branches"
tags: [research, codebase, trading-algorithms, martingale, sax-patterns, ellipse, multi-timeframe]
---

# Research: Trading Algorithms in `grant-d/trading` Repository

## Research Question

Comprehensive analysis of all trading algorithms across repository branches (main, rskr, feat/ellipse, feat/perplexity-ideas) to document core algorithms, approaches, patterns, architecture, viability, and testing infrastructure for pure code-based trading strategies.

## Summary

Repository contains four distinct algorithmic trading approaches across branches:

1. **MAIN** - Adaptive Martingale with Multi-Timeframe Analysis (production-ready, comprehensive risk controls)
2. **RSKR** - SAX Pattern Trading + ML Enhancement (service-oriented architecture, multi-strategy support)
3. **FEAT/ELLIPSE** - SAX Production + Geometric Research (90.7% win rate backtested + ellipse perspective PoC)
4. **FEAT/PERPLEXITY-IDEAS** - ML-Enhanced SAX + Research Ideation (1737% backtest returns, 18 advanced concepts)

All branches focus on pure algorithmic approaches (no LLM/transformer implementations). Each branch represents different stages of evolution and experimentation with pattern-based trading strategies.

---

## Detailed Findings

### Branch 1: MAIN - Adaptive Martingale Multi-Timeframe System

**Status**: Production-ready, actively maintained
**Core Algorithm**: Adaptive Martingale DCA (Dollar-Cost Averaging) with Multi-Timeframe Signal Aggregation

#### 1.1 Core Trading Approach

**Primary Strategy**: Martingale-based position scaling with exponential rung spacing

**Key Innovation**: Multi-timeframe analysis (5h/15h/1h) with weighted signal aggregation and risk-adaptive parameter scaling.

**Entry Logic**:

- Multi-timeframe signal strength ≥ 0.6 (configurable)
- Signal confidence ≥ 0.5
- Volume confirmation (min score: 45.0)
- Bullish direction (STRONG_BUY, BUY, WEAK_BUY)
- Risk assessment approval

**Position Sizing**:

- Base position: configurable USD amount
- Maximum DCA levels: 5-6 buys
- Rung spacing: Exponential (N=2.618) - rungs at 1x, 1.27x, 1.41x, 1.68x, 2.0x depth
- Dynamic spacing based on ATR volatility (min 1.5%, max 5%)

#### 1.2 Multi-Timeframe Analysis Framework

**Integrated Timeframe System**:

| Timeframe | Purpose | Analyzer | Weight |
| --- | --- | --- | --- |
| 5h (Micro) | Entry timing | micro_structure_analyzer.py | 30% |
| 15h (Trend) | Primary trend | trend_momentum_analyzer.py | 40% |
| 1h (S/R) | Key levels | support_resistance_analyzer.py | 30% |

**Micro-Structure (5h)**:

- Candlestick patterns: engulfing, hammer, doji, breakouts
- Pattern strength: weak/moderate/strong/very_strong
- Micro-trend direction with R² fit quality

**Trend/Momentum (15h)**:

- Moving averages: SMA(20), SMA(50), EMA(12), EMA(26)
- MACD with divergence detection
- RSI oversold/overbought with divergence
- 7-level trend classification (strong_uptrend → strong_downtrend)

**Support/Resistance (1h)**:

- Pivot highs/lows, swing levels, psychological levels
- Fibonacci retracements, volume profile POC
- Confluence zone identification
- Level strength classification with touch count

**Signal Aggregation** (main:martingale-bot/multi_timeframe_signal_scorer.py):

```text
composite_score = (micro*0.3 + trend*0.4 + sr*0.3)
+ alignment_bonus (20% if all signals align)
+ confluence_bonus (15% for S/R proximity)
```

#### 1.3 Risk Management System

**Multi-Layer Risk Architecture**:

**Risk Scoring (0-10 scale)** based on:

- GARCH volatility (conditional heteroskedasticity)
- ATR-based volatility percentage
- RSI extremes
- VWAP distance
- Autocorrelation (mean reversion vs trending)
- Stationarity testing
- Market structure classification

**Risk-Adaptive Parameters**:

| Risk Score | Strategy | Trail Factor | Max DCA | Position Size | Status |
| --- | --- | --- | --- | --- | --- |
| 0-2 | Scalping | 0.5-1.5% | 5 | 150% | Aggressive |
| 3-4 | Momentum | 1.0-2.0% | 5 | 125% | Growth |
| 5-6 | Balanced | 1.5-2.5% | 4 | 100% | Normal |
| 7-8 | Conservative | 2.0-3.0% | 3 | 75% | Defensive |
| 9-10 | Halt | 3.0%+ | 2 | 50% | Emergency |

**Daily Loss Limits**:

- Default: 5% daily loss limit
- Portfolio-wide tracking in data/bot.risk_state.json
- Trading halt on breach
- Weekly drawdown monitoring (10% default)

**Smart DCA Conditions**:

- Won't DCA if position down >10%
- Won't exceed 2.5x base order size
- RSI capitulation scaling (RSI <20 → 1.5x DCA size)
- Technical structure validation (blocks DCA in strong bearish trends)

#### 1.4 Exit Strategy

**Two-Part Stop Calculation**:

1. **Break-even**: Entry price + fees (0.5% total)
2. **Trailing stop**: Dynamic based on position state

**Smart Trail Activation**:

- Volatility-adjusted (ATR-based)
- Position age (48h+ → accept break-even)
- Profit level (tightens as profit increases)
- Risk score (tight in high risk environments)

**Profit Targeting** (main:martingale-bot/dynamic_profit_target.py):

- Base target: 2.5-3.0% position-dependent
- Small positions (<$200): 3% base
- Medium positions ($200-$1000): 2.5% base
- Large positions (>$1000): 2% base
- Volatility multiplier: 1.5x in high volatility
- Setup strength: 1.25x for strong signals
- Time-based: 0.5% per day (max 2x multiplier)

**Deep DCA Override**: After 60% of max DCA levels, force break-even exit to prevent compounding losses

#### 1.5 Trailing Stop Management

**Adaptive Trail Manager** (main:martingale-bot/adaptive_trail_manager.py):

**Trail Parameters by Risk Level**:

| Risk Level | Base Trail | Min Trail | Max Trail | Profit Tightening |
| --- | --- | --- | --- | --- |
| Low | 0.5% | 0.25% | 1.5% | 5% per 1% profit |
| Medium | 1.0% | 0.5% | 2.5% | 3% per 1% profit |
| High | 1.5% | 1.0% | 3.5% | 2% per 1% profit |

**Trail Calculation Formula**:

```text
trail_pct = base_trail
          + (volatility_adjustment * ATR_pct)
          - (profit_tightening_factor * unrealized_profit_pct)
          - (time_decay_factor * hours_in_position)

clamped to [min_trail_pct, max_trail_pct]
```

#### 1.6 Market Regime Detection

**Regime Classification** (main:martingale-bot/market_regime_detector.py):

- **Trend Regime**: ADX-based (ADX >25 = strong trend)
- **Volatility Regime**: ATR percentage (ATR >2.5% = high vol)
- **Direction**: +DI vs -DI comparison
- **Mean Reversion**: Autocorrelation testing
- **GARCH Volatility**: Conditional volatility modeling

**Regime-based Adjustments**:

- Trending: Reduced DCA levels, wider trails
- Mean reverting: Increased DCA levels, tighter entry
- High volatility: Scaled-down positions, wider trails
- Low volatility: Normal parameters

#### 1.7 Architecture

**Component Structure**:

```text
martingale-bot/
├── adaptive_risk_seeker.py         # Main decision loop (2202 lines)
├── adaptive_risk_buyer.py          # Entry execution
├── adaptive_risk_seller.py         # Exit execution
├── risk_manager.py                 # Risk calculations
├── multi_timeframe_signal_scorer.py # Signal aggregation
├── adaptive_trail_manager.py       # Dynamic stops
├── micro_structure_analyzer.py     # 5h patterns
├── trend_momentum_analyzer.py      # 15h trends
├── support_resistance_analyzer.py  # 1h S/R levels
├── volume_analyzer.py              # Volume confirmation
├── market_regime_detector.py       # Regime classification
├── profit_lock_manager.py          # Profit protection
├── dynamic_profit_target.py        # Adaptive targets
├── performance_tracker.py          # Real-time metrics
└── tests/                          # Comprehensive test suite
```

**Data Models**: Use dataclasses for type safety, immutable signal objects, persistent state files

**Integration**: Alpaca API for live trading, CSV logging (bot.order.csv), risk state persistence

#### 1.8 Testing Infrastructure

**Comprehensive Test Coverage** (main:martingale-bot/tests/):

- test_multi_timeframe_system.py - Full system integration
- test_integration_volume_backtest.py - Volume validation
- test_adaptive_trail_manager.py - Trail calculations
- test_risk_manager.py - Risk scoring and parameters
- test_micro_structure_analyzer.py - Pattern detection
- test_trend_momentum_analyzer.py - Trend analysis
- test_performance_tracker.py - Metrics calculation
- test_market_regime_detector.py - Regime classification

#### 1.9 Key Strengths

✓ Production-grade risk controls (daily limits, position sizing, circuit breaker)
✓ Multi-timeframe validation reduces false signals
✓ Adaptive parameters respond to market conditions
✓ Comprehensive testing infrastructure
✓ Dynamic trailing stops manage exits intelligently
✓ Smart DCA prevents runaway losses

#### 1.10 Key Files Reference

| File | Lines | Purpose |
| --- | --- | --- |
| adaptive_risk_seeker.py | 934-2202 | Main trading loop, entry/exit logic |
| risk_manager.py | 150+ | Risk scoring (0-10), adaptive parameters |
| multi_timeframe_signal_scorer.py | 103-150 | Signal weighting and aggregation |
| adaptive_trail_manager.py | 75+ | Dynamic trailing stop calculation |
| performance_tracker.py | 80+ | Real-time metrics (Sharpe, drawdown, etc) |

---

### Branch 2: RSKR - SAX Pattern Trading + Multi-Strategy Architecture

**Status**: Production architecture, service-oriented design
**Core Algorithm**: SAX (Symbolic Aggregate Approximation) Pattern Matching + ML Enhancement

#### 2.1 Core Trading Approach

**Primary Algorithm**: SAX converts price time-series into symbolic patterns for universal pattern discovery and statistical validation.

**SAX Encoding Process** (rskr:bot/sax/sax_pattern.py):

1. **Z-normalize**: (price - mean) / std with outlier clipping (±3σ via MAD)
2. **PAA**: Compress to segments (e.g., 30 bars → 5 segments)
3. **Discretize**: Map to alphabet {a,b,c,d} using Gaussian breakpoints
4. **Enhance**: Add momentum (↑/↓/→), range indicators (F/W/N), consecutive extremes

**Example Pattern**: "abdc↑F2" = Low-High-VeryHigh-High + Rising + FullRange + 2 consecutive extremes

#### 2.2 Pattern Discovery Algorithm

**Golden Pattern Discovery** (rskr:bot/sax/sax_pattern_discovery.py):

1: **Generate Training Patterns**

- Load pre-cutoff data (e.g., pre-2024-01-01)
- Generate SAX patterns across configured parameters
- Pool patterns across 3+ stocks for universal validation

2: **Golden Pattern Criteria**

- Minimum occurrences: 10-20 (varies by timeframe)
- Win rate threshold: 55-65% depending on timeframe
- Mean return: 0.5-2% per trade
- Cross-symbol requirement: Appears in 2+ different assets

3: **Hold Period Optimization**

- Test hold periods [3-14 bars] per timeframe
- Select optimal hold maximizing: E[return] = mean_return × win_rate
- Compute Sharpe ratio and max drawdown per pattern

**Output**: GoldenPattern dataclass with pattern, hold_bars, mean_return, win_rate, sample_size, num_stocks, sharpe, max_drawdown

**Critical**: Uses pre-discovery cutoff date to prevent lookahead bias

#### 2.3 ML-Enhanced Pattern Trading

**ML Classifier Stack** (rskr:bot/sax/ml_strategy.py):

- Primary: XGBoost with class weighting
- Secondary: Random Forest
- Tertiary: Neural Network (MLP)
- Ensemble: Voting classifier combining all three

**Feature Engineering**:

- SAX Pattern features: length, unique chars, entropy, transition frequency
- Momentum: direction (↑/↓/→), range type (F/W/N)
- Market context: volatility, trend, RSI, MACD, Bollinger Bands, ATR, GARCH, S/R distance, volume, Williams %R, CCI, MFI, ADX

**Training Strategy**:

- Time-series split (no future leak)
- Class weight balancing
- Calibrated probability outputs
- Hold-out test period validation

#### 2.4 Multi-Strategy Support

**Additional Strategies**:

**RSI Strategy** (rskr:bot/strategies/rsi_strategy.py):

- Buy: RSI < 35 (oversold) + optional trend filter
- Sell: RSI > 65 (overbought)
- Parameters: rsi_period (14), profit_target (5%), stop_loss (3%), trailing_stop (2%)

**GARCH Strategy** (rskr:bot/strategies/garch_strategy.py):

- Fit ARCH model to recent price changes
- Enter when volatility in favorable range [2%, 3%]
- Position size adjusts with volatility (smaller in high vol)

#### 2.5 Architecture

**Service-Oriented Design**:

```text
bot/
├── strategies/
│   ├── strategy.py                 # Base Strategy class
│   ├── rsi_strategy.py
│   ├── garch_strategy.py
│   └── indicators.py               # Technical indicators
├── sax/
│   ├── sax_pattern.py              # SAX generation
│   ├── sax_pattern_discovery.py   # Pattern mining
│   ├── sax_backtester.py           # Backtesting
│   ├── sax_analyzer.py             # Interactive analysis
│   ├── sax_trader.py               # Live trading
│   ├── ml_module.py                # ML integration
│   ├── ml_strategy.py              # ML classifier
│   ├── signal_processor.py         # Signal execution
│   └── config.py                   # Configuration
├── trading/
│   ├── trading_bot.py              # Main async loop
│   ├── signal_processing_service.py
│   ├── order_service.py
│   ├── order_trail_update_service.py
│   ├── market_data_service.py
│   ├── position_sync_service.py
│   ├── performance_service.py
│   ├── reconciliation_service.py
│   └── health_monitor.py
├── optimizer/
│   ├── strategy_optimizer.py       # Parameter optimization
│   ├── backtester.py               # Fast backtest
│   ├── walk_forward.py             # WFO implementation
│   └── safety_validator.py         # Parameter limits
├── positions/
│   └── position_manager.py         # Position tracking
├── exchanges/
│   ├── alpaca_live_exchange.py
│   └── paper_exchange.py
├── data/
│   ├── data_manager.py             # Fetch/cache data
│   └── market_data_cache.py
└── reconciliation/
    └── reconciler.py               # Order/position sync
```

**Design Patterns**:

- Strategy Pattern: Pluggable strategies via base class
- Service-Oriented: Separate services for specific concerns
- State Management: BotState, SymbolState with JSON persistence
- Twin Models: TwinOrder (BUY + SELL pair), TwinPosition
- Reconciliation: ExchangeReconciler syncs bot state with exchange

#### 2.6 Risk Management

**Position Sizing**:

- Base: position_size_pct of capital (e.g., 2%)
- Adjusted by signal strength: WEAK (0.5x), MODERATE (0.75x), STRONG (1.0x), VERY_STRONG (1.25x)
- Volatility adjustment for GARCH
- Max concurrent positions cap (e.g., 3)

**Trailing Stops** (ATR-based):

- Entry stop: ABOVE current price (long positions)
- Initial distance: 2.0 × ATR
- Tightening: As profit increases, ATR multiplier decays
- Per-timeframe configuration
- Only ratchets up (never down for longs)

**Hard Stops**:

- Profit target: e.g., 5%
- Stop loss: e.g., 3%
- Per-strategy override

**HealthMonitor**:

- Error rate tracking (fail if >20%)
- Daily loss limit hard stop
- Data staleness checks (<5 min old)
- Order success rate (>80%)
- Circuit breaker on health degradation

#### 2.7 Walk-Forward Optimization

**WFO Process** (rskr:bot/optimizer/walk_forward.py):

1. Train Phase: 80% historical data for pattern discovery + parameter optimization
2. Test Phase: Apply to hold-out 20%
3. Walk Forward: Slide window, repeat
4. Report: Aggregate across all test periods

**Safety Validator**: Limits parameter changes (max 200% change), enforces minimum trades, cooldown periods

#### 2.8 Multi-Timeframe Implementation

**Configuration**:

```yaml
timeframes:
  1h:
    window_sizes: [20, 30, 40]
    paa_sizes: [4, 5, 6]
    min_win_rate: 0.55
  4h:
    window_sizes: [20, 35]
    paa_sizes: [5, 6, 7]
    min_win_rate: 0.65
```

**Multi-TF Scoring**: Sum of (pattern win_rate × lookback_multiplier) across timeframes

**Optimal Combination**: 1h trading with 4h context yielded 65.62% returns in backtesting

#### 2.9 Testing & Validation

**SAX Backtester**:

- Train/test split with cutoff date
- Fee & slippage included (0.25% fees, 0.1% slippage)
- Compound returns
- Multiple positions (respects max limit)
- Trailing stops simulated
- Metrics: win rate, avg return, Sharpe, Sortino, Calmar, max drawdown

**Backtest Results (1-year test)**:

- Return: 65.62%
- Win Rate: 60.5%
- Sharpe Ratio: 1.23
- Max Drawdown: -12.5%
- Avg trades/month: 12-15

**Paper Trading**: PaperExchange simulates live fills with realistic logic

#### 2.10 Key Differences from MAIN

| Aspect | main | rskr |
| --- | --- | --- |
| Core Algorithm | Martingale seeker | SAX patterns + ML |
| Strategies | 1 (martingale) | 3 (SAX, RSI, GARCH) + ML |
| Entry Logic | Volume/trend/RSI | Pattern matching confidence |
| Exit Strategy | Dynamic profit lock | ATR-based trailing |
| Optimization | GA (disabled) | WFO (enabled) |
| Architecture | Monolithic | Service-oriented (10+ services) |
| State | File-based | JSON + reconciliation |

#### 2.11 Key Files Reference

| File | Purpose |
| --- | --- |
| rskr:bot/sax/sax_pattern_discovery.py | Pattern discovery & filtering |
| rskr:bot/sax/ml_strategy.py | ML classifier training |
| rskr:bot/sax/sax_pattern.py | SAX generation |
| rskr:bot/trading/trading_bot.py | Main async loop |
| rskr:bot/trading/signal_processing_service.py | Signal → Order conversion |
| rskr:bot/optimizer/walk_forward.py | WFO optimization |

---

### Branch 3: FEAT/ELLIPSE - SAX Production + Geometric Research

**Status**: Production SAX + Research Ellipse PoC
**Core Algorithms**: (1) SAX Pattern Trading (mature), (2) Perspective Reconstruction Engine (experimental)

#### 3.1 SAX Pattern Trading System

**Identical to RSKR** with following enhancements:

**Symbol-Specific Patterns**: Patterns discovered per symbol (no cross-asset contamination)

**Multi-Timeframe Pooling**:

- 1h trading uses patterns from 1h + 4h + 1d data
- 4h trading uses patterns from 4h + 1d data
- Daily trading uses 1d patterns only

**Holding Period Testing**: Optimal hold at 6 bars (24h on 4h timeframe)

**Performance** (feat/ellipse:bot/sax/PRD.md):

- Total Return: 1737%
- Win Rate: 90.7%
- Sharpe Ratio: 6.38
- Max Drawdown: <10%
- Trades: 107 executed
- Avg Return per Trade: 2.80%

**Validation**: Pattern discovery uses pre-2024 data, backtest on 2024+ (out-of-sample)

#### 3.2 Ellipse/Perspective Algorithm (Research PoC)

**Novel Hypothesis**: Price charts are 2D projections of higher-dimensional manifolds containing predictive information

**Concept**: Detect and analyze ellipses in price data to:

1. Estimate perspective distortion parameters
2. Reconstruct underlying manifold geometry
3. Extract trading signals from curvature and focal dynamics

**Ellipse Detection** (feat/ellipse:poc/src/ellipse_detection.py):

**Process**:

1. Sliding windows (48-192 bars)
2. Z-normalization to [-1, 1] with outlier clipping
3. PAA compression (48 bars → 5 segments)
4. Conic fitting: Ax²+Bxy+Cy²+Dx+Ey+F=0 (least-squares)
5. Ellipse extraction: center (h,k), semi-axes (a,b), rotation (θ), eccentricity (e)
6. Quality scoring: residual-based filtering

**Parameters**:

- Window: 48-192 bars
- Stride: 4-8 bars
- Residual threshold: 0.12 (dynamic per window length)
- Min eccentricity: 0.05

**Perspective Reconstruction** (feat/ellipse:poc/src/perspective_solver.py):

From detected ellipse, solve for:

- Intrinsic radius: R (original 3D circle)
- Viewing angle: θ (tilt from projection plane)
- Azimuth: φ (rotation around normal)
- Projection matrix: 3×3 matrix (3D→2D)
- Normal vector: Surface normal in 3D

**Formula**:

```text
aspect_ratio = semi_major / semi_minor
tilt_angle θ = arccos(1 / aspect_ratio)
normal = [sin(θ)·cos(φ), sin(θ)·sin(φ), cos(θ)]
view_distance = |intrinsic_radius / sin(θ)|
```

**Manifold-Derived Signals** (feat/ellipse:poc/src/perspective_metrics.py):

Features extracted:

- view_angle: Viewing angle degrees (0°=head-on, 90°=edge-on)
- distortion: Aspect ratio - 1
- scale_factor: √(a·b)
- curvature_gaussian: Estimated Gaussian curvature
- curvature_delta: Change in curvature
- focal_distance: 2ae
- focal_velocity: Rate of focal point movement
- rotation_velocity: Rate of angle change
- distortion_acceleration: Second derivative of aspect ratio

#### 3.3 Ellipse PoC Validation Results

**Performance Summary** (feat/ellipse:poc/README.md):

| Asset | Timeframe | Best Feature | Return/Trade | Sharpe | Status |
| --- | --- | --- | --- | --- | --- |
| AAPL | 1h | close | ~3.8 bps | 0.28 | Weak |
| SPY | 4h | curvature_delta | ~30 bps | 1.9 | Promise |
| BTC | 1d | close | 200-400 bps | N/A | N/A |

**Key Finding**: Curvature features showed modest promise on SPY 4h but did NOT provide incremental lift when combined with price-based signals in linear fusion models.

**Conclusion**: PoC viable for continued research but not production-ready. Focus areas:

- Stronger curve family classification (parabola, hyperbola, catenary, log spiral)
- Non-linear signal fusion
- Multi-asset ensemble approaches

#### 3.4 Architecture

**Dual Module Structure**:

```text
feat/ellipse:
├── bot/sax/                    # Production SAX system (identical to rskr)
│   ├── sax_pattern.py
│   ├── sax_pattern_discovery.py
│   ├── sax_backtester.py
│   ├── sax_trader.py
│   └── [...other SAX files]
└── poc/                        # Ellipse research PoC
    ├── src/
    │   ├── ellipse_detection.py       # Conic fitting + extraction
    │   ├── perspective_solver.py      # 3D projection recovery
    │   ├── perspective_metrics.py     # Feature engineering
    │   ├── pipeline.py                # End-to-end PoC
    │   ├── run_poc.py                 # CLI entry point
    │   └── feature_backtest.py        # Quick validation
    ├── design.md                      # Geometric theory
    └── README.md                      # PoC guide
```

#### 3.5 Key Differences from RSKR

| Aspect | rskr | feat/ellipse |
| --- | --- | --- |
| SAX System | Core production | Enhanced with symbol-specific patterns |
| Pattern Scope | Cross-symbol validation | Symbol-specific + multi-TF pooling |
| Performance | 65.62% return | 1737% return (different test period) |
| Additional Research | None | Ellipse/perspective PoC |
| Documentation | Technical only | Technical + research ideation |

#### 3.6 Key Files Reference

**SAX Production**:

- feat/ellipse:bot/sax/sax_trader.py - Live trading (1851+ lines)
- feat/ellipse:bot/sax/sax_pattern_discovery.py - Pattern mining
- feat/ellipse:bot/sax/sax_backtester.py - Validation

**Ellipse Research**:

- feat/ellipse:poc/src/ellipse_detection.py - Core detection algorithm
- feat/ellipse:poc/src/perspective_solver.py - Projection recovery
- feat/ellipse:poc/src/perspective_metrics.py - Feature engineering
- feat/ellipse:poc/design.md - Geometric theory & implementation

---

### Branch 4: FEAT/PERPLEXITY-IDEAS - ML-Enhanced SAX + Advanced Concepts

**Status**: Production SAX + Research Ideation Framework
**Core Algorithms**: (1) SAX Pattern Trading (mature), (2) ML-Enhanced Pattern Prediction, (3) 18 Advanced Trading Concepts

#### 4.1 SAX Pattern Trading System

**Identical to feat/ellipse** SAX implementation with same performance characteristics:

- Total Return: 1737%
- Win Rate: 90.7%
- Sharpe Ratio: 6.38
- Symbol-specific patterns on 4h timeframe (stocks only, zero-commission)

#### 4.2 ML-Enhanced Pattern Trading

**Full ML Integration** (feat/perplexity-ideas:bot/sax/ml_strategy.py):

**Classifier Stack**:

- Primary: XGBoost with class weighting for imbalanced data
- Secondary: Random Forest for feature robustness
- Tertiary: Neural Network (MLP) for non-linear patterns
- Ensemble: Voting classifier combining all three

**Feature Engineering**:

From SAX Pattern:

- Pattern length, unique characters, entropy, transition frequency
- Momentum direction (↑/↓/→), range type (F/W/N)
- Consecutive extremes count

Market Context (preserved):

- Volatility (realized), trend, price position, momentum
- RSI, SMA crossovers, MACD, Bollinger Bands
- ATR, GARCH volatility, support/resistance distance
- Volume patterns, Williams %R, CCI, MFI, ADX

**Feature Selection**: MarketAwareFeatureSelector preserves critical market features while selecting top-K pattern features using mutual information

**Training Strategy**:

- Time-series split (no future leak)
- Class weight balancing
- Calibrated probability outputs for confidence scoring
- Hold-out test period validation

**Signal Generation**:

```python
signal = ml_module.generate_signal(
    symbol=symbol,
    current_ohlcv=current_window,
    pattern=detected_pattern,
    timeframe=timeframe
)
# Returns:
# - action: 'BUY' / 'SELL' / 'HOLD'
# - confidence: 0.0-1.0 (model probability)
# - predicted_return: Expected return %
# - hold_bars: Optimal holding period
```

#### 4.3 Perplexity Ideas Framework

**18 Advanced Trading Concepts** (feat/perplexity-ideas:PERPLEXITY-IDEAS.md):

**Quantitative Concepts** (implemented or ideated):

1. **Quantum Price Formation Theory** - Price as eigenvalues of bid/ask operator matrices
2. **Fractal Market Hypothesis** - Hurst exponent & MFDFA for regime detection
3. **Lévy Flight Recognition** - Heavy-tailed price movements
4. **Wavelet Decomposition** - Time-frequency analysis of price shocks
5. **Multi-EMA Quantum Synthesis** - Interference patterns across EMA periods
6. **Attention Matrix Microstructure** - Transformer attention as dependency graphs
7. **Latent Space Arbitrage** - Embedding distance as predictor
8. **Frequency-Domain Attention** - FFT + multi-head attention on spectral components
9. **Cross-Attention Market Coupling** - Inter-market information flow
10. **Stochastic Embedding Transitions** - Probability distribution evolution

**LLM-Augmented Strategies** (ideation only, not implemented per instructions):
11. **Chain-of-Alpha** - Dual-chain LLM for factor discovery
12. **Adaptive Kalman Filters** - Transformer-informed filtering
13. **Transformer-Enhanced HMM** - Attention-modulated state transitions
14. **Multi-Asset Embedding Topology** - Joint encodings for arbitrage
15. **X-Trend Few-Shot** - 500-pattern library with cross-attention
16. **Attention-Weighted Portfolio** - Centrality-based sizing
17. **Multi-Timeframe Order Flow** - Auction market theory
18. **Granger Causality Attention** - Causal discovery via attention

**Purpose**: Documents cutting-edge algorithmic concepts for future research and experimentation. Provides theoretical foundation for next-generation features.

**Status**: Local LLM implementations IGNORED per instructions. Focus on pure quantitative algorithms.

#### 4.4 Architecture Enhancements

**Per-Timeframe Trailing Stops** (feat/perplexity-ideas:bot/sax/config.py):

Each timeframe has independent trailing stop configuration:

```yaml
timeframes:
  1h:
    trailing_stop:
      activation: 0.0        # Activate immediately
      initial_stop_atr: 2.0  # 2 × ATR
      tightness: 0.0         # Fixed, no tightening
  4h:
    trailing_stop:
      activation: 0.0
      initial_stop_atr: 3.0  # Wider for 4h
      tightness: 0.0
```

**Trailing Sell Feature**: Exit-side trailing stops (not just entry)

**Enhanced Signal Processor**: Position sizing, fees, slippage, compounding integrated

#### 4.5 Key Differences from FEAT/ELLIPSE

| Aspect | feat/ellipse | feat/perplexity-ideas |
| --- | --- | --- |
| SAX System | Production-ready | Production-ready (identical) |
| ML Integration | Optional ML module | Full ML ensemble integration |
| Research Focus | Geometric (ellipse) | Algorithmic ideation (18 concepts) |
| Trailing Stops | Basic ATR | Per-timeframe + trailing sell |
| Documentation | PoC-focused | Ideation framework |

#### 4.6 Key Files Reference

**Core Algorithm**:

- feat/perplexity-ideas:bot/sax/ml_strategy.py - ML classifier training
- feat/perplexity-ideas:bot/sax/ml_module.py - Unified ML/rule-based interface
- feat/perplexity-ideas:bot/sax/signal_processor.py - Enhanced execution engine
- feat/perplexity-ideas:PERPLEXITY-IDEAS.md - 18 advanced concepts documentation

---

## Architectural Comparison Across Branches

### Signal Flow Comparison

**MAIN (Martingale Multi-Timeframe)**:

```text
OHLCV → Multi-TF Fetcher → [Micro/Trend/S/R Analyzers] →
Signal Scorer → Risk Manager → Seeker (Entry/DCA/Exit) →
Buyer/Seller → Performance Tracking
```

**RSKR (SAX Service-Oriented)**:

```text
OHLCV → DataManager → SAX Generator → Pattern Discovery →
ML Strategy (optional) → Signal Processing Service →
Order Service → Exchange → Reconciliation → Performance
```

**FEAT/ELLIPSE (Dual System)**:

```text
SAX Path: [Same as RSKR]
Ellipse Path: OHLCV → Window Scan → Ellipse Detection →
Perspective Solver → Feature Extraction → Correlation Analysis
```

**FEAT/PERPLEXITY-IDEAS (ML-Enhanced SAX)**:

```text
OHLCV → DataManager → SAX Generator → Pattern Discovery →
ML Ensemble Classifier → Signal Processor (enhanced) →
Order Service → Trailing Stop Manager (per-TF) → Performance
```

### Risk Management Comparison

| Feature | MAIN | RSKR | FEAT/ELLIPSE | FEAT/PERPLEXITY-IDEAS |
| --- | --- | --- | --- | --- |
| Position Sizing | Risk-adaptive (0-10 scale) | Signal strength-based | Signal strength-based | Signal strength-based |
| Trailing Stops | Adaptive Trail Manager | ATR-based, per-TF | ATR-based, per-TF | ATR-based + trailing sell, per-TF |
| Stop Loss | Smart DCA conditions | Hard stops | Hard stops | Hard stops |
| Daily Limits | 5% daily loss limit | Health monitor | Health monitor | Health monitor |
| Circuit Breaker | Yes (risk state) | Yes (health monitor) | Yes (health monitor) | Yes (health monitor) |
| DCA/Martingale | Yes (exponential spacing) | No | No | No |

### Testing Infrastructure Comparison

| Branch | Test Framework | Validation Method | Performance Metrics |
| --- | --- | --- | --- |
| MAIN | pytest suite (15+ files) | Integration tests | Real-time tracker |
| RSKR | pytest + walk-forward | WFO, paper trading | Backtest + live metrics |
| FEAT/ELLIPSE | SAX backtester + PoC tests | Out-of-sample validation | 1737% return (2024-2025) |
| FEAT/PERPLEXITY-IDEAS | SAX backtester + ML validation | Time-series CV + OOS | 1737% return (2024-2025) |

---

## Code References

### MAIN Branch Key Files

- `main:martingale-bot/adaptive_risk_seeker.py:934-2202` - Main trading loop
- `main:martingale-bot/risk_manager.py:150+` - Risk scoring
- `main:martingale-bot/multi_timeframe_signal_scorer.py:103-150` - Signal aggregation
- `main:martingale-bot/adaptive_trail_manager.py:75+` - Dynamic trailing stops
- `main:martingale-bot/performance_tracker.py:80+` - Real-time metrics

### RSKR Branch Key Files

- `rskr:bot/sax/sax_pattern_discovery.py:100-400` - Pattern discovery
- `rskr:bot/sax/ml_strategy.py:50-300` - ML classifier
- `rskr:bot/trading/trading_bot.py:100-200` - Main async loop
- `rskr:bot/optimizer/walk_forward.py` - WFO optimization
- `rskr:bot/common/trailing_stop_tracker.py` - Trailing stop state

### FEAT/ELLIPSE Branch Key Files

**SAX Production**:

- `feat/ellipse:bot/sax/sax_trader.py:600+` - Live trading executor
- `feat/ellipse:bot/sax/sax_backtester.py` - Validation engine
- `feat/ellipse:bot/sax/PRD.md` - Performance results

**Ellipse Research**:

- `feat/ellipse:poc/src/ellipse_detection.py` - Core detection algorithm
- `feat/ellipse:poc/src/perspective_solver.py` - Projection recovery
- `feat/ellipse:poc/design.md` - Geometric theory

### FEAT/PERPLEXITY-IDEAS Branch Key Files

- `feat/perplexity-ideas:bot/sax/ml_strategy.py` - ML ensemble
- `feat/perplexity-ideas:bot/sax/signal_processor.py` - Enhanced execution
- `feat/perplexity-ideas:PERPLEXITY-IDEAS.md` - 18 advanced concepts

---

## Architecture Documentation

### MAIN Branch Patterns

**Multi-Component Architecture**:

- Seeker (adaptive_risk_seeker.py) - Main decision loop
- Buyer (adaptive_risk_buyer.py) - Entry execution
- Seller (adaptive_risk_seller.py) - Exit execution
- Risk Manager (risk_manager.py) - All risk calculations
- Signal Scorer (multi_timeframe_signal_scorer.py) - Signal aggregation
- Trail Manager (adaptive_trail_manager.py) - Dynamic stops

**Data Models**: Dataclasses for type safety, immutable signals, persistent state files

### RSKR/FEAT Branches Patterns

**Service-Oriented Architecture**:

- Strategy Pattern: Base Strategy class with pluggable subclasses
- Service Separation: OrderService, SignalProcessingService, MarketDataService, ReconciliationService
- State Management: BotState, SymbolState with JSON persistence
- Twin Models: TwinOrder (BUY+SELL pair), TwinPosition
- Reconciliation Pattern: ExchangeReconciler syncs bot state with exchange

**Configuration System**: Pydantic dataclasses with YAML loading, per-timeframe configs

---

## Viability Assessment

### MAIN Branch Viability

**Strengths**:
✓ Production-grade with comprehensive risk controls
✓ Multi-timeframe validation reduces false signals
✓ Adaptive parameters respond to market conditions
✓ Smart DCA prevents runaway losses
✓ Extensive testing infrastructure
✓ Real-time performance tracking

**Weaknesses**:
✗ Martingale risk if multiple DCA levels triggered in trending market
✗ Requires sufficient capital for DCA scaling
✗ Complex parameter tuning (many knobs)

**Production Readiness**: HIGH - Ready for live deployment with appropriate capital and position sizing

### RSKR Branch Viability

**Strengths**:
✓ Service-oriented architecture enables maintainability
✓ Walk-forward optimization prevents overfitting
✓ Multiple strategies (SAX, RSI, GARCH) provide diversification
✓ ML integration adds adaptive signal generation
✓ Paper trading validation path
✓ Health monitoring and circuit breaker

**Weaknesses**:
✗ More complex architecture (higher maintenance burden)
✗ WFO requires significant historical data
✗ ML models need retraining periodically

**Production Readiness**: MEDIUM-HIGH - Architecture solid, needs live validation of WFO parameters

### FEAT/ELLIPSE Branch Viability

**SAX System Strengths**:
✓ Exceptional backtest results (1737% return, 90.7% win rate)
✓ Out-of-sample validation on 2024-2025 data
✓ Symbol-specific patterns prevent contamination
✓ Multi-timeframe pooling enriches pattern library
✓ Zero-commission environment (critical for viability)

**SAX System Weaknesses**:
✗ Requires zero-commission broker (Alpaca stocks only)
✗ Performance may not generalize beyond stocks
✗ Pattern discovery computationally intensive
✗ Needs periodic retraining as market conditions change

**Ellipse PoC Strengths**:
✓ Novel geometric approach with theoretical foundation
✓ Mathematically sound perspective reconstruction
✓ SPY 4h showed promise (Sharpe 1.9)

**Ellipse PoC Weaknesses**:
✗ No incremental lift in linear fusion models
✗ Weak performance on AAPL, mixed on other assets
✗ Requires further research (non-linear fusion, curve families)
✗ Not production-ready

**Production Readiness**:

- SAX System: HIGH (with zero-commission stocks)
- Ellipse PoC: LOW (research stage)

### FEAT/PERPLEXITY-IDEAS Branch Viability

**Strengths**:
✓ Same exceptional SAX backtest results (1737%)
✓ ML ensemble adds confidence scoring
✓ Per-timeframe trailing stops optimize exits
✓ Enhanced signal processor with compounding
✓ 18 advanced concepts provide research roadmap
✓ Trailing sell feature improves risk management

**Weaknesses**:
✗ ML models add complexity
✗ Perplexity ideas mostly ideation (not implemented)
✗ Zero-commission requirement same as feat/ellipse

**Production Readiness**: HIGH (SAX + ML production-ready, perplexity ideas are research framework)

---

## Algorithmic Recreateability

### MAIN Branch (Adaptive Martingale Multi-Timeframe)

**Core Algorithm Pseudocode**:

```python
Initialize:
  capital = starting_capital
  max_dca_levels = 5
  rung_spacing_factor = 2.618

On Each Bar:
  # 1. Multi-Timeframe Signal Generation
  micro_signal = analyze_5h_patterns(bars_5h[-30:])
  trend_signal = analyze_15h_trends(bars_15h[-50:])
  sr_signal = analyze_1h_support_resistance(bars_1h[-100:])

  # 2. Composite Signal Calculation
  composite = (micro*0.3 + trend*0.4 + sr*0.3)
  if all_signals_aligned:
    composite += 0.2 * composite  # 20% alignment bonus
  if near_sr_confluence:
    composite += 0.15 * composite  # 15% confluence bonus

  # 3. Risk Assessment
  risk_score = calculate_risk_score(
    garch_volatility, atr, rsi, vwap_distance,
    autocorrelation, stationarity
  )  # Returns 0-10

  # 4. Entry Decision
  if composite >= 0.6 and signal_confidence >= 0.5 and volume_score >= 45:
    if open_positions == 0:
      position_size = capital * risk_adjusted_multiplier(risk_score)
      enter_position(price=current_close, size=position_size)
      set_trailing_stop(entry - 2.0*ATR)
    elif price_dropped_below_dca_trigger and dca_level < max_dca_levels:
      if smart_dca_conditions_pass(position, rsi, market_structure):
        dca_size = calculate_dca_size(dca_level, rung_spacing_factor)
        add_to_position(price=current_close, size=dca_size)

  # 5. Exit Management
  for position in open_positions:
    profit_pct = (current_price - position.entry) / position.entry

    # Update trailing stop
    new_trail = calculate_adaptive_trail(
      current_price, ATR, profit_pct, position_age, risk_score
    )
    if new_trail > position.trailing_stop:
      position.trailing_stop = new_trail

    # Check exit conditions
    if current_low <= position.trailing_stop:
      close_position(position, exit_price=position.trailing_stop)
    elif profit_pct >= dynamic_profit_target(position):
      close_position(position, exit_price=current_close)
    elif position_age >= 48_hours and profit_pct >= breakeven:
      close_position(position, exit_price=current_close)
```

**Key Parameters for Recreation**:

- Timeframes: 5h (micro), 15h (trend), 1h (S/R)
- Signal weights: 30% / 40% / 30%
- Entry threshold: 0.6 composite signal
- Volume score minimum: 45.0
- Rung spacing: exponential with N=2.618
- ATR trailing stop: 2.0 × ATR initial
- Risk scoring: GARCH, ATR, RSI, VWAP, autocorrelation, stationarity
- DCA conditions: position down <10%, RSI capitulation scaling, technical structure validation

### RSKR/FEAT Branches (SAX Pattern Trading)

**Core Algorithm Pseudocode**:

```python
# Phase 1: Pattern Discovery (Training)
Training Data (Pre-Cutoff):
  for each symbol in [AAPL, MSFT, GOOG, ...]:
    for window_size in [20, 30, 40]:
      for stride in [10]:
        for each window in data:
          # Generate SAX pattern
          normalized = z_normalize(window, clip_outliers=3*sigma)
          paa_segments = piecewise_aggregate(normalized, n_segments=5)
          alphabet = discretize(paa_segments, alphabet_size=4)  # a,b,c,d
          enhanced = add_momentum_range_extremes(alphabet)

          patterns.append((enhanced, symbol, window_index))

  # Frequency filtering
  pattern_counts = count_occurrences(patterns)
  candidates = filter(pattern_counts, min_occurrences=10)

  # Profitability testing
  for pattern in candidates:
    for hold_period in [3, 4, 5, 6, 7, 8]:
      occurrences = find_all(pattern, training_data)
      returns = []
      for occ in occurrences:
        entry = data[occ.index + 1].open  # Next bar open
        exit = data[occ.index + 1 + hold_period].close
        returns.append((exit - entry) / entry - slippage)

      mean_return = mean(returns)
      win_rate = sum(r > 0 for r in returns) / len(returns)
      sharpe = mean_return / std(returns)

      if win_rate >= 0.65 and mean_return >= 0.005:
        golden_patterns[pattern] = {
          'hold_bars': hold_period,
          'mean_return': mean_return,
          'win_rate': win_rate,
          'sharpe': sharpe
        }

# Phase 2: Live Trading (Test)
Test Data (Post-Cutoff):
  for each bar:
    # Generate current pattern
    current_pattern = generate_sax_pattern(bars[-30:])

    # Check if matches golden pattern
    if current_pattern in golden_patterns:
      golden = golden_patterns[current_pattern]

      # Entry decision
      if len(positions) < max_positions and symbol not in positions:
        entry_price = next_bar.open
        position_size = capital * 0.10  # 10% per trade
        hold_until = current_bar + golden.hold_bars

        # Set trailing stop
        atr = calculate_atr(bars[-14:])
        stop_price = entry_price - (2.0 * atr)

        positions[symbol] = Position(
          entry=entry_price,
          size=position_size,
          hold_until=hold_until,
          stop=stop_price
        )

    # Exit management
    for symbol, position in positions.items():
      # Update trailing stop
      new_stop = update_trailing_stop(position, current_price, atr)
      if new_stop > position.stop:
        position.stop = new_stop

      # Check exit
      if current_low <= position.stop:
        exit_position(symbol, price=position.stop)
      elif current_bar >= position.hold_until:
        exit_position(symbol, price=current_close)
```

**Key Parameters for Recreation**:

- Window sizes: [20, 30, 40] bars
- PAA segments: 5
- Alphabet size: 4 (a, b, c, d)
- Stride: 10 bars
- Min occurrences: 10
- Min win rate: 65%
- Min mean return: 0.5%
- Hold periods tested: [3, 4, 5, 6, 7, 8] bars
- Optimal hold: 6 bars (typically)
- Trailing stop: 2.0 × ATR
- Position size: 10% of capital
- Max positions: 4
- Slippage: ±0.1%

### FEAT/ELLIPSE (Ellipse Detection)

**Core Algorithm Pseudocode**:

```python
# Ellipse Detection
for window_size in [48, 96, 192]:
  for stride in [4, 8]:
    for each window in price_data:
      # Normalize
      normalized = z_normalize(window, clip=3*MAD)
      paa_compressed = piecewise_aggregate(normalized, n_segments=5)

      # Fit conic equation: Ax² + Bxy + Cy² + Dx + Ey + F = 0
      A, B, C, D, E, F = least_squares_fit(
        paa_compressed,
        constraint=(4*A*C - B**2 > 0)  # Ensures ellipse
      )

      # Extract ellipse parameters
      center_x = (C*D - B*E/2) / (B**2/4 - A*C)
      center_y = (A*E/2 - B*D) / (B**2/4 - A*C)
      rotation = 0.5 * atan2(B, A - C)

      # Transform to rotated frame
      rotated_points = rotate(paa_compressed - center, -rotation)
      alpha, beta = fit_axes(rotated_points)
      semi_major = 1 / sqrt(alpha)
      semi_minor = 1 / sqrt(beta)

      # Quality check
      residual = mean_squared_error(fitted, paa_compressed) / scale
      eccentricity = sqrt(1 - (semi_minor / semi_major)**2)

      threshold = 0.12 + 0.22 * (window_size - 48) / 48
      if residual < threshold and eccentricity > 0.05:
        ellipses.append(EllipseParams(
          center, semi_major, semi_minor, rotation, eccentricity
        ))

# Perspective Reconstruction
for ellipse in ellipses:
  # Viewing angle from aspect ratio
  aspect_ratio = ellipse.semi_major / ellipse.semi_minor
  tilt_angle = arccos(1 / aspect_ratio)

  # Normal vector
  azimuth = ellipse.rotation
  normal = [
    sin(tilt_angle) * cos(azimuth),
    sin(tilt_angle) * sin(azimuth),
    cos(tilt_angle)
  ]

  # Projection matrix
  R_z = rotation_matrix_z(azimuth)
  R_y = rotation_matrix_y(tilt_angle)
  projection_matrix = R_z @ R_y

  # View distance (if intrinsic radius R known)
  view_distance = R / sin(tilt_angle)

  features[ellipse] = {
    'view_angle': degrees(tilt_angle),
    'distortion': aspect_ratio - 1,
    'scale_factor': sqrt(semi_major * semi_minor),
    'curvature_gaussian': estimate_gaussian_curvature(ellipse),
    'focal_distance': 2 * semi_major * eccentricity
  }
```

**Key Parameters for Recreation**:

- Window sizes: [48, 96, 192] bars
- Stride: [4, 8] bars
- Residual threshold: 0.12 + 0.22 * (window_size - 48) / 48
- Min eccentricity: 0.05
- PAA segments: 5
- Outlier clipping: 3 × MAD

---

## References

- Research: `thoughts/grantdickinson/2026-01-01-research-branch-algorithms.md` (this document)
- MAIN Branch: <git@github.com>:grant-d/trading.git, branch: main, commit: 0fe8f3d4
- RSKR Branch: <git@github.com>:grant-d/trading.git, branch: rskr
- FEAT/ELLIPSE Branch: <git@github.com>:grant-d/trading.git, branch: feat/ellipse
- FEAT/PERPLEXITY-IDEAS Branch: <git@github.com>:grant-d/trading.git, branch: feat/perplexity-ideas

---

## Open Questions

1. **Cross-Branch Strategy Selection**: Which branch's algorithm is optimal for specific market conditions (trending vs ranging, high vs low volatility)?

2. **SAX Pattern Generalization**: Do SAX patterns discovered on US equities generalize to other asset classes (forex, crypto, commodities)?

3. **Ellipse Research Continuation**: Is the geometric approach worth pursuing with non-linear fusion models, or should resources focus on SAX/ML enhancement?

4. **ML Model Drift**: How frequently do ML models in RSKR/PERPLEXITY-IDEAS branches need retraining to maintain edge?

5. **Production Integration**: Can MAIN's adaptive risk management be integrated with RSKR/FEAT's SAX pattern detection for a hybrid system?

6. **Zero-Commission Dependency**: How sensitive are SAX strategies to commission changes? What's the maximum commission rate that preserves edge?

7. **Capital Requirements**: What's the minimum capital required for each branch's strategy to be viable (considering position sizing and risk management)?

8. **Walk-Forward Stability**: How stable are WFO-optimized parameters over time? What's the optimal retraining frequency?
