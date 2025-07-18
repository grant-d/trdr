# Matrix Profile Trading System

## Overview

This document describes a sophisticated trading system that combines Matrix Profile analysis with Genetic Algorithm (GA) optimization and Walk-Forward Optimization (WFO). The system is designed to identify and trade recurring patterns in financial time series data using state-of-the-art time series analysis techniques.

## Matrix Profile Theory

### What is Matrix Profile?

The Matrix Profile (MP) is a data structure that stores the z-normalized Euclidean distance between any subsequence within a time series and its nearest neighbor. It enables:

1. **Motif Discovery**: Finding recurring patterns that may indicate profitable trading opportunities
2. **Discord Discovery**: Identifying anomalies or unusual market behavior
3. **Regime Change Detection**: Detecting shifts in market dynamics
4. **Pattern Matching**: Finding specific known patterns in historical data

### Why Matrix Profile for Trading?

1. **Exact Solutions**: Unlike many ML approaches, MP provides exact pattern matches
2. **Scalability**: O(n²) complexity makes it feasible for large datasets
3. **Single Parameter**: Only requires window size, reducing overfitting risk
4. **No Training Required**: Works directly on time series data

## System Architecture

### Components

1. **Data Layer**
   - Uses fractionally differentiated price data for stationarity
   - Columns: `open_fd`, `high_fd`, `low_fd`, `close_fd`, `hlc3_fd`, `dv_fd`, `volume_lr`
   - Supports multiple timeframes (1m, 5m, 15m, 1h, 1d)
   - Multi-timeframe synchronization for pattern matching

2. **Strategy Layer** (`matrix_profile_strategy.py`)
   - Pattern recognition using STUMPY across multiple timeframes
   - Multi-timeframe pattern confirmation (e.g., 1m + 15m, 5m + 1d)
   - Signal generation based on motifs and discords
   - Risk management with configurable stops

3. **Optimization Layer**
   - **Genetic Algorithm**: Optimizes strategy parameters
   - **Walk-Forward**: Validates performance out-of-sample
   - Parameters: window size per timeframe, thresholds, lookback periods
   - Multi-timeframe weight optimization

4. **Evaluation Layer**
   - Comprehensive metrics via QuantStats
   - Position management and trade execution simulation
   - Performance attribution across timeframes

### Trading Logic

1. **Multi-Timeframe Pattern Recognition**
   ```python
   # Compute matrix profiles for multiple timeframes
   mp_1m = stumpy.stump(price_series_1m, window_size_1m)
   mp_15m = stumpy.stump(price_series_15m, window_size_15m)
   
   # Find motifs (recurring patterns) in each timeframe
   motifs_1m = np.argsort(mp_1m[:, 0])[:num_motifs]
   motifs_15m = np.argsort(mp_15m[:, 0])[:num_motifs]
   
   # Confirm patterns across timeframes
   confirmed_signal = (pattern_match_1m and pattern_match_15m)
   ```

2. **Signal Generation**
   - **Strong Buy**: Pattern match in both lower (1m) and higher (15m) timeframes
   - **Weak Buy**: Pattern match in single timeframe with supporting indicators
   - **Sell Signal**: Divergence between timeframes or discord patterns
   - **Exit**: Stop-loss, take-profit, or pattern completion in any timeframe

3. **Multi-Timeframe Confirmation**
   - Primary timeframe generates initial signal
   - Secondary timeframe confirms or invalidates
   - Weightings: e.g., 1m (40%), 5m (30%), 15m (30%)
   - Higher timeframes provide trend context

4. **Risk Management**
   - Position sizing based on multi-timeframe confidence
   - Dynamic stop-loss based on lowest timeframe volatility
   - Scale-in/scale-out based on timeframe alignment
   - Maximum drawdown limits per timeframe

## Parameter Optimization

### GA Parameters

1. **Window Sizes per Timeframe**:
   - 1m: 10-100 bars
   - 5m: 10-50 bars  
   - 15m: 5-30 bars
   - 1h: 5-20 bars
   
2. **Thresholds per Timeframe**:
   - Primary threshold: 0.1-0.6
   - Confirmation threshold: 0.2-0.8
   
3. **Lookback Periods**:
   - 1m: 500-2000 bars
   - 5m: 200-800 bars
   - 15m: 100-400 bars
   
4. **Timeframe Weights** (sum to 1.0):
   - Weight_1m: 0.2-0.6
   - Weight_5m: 0.2-0.5
   - Weight_15m: 0.1-0.4
   
5. **Risk Parameters**:
   - Stop Loss: 0.1%-2%
   - Take Profit: 0.2%-5%
   - Position Size Multiplier: 0.5-2.0 based on timeframe alignment

### Fitness Function

```python
def fitness_function(params):
    # Generate signals for each timeframe
    signals_1m = generate_signals(data_1m, params['tf_1m'])
    signals_5m = generate_signals(data_5m, params['tf_5m'])
    signals_15m = generate_signals(data_15m, params['tf_15m'])
    
    # Combine signals with weights
    combined_signals = (
        signals_1m * params['weight_1m'] +
        signals_5m * params['weight_5m'] +
        signals_15m * params['weight_15m']
    )
    
    # Calculate returns
    returns = calculate_returns(combined_signals)
    
    # Multi-objective optimization
    sharpe = calculate_sharpe(returns)
    max_dd = calculate_max_drawdown(returns)
    win_rate = calculate_win_rate(returns)
    consistency = calculate_timeframe_consistency(signals_1m, signals_5m, signals_15m)
    
    # Weighted fitness score with timeframe consistency bonus
    fitness = (sharpe * 0.4 + 
              (1 - max_dd) * 0.3 + 
              win_rate * 0.2 + 
              consistency * 0.1)
    
    return fitness
```

## Walk-Forward Optimization

### Configuration

- **In-Sample Period**: 2000 bars (~1.4 days for 1m data)
- **Out-Sample Period**: 500 bars (~8 hours)
- **Step Size**: 500 bars
- **Minimum Samples**: 1000 bars

### Process

1. Optimize parameters on in-sample data using GA
2. Validate on out-of-sample data
3. Step forward and repeat
4. Aggregate results across all periods

## Performance Metrics

### Primary Metrics

1. **Sharpe Ratio**: Risk-adjusted returns
2. **Sortino Ratio**: Downside risk-adjusted returns
3. **Maximum Drawdown**: Worst peak-to-trough decline
4. **Win Rate**: Percentage of profitable trades
5. **Profit Factor**: Gross profits / Gross losses

### Secondary Metrics

1. **Calmar Ratio**: Return / Max Drawdown
2. **Value at Risk (VaR)**: 5% worst-case loss
3. **Kelly Criterion**: Optimal position sizing
4. **Consistency**: % of profitable WFO periods

## Implementation Notes

### Data Preprocessing

1. Use fractionally differentiated data for stationarity
2. Handle missing values with forward fill
3. Normalize features for pattern matching
4. Consider volume-weighted patterns

### Computational Efficiency

1. Use STUMPY's optimized algorithms
2. Parallelize GA population evaluation
3. Cache computed matrix profiles
4. Use GPU acceleration when available

### Risk Controls

1. Maximum position limits
2. Correlation checks for portfolio
3. Regime-based position sizing
4. Dynamic parameter adaptation

## Usage Example

```python
# Load data
loader = AlpacaDataLoader(config)
df = loader.load_data()
df = loader.transform(df, frac_diff="_fd", log_volume="_lr")

# Initialize strategy
strategy = MatrixProfileStrategy(
    window_range=(20, 100),
    threshold_range=(0.2, 0.7),
    lookback_range=(100, 300)
)

# Setup optimization
ga_config = GeneticOptimizer(
    param_ranges=strategy.get_param_ranges(),
    population_size=50,
    generations=20
)

wfo_config = WFOConfig(
    in_sample_periods=2000,
    out_sample_periods=500,
    step_periods=500
)

# Run optimization
results = run_matrix_profile_optimization(
    df, strategy, ga_config, wfo_config
)

# Generate report
print(results['performance_report'])
```

## Expected Results

### Backtesting Performance (Typical)

- **Sharpe Ratio**: 1.5 - 3.0
- **Max Drawdown**: 10% - 20%
- **Win Rate**: 45% - 55%
- **Profit Factor**: 1.3 - 2.0

### Live Trading Considerations

1. Slippage and transaction costs
2. Market impact for large positions
3. Pattern degradation over time
4. Regime changes requiring reoptimization

## Multi-Timeframe Pattern Examples

### Confluence Trading

1. **Bullish Confluence**:
   - 1m: Motif match with historical rally pattern
   - 5m: Ascending triangle pattern detected
   - 15m: Oversold bounce pattern
   - Signal: Strong Buy with 1.5x position size

2. **Bearish Divergence**:
   - 1m: Discord (unusual pattern)
   - 5m: Resistance pattern match
   - 15m: Downtrend continuation pattern
   - Signal: Sell with tight stop-loss

3. **Neutral/Hold**:
   - Conflicting patterns across timeframes
   - Low confidence scores
   - Wait for alignment

### Timeframe Synchronization

```python
# Example: Aligning different timeframe data
def align_timeframes(df_1m, df_5m, df_15m):
    # Resample to common timestamps
    base_time = df_1m['timestamp']
    
    # Map 5m and 15m signals to 1m timestamps
    df_5m_aligned = resample_signals(df_5m, base_time)
    df_15m_aligned = resample_signals(df_15m, base_time)
    
    return df_1m, df_5m_aligned, df_15m_aligned
```

## Future Enhancements

1. **Advanced Multi-Timeframe Features**:
   - Dynamic timeframe selection based on volatility
   - Fractal pattern recognition across scales
   - Timeframe momentum indicators
   
2. **Machine Learning Integration**:
   - Learn optimal timeframe weights
   - Pattern classification with deep learning
   - Reinforcement learning for dynamic adjustment

3. **Advanced Pattern Matching**:
   - Cross-correlation between timeframes
   - Wavelet transforms for multi-scale analysis
   - DTW (Dynamic Time Warping) for flexible pattern matching

4. **Market Microstructure**:
   - Order flow patterns in lower timeframes
   - Volume profile integration
   - Liquidity-based signal filtering

5. **Real-time Optimization**:
   - Streaming matrix profile updates
   - Online parameter adaptation
   - Continuous walk-forward optimization

## References

1. Yeh, C. M., et al. (2016). "Matrix Profile I: All Pairs Similarity Joins for Time Series"
2. Law, S. (2019). "STUMPY: A Powerful and Scalable Python Library for Time Series Data Mining"
3. Financial Time Series: Market Analysis Techniques Based on Matrix Profiles (2024)