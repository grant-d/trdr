# Helios Trading Analysis Toolkit

A comprehensive trading analysis system implementing advanced market state scoring, regime-based trading strategies, and persistent multi-strategy management as specified in the Product Requirements Document.

## Features

### Core Trading Features
- **Dollar Bars**: Activity-based price aggregation ensuring equal information content per bar
- **Market State Score (MSS)**: Multi-factor regime classification using:
  - Trend/Momentum (linear regression slope)
  - Volatility (ATR-based)
  - Exhaustion/Mean Reversion (distance from SMA relative to ATR)
- **Technical Indicators**: MACD, RSI, Standard Deviation, ATR
- **Regime Classification**: Strong Bull, Weak Bull, Neutral, Weak Bear, Strong Bear
- **Dynamic Trading Strategy**: 
  - Regime-adaptive position sizing
  - Action matrix based on regime and indicators
  - Fractional position support

### Advanced Features (PRD Implementation)
- **Trading Contexts**: Isolated strategy instances with persistent state
  - Format: `instrument_exchange_timeframe_experiment`
  - Atomic state persistence after every change
  - Crash recovery and session resumption
- **Regime-Specific Playbooks**: Optimized parameters for each market regime
  - Instrument and timeframe specific defaults
  - Dynamic weight adjustments
  - Stop-loss multipliers per regime
- **Risk Management**:
  - Dynamic trailing stops (1x-2x ATR based on regime)
  - Position size constraints
  - Stop-loss monitoring
- **Genetic Algorithm Optimization**:
  - Walk-forward optimization to prevent overfitting
  - Customizable fitness functions (Sortino/Calmar ratio)
  - Parameter discovery across multiple dimensions
  - **Threshold constraint enforcement** to ensure valid strategy behavior
  - **Adaptive fitness calibration** for different asset volatility profiles
- **Performance Metrics**: 
  - Sortino Ratio (downside deviation focus)
  - Calmar Ratio (drawdown adjusted returns)
  - Win rate, profit factor, trade analysis
  - **Geometric mean returns** for accurate multi-period performance

## Installation

The project can be installed using either `uv` or standard pip:

### Using uv (recommended)
```bash
cd helios
uv sync
```

### Using pip
```bash
cd helios
pip install -r requirements.txt
```

### Required Environment Variables

Create a `.env` file with any API keys you might need:
```bash
HELIOS_API_KEY=your_api_key_here  # Optional
HELIOS_DATA_PATH=./data           # Optional, defaults to ./data
```

## Usage

### Command Line Interface

Helios v0.2.0 provides multiple commands for different workflows:

#### 1. Basic Analysis (Original Mode)

Run a simple backtest without state persistence:
```bash
python -m helios analyze --data your_data.csv
```

With dollar bars:
```bash
python -m helios analyze --data your_data.csv --dollar-bars --dollar-threshold 500000
```

Save results:
```bash
python -m helios analyze --data your_data.csv --save-results --output-dir ./results
```

#### 2. Trading Context Management (PRD Mode)

Create a trading context:
```bash
python -m helios context create --instrument BTC --exchange COINBASE --timeframe 1h --experiment default
```

List all contexts:
```bash
python -m helios context list
```

View context status:
```bash
python -m helios context status --id BTC_COINBASE_1h_default
```

Pause/Resume context:
```bash
python -m helios context pause --id BTC_COINBASE_1h_default
python -m helios context resume --id BTC_COINBASE_1h_default
```

#### 3. Strategy Optimization

Run genetic algorithm optimization:
```bash
python -m helios optimize --data data.csv --population 50 --generations 20
```

Save optimization results with custom name:
```bash
python -m helios optimize --data data.csv --population 50 --generations 20 --output msft
# Creates: optimization_results/msft-params.json
```

Enable short positions:
```bash
python -m helios optimize --data data.csv --population 50 --generations 20 --allow-shorts
```

Chain optimization with immediate testing:
```bash
python -m helios optimize --data data.csv --population 50 --generations 20 --output btc --test
```

Chain optimization, testing, and plotting:
```bash
python -m helios optimize --data data.csv --population 50 --generations 20 --output btc --test --plot
```

Walk-forward optimization:
```bash
python -m helios optimize --data data.csv --walk-forward --window-days 365 --step-days 90
```

Optimize for specific context:
```bash
python -m helios optimize --data data.csv --context-id BTC_COINBASE_1h_default --walk-forward
```

#### 4. Testing Optimized Strategies

Test with saved parameters:
```bash
python -m helios test --params optimization_results/msft-params.json
```

Test with plots:
```bash
python -m helios test --params optimization_results/btc-params.json --plot
```

Test with different data:
```bash
python -m helios test --data new_data.csv --params optimization_results/msft-params.json
```

#### 5. Context-Based Backtesting

Run backtest with persistent context:
```bash
python -m helios backtest --context-id BTC_COINBASE_1h_default --data data.csv
```

### Data Format

Your data file (CSV or Parquet) must contain the following columns:
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price (or `adj close` for adjusted prices)
- `volume`: Trading volume

The first column should be a datetime index.

### Output Files

#### Optimization Results (`./optimization_results/`)
- `<name>-params.json`: Best parameters found by genetic algorithm
- Contains parameters, fitness score, and parameter ranges used

#### Strategy Results (`./strategy_results/`)
- `backtest_results.csv`: Full backtest data with positions, prices, etc.
- `performance_report.txt`: Human-readable performance summary
- `trades_log.csv`: Detailed trade log (enhanced strategy only)

### Command Options

#### Analyze Command
- `--data, -d`: Path to data file (CSV or Parquet) [required]
- `--dollar-bars`: Convert to dollar bars before analysis
- `--dollar-threshold`: Dollar volume threshold for bars (default: 1000000)
- `--lookback`: Lookback period for indicators (default: 20)
- `--capital`: Initial capital (default: 100000)
- `--save-results`: Save results to files
- `--output-dir`: Output directory for results (default: ./results)

#### Context Command
- `action`: create, list, status, pause, resume, delete
- `--id`: Context ID (for status/pause/resume/delete)
- `--instrument`: Trading instrument (e.g., BTC, AAPL)
- `--exchange`: Exchange name (e.g., COINBASE, NASDAQ)
- `--timeframe`: Time period (e.g., 1m, 5m, 15m, 1h, 4h, 1d)
- `--experiment`: Experiment name (default: 'default')
- `--capital`: Initial capital (default: 100000)

#### Optimize Command
- `--data, -d`: Path to data file [required]
- `--dollar-threshold`: Dollar volume threshold ('auto', number, or 'none' to disable)
- `--context-id`: Context to optimize for
- `--walk-forward`: Use walk-forward optimization
- `--window-days`: Training window size (default: 365)
- `--step-days`: Step size for walk-forward (default: 90)
- `--population`: GA population size (default: 50)
- `--generations`: Number of generations (default: 20)
- `--fitness`: Fitness metric: 'sortino' or 'calmar' (default: sortino)
- `--allow-shorts`: Enable short positions (disabled by default)
- `--output`: Save results with optional prefix (e.g., --output msft)
- `--test`: Run test immediately after optimization
- `--plot`: Generate plots after optimization (requires --test)

#### Test Command (formerly run-optimized)
- `--params, -p`: Path to optimized parameters JSON file [required]
- `--data, -d`: Path to data file (optional - uses saved path from optimization)
- `--dollar-threshold`: Override saved dollar threshold setting
- `--capital`: Initial capital (default: 100000)
- `--plot`: Generate performance plots
- `--save-results`: Save test results

#### Backtest Command
- `--context-id`: Context ID to backtest [required]
- `--data, -d`: Path to data file [required]
- `--save-results`: Save backtest results

## Module Structure

### Core Modules
- `data_processing.py`: Data preparation and dollar bars conversion
- `factors.py`: Technical indicators and Market State Score calculation
- `strategy.py`: Trading strategy implementation with action matrix
- `performance.py`: Performance evaluation metrics (Sortino, Calmar, etc.)
- `main.py`: CLI and orchestration

### Advanced Modules (PRD Implementation)
- `trading_context.py`: Persistent trading context management
  - `TradingContext`: Individual strategy instance with state persistence
  - `TradingContextManager`: Multi-context orchestration
- `playbook.py`: Regime-specific parameter management
  - `Playbook`: Parameter sets for each regime
  - `PlaybookManager`: Multi-playbook coordination
- `optimization.py`: Genetic algorithm and walk-forward optimization
  - `GeneticAlgorithm`: Population-based parameter search
  - `WalkForwardOptimizer`: Rolling window optimization

### State Persistence
- `./state/`: Directory for context state files
  - `{context_id}.json`: Individual context state
- `./playbooks/`: Directory for regime playbooks
  - `{context_id}_playbook.json`: Optimized parameters per regime
- `./optimization_results/`: GA optimization results

## Example Code

### Basic Analysis
```python
import pandas as pd
from helios import (
    create_dollar_bars,
    calculate_mss,
    EnhancedTradingStrategy,
    generate_performance_report
)

# Load your data
df = pd.read_csv('ohlcv_data.csv', parse_dates=True, index_col=0)

# Create dollar bars
dollar_bars = create_dollar_bars(df, dollar_threshold=1000000)

# Calculate Market State Score
factors_df, regimes = calculate_mss(dollar_bars)

# Run trading strategy
strategy = EnhancedTradingStrategy(initial_capital=100000)
results = strategy.run_backtest(dollar_bars, factors_df)

# Generate report
trades_summary = strategy.get_trade_summary()
report = generate_performance_report(results, trades_summary)
print(report)
```

### Trading Context Usage
```python
from helios import TradingContextManager, prepare_data, calculate_mss

# Create context manager
manager = TradingContextManager()

# Create a new context
context = manager.create_context(
    instrument='BTC',
    exchange='COINBASE',
    timeframe='1h',
    experiment_name='momentum_test',
    initial_capital=100000
)

# Process historical data
df = pd.read_csv('btc_data.csv', parse_dates=True, index_col=0)
df = prepare_data(df)

# Calculate factors
factors_df, regimes = calculate_mss(df)

# Process each bar (simulating live trading)
for i in range(len(df)):
    if i < 20:  # Skip warmup period
        continue
    
    # Process bar with context
    result = context.process_bar(df.iloc[i], factors_df.iloc[i])
    
    # Check stop loss
    if context.check_stop_loss(df.iloc[i]['close']):
        print(f"Stop loss hit at {df.index[i]}")
```

### Genetic Algorithm Optimization
```python
from helios import (
    GeneticAlgorithm,
    WalkForwardOptimizer,
    create_default_parameter_ranges
)

# Create GA with custom parameters
param_ranges = create_default_parameter_ranges()
ga = GeneticAlgorithm(
    parameter_ranges=param_ranges,
    population_size=100,
    generations=50,
    fitness_metric='sortino'
)

# Run walk-forward optimization
wfo = WalkForwardOptimizer(
    window_size=365,  # 1 year training
    step_size=90,     # 3 month steps
    test_size=90      # 3 month test
)

# Load data
df = pd.read_csv('historical_data.csv', parse_dates=True, index_col=0)

# Optimize
results = wfo.optimize(df, ga)
print(f"Average out-of-sample fitness: {results['avg_test_fitness']:.4f}")
```

## Architecture Design (PRD Implementation)

### State Persistence
The system implements atomic state persistence as specified in the PRD:
- Every state change is immediately persisted to disk
- Atomic writes using temp file + rename pattern
- Full crash recovery on restart
- No in-memory state dependencies

### Trading Context Format
```
context_id = instrument_exchange_timeframe_experiment
Example: BTC_COINBASE_1h_momentum_v2
```

### Action Matrix (Section 4.3.3)
| MSS Score | Regime | Action | Stop Loss |
|-----------|--------|--------|------------|
| > 60 | Strong Bull | Enter/Hold Long | 2x ATR |
| 20-60 | Weak Bull | Hold Longs Only | 1x ATR |
| -20-20 | Neutral | Exit All Positions | - |
| -60--20 | Weak Bear | Hold Shorts Only | 1x ATR |
| < -60 | Strong Bear | Enter/Hold Short | 2x ATR |

### Playbook Structure
Each context maintains regime-specific parameters:
- Factor weights (trend, volatility, exhaustion)
- Lookback periods
- Stop-loss multipliers
- Position size adjustments

### Walk-Forward Optimization
Prevents overfitting through:
- Rolling training windows
- Out-of-sample testing
- No look-ahead bias
- Parameter robustness validation

## Recent Improvements (v0.2.1)

### 1. Threshold Constraint Enforcement
The genetic algorithm now enforces proper threshold ordering during parameter generation:
- `strong_bull > weak_bull > neutral_upper > neutral_lower > weak_bear > strong_bear`
- Implemented in `create_individual()`, `mutate()`, and `crossover()` methods
- Prevents invalid strategies that would behave opposite of intended
- Ensures MSFT and other stocks optimize correctly

### 2. Fitness Function Calibration
Fixed scale mismatch between Sortino ratios and drawdown percentages:
- Separate calibration for stocks vs crypto assets
- Reduced Sharpe weight (5% for stocks, 0% for crypto)
- Drawdown penalty only applies above 20% for stocks
- Volatility-based fitness multipliers (1.2x-1.5x for stocks)

### 3. Command Chaining
New ability to chain optimization with testing and plotting:
```bash
# Optimize and immediately test
python -m helios optimize --data data.csv --output msft --test

# Optimize, test, and plot in one command
python -m helios optimize --data data.csv --output btc --test --plot --allow-shorts
```

### 4. Simplified Output Naming
- Use `--output msft` instead of `--output-prefix msft_`
- Automatic dash insertion: creates `msft-params.json`
- Removed separate fitness history file to reduce clutter

### 5. Geometric Mean Returns
Added accurate multi-period performance calculation:
- Accounts for compounding effects
- More accurate representation of long-term returns
- Displayed in performance reports alongside arithmetic mean

### 6. Enhanced Short Selling Support
- `--allow-shorts` flag properly integrated throughout
- Separate optimization paths for long-only vs long/short strategies
- Improved position management for short positions

## Example Workflows

### Quick Testing Workflow
```bash
# Fast optimization for testing
python main.py optimize \
  --data data.csv \
  --population 10 \
  --generations 5 \
  --output quick_test \
  --test

# Results displayed immediately
```

### Production Optimization Workflow
```bash
# Thorough optimization with testing and plotting
python main.py optimize \
  --data data.csv \
  --population 100 \
  --generations 50 \
  --output production \
  --test \
  --plot \
  --allow-shorts

# Test on different data
python main.py test \
  --data out_of_sample_data.csv \
  --params optimization_results/production-params.json \
  --plot
```

### Compare Strategies Workflow
```bash
# Optimize MSFT
python main.py optimize \
  --data msft_data.csv \
  --population 50 \
  --generations 20 \
  --output msft \
  --test

# Optimize BTC with shorts
python main.py optimize \
  --data btc_data.csv \
  --population 50 \
  --generations 20 \
  --output btc \
  --allow-shorts \
  --test

# Compare results from the performance reports
```

## Optimization Tips

1. **Start small**: Use `--population 10 --generations 5` for quick tests
2. **Use dollar bars**: Auto-threshold usually provides better results than time-based bars
3. **Save everything**: Always use `--output` to preserve your work
4. **Test out-of-sample**: Use optimized parameters on different data periods
5. **Chain commands**: Use `--test --plot` to see results immediately
6. **Enable shorts for crypto**: Use `--allow-shorts` for volatile assets like BTC
7. **Fitness metric**: `sortino` usually works better than `calmar` for trend-following strategies

## Performance Expectations

Based on testing with various assets:

### MSFT (Stock) - Optimized
- Total Return: ~240-340%
- Sortino Ratio: ~0.8-1.2
- Sharpe Ratio: ~0.6-0.9
- Max Drawdown: ~10-20%
- Win Rate: ~90-95%

### BTC (Crypto) - Optimized with Shorts
- Total Return: ~100-300%
- Sortino Ratio: ~2.0-3.5
- Sharpe Ratio: ~1.4-1.8
- Max Drawdown: ~5-15%
- Win Rate: ~60-75%

The constraint enforcement ensures strategies behave as intended, leading to more predictable and reliable performance across different asset classes.

## Future Enhancements (Phase 3)

- **Live Trading Engine**: Poll exchanges for new bars
- **Database Backend**: Upgrade from JSON to SQLite/PostgreSQL
- **Cloud Deployment**: Dockerize for cloud hosting
- **LLM Integration**: Sentiment analysis and performance review
- **Web Dashboard**: Interactive UI with Streamlit/Dash
- **Multi-Instrument**: Concurrent strategy management

## License

MIT