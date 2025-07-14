# Helios Command Line Usage Guide

This guide shows you how to use the Helios command-line interface for optimization and backtesting.

## Complete Workflow: Optimize → Run

Here's the complete sequence of commands a user would run:

### Step 1: Run Optimization

```bash
# Basic optimization (uses simple strategy)
uv run python main.py optimize \
  --data /path/to/your/data.csv \
  --population 50 \
  --generations 20 \
  --save-results

# Enhanced optimization (uses gradual entries, stop-losses, regime thresholds)
uv run python main.py optimize \
  --data /path/to/your/data.csv \
  --enhanced \
  --dollar-bars \
  --population 50 \
  --generations 20 \
  --fitness sortino \
  --save-results
```

This will:
- Load your data and convert to dollar bars (if `--dollar-bars` specified)
- Run genetic algorithm optimization
- Save optimized parameters to `./optimization_results/optimized_parameters.json`
- Save fitness history to `./optimization_results/fitness_history.json`

### Step 2: Use Optimized Parameters

```bash
# Run backtest with optimized parameters
uv run python main.py run-optimized \
  --data /path/to/your/data.csv \
  --params ./optimization_results/optimized_parameters.json \
  --dollar-bars \
  --save-results
```

This will:
- Load the optimized parameters
- Run backtest using those parameters
- Generate performance report
- Save results to `./strategy_results/`

### Step 3: View Results

```bash
# View performance report
cat strategy_results/performance_report.txt

# View trade log (enhanced strategy only)
head strategy_results/trades_log.csv

# View backtest data
head strategy_results/backtest_results.csv
```

## Available Commands

### `analyze` - Basic Analysis
Run basic analysis without optimization:

```bash
uv run python main.py analyze \
  --data data.csv \
  --dollar-bars \
  --save-results
```

### `optimize` - Genetic Algorithm Optimization
Run optimization to find best parameters:

```bash
# All available options
uv run python main.py optimize \
  --data data.csv \
  --enhanced \
  --dollar-bars \
  --dollar-threshold 1000000 \
  --population 50 \
  --generations 20 \
  --fitness sortino \
  --save-results
```

Options:
- `--enhanced`: Use enhanced strategy (gradual entries, regime thresholds)
- `--dollar-bars`: Convert to dollar bars before optimization
- `--dollar-threshold`: Dollar volume threshold (default: 1,000,000)
- `--population`: GA population size (default: 50)
- `--generations`: Number of generations (default: 20)
- `--fitness`: Fitness metric (`sortino` or `calmar`, default: sortino)
- `--save-results`: Save optimization results to files

### `run-optimized` - Use Saved Parameters
Run backtest with previously optimized parameters:

```bash
# All available options
uv run python main.py run-optimized \
  --data data.csv \
  --params optimization_results/optimized_parameters.json \
  --dollar-bars \
  --dollar-threshold 1000000 \
  --capital 100000 \
  --save-results \
  --output-dir ./my_results
```

Options:
- `--params`: Path to saved optimization parameters JSON file
- `--dollar-bars`: Convert to dollar bars (must match optimization settings)
- `--capital`: Initial capital (default: 100,000)
- `--save-results`: Save backtest results
- `--output-dir`: Output directory (default: ./strategy_results)

## Enhanced vs Basic Strategy

### Basic Strategy
- Simple position sizing
- Fixed action matrix
- Basic stop-losses

**Use when**: Testing simple ideas or baseline comparisons

### Enhanced Strategy (`--enhanced`)
- Gradual position entry/exit
- Regime-specific stop-loss multipliers  
- Dynamic position sizing based on MSS magnitude
- Optimizable regime thresholds

**Use when**: Serious trading strategy development

## File Outputs

### Optimization Results (`./optimization_results/`)
- `optimized_parameters.json`: Best parameters found by GA
- `fitness_history.json`: Fitness progression over generations

### Strategy Results (`./strategy_results/`)
- `backtest_results.csv`: Full backtest data with positions, prices, etc.
- `performance_report.txt`: Human-readable performance summary
- `trades_log.csv`: Detailed trade log (enhanced strategy only)

## Example Workflows

### Quick Test
```bash
# Fast optimization for testing
uv run python main.py optimize \
  --data data.csv \
  --enhanced \
  --population 10 \
  --generations 5 \
  --save-results

# Test the results
uv run python main.py run-optimized \
  --data data.csv \
  --params optimization_results/optimized_parameters.json
```

### Production Optimization
```bash
# Thorough optimization
uv run python main.py optimize \
  --data data.csv \
  --enhanced \
  --dollar-bars \
  --population 100 \
  --generations 50 \
  --fitness sortino \
  --save-results

# Backtest on same data
uv run python main.py run-optimized \
  --data data.csv \
  --params optimization_results/optimized_parameters.json \
  --dollar-bars \
  --save-results

# Test on different data
uv run python main.py run-optimized \
  --data new_data.csv \
  --params optimization_results/optimized_parameters.json \
  --dollar-bars \
  --save-results \
  --output-dir ./out_of_sample_results
```

### Compare Strategies
```bash
# Optimize basic strategy
uv run python main.py optimize \
  --data data.csv \
  --population 50 \
  --generations 20 \
  --save-results
mv optimization_results basic_optimization_results

# Optimize enhanced strategy  
uv run python main.py optimize \
  --data data.csv \
  --enhanced \
  --population 50 \
  --generations 20 \
  --save-results
mv optimization_results enhanced_optimization_results

# Compare results
uv run python main.py run-optimized \
  --data data.csv \
  --params basic_optimization_results/optimized_parameters.json \
  --output-dir ./basic_results

uv run python main.py run-optimized \
  --data data.csv \
  --params enhanced_optimization_results/optimized_parameters.json \
  --output-dir ./enhanced_results
```

## Tips

1. **Start small**: Use `--population 10 --generations 5` for quick tests
2. **Use `--dollar-bars`**: Usually provides better results than time-based bars
3. **Enhanced strategy**: Almost always outperforms basic strategy
4. **Save everything**: Always use `--save-results` to preserve your work
5. **Test out-of-sample**: Use optimized parameters on different data periods
6. **Fitness metric**: `sortino` usually works better than `calmar` for this strategy

## Performance Expectations

Based on our testing with Bitcoin data:

### Basic Strategy
- Total Return: ~212%
- Sortino Ratio: ~1.10
- Max Drawdown: ~19%

### Enhanced Strategy (Optimized)
- Total Return: ~375%+ 
- Sortino Ratio: ~1.58+
- Max Drawdown: ~41% (higher due to more aggressive trading)
- Win Rate: ~61%

The enhanced strategy typically achieves 50-80% higher returns than the basic strategy, with better risk-adjusted metrics, at the cost of higher drawdowns.