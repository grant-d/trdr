"""
Example usage of the strategy optimization framework
Demonstrates optimizing a simple MA crossover strategy using WFO
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from strategy_optimization_framework import (
    WalkForwardOptimizer,
    DataLoaderAdapter,
    AlpacaStockPortfolioEngine,
)
from ma_strategy import MovingAverageStrategy
from alpaca_data_loader import AlpacaDataLoader
from config_manager import Config


def run_ma_optimization_example(
    symbol: str = "SPY",
    timeframe: str = "1d",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    train_days: int = 252,  # 1 year
    test_days: int = 63,  # 3 months
    step_days: int = 21,  # 1 month
    max_evaluations: int = 500,
):
    """
    Run a complete example of MA strategy optimization using WFO

    Args:
        symbol: Trading symbol (default: SPY)
        timeframe: Bar timeframe (default: 1d for daily)
        start_date: Start date for analysis (default: 2 years ago)
        end_date: End date for analysis (default: today)
        train_days: Training window size in days
        test_days: Testing window size in days
        step_days: Step size between windows in days
        max_evaluations: Maximum evaluations per optimization window
    """

    # Default dates if not provided
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=730)  # 2 years

    print(f"=== MA Strategy Optimization Example ===")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Timeframe: {timeframe}")
    print(f"Train/Test/Step: {train_days}/{test_days}/{step_days} days")
    print()

    # Step 1: Create configuration
    config = Config(
        symbol=symbol,
        timeframe=timeframe,
        min_bars=train_days + test_days + 100,  # Extra buffer
    )

    # Step 2: Create data loader
    print("Setting up data loader...")
    base_loader = AlpacaDataLoader(config)
    data_loader = DataLoaderAdapter(base_loader)

    # Step 3: Create strategy
    print("Creating MA crossover strategy...")
    strategy = MovingAverageStrategy(min_fast=5, max_fast=30, min_slow=20, max_slow=100)

    # Step 4: Create optimizer
    print("Setting up walk-forward optimizer...")
    optimizer = WalkForwardOptimizer(
        strategy=strategy,
        data_loader=data_loader,
        portfolio_engine_class=AlpacaStockPortfolioEngine,
        initial_balance=100000.0,
    )

    # Step 5: Run optimization
    print("\nStarting walk-forward optimization...")
    print("=" * 50)

    results = optimizer.run(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        max_evaluations=max_evaluations,
    )

    # Step 6: Analyze results
    print("\n=== Optimization Results ===")
    analysis = optimizer.analyze_results(results)

    print(f"\nWindows analyzed: {analysis['num_windows']}")
    print(
        f"Positive windows: {analysis['positive_windows']} ({analysis['win_ratio']:.1%})"
    )
    print(f"Negative windows: {analysis['negative_windows']}")

    print(f"\nOut-of-Sample Performance:")
    print(f"  Average Return: {analysis['avg_oos_return']:.2f}%")
    print(f"  Std Dev Return: {analysis['std_oos_return']:.2f}%")
    print(
        f"  Min/Max Return: {analysis['min_oos_return']:.2f}% / {analysis['max_oos_return']:.2f}%"
    )
    print(f"  Average Sharpe: {analysis['avg_sharpe']:.2f}")
    print(f"  Average Max DD: {analysis['avg_max_drawdown']:.2f}%")
    print(f"  Average Win Rate: {analysis['avg_win_rate']:.2%}")

    # Step 7: Display individual window results
    print("\n=== Individual Window Results ===")
    print(
        f"{'Window':<8} {'Train Period':<25} {'Test Period':<25} "
        f"{'Fast MA':<8} {'Slow MA':<8} {'OOS Return':<12} {'Sharpe':<8} {'Max DD':<8}"
    )
    print("-" * 120)

    for i, (window, params, metrics) in enumerate(results):
        print(
            f"{i+1:<8} "
            f"{window.train_start.date()} to {window.train_end.date()} "
            f"{window.test_start.date()} to {window.test_end.date()} "
            f"{params.fast_period:<8} {params.slow_period:<8} "
            f"{metrics['total_return_pct']:>11.2f}% "
            f"{metrics['sharpe_ratio']:>7.2f} "
            f"{metrics['max_drawdown_pct']:>7.2f}%"
        )

    # Step 8: Summary statistics
    print("\n=== Summary ===")
    total_return = sum(r[2]["total_return_pct"] for r in results)
    print(f"Cumulative OOS Return: {total_return:.2f}%")

    # Calculate annualized return
    total_days = (results[-1][0].test_end - results[0][0].test_start).days
    years = total_days / 365.25
    annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
    print(f"Annualized OOS Return: {annualized_return:.2f}%")

    return results, analysis


def run_parameter_stability_analysis(results):
    """
    Analyze parameter stability across optimization windows

    Args:
        results: Results from WFO optimization
    """
    print("\n=== Parameter Stability Analysis ===")

    # Extract parameters
    fast_periods = [r[1].fast_period for r in results]
    slow_periods = [r[1].slow_period for r in results]

    # Calculate statistics
    import numpy as np

    print(f"\nFast MA Period:")
    print(f"  Mean: {np.mean(fast_periods):.1f}")
    print(f"  Std Dev: {np.std(fast_periods):.1f}")
    print(f"  Min/Max: {min(fast_periods)} / {max(fast_periods)}")

    print(f"\nSlow MA Period:")
    print(f"  Mean: {np.mean(slow_periods):.1f}")
    print(f"  Std Dev: {np.std(slow_periods):.1f}")
    print(f"  Min/Max: {min(slow_periods)} / {max(slow_periods)}")

    # Check for parameter consistency
    fast_cv = np.std(fast_periods) / np.mean(fast_periods)
    slow_cv = np.std(slow_periods) / np.mean(slow_periods)

    print(f"\nCoefficient of Variation:")
    print(f"  Fast MA: {fast_cv:.2%}")
    print(f"  Slow MA: {slow_cv:.2%}")

    if fast_cv < 0.3 and slow_cv < 0.3:
        print("\n✓ Parameters show good stability across windows")
    else:
        print("\n⚠ Parameters show high variation - strategy may be overfitting")


if __name__ == "__main__":
    # Run the optimization example
    results, analysis = run_ma_optimization_example(
        symbol="SPY",
        timeframe="1d",
        train_days=252,  # 1 year training
        test_days=63,  # 3 months testing
        step_days=21,  # 1 month step
        max_evaluations=300,  # Reduced for faster demo
    )

    # Run parameter stability analysis
    run_parameter_stability_analysis(results)

    print("\n=== Optimization Complete ===")
