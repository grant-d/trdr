#!/usr/bin/env python3
"""
Debug financial metrics calculation issues.
"""
import pandas as pd
import numpy as np
from strategy_evaluator import StrategyEvaluator
from simple_mp_strategy import SimpleMatrixProfileStrategy
from financial_metrics import FinancialMetrics


def debug_metrics_calculation():
    """Debug why metrics calculation is failing."""
    print("Debugging Financial Metrics Calculation")
    print("=" * 50)

    # Load small sample of data
    df = pd.read_csv("data/sol_usd_1m.transform.csv").head(200)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Generate simple signals
    strategy = SimpleMatrixProfileStrategy(window_size=20, buy_threshold=0.3)
    signals = strategy.generate_signals(df)

    # Run backtest
    evaluator = StrategyEvaluator()
    results = evaluator.backtest(df, signals)

    # Check the returns
    daily_returns = results["daily_returns"]
    print(f"\n1. Daily Returns Analysis:")
    print(f"   Total returns: {len(daily_returns)}")
    print(f"   Non-zero returns: {sum(1 for r in daily_returns if r != 0)}")
    print(f"   Mean return: {np.mean(daily_returns):.6f}")
    print(f"   Std deviation: {np.std(daily_returns):.6f}")
    print(f"   Min return: {np.min(daily_returns):.6f}")
    print(f"   Max return: {np.max(daily_returns):.6f}")

    # Check if all returns are zero
    if all(r == 0 for r in daily_returns):
        print("   ⚠️ All returns are zero!")

    # Try calculating metrics manually
    print("\n2. Manual Metrics Calculation:")
    fm = FinancialMetrics()

    # Create test returns with some variation
    test_returns = pd.Series([0.001, -0.002, 0.003, -0.001, 0.002])
    print(f"   Test returns: {test_returns.tolist()}")

    try:
        test_metrics = fm.calculate_all_metrics(test_returns)
        print(f"   ✓ Test metrics calculated successfully")
        print(f"   Sharpe: {test_metrics.get('sharpe', 'N/A')}")
    except Exception as e:
        print(f"   ✗ Error with test returns: {e}")

    # Try with actual returns
    print("\n3. Actual Returns Metrics:")
    returns_series = pd.Series(daily_returns)

    # Remove any NaN or infinite values
    returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns_series) > 0:
        try:
            actual_metrics = fm.calculate_all_metrics(returns_series)
            print(f"   ✓ Metrics calculated")
            print(f"   Total metrics: {len(actual_metrics)}")

            # Show key metrics
            for key in ["total_return", "sharpe", "max_drawdown", "volatility"]:
                if key in actual_metrics:
                    print(f"   {key}: {actual_metrics[key]:.4f}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print("   ⚠️ No valid returns to calculate metrics")

    # Check specific problematic metrics
    print("\n4. Checking Specific Metrics:")
    import quantstats as qs

    if len(returns_series) > 0:
        try:
            # These often cause issues
            print(f"   Win rate: {qs.stats.win_rate(returns_series):.4f}")
        except Exception as e:
            print(f"   ✗ Win rate error: {e}")

        try:
            print(f"   Profit factor: {qs.stats.profit_factor(returns_series):.4f}")
        except Exception as e:
            print(f"   ✗ Profit factor error: {e}")

        try:
            print(f"   Tail ratio: {qs.stats.tail_ratio(returns_series):.4f}")
        except Exception as e:
            print(f"   ✗ Tail ratio error: {e}")


if __name__ == "__main__":
    debug_metrics_calculation()
