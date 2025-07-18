#!/usr/bin/env python3
"""
Walk-Forward Optimization framework for trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class WFOConfig:
    """Configuration for walk-forward optimization."""

    in_sample_periods: int  # Number of periods for optimization
    out_sample_periods: int  # Number of periods for testing
    step_periods: int  # Number of periods to step forward
    min_samples: int = 1000  # Minimum samples needed for optimization


class WalkForwardOptimizer:
    """
    Implements walk-forward optimization for trading strategies.
    """

    def __init__(self, config: WFOConfig):
        self.config = config

    def split_data(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into in-sample/out-sample pairs for walk-forward analysis.

        Returns:
            List of (in_sample_df, out_sample_df) tuples
        """
        splits = []
        total_len = len(df)

        start_idx = 0
        while (
            start_idx + self.config.in_sample_periods + self.config.out_sample_periods
            <= total_len
        ):
            # In-sample data
            in_sample_end = start_idx + self.config.in_sample_periods
            in_sample = df.iloc[start_idx:in_sample_end]

            # Out-sample data
            out_sample_end = in_sample_end + self.config.out_sample_periods
            out_sample = df.iloc[in_sample_end:out_sample_end]

            if len(in_sample) >= self.config.min_samples:
                splits.append((in_sample, out_sample))

            # Step forward
            start_idx += self.config.step_periods

        return splits

    def optimize(
        self,
        df: pd.DataFrame,
        strategy_func: Callable,
        optimize_func: Callable,
        evaluate_func: Callable,
    ) -> Dict[str, Any]:
        """
        Run walk-forward optimization.

        Args:
            df: Complete dataset
            strategy_func: Function that generates signals given params
            optimize_func: Function that finds optimal params given in-sample data
            evaluate_func: Function that evaluates performance on out-sample data

        Returns:
            Dictionary with optimization results
        """
        splits = self.split_data(df)

        if not splits:
            raise ValueError("Not enough data for walk-forward optimization")

        results = {
            "periods": [],
            "in_sample_performance": [],
            "out_sample_performance": [],
            "optimal_params": [],
            "all_signals": [],
            "all_returns": [],
        }

        print(f"Running walk-forward optimization with {len(splits)} periods...")

        for i, (in_sample, out_sample) in enumerate(splits):
            print(f"\nPeriod {i+1}/{len(splits)}")

            # Find optimal parameters on in-sample data
            optimal_params = optimize_func(in_sample)
            results["optimal_params"].append(optimal_params)

            # Evaluate on in-sample (for comparison)
            in_signals = strategy_func(in_sample, optimal_params)
            in_performance = evaluate_func(in_sample, in_signals)
            results["in_sample_performance"].append(in_performance)

            # Evaluate on out-of-sample
            out_signals = strategy_func(out_sample, optimal_params)
            out_performance = evaluate_func(out_sample, out_signals)
            results["out_sample_performance"].append(out_performance)

            # Store signals and returns for later analysis
            results["all_signals"].extend(out_signals)

            # Calculate returns
            if "close" in out_sample.columns:
                returns = out_sample["close"].pct_change().fillna(0)
                strategy_returns = (
                    pd.Series(out_signals).shift(1).fillna(0) * returns.values
                )
                results["all_returns"].extend(strategy_returns)

            results["periods"].append(
                {
                    "in_sample_start": in_sample.index[0],
                    "in_sample_end": in_sample.index[-1],
                    "out_sample_start": out_sample.index[0],
                    "out_sample_end": out_sample.index[-1],
                }
            )

            print(f"  In-sample Sharpe: {in_performance.get('sharpe', 0):.3f}")
            print(f"  Out-sample Sharpe: {out_performance.get('sharpe', 0):.3f}")

        # Calculate aggregate statistics
        results["aggregate_stats"] = self._calculate_aggregate_stats(results)

        return results

    def _calculate_aggregate_stats(self, results: Dict) -> Dict:
        """Calculate aggregate statistics across all periods."""
        all_returns = np.array(results["all_returns"])

        # Remove any NaN values
        all_returns = all_returns[~np.isnan(all_returns)]

        if len(all_returns) == 0:
            return {}

        # Calculate overall performance
        cumulative_return = np.prod(1 + all_returns) - 1

        # Sharpe ratio (annualized for minute data)
        sharpe = 0
        if np.std(all_returns) > 0:
            sharpe = np.sqrt(525600) * np.mean(all_returns) / np.std(all_returns)

        # Calculate consistency
        out_sharpes = [p.get("sharpe", 0) for p in results["out_sample_performance"]]
        consistency = (
            np.sum(np.array(out_sharpes) > 0) / len(out_sharpes) if out_sharpes else 0
        )

        return {
            "total_return": cumulative_return * 100,
            "sharpe_ratio": sharpe,
            "consistency": consistency * 100,  # Percentage of profitable periods
            "avg_in_sample_sharpe": np.mean(
                [p.get("sharpe", 0) for p in results["in_sample_performance"]]
            ),
            "avg_out_sample_sharpe": np.mean(out_sharpes),
            "num_periods": len(results["periods"]),
        }


def example_usage():
    """Example of how to use WalkForwardOptimizer."""
    # Configuration
    config = WFOConfig(
        in_sample_periods=2000,  # ~1.4 days of minute data
        out_sample_periods=500,  # ~8 hours
        step_periods=500,  # Step forward 8 hours
    )

    # Create optimizer
    wfo = WalkForwardOptimizer(config)

    # Load data
    df = pd.read_csv("data/sol_usd_1m.transform.csv")

    # Example strategy function
    def strategy_func(data, params):
        # This would call your Matrix Profile strategy
        return np.random.choice([-1, 0, 1], size=len(data))

    # Example optimization function
    def optimize_func(data):
        # This would use DEAP to find optimal params
        return {"window": 50, "threshold": 0.4}

    # Example evaluation function
    def evaluate_func(data, signals):
        returns = data["close"].pct_change().fillna(0)
        strategy_returns = pd.Series(signals).shift(1).fillna(0) * returns.values

        sharpe = 0
        if np.std(strategy_returns) > 0:
            sharpe = (
                np.sqrt(525600) * np.mean(strategy_returns) / np.std(strategy_returns)
            )

        return {"sharpe": sharpe, "return": (np.prod(1 + strategy_returns) - 1) * 100}

    # Run optimization
    # results = wfo.optimize(df, strategy_func, optimize_func, evaluate_func)

    print("Walk-Forward Optimizer ready for integration with Matrix Profile strategy.")


if __name__ == "__main__":
    example_usage()
