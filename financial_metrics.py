#!/usr/bin/env python3
"""
Financial performance metrics using QuantStats library.
"""
import pandas as pd
import numpy as np
import quantstats as qs
from typing import Dict, Union, Optional
import warnings

# Suppress QuantStats warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class FinancialMetrics:
    """
    Calculate comprehensive financial performance metrics using QuantStats.
    """

    def __init__(self, periods_per_year: int = 525600):
        """
        Initialize with frequency.

        Args:
            periods_per_year: Number of periods per year (default: 525600 for minute data)
        """
        self.periods_per_year = periods_per_year

    def calculate_all_metrics(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary with all performance metrics
        """
        # Convert to pandas Series if needed
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        # Remove any NaN values
        returns = returns.dropna()

        if len(returns) == 0:
            return self._empty_metrics()

        # Check if all returns are zero
        if returns.std() == 0 or len(returns) < 2:
            # Return basic metrics only
            metrics = self._empty_metrics()
            metrics["total_return"] = (np.prod(1 + returns) - 1) * 100
            return metrics

        # Calculate all metrics
        metrics = {}

        try:
            # Suppress specific warnings we expect
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*division by zero.*")
                warnings.filterwarnings(
                    "ignore", message=".*invalid value encountered.*"
                )
                warnings.filterwarnings(
                    "ignore", message=".*No non-zero returns found.*"
                )

                # Basic return metrics
                metrics["total_return"] = qs.stats.comp(returns) * 100
                metrics["cagr"] = qs.stats.cagr(returns) * 100
                metrics["volatility"] = (
                    qs.stats.volatility(returns, annualize=True) * 100
                )

                # Risk-adjusted metrics
                metrics["sharpe"] = qs.stats.sharpe(returns, annualize=True)
                metrics["sortino"] = qs.stats.sortino(returns, annualize=True)
                metrics["calmar"] = qs.stats.calmar(returns)

                # Drawdown metrics
                metrics["max_drawdown"] = qs.stats.max_drawdown(returns) * 100
                metrics["avg_drawdown"] = qs.stats.avg_drawdown(returns) * 100
                metrics["avg_drawdown_days"] = qs.stats.avg_drawdown_days(returns)

                # Win/Loss metrics
                metrics["win_rate"] = qs.stats.win_rate(returns) * 100
                metrics["avg_win"] = qs.stats.avg_win(returns) * 100
                metrics["avg_loss"] = qs.stats.avg_loss(returns) * 100
                metrics["profit_factor"] = qs.stats.profit_factor(returns)

                # Distribution metrics
                metrics["skewness"] = qs.stats.skew(returns)
                metrics["kurtosis"] = qs.stats.kurtosis(returns)

                # Value at Risk
                metrics["var_5"] = qs.stats.value_at_risk(returns, cutoff=0.05) * 100
                metrics["cvar_5"] = (
                    qs.stats.conditional_value_at_risk(returns, cutoff=0.05) * 100
                )

                # Additional metrics
                metrics["kelly_criterion"] = qs.stats.kelly_criterion(returns) * 100
                metrics["tail_ratio"] = qs.stats.tail_ratio(returns)
                metrics["common_sense_ratio"] = qs.stats.common_sense_ratio(returns)

                # Benchmark comparison if provided
                if benchmark_returns is not None:
                    if isinstance(benchmark_returns, np.ndarray):
                        benchmark_returns = pd.Series(benchmark_returns)
                    benchmark_returns = benchmark_returns.dropna()

                    if len(benchmark_returns) > 0:
                        metrics["alpha"] = (
                            qs.stats.alpha(returns, benchmark_returns) * 100
                        )
                        metrics["beta"] = qs.stats.beta(returns, benchmark_returns)
                        metrics["information_ratio"] = qs.stats.information_ratio(
                            returns, benchmark_returns
                        )

        except Exception as e:
            # Only print warning if it's not an expected error
            error_msg = str(e)
            if not any(
                expected in error_msg
                for expected in [
                    "division by zero",
                    "invalid value",
                    "No non-zero returns",
                ]
            ):
                print(f"Warning: Error calculating some metrics: {e}")
            # Fill in basic metrics manually if QuantStats fails
            if "total_return" not in metrics:
                metrics["total_return"] = (np.prod(1 + returns) - 1) * 100
            if "sharpe" not in metrics:
                metrics["sharpe"] = (
                    np.sqrt(self.periods_per_year) * returns.mean() / returns.std()
                    if returns.std() > 0
                    else 0
                )
            if "max_drawdown" not in metrics:
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                metrics["max_drawdown"] = np.min(drawdown) * 100

        return metrics

    def _empty_metrics(self) -> Dict[str, float]:
        """Return dictionary with zero/NaN metrics for empty data."""
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "avg_drawdown_days": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "var_5": 0.0,
            "cvar_5": 0.0,
            "kelly_criterion": 0.0,
            "tail_ratio": 0.0,
            "common_sense_ratio": 0.0,
        }

    def format_metrics_report(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics into a readable report.
        """
        report = []
        report.append("PERFORMANCE METRICS")
        report.append("=" * 40)

        # Return metrics
        report.append("\n📈 RETURNS:")
        report.append(f"  Total Return:     {metrics.get('total_return', 0):.2f}%")
        report.append(f"  CAGR:             {metrics.get('cagr', 0):.2f}%")
        report.append(f"  Volatility:       {metrics.get('volatility', 0):.2f}%")

        # Risk metrics
        report.append("\n⚖️  RISK-ADJUSTED:")
        report.append(f"  Sharpe Ratio:     {metrics.get('sharpe', 0):.3f}")
        report.append(f"  Sortino Ratio:    {metrics.get('sortino', 0):.3f}")
        report.append(f"  Calmar Ratio:     {metrics.get('calmar', 0):.3f}")

        # Drawdown metrics
        report.append("\n📉 DRAWDOWN:")
        report.append(f"  Max Drawdown:     {metrics.get('max_drawdown', 0):.2f}%")
        report.append(f"  Avg Drawdown:     {metrics.get('avg_drawdown', 0):.2f}%")
        report.append(f"  Avg DD Days:      {metrics.get('avg_drawdown_days', 0):.1f}")

        # Win/Loss metrics
        report.append("\n🎯 WIN/LOSS:")
        report.append(f"  Win Rate:         {metrics.get('win_rate', 0):.1f}%")
        report.append(f"  Avg Win:          {metrics.get('avg_win', 0):.2f}%")
        report.append(f"  Avg Loss:         {metrics.get('avg_loss', 0):.2f}%")
        report.append(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}")

        # Risk metrics
        report.append("\n⚠️  RISK:")
        report.append(f"  VaR (5%):         {metrics.get('var_5', 0):.2f}%")
        report.append(f"  CVaR (5%):        {metrics.get('cvar_5', 0):.2f}%")
        report.append(f"  Kelly Criterion:  {metrics.get('kelly_criterion', 0):.1f}%")

        return "\n".join(report)


def example_usage():
    """Example of how to use FinancialMetrics."""
    # Generate sample returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 1000))  # Daily returns

    # Calculate metrics
    fm = FinancialMetrics(periods_per_year=252)  # Daily data
    metrics = fm.calculate_all_metrics(returns)

    # Print report
    print(fm.format_metrics_report(metrics))


if __name__ == "__main__":
    example_usage()
