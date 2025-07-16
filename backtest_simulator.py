"""
Backtest simulator for running trading simulations with optimized parameters.

This module provides functionality to simulate real trading using the best
parameters found by the genetic algorithm optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from regime_strategy import RegimeStrategy, RegimeStrategyParameters
from indicators import calculate_mss
from performance import calculate_sortino_ratio, calculate_calmar_ratio


class BacktestSimulator:
    """
    Simulates trading with optimized parameters.
    
    This class takes the best parameters from optimization and runs
    a realistic backtest showing actual trades, costs, and performance.
    """

    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize the backtest simulator.
        
        Args:
            initial_capital: Starting capital for simulation
        """
        self.initial_capital = initial_capital

    def run(
        self, 
        parameters: Dict[str, float], 
        data: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run backtest simulation with given parameters.
        
        Args:
            parameters: Dictionary of strategy parameters
            data: Market data (dollar bars)
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary of performance metrics
        """
        if verbose:
            print("\n📊 Running Backtest Simulation with Best Parameters")
            print("="*80)

        # Create properly typed parameters
        params = RegimeStrategyParameters.from_dict(parameters)

        # Calculate market indicators
        factors_df, _ = calculate_mss(data)
        factors_df = factors_df.dropna()

        if len(factors_df) < 50:
            if verbose:
                print("⚠️  Insufficient data for backtest")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0
            }

        # Initialize strategy with the parameters
        param_dict = params.to_dict()
        strategy = RegimeStrategy(
            initial_capital=self.initial_capital,
            max_position_fraction=param_dict['max_position_fraction'],
            entry_step_size=param_dict['entry_step_size'],
            stop_loss_multiplier_strong=param_dict['stop_loss_multiplier_strong'],
            stop_loss_multiplier_weak=param_dict['stop_loss_multiplier_weak'],
            strong_bull_threshold=param_dict['strong_bull_threshold'],
            weak_bull_threshold=param_dict['weak_bull_threshold'],
            neutral_upper=param_dict['neutral_upper'],
            neutral_lower=param_dict['neutral_lower'],
            weak_bear_threshold=param_dict['weak_bear_threshold'],
            strong_bear_threshold=param_dict['strong_bear_threshold'],
            allow_shorts=False
        )

        # Run the backtest
        results_df = strategy.run_backtest(data, factors_df)

        # Get trade summary
        trade_summary = strategy.get_trade_summary()

        # Calculate key metrics
        final_value = results_df['portfolio_value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100

        # Calculate returns for Sharpe ratio
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df = results_df.dropna()

        # Calculate annualization factor
        if 'timestamp' in data.columns and len(results_df) > 1:
            time_diffs = pd.to_datetime(data['timestamp']).diff().dropna()
            avg_bar_duration = pd.Timedelta(time_diffs.mean())
            bars_per_year = pd.Timedelta(days=252).total_seconds() / avg_bar_duration.total_seconds()
        else:
            bars_per_year = 252 * 390  # Assume minute bars

        # Calculate Sharpe ratio
        mean_return = results_df['returns'].mean()
        std_return = results_df['returns'].std()
        sharpe_ratio = (mean_return / std_return) * (bars_per_year ** 0.5) if std_return > 0 else 0
        
        # Calculate Sortino ratio
        sortino_ratio = calculate_sortino_ratio(results_df['returns'], periods_per_year=int(bars_per_year))
        
        # Calculate Calmar ratio
        calmar_ratio = calculate_calmar_ratio(results_df['returns'], periods_per_year=int(bars_per_year))

        # Calculate max drawdown
        peak = results_df['portfolio_value'].cummax()
        drawdown = (results_df['portfolio_value'] - peak) / peak
        max_drawdown = drawdown.min() * 100

        # Calculate actual trade costs
        position_changes = results_df['position_units'].diff().abs()
        trade_values = position_changes * results_df['close']
        total_costs = (trade_values * 0.0025).sum()  # Alpaca's 0.25% rate

        # Prepare metrics
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': trade_summary['win_rate'],
            'total_trades': trade_summary['total_trades'],
            'profit_factor': trade_summary['profit_factor'],
            'final_value': final_value,
            'total_costs': total_costs
        }

        if verbose:
            self._print_results(metrics, results_df, strategy, trade_summary)

        return metrics

    def _print_results(
        self, 
        metrics: Dict[str, float], 
        results_df: pd.DataFrame,
        strategy: RegimeStrategy,
        trade_summary: Dict
    ):
        """Print detailed backtest results."""
        print(f"\n💰 Backtest Results:")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Final Value: ${metrics['final_value']:,.2f}")
        print(f"   Total Return: {metrics['total_return']:.2f}%")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"   Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"   Transaction Costs: ${metrics['total_costs']:,.2f}")

        print(f"\n📈 Trading Summary:")
        print(f"   Total Trades: {trade_summary['total_trades']}")
        print(f"   Win Rate: {trade_summary['win_rate']:.1f}%")
        print(f"   Profit Factor: {trade_summary['profit_factor']:.2f}")

        # Show recent trades
        if strategy.trades:
            print(f"\n🔄 Last 5 Trades:")
            recent_trades = strategy.trades[-5:]
            for trade in recent_trades:
                action_icon = "🟢" if trade.action == "Buy" else "🔴"
                print(f"   {action_icon} {trade.timestamp.strftime('%Y-%m-%d %H:%M')} - {trade.action} "
                      f"{abs(trade.units):.2f} @ ${trade.price:.2f} - {trade.reason}")

        # Show position distribution across regimes
        print(f"\n🎯 Position Distribution by Regime:")
        regime_stats = results_df.groupby('regime').agg({
            'position_units': ['mean', 'count']
        }).round(2)

        for regime in ['Strong Bull', 'Weak Bull', 'Neutral', 'Weak Bear', 'Strong Bear']:
            if regime in regime_stats.index:
                avg_pos = regime_stats.loc[regime, ('position_units', 'mean')]
                count = regime_stats.loc[regime, ('position_units', 'count')]
                pct = (count / len(results_df)) * 100
                print(f"   {regime}: Avg Position = {avg_pos:.2f} units ({pct:.1f}% of time)")

        print("="*80)
