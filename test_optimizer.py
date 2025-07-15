#!/usr/bin/env python3
"""
Test script for the Optimizer class.

Demonstrates walk-forward optimization with genetic algorithm on a simple
moving average crossover strategy.
"""

import pandas as pd
import numpy as np
from optimizer import Optimizer, OptimizerConfig
from strategy_parameters import BaseStrategyParameters, IntegerRange
from pydantic import Field


class MAStrategyParameters(BaseStrategyParameters):
    """Moving Average strategy parameters."""
    
    fast_ma: IntegerRange = Field(
        default_factory=lambda: IntegerRange(min_value=5, max_value=50),
        description="Fast moving average period"
    )
    slow_ma: IntegerRange = Field(
        default_factory=lambda: IntegerRange(min_value=20, max_value=200),
        description="Slow moving average period"
    )
    
    def validate_constraints(self) -> bool:
        """Ensure fast MA is shorter than slow MA."""
        # For constant ranges, get the value directly
        if hasattr(self.fast_ma, 'value'):
            fast_val = self.fast_ma.value
        else:
            fast_val = self.fast_ma.sample()
            
        if hasattr(self.slow_ma, 'value'):
            slow_val = self.slow_ma.value
        else:
            slow_val = self.slow_ma.sample()
            
        return fast_val < slow_val


def simple_ma_strategy_fitness(params: MAStrategyParameters, data: pd.DataFrame) -> float:
    """
    Simple moving average crossover strategy fitness function.
    
    Args:
        params: MAStrategyParameters with 'fast_ma' and 'slow_ma' periods
        data: Market data with 'close' prices
        
    Returns:
        Sharpe ratio as fitness score
    """
    param_dict = params.to_dict()
    fast_ma = int(param_dict['fast_ma'])
    slow_ma = int(param_dict['slow_ma'])
    
    # Ensure fast < slow
    if fast_ma >= slow_ma:
        return -1000.0
        
    # Calculate moving averages
    data = data.copy()
    data['fast_ma'] = data['close'].rolling(window=fast_ma).mean()
    data['slow_ma'] = data['close'].rolling(window=slow_ma).mean()
    
    # Generate signals
    data['signal'] = 0
    data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1
    data.loc[data['fast_ma'] < data['slow_ma'], 'signal'] = -1
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    data['strategy_returns'] = data['signal'].shift(1) * data['returns']
    
    # Drop NaN values
    data = data.dropna()
    
    if len(data) < 20:
        return -1000.0
        
    # Calculate Sharpe ratio (annualized for minute data)
    # Assuming 252 trading days * 390 minutes per day
    periods_per_year = 252 * 390
    
    mean_return = data['strategy_returns'].mean()
    std_return = data['strategy_returns'].std()
    
    if std_return == 0:
        return 0.0
        
    sharpe_ratio = np.sqrt(periods_per_year) * mean_return / std_return
    
    return float(sharpe_ratio)


def main():
    """Test the optimizer with a simple strategy."""
    
    # Load data
    print("Loading market data...")
    try:
        df = pd.read_csv('data/btc_usd_1m_bars.csv', parse_dates=['timestamp'])
        print(f"Loaded {len(df)} bars")
    except FileNotFoundError:
        print("Creating synthetic data for demonstration...")
        # Create synthetic price data
        np.random.seed(42)
        n_bars = 2000
        timestamps = pd.date_range(start='2024-01-01', periods=n_bars, freq='1min')
        
        # Generate realistic price movement
        returns = np.random.normal(0.0001, 0.002, n_bars)
        prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
            'high': prices * (1 + np.random.uniform(0, 0.002, n_bars)),
            'low': prices * (1 + np.random.uniform(-0.002, 0, n_bars)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, n_bars)
        })
        
    # Configure optimizer
    config = OptimizerConfig(
        # Walk-forward parameters
        n_splits=3,  # Use 3 splits for faster demo
        test_size=None,  # Expanding window
        gap=0,
        
        # GA parameters
        param_class=MAStrategyParameters,
        population_size=30,  # Smaller for demo
        generations=20,      # Fewer for demo
        crossover_prob=0.8,
        mutation_prob=0.2,
        tournament_size=3,
        
        # General
        verbose=True,
        seed=42
    )
    
    # Create optimizer
    optimizer = Optimizer(config)
    
    # Run optimization
    print("\nStarting optimization...")
    results = optimizer.optimize(
        data=df,
        fitness_function=simple_ma_strategy_fitness,
        evaluation_function=simple_ma_strategy_fitness  # Same function for train and test
    )
    
    # Get consensus parameters
    print("\n" + "="*80)
    print("Final Parameter Recommendations")
    print("="*80)
    
    consensus_params = optimizer.get_consensus_parameters()
    print(f"\nConsensus Parameters (median):")
    for param, value in consensus_params.items():
        print(f"  {param}: {value:.1f}")
        
    weighted_params = optimizer.get_performance_weighted_parameters()
    print(f"\nPerformance-Weighted Parameters:")
    for param, value in weighted_params.items():
        print(f"  {param}: {value:.1f}")
        
    # Get results DataFrame
    results_df = optimizer.get_results_dataframe()
    print("\nDetailed Results:")
    print(results_df.to_string())
    
    # Analyze parameter stability
    print("\n" + "="*80)
    print("Parameter Stability Analysis")
    print("="*80)
    
    param_names = list(config.param_class.get_param_ranges().keys())
    for param in param_names:
        param_values = [r.best_params.to_dict()[param] for r in results]
        print(f"\n{param}:")
        print(f"  Mean: {np.mean(param_values):.1f}")
        print(f"  Std:  {np.std(param_values):.1f}")
        print(f"  CV:   {np.std(param_values)/np.mean(param_values)*100:.1f}%")


if __name__ == "__main__":
    main()