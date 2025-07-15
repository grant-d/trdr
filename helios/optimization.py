"""
Genetic Algorithm optimization for Helios Trader
Implements walk-forward optimization of strategy parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Mapping
from dataclasses import dataclass
import random
from datetime import datetime, timedelta
import json
from pathlib import Path
import signal
import time
import threading
from chalk import black, cyan, green, yellow, red, bold
from .factors import calculate_mss, calculate_macd, calculate_rsi
from .strategy_enhanced import EnhancedTradingStrategy
from .performance import calculate_sortino_ratio, calculate_calmar_ratio
from .data_processing import prepare_data
from .ranges import ParameterRange, MinMaxRange, LogRange, DiscreteRange, create_log_range, create_discrete_range
from .generic_algorithm import GeneticAlgorithm, estimate_asset_volatility_scale

class WalkForwardOptimizer:
    """
    Walk-forward optimization for genetic algorithm
    """
    
    def __init__(self, 
                 window_size: int = 365,
                 step_size: int = 90,
                 test_size: int = 90):
        """
        Initialize walk-forward optimizer
        
        Parameters:
        -----------
        window_size : int
            Training window size in days
        step_size : int
            Step size for moving window
        test_size : int
            Test period size in days
        """
        self.window_size = window_size
        self.step_size = step_size
        self.test_size = test_size
    
    def generate_windows(self, data: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Generate train/test windows
        
        Returns:
        --------
        List of tuples (train_start, train_end, test_start, test_end)
        """
        windows = []
        
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_start = start_date
        
        while current_start + timedelta(days=self.window_size + self.test_size) <= end_date:
            train_start = current_start
            train_end = current_start + timedelta(days=self.window_size)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_size)
            
            windows.append((train_start, train_end, test_start, test_end))
            
            current_start += timedelta(days=self.step_size)
        
        return windows
    
    def optimize(self, data: pd.DataFrame, ga: GeneticAlgorithm, 
                save_results: bool = True, output_dir: str = "./optimization_results") -> Dict:
        """
        Run walk-forward optimization
        
        Parameters:
        -----------
        data : pd.DataFrame
            Full dataset
        ga : GeneticAlgorithm
            Genetic algorithm instance
        save_results : bool
            Save optimization results
        output_dir : str
            Directory to save optimization results
        
        Returns:
        --------
        Dict
            Optimization results
        """
        windows = self.generate_windows(data)
        results = []
        
        print(f"Running walk-forward optimization with {len(windows)} windows")
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\nWindow {i+1}/{len(windows)}")
            print(f"Train: {train_start} to {train_end}")
            print(f"Test: {test_start} to {test_end}")
            
            # Get train and test data
            train_data = data[train_start:train_end]
            test_data = data[test_start:test_end]
            
            # Optimize on training data
            best_individual, fitness_history, ensemble_params = ga.optimize(train_data, verbose=False)
            
            # Evaluate on test data (let it auto-detect volatility for test period)
            test_fitness = ga.evaluate_fitness(best_individual, test_data)
            
            window_result = {
                'window': i,
                'train_start': train_start.isoformat(),
                'train_end': train_end.isoformat(),
                'test_start': test_start.isoformat(),
                'test_end': test_end.isoformat(),
                'best_parameters': best_individual.genes,
                'train_fitness': best_individual.fitness,
                'test_fitness': test_fitness,
                'fitness_history': fitness_history
            }
            
            results.append(window_result)
            
            print(f"Train Fitness: {best_individual.fitness:.4f}, "
                  f"Test Fitness: {test_fitness:.4f}")
        
        # Aggregate results
        avg_train_fitness = np.mean([r['train_fitness'] for r in results])
        avg_test_fitness = np.mean([r['test_fitness'] for r in results])
        
        optimization_results = {
            'windows': results,
            'avg_train_fitness': avg_train_fitness,
            'avg_test_fitness': avg_test_fitness,
            'parameter_ranges': ga.parameter_ranges,
            'ga_settings': {
                'population_size': ga.population_size,
                'generations': ga.generations,
                'mutation_rate': ga.mutation_rate
            }
        }
        
        if save_results:
            # Save results
            results_dir = Path(output_dir)
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = results_dir / f'walk_forward_{timestamp}.json'
            
            with open(results_file, 'w') as f:
                json.dump(optimization_results, f, indent=2)
            
            print(f"\nResults saved to: {results_file}")
        
        return optimization_results

def auto_detect_dollar_thresholds(data: pd.DataFrame, sample_size: int = 1000) -> float:
    """
    Automatically detect appropriate dollar bar thresholds based on asset characteristics
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data with columns: open, high, low, close, volume
    sample_size : int
        Number of recent bars to analyze for threshold detection
        
    Returns:
    --------
    float
        Single recommended threshold value
    """
    # Use most recent data for analysis
    sample_data = data.tail(sample_size)
    
    # Calculate dollar volume per bar
    dollar_volume = (sample_data['high'] + sample_data['low'] + sample_data['close']) / 3 * sample_data['volume']
    
    # Calculate statistics - ensure we get numeric values
    # Use pandas methods for better type handling
    median_dollar_vol = float(dollar_volume.median())
    mean_dollar_vol = float(dollar_volume.mean())
    
    # Determine asset class based on price level and volume characteristics
    avg_price = float(sample_data['close'].mean())
    avg_volume = float(sample_data['volume'].mean())
    
    print(f"\n{black('Auto-threshold analysis:')}")
    print(f"  {black('Average price:'.ljust(22))} ${avg_price:.2f}")
    print(f"  {black('Average volume:'.ljust(22))} {avg_volume:,.0f}")
    print(f"  {black('Median dollar volume:'.ljust(22))} ${median_dollar_vol:,.0f}")
    print(f"  {black('Mean dollar volume:'.ljust(22))} ${mean_dollar_vol:,.0f}")
    
    # Calculate target bars per day (aim for 20-50 bars)
    target_bars_per_day = 30  # Good balance for most assets
    
    # Calculate threshold to achieve target bars per day
    target_threshold = median_dollar_vol / target_bars_per_day
    
    # Set reasonable bounds (0.5x to 2x the target)
    min_threshold = float(target_threshold * 0.5)
    max_threshold = float(target_threshold * 2.0)
    
    # Apply absolute limits to prevent extreme values
    # Use a more reasonable minimum based on the actual data
    data_max_dollar_vol = float(dollar_volume.max())
    reasonable_min = max(1000.0, data_max_dollar_vol * 0.01)  # At least $1k or 1% of max volume
    min_threshold = max(reasonable_min, min(min_threshold, 100_000_000.0))
    max_threshold = max(min_threshold * 2, min(max_threshold, 500_000_000.0))
    
    print(f"  {black('Target:'.ljust(22))} ~{target_bars_per_day} bars per day")
    
    # Create logarithmic range with 4 points
    threshold_range = create_log_range(float(min_threshold), float(max_threshold), 4)
    
    # Return the second value (index 1) from the 4-point range
    # This gives us a more conservative threshold for more bars per day
    # With 4 values: [min, lower-mid, upper-mid, max], we pick lower-mid
    return threshold_range.values[1]

def create_enhanced_parameter_ranges(volatility_scale: float = 1.0) -> Dict[str, ParameterRange]:
    """
    Create enhanced parameter ranges using polymorphic configuration
    Uses LogRange for discrete threshold values and MinMaxRange for continuous weights
    
    Parameters:
    -----------
    volatility_scale : float
        Scale factor for regime thresholds based on asset volatility (0.1 to 3.0)
        Lower values for low-volatility assets like stocks
        Higher values for high-volatility assets like crypto
    """
    # Clamp volatility scale to reasonable bounds
    volatility_scale = max(0.1, min(3.0, volatility_scale))
    
    return {
        # Factor weights (continuous ranges for fine-tuning)
        'weight_trend': MinMaxRange(0.0, 1.0),
        'weight_volatility': MinMaxRange(0.0, 1.0),
        'weight_exhaustion': MinMaxRange(0.0, 1.0),
        
        # Lookback periods (continuous range)
        'lookback_int': MinMaxRange(10, 50),
        
        # Regime thresholds (scaled by volatility)
        'strong_bull_threshold': create_log_range(15.0 * volatility_scale, 50.0 * volatility_scale, 4),
        'weak_bull_threshold': create_log_range(5.0 * volatility_scale, 25.0 * volatility_scale, 4),
        'neutral_threshold_upper': create_log_range(3.0 * volatility_scale, 15.0 * volatility_scale, 4),
        'neutral_threshold_lower': create_log_range(-15.0 * volatility_scale, -3.0 * volatility_scale, 4),
        'weak_bear_threshold': create_log_range(-25.0 * volatility_scale, -5.0 * volatility_scale, 4),
        'strong_bear_threshold': create_log_range(-50.0 * volatility_scale, -15.0 * volatility_scale, 4),
        
        # Stop-loss multipliers (discrete logarithmic spacing)
        'stop_loss_multiplier_strong': create_log_range(1.5, 4.5, 4),
        'stop_loss_multiplier_weak': create_log_range(0.8, 2.7, 4),
        
        # Gradual entry parameters (discrete logarithmic spacing)
        'entry_step_size': create_log_range(0.15, 0.7, 4),
        'max_position_pct': create_log_range(0.6, 1.0, 4),
    }
