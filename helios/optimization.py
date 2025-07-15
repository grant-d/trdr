"""
Genetic Algorithm optimization for Helios Trader
Implements walk-forward optimization of strategy parameters
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
import json
from pathlib import Path
from chalk import black
from ranges import ParameterRange, MinMaxRange, create_log_range
from generic_algorithm import GeneticAlgorithm

class WalkForwardOptimizer:
    """
    Walk-forward optimization for genetic algorithm
    """
    
    def __init__(self, n_windows: int = 5, train_ratio: float = 0.7):
        """
        Initialize walk-forward optimizer (always uses bar-based, non-overlapping windows)
        
        Parameters:
        -----------
        n_windows : int
            Number of windows to create (minimum 1)
        train_ratio : float
            Train/test split ratio (e.g., 0.7 means 70% train, 30% test)
        """
        self.n_windows = math.trunc(max(n_windows, 1))  # Ensure at least 1 window
        self.train_ratio = min(max(0.05, train_ratio), 0.95)
    
    def generate_windows(self, data: pd.DataFrame) -> List[Tuple[int, int, int, int]]:
        """
        Generate train/test windows (always bar-based, non-overlapping)
        
        Returns:
        --------
        List of tuples (train_start, train_end, test_start, test_end) as bar indices
        """
        windows = []
        total_bars = len(data)
        
        # Calculate minimum bars needed based on test split requirement
        # We need at least 100 bars in the TEST set for meaningful evaluation
        min_test_bars = 100
        min_bars_per_window = int(min_test_bars / (1 - self.train_ratio))  # e.g., 100 / 0.3 = 333
        
        # Check if we should adjust the number of windows
        if total_bars < min_bars_per_window * self.n_windows:
            # Not enough data for requested windows with minimum size
            # Recalculate to use fewer windows
            actual_n_windows = max(1, total_bars // min_bars_per_window)
            
            if actual_n_windows < self.n_windows:
                print(f"⚠️  Adjusting from {self.n_windows} to {actual_n_windows} windows due to insufficient data")
                if actual_n_windows == 0:
                    actual_n_windows = 1
                    print(f"   Using 1 window with only {total_bars} bars (minimum recommended: {min_bars_per_window})")
        else:
            actual_n_windows = self.n_windows
        
        # Calculate actual window size
        window_size = total_bars // actual_n_windows
        
        if window_size < min_bars_per_window:
            print(f"Warning: Only {window_size} bars per window (minimum recommended: {min_bars_per_window})")
            print(f"  With {self.train_ratio:.0%} train split, test set has only {int(window_size * (1 - self.train_ratio))} bars")
        
        train_size = int(window_size * self.train_ratio)
        test_size = window_size - train_size
        
        for i in range(actual_n_windows):
            train_start = i * window_size
            train_end = train_start + train_size
            test_start = train_end
            test_end = min(test_start + test_size, total_bars)
            
            if test_end > total_bars:
                break
                
            windows.append((train_start, train_end, test_start, test_end))
        
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
            
            # Get train and test data (always bar-based)
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Optimize on training data
            best_individual, fitness_history, ensemble_params = ga.optimize(train_data, verbose=False)
            
            # Evaluate on test data (let it auto-detect volatility for test period)
            test_fitness = ga.evaluate_fitness(best_individual, test_data)
            
            window_result = {
                'window': i,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
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

def auto_detect_dollar_thresholds(data: pd.DataFrame, walk_forward_windows: int = 1, symbol: str = "UNKNOWN", cache_dir: str = "./helios/data") -> float:
    """
    Automatically detect appropriate dollar bar thresholds based on asset characteristics
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data with columns: open, high, low, close, volume
    walk_forward_windows : int
        Number of walk-forward windows needed (affects bar count requirements)
    symbol : str
        Symbol name for caching the threshold
    cache_dir : str
        Directory to cache the threshold file
        
    Returns:
    --------
    float
        Single recommended threshold value
    """
    # Check if we have a cached threshold for this symbol
    cache_file = Path(cache_dir) / f"{symbol}_dollar_threshold.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                cached_threshold = cache_data.get('dollar_threshold')
                if cached_threshold:
                    print(f"\n{black('Using cached dollar threshold for {}'.format(symbol))}: ${cached_threshold:,.0f}")
                    return float(cached_threshold)
        except Exception as e:
            print(f"Warning: Could not load cached threshold: {e}")
    # Calculate dollar volume for ALL bars in the dataset
    dollar_volume = (data['high'] + data['low'] + data['close']) / 3 * data['volume']
    
    # Calculate total dollar volume
    total_dollar_volume = float(dollar_volume.sum())
    
    # Calculate basic statistics
    median_dollar_vol = float(dollar_volume.median())
    mean_dollar_vol = float(dollar_volume.mean())
    std_dollar_vol = float(dollar_volume.std())
    
    # Calculate target number of bars needed
    # We need at least 100 bars in TEST set, with 70/30 split that means ~333 bars per window
    min_test_bars = 100
    train_ratio = 0.7  # Standard 70/30 split
    min_bars_per_window = int(min_test_bars / (1 - train_ratio))  # 100 / 0.3 = 333
    safety_factor = 1.5  # Standard safety factor
    target_total_bars = walk_forward_windows * min_bars_per_window * safety_factor
    
    # Simple calculation: total volume / target bars = threshold
    # This ensures we get approximately the right number of bars
    base_threshold = total_dollar_volume / target_total_bars
    
    # Adjust for volume distribution
    # If volume is highly skewed (common in crypto), we need a lower threshold
    volume_skewness = std_dollar_vol / (mean_dollar_vol + 1e-9)
    
    if volume_skewness > 3.0:
        # High skewness - many small bars, few large bars
        adjustment_factor = 0.3
    elif volume_skewness > 1.5:
        # Moderate skewness
        adjustment_factor = 0.6
    else:
        # Low skewness - relatively uniform volume
        adjustment_factor = 0.9
    
    target_threshold = base_threshold * adjustment_factor
    
    # Simple bounds check
    if median_dollar_vol > 0:
        # Don't let threshold be too extreme relative to median
        min_threshold = median_dollar_vol * 0.01
        max_threshold = median_dollar_vol * 100
        target_threshold = float(np.clip(target_threshold, min_threshold, max_threshold))
    
    # Print analysis
    print(f"\n{black('Auto-threshold analysis:')}")
    print(f"  {black('Total bars:'.ljust(22))} {len(data)}")
    print(f"  {black('Total dollar volume:'.ljust(22))} ${total_dollar_volume:,.0f}")
    print(f"  {black('Mean dollar vol/bar:'.ljust(22))} ${mean_dollar_vol:,.0f}")
    print(f"  {black('Median dollar vol/bar:'.ljust(22))} ${median_dollar_vol:,.0f}")
    print(f"  {black('Volume skewness:'.ljust(22))} {volume_skewness:.2f}")
    print(f"  {black('Walk-forward windows:'.ljust(22))} {walk_forward_windows}")
    print(f"  {black('Target bars needed:'.ljust(22))} {int(target_total_bars)}")
    print(f"  {black('Base threshold:'.ljust(22))} ${base_threshold:,.0f}")
    print(f"  {black('Adjustment factor:'.ljust(22))} {adjustment_factor}")
    print(f"  {black('Target threshold:'.ljust(22))} ${target_threshold:,.0f}")
    
    # Save the calculated threshold for future use
    try:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir_path / f"{symbol}_dollar_threshold.json"
        
        cache_data = {
            'dollar_threshold': float(target_threshold),
            'symbol': symbol,
            'calculated_at': datetime.now().isoformat(),
            'walk_forward_windows': walk_forward_windows,
            'total_bars': len(data),
            'mean_dollar_volume': mean_dollar_vol,
            'median_dollar_volume': median_dollar_vol,
            'volume_skewness': volume_skewness
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"  {black('Saved threshold to:'.ljust(22))} {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save threshold to cache: {e}")
    
    return float(target_threshold)

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
