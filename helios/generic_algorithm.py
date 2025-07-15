"""
Genetic Algorithm optimization for Helios Trader
Implements walk-forward optimization of strategy parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Mapping
from dataclasses import dataclass
import random
from datetime import datetime
import json
import threading
from chalk import black, cyan, green, yellow, red, bold
from factors import calculate_mss, calculate_macd, calculate_rsi
from strategy_enhanced import EnhancedTradingStrategy
from performance import calculate_sortino_ratio, calculate_calmar_ratio
from ranges import ParameterRange, MinMaxRange, LogRange, DiscreteRange, create_log_range, create_discrete_range


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population"""
    genes: Dict[str, float]
    fitness: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'genes': self.genes,
            'fitness': self.fitness
        }


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing trading strategy parameters
    """
    
    def __init__(self, 
                 parameter_config: Mapping[str, Union[ParameterRange, Tuple[float, float]]],
                 symbol: str,
                 timeframe: str,
                 population_size: int = 50,
                 generations: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1,
                 allow_shorts: bool = False) -> None:
        """
        Initialize genetic algorithm
        
        Parameters:
        -----------
        parameter_config : Mapping[str, Union[ParameterRange, Tuple[float, float]]]
            Parameter configuration using polymorphic range types or legacy tuples
        population_size : int
            Number of individuals in population
        generations : int
            Number of generations to evolve
        mutation_rate : float
            Probability of mutation
        crossover_rate : float
            Probability of crossover
        elitism_rate : float
            Percentage of best individuals to keep
        allow_shorts : bool
            Whether to allow short positions
        symbol : str
            Trading symbol (e.g., 'BTC/USD', 'AAPL')
        timeframe : str
            Timeframe (e.g., '1min', '1hour', 'dollar_bars_1000')
        """
        # Convert parameter config to uniform ParameterRange objects with validation
        self.parameter_ranges = {}
        for param, config in parameter_config.items():
            try:
                # Validate parameter name
                if not isinstance(param, str) or not param.strip():
                    raise ValueError(f"Parameter name must be a non-empty string, got: {param}")
                
                if isinstance(config, ParameterRange):
                    # Validate ParameterRange object
                    self._validate_parameter_range(param, config)
                    self.parameter_ranges[param] = config
                elif isinstance(config, tuple) and len(config) == 2:
                    # Legacy tuple format - validate and convert to MinMaxRange
                    min_val, max_val = config
                    
                    # Validate tuple values
                    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                        raise ValueError(f"Parameter range values must be numeric, got: {config}")
                    
                    if np.isnan(min_val) or np.isinf(min_val) or np.isnan(max_val) or np.isinf(max_val):
                        raise ValueError(f"Parameter range values cannot be NaN or infinite: {config}")
                    
                    if min_val >= max_val:
                        raise ValueError(f"Parameter range min_val must be < max_val: {config}")
                    
                    # Create and validate MinMaxRange
                    range_obj = MinMaxRange(min_val, max_val)
                    self._validate_parameter_range(param, range_obj)
                    self.parameter_ranges[param] = range_obj
                else:
                    raise ValueError(f"Invalid parameter config for {param}: {config}. Must be ParameterRange object or (min, max) tuple")
                    
            except Exception as e:
                raise ValueError(f"Error validating parameter '{param}': {e}")
        
        # Validate we have at least one parameter
        if not self.parameter_ranges:
            raise ValueError("At least one parameter range must be specified")
        
        # Validate parameter set for threshold constraints
        self._validate_threshold_parameter_consistency()
        
        # Validate GA configuration parameters
        self._validate_ga_parameters(population_size, generations, mutation_rate, crossover_rate, elitism_rate)
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.allow_shorts = allow_shorts
        self.symbol = symbol
        self.timeframe = timeframe
        self.candidate_archive: List[Dict] = []  # Archive of recent top candidates
        self.archive_size = 5  # Keep top 5 candidates from recent runs
        # self.penalty_table_shown = False  # Track if penalty table has been displayed
        
        # Calculate elite size
        self.elite_size = max(1, int(population_size * elitism_rate))
        
        # Adaptive parameters
        self.base_mutation_rate = mutation_rate
        self.base_crossover_rate = crossover_rate
        self.generations_without_improvement = 0
        self.best_fitness_history = []
        self.max_fitness_history = 50  # Limit best fitness history to prevent memory leaks
        
        # Age-based elitism
        self.individual_ages = {}  # Track age of individuals
        self.max_elite_age = 5  # Maximum generations an elite can survive
        self.max_individual_ages = 1000  # Limit individual ages tracking to prevent memory leaks
        
        # Timeout handling for long evaluations
        self.evaluation_timeout = 5 * 60  # Maximum seconds per fitness evaluation (5 minutes)
        self.timeout_count = 0  # Track number of timeouts
        self.max_timeouts = 10  # Maximum timeouts before stopping optimization
        
        # Race condition prevention - thread synchronization
        self._fitness_history_lock = threading.Lock()  # Protects fitness history file I/O
        self._state_lock = threading.Lock()  # Protects shared state variables
        
        # Adaptive penalty system with persistence (instrument/timeframe specific)
        from utils import generate_filename
        self.fitness_history_file = generate_filename(
            symbol=symbol, 
            timeframe=timeframe, 
            postfix="fitness_history",
            extension=".json"
        )
        self.fitness_history = []  # Store historical fitness scores
        self.penalty_history = {
            'trade_quality': [],
            'drawdown': [],
            'parameter_extreme': []
        }
        self.adaptive_window = 100  # Number of evaluations to consider for adaptation
        self._load_fitness_history()
        
        # Log adaptive system initialization
        self._log_adaptive_system_info()
        
        # Dynamic population
        self.min_population_size = max(20, population_size // 2)
        self.max_population_size = min(200, population_size * 2)
        self.target_diversity = 0.3  # Target diversity level
    
    def _validate_parameter_range(self, param_name: str, param_range: ParameterRange) -> None:
        """Validate a single parameter range for common issues"""
        try:
            # Test basic functionality
            sample_val = param_range.sample()
            if not isinstance(sample_val, (int, float)):
                raise ValueError(f"Parameter range sample() must return numeric value, got {type(sample_val)}")
            
            if np.isnan(sample_val) or np.isinf(sample_val):
                raise ValueError(f"Parameter range sample() returned invalid value: {sample_val}")
            
            # Test mutation if available
            if hasattr(param_range, 'mutate'):
                mutated_val = param_range.mutate(sample_val)
                if not isinstance(mutated_val, (int, float)):
                    raise ValueError(f"Parameter range mutate() must return numeric value, got {type(mutated_val)}")
                
                if np.isnan(mutated_val) or np.isinf(mutated_val):
                    raise ValueError(f"Parameter range mutate() returned invalid value: {mutated_val}")
                    
        except Exception as e:
            raise ValueError(f"Parameter range validation failed for '{param_name}': {e}")
    
    def _validate_threshold_parameter_consistency(self) -> None:
        """Validate that threshold parameters have consistent ordering constraints"""
        # List of threshold parameters in required order (highest to lowest)
        threshold_params = [
            'strong_bull_threshold',
            'weak_bull_threshold', 
            'neutral_threshold_upper',
            'neutral_threshold_lower',
            'weak_bear_threshold',
            'strong_bear_threshold'
        ]
        
        # Check if we have any threshold parameters
        present_thresholds = [p for p in threshold_params if p in self.parameter_ranges]
        
        if not present_thresholds:
            return  # No threshold parameters, validation not needed
        
        # If we have threshold parameters, we should have a complete set for proper ordering
        missing_thresholds = [p for p in threshold_params if p not in self.parameter_ranges]
        if missing_thresholds and len(present_thresholds) > 1:
            print(yellow(f"Warning: Incomplete threshold parameter set. Missing: {missing_thresholds}"))
            print("This may result in invalid threshold ordering during optimization.")
        
        # Validate that threshold ranges are reasonable
        for param in present_thresholds:
            param_range = self.parameter_ranges[param]
            try:
                # Sample multiple times to check consistency
                samples = [param_range.sample() for _ in range(10)]
                min_sample, max_sample = min(samples), max(samples)
                
                # Basic range checks
                if 'strong_bull' in param and (min_sample < 10 or max_sample > 200):
                    print(yellow(f"Warning: {param} range [{min_sample:.1f}, {max_sample:.1f}] may be too extreme"))
                elif 'weak_bull' in param and (min_sample < 0 or max_sample > 100):
                    print(yellow(f"Warning: {param} range [{min_sample:.1f}, {max_sample:.1f}] may be too extreme"))
                elif 'neutral' in param and (abs(min_sample) > 50 or abs(max_sample) > 50):
                    print(yellow(f"Warning: {param} range [{min_sample:.1f}, {max_sample:.1f}] may be too extreme"))
                elif 'bear' in param and (min_sample > 0 or min_sample < -200 or max_sample > 0):
                    print(yellow(f"Warning: {param} range [{min_sample:.1f}, {max_sample:.1f}] may be invalid for bear threshold"))
                    
            except Exception as e:
                print(yellow(f"Warning: Could not validate threshold parameter '{param}': {e}"))
    
    def _validate_ga_parameters(self, population_size: int, generations: int, 
                               mutation_rate: float, crossover_rate: float, elitism_rate: float) -> None:
        """Validate genetic algorithm configuration parameters"""
        # Population size validation
        if not isinstance(population_size, int) or population_size < 2:
            raise ValueError(f"Population size must be an integer >= 2, got: {population_size}")
        
        if population_size > 1000:
            print(yellow(f"Warning: Large population size ({population_size}) may be slow"))
        
        # Generations validation
        if not isinstance(generations, int) or generations < 1:
            raise ValueError(f"Generations must be an integer >= 1, got: {generations}")
        
        if generations > 200:
            print(yellow(f"Warning: Large number of generations ({generations}) may be slow"))
        
        # Rate validations
        for rate_name, rate_value in [
            ('mutation_rate', mutation_rate),
            ('crossover_rate', crossover_rate), 
            ('elitism_rate', elitism_rate)
        ]:
            if not isinstance(rate_value, (int, float)):
                raise ValueError(f"{rate_name} must be numeric, got: {rate_value}")
            
            if np.isnan(rate_value) or np.isinf(rate_value):
                raise ValueError(f"{rate_name} cannot be NaN or infinite: {rate_value}")
            
            if rate_value < 0 or rate_value > 1:
                raise ValueError(f"{rate_name} must be between 0 and 1, got: {rate_value}")
        
        # Logical validations
        if elitism_rate > 0.5:
            print(yellow(f"Warning: High elitism rate ({elitism_rate}) may reduce exploration"))
        
        if mutation_rate + crossover_rate > 1.2:
            print(yellow(f"Warning: High mutation + crossover rates ({mutation_rate + crossover_rate:.2f}) may cause instability"))
        
        if mutation_rate < 0.01 and crossover_rate < 0.01:
            print(yellow(f"Warning: Very low mutation and crossover rates may cause premature convergence"))
    
    def _load_fitness_history(self):
        """Load historical fitness data from disk for adaptive penalties"""
        def init_empty_history():
            """Initialize empty history structure"""
            self.fitness_history = []
            self.penalty_history = {
                'trade_quality': [],
                'drawdown': [],
                'parameter_extreme': []
            }
        
        # Use lock to prevent race conditions during file I/O
        with self._fitness_history_lock:
            try:
                if not self.fitness_history_file.exists():
                    init_empty_history()
                    return
                    
                # Check if file is empty
                if self.fitness_history_file.stat().st_size == 0:
                    print(yellow(f"Warning: Empty fitness history file found. Starting fresh."))
                    init_empty_history()
                    return
                    
                with open(self.fitness_history_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        print(yellow(f"Warning: Fitness history file is empty. Starting fresh."))
                        init_empty_history()
                        return
                        
                    data = json.loads(content)
                    
                    # Validate data structure
                    if not isinstance(data, dict):
                        raise ValueError("Invalid data format: expected dict")
                        
                    self.fitness_history = data.get('fitness_scores', [])
                    if not isinstance(self.fitness_history, list):
                        self.fitness_history = []
                        
                    penalty_data = data.get('penalty_history', {})
                    if not isinstance(penalty_data, dict):
                        penalty_data = {}
                        
                    self.penalty_history = {
                        'trade_quality': penalty_data.get('trade_quality', []),
                        'drawdown': penalty_data.get('drawdown', []),
                        'parameter_extreme': penalty_data.get('parameter_extreme', [])
                    }
                    
                    # Ensure all penalty history values are lists
                    for key in self.penalty_history:
                        if not isinstance(self.penalty_history[key], list):
                            self.penalty_history[key] = []
                    
                    # Keep only recent history within adaptive window
                    for key in self.penalty_history:
                        self.penalty_history[key] = self.penalty_history[key][-self.adaptive_window:]
                    self.fitness_history = self.fitness_history[-self.adaptive_window:]
                    
            except FileNotFoundError:
                # Expected for first run - no warning needed
                init_empty_history()
            except (json.JSONDecodeError, ValueError) as e:
                print(yellow(f"Warning: Corrupted fitness history file ({e}). Starting fresh."))
                # Backup corrupted file
                try:
                    backup_name = self.fitness_history_file.with_suffix('.json.backup')
                    if backup_name.exists():
                        # Add timestamp to avoid conflicts
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_name = self.fitness_history_file.with_suffix(f'.json.backup_{timestamp}')
                    self.fitness_history_file.rename(backup_name)
                    print(f"  Corrupted file backed up as: {backup_name.name}")
                except Exception as backup_e:
                    print(yellow(f"  Warning: Could not backup corrupted file: {backup_e}"))
                init_empty_history()
            except Exception as e:
                print(yellow(f"Warning: Unexpected error loading fitness history: {e}"))
                init_empty_history()
    
    def _save_fitness_history(self):
        """Save historical fitness data to disk"""
        # Use lock to prevent race conditions during file I/O
        with self._fitness_history_lock:
            try:
                # Ensure directory exists
                self.fitness_history_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Keep only recent history to prevent file from growing too large
                recent_fitness = self.fitness_history[-self.adaptive_window:]
                recent_penalties = {}
                for key, values in self.penalty_history.items():
                    recent_penalties[key] = values[-self.adaptive_window:]
                
                data = {
                    'fitness_scores': recent_fitness,
                    'penalty_history': recent_penalties,
                    'last_updated': datetime.now().isoformat()
                }
                
                with open(self.fitness_history_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
            except PermissionError as e:
                print(red(f"Warning: Permission denied saving fitness history to {self.fitness_history_file}: {e}"))
            except OSError as e:
                print(f"Warning: Disk error saving fitness history: {e}")
            except Exception as e:
                print(f"Warning: Unexpected error saving fitness history: {e}")
    
    def _calculate_adaptive_penalty_scale(self, penalty_type: str, raw_penalty: float) -> float:
        """Calculate adaptive penalty scaling based on historical distribution"""
        try:
            if not self.penalty_history[penalty_type] or len(self.penalty_history[penalty_type]) < 10:
                # Not enough history, use raw penalty
                # Record this penalty for future learning
                self.penalty_history[penalty_type].append(raw_penalty)
                # Trim to adaptive window to prevent unbounded growth
                self.penalty_history[penalty_type] = self.penalty_history[penalty_type][-self.adaptive_window:]
                return raw_penalty
            
            # Get historical penalties for this type
            historical = np.array(self.penalty_history[penalty_type])
            
            if len(historical) == 0:
                self.penalty_history[penalty_type].append(raw_penalty)
                # Trim to adaptive window to prevent unbounded growth
                self.penalty_history[penalty_type] = self.penalty_history[penalty_type][-self.adaptive_window:]
                return raw_penalty
            
            # Calculate percentiles for adaptive scaling
            try:
                p25 = np.percentile(historical, 25)
                p50 = np.percentile(historical, 50)  # median
                p75 = np.percentile(historical, 75)
                p90 = np.percentile(historical, 90)
            except Exception as e:
                print(f"Warning: Error calculating percentiles for {penalty_type} penalties: {e}")
                self.penalty_history[penalty_type].append(raw_penalty)
                # Trim to adaptive window to prevent unbounded growth
                self.penalty_history[penalty_type] = self.penalty_history[penalty_type][-self.adaptive_window:]
                return raw_penalty
            
            # Adaptive scaling based on where current penalty falls in distribution
            if raw_penalty <= p25:
                # Very good (low penalty) - reduce penalty further to encourage
                scale_factor = 0.5
            elif raw_penalty <= p50:
                # Good - slight reduction
                scale_factor = 0.75
            elif raw_penalty <= p75:
                # Average - normal penalty
                scale_factor = 1.0
            elif raw_penalty <= p90:
                # Bad - increase penalty
                scale_factor = 1.5
            else:
                # Very bad - strong penalty
                scale_factor = 2.0
            
            # Record this penalty for future adaptive scaling
            self.penalty_history[penalty_type].append(raw_penalty)
            # Trim to adaptive window to prevent unbounded growth
            self.penalty_history[penalty_type] = self.penalty_history[penalty_type][-self.adaptive_window:]
            
            return raw_penalty * scale_factor
            
        except Exception as e:
            print(f"Warning: Error in adaptive penalty calculation for {penalty_type}: {e}")
            # Fallback to raw penalty
            try:
                self.penalty_history[penalty_type].append(raw_penalty)
                # Trim to adaptive window to prevent unbounded growth
                self.penalty_history[penalty_type] = self.penalty_history[penalty_type][-self.adaptive_window:]
            except:
                pass
            return raw_penalty
    
    def _log_adaptive_system_info(self):
        """Log adaptive penalty system information for transparency"""
        try:
            
            
            # Load existing history count
            history_counts = {}
            for penalty_type, history in self.penalty_history.items():
                history_counts[penalty_type] = len(history)
            
            fitness_count = len(self.fitness_history)
            
            if fitness_count > 0 or any(count > 0 for count in history_counts.values()):
                print(f"\n{cyan('Adaptive Penalty System:'.ljust(22))} {self.symbol}/{self.timeframe}")
                print(f"  {black('Fitness history file:'.ljust(22))} {self.fitness_history_file.name}")
                print(f"  {black('Historical eval:'.ljust(22))} {fitness_count}")
                
                if any(count > 0 for count in history_counts.values()):
                    print(f"  {black('Penalty histories:'.ljust(22))}")
                    for penalty_type, count in history_counts.items():
                        if count > 0:
                            status = green(f"{count} samples") if count >= 10 else yellow(f"{count} samples (need 10+)")
                            print(f"    {penalty_type.replace('_', ' ').title():.<20} {status}")
                else:
                    print(f"  {yellow('Status: Learning mode (no penalty history yet)')}")
            
        except Exception as e:
            print(f"Warning: Could not log adaptive system info: {e}")
    
    def _log_volatility_regime_detection(self, volatility_scale: float, verbose: bool = True):
        """Log detected volatility regime for user awareness"""
        if not verbose:
            return
            
        try:
            # Determine regime and color
            if volatility_scale < 0.3:
                regime = "Very Low (Blue-chip stocks)"
                color = green
            elif volatility_scale < 0.5:
                regime = "Low (Traditional stocks)"
                color = green
            elif volatility_scale < 1.0:
                regime = "Medium (Growth stocks/volatile assets)"
                color = yellow
            elif volatility_scale < 1.5:
                regime = "High (Crypto/emerging markets)"
                color = yellow
            else:
                regime = "Very High (Extreme volatility)"
                color = red
            
            print(f"  {black('Volatility regime:'.ljust(22))} {color(regime)} ({volatility_scale:.2f})")
            
        except Exception as e:
            print(f"Warning: Could not log volatility regime: {e}")
    
    def _log_adaptive_penalty_ranges(self, verbose: bool = False):
        """Log current adaptive penalty ranges for transparency"""
        if not verbose:
            return
            
        try:
            print(f"\n{cyan('Current Adaptive Penalty Ranges:')}")
            
            for penalty_type, history in self.penalty_history.items():
                if len(history) >= 10:
                    historical = np.array(history)
                    p25 = np.percentile(historical, 25)
                    p50 = np.percentile(historical, 50)
                    p75 = np.percentile(historical, 75)
                    p90 = np.percentile(historical, 90)
                    
                    print(f"  {black(penalty_type.replace('_', ' ').title() + ':'.ljust(20))}")
                    print(f"    Excellent (≤{p25:.3f}):  {green('0.5x penalty')}")
                    print(f"    Good      (≤{p50:.3f}):  {yellow('0.75x penalty')}")
                    print(f"    Average   (≤{p75:.3f}):  {black('1.0x penalty')}")
                    print(f"    Poor      (≤{p90:.3f}):  {yellow('1.5x penalty')}")
                    print(f"    Very Poor (>{p90:.3f}):  {red('2.0x penalty')}")
                else:
                    status = yellow(f"Learning mode ({len(history)}/10 samples)")
                    print(f"  {black(penalty_type.replace('_', ' ').title() + ':'.ljust(20))} {status}")
                    
        except Exception as e:
            print(f"Warning: Could not log adaptive penalty ranges: {e}")

    def create_individual(self) -> Individual:
        """Create a random individual with enforced threshold constraints
        
        Generates parameters in two phases:
        1. Non-threshold parameters are sampled normally
        2. Threshold parameters are generated in a loop until valid ordering is achieved
           (strong_bull > weak_bull > neutral_upper > neutral_lower > weak_bear > strong_bear)
        
        Falls back to default valid thresholds after 100 attempts to prevent infinite loops.
        """
        genes = {}
        
        # First, generate non-threshold parameters
        for param, param_range in self.parameter_ranges.items():
            if 'threshold' not in param:
                if param.endswith('_int'):
                    # Integer parameters - sample and convert to int
                    value = param_range.sample()
                    genes[param] = int(value)
                else:
                    # All other parameters use the parameter range's sample method
                    genes[param] = param_range.sample()
        
        # Generate threshold parameters with constraints
        # Keep trying until we get valid thresholds
        max_attempts = 100
        for attempt in range(max_attempts):
            # Sample all thresholds
            for param, param_range in self.parameter_ranges.items():
                if 'threshold' in param:
                    genes[param] = param_range.sample()
            
            # Check constraints
            strong_bull = genes.get('strong_bull_threshold', 50.0)
            weak_bull = genes.get('weak_bull_threshold', 20.0)
            neutral_upper = genes.get('neutral_threshold_upper', 10.0)
            neutral_lower = genes.get('neutral_threshold_lower', -10.0)
            weak_bear = genes.get('weak_bear_threshold', -20.0)
            strong_bear = genes.get('strong_bear_threshold', -50.0)
            
            # Validate ordering
            if (strong_bull > weak_bull and 
                weak_bull > neutral_upper and
                neutral_upper > neutral_lower and
                neutral_lower > weak_bear and
                weak_bear > strong_bear):
                # All constraints satisfied
                break
        else:
            # If we couldn't generate valid thresholds randomly, fix them
            # Set default valid thresholds
            genes['strong_bull_threshold'] = 50.0
            genes['weak_bull_threshold'] = 20.0
            genes['neutral_threshold_upper'] = 10.0
            genes['neutral_threshold_lower'] = -10.0
            genes['weak_bear_threshold'] = -20.0
            genes['strong_bear_threshold'] = -50.0
        
        return Individual(genes=genes)
    
    def create_population(self) -> List[Individual]:
        """Create initial population with smart seeding strategy
        
        Always seeds from archive when available to maintain continuity
        """
        population = []
        
        # Strategy: Use about 20% of population for seeded/elite individuals
        elite_slots = max(1, int(self.population_size * 0.2))
        
        # Always seed from archive if available
        if self.candidate_archive:
            # 1. Add individuals from archive
            archive_to_add = min(len(self.candidate_archive), elite_slots)
            for i in range(archive_to_add):
                archived = Individual(self.candidate_archive[i].copy())
                population.append(archived)
            
            # 2. Add mutated versions of best archived individuals for diversity
            if self.candidate_archive:
                best_archived = self.candidate_archive[0]  # Archive is sorted by fitness
                mutations_to_add = min(elite_slots // 2, 3)  # Add up to 3 mutated versions
                
                for i in range(mutations_to_add):
                    mutated = Individual(best_archived.copy())
                    # Apply light mutation (50% chance per parameter, small changes)
                    for param in mutated.genes:
                        if random.random() < 0.5:
                            param_range = self.parameter_ranges[param]
                            current_val = mutated.genes[param]
                            # Small mutation: +/- 10% of parameter range
                            mutation_range = param_range.get_range() * 0.1
                            mutation = random.uniform(-mutation_range, mutation_range)
                            new_val = param_range.clip(current_val + mutation)
                            mutated.genes[param] = new_val
                    population.append(mutated)
        
        # 3. Fill remaining population with random individuals for exploration
        remaining_size = self.population_size - len(population)
        population.extend([self.create_individual() for _ in range(remaining_size)])
        
        return population
    
    def calculate_population_diversity(self, population: List[Individual]) -> float:
        """Calculate diversity of population based on parameter variance"""
        if len(population) < 2:
            return 0.0
        
        # Calculate variance for each parameter
        variances = []
        for param in self.parameter_ranges.keys():
            # Use .get() for defensive coding
            values = [ind.genes.get(param, 0.0) for ind in population if param in ind.genes]
            if len(values) < 2:
                continue
                
            min_val, max_val = min(values), max(values)
            # Fix divide-by-zero: if all values are identical, skip this parameter
            if max_val == min_val:
                continue
                
            # Normalize values to [0, 1] range
            normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
            variance = np.var(normalized_values)
            variances.append(variance)
        
        return float(np.mean(variances)) if variances else 0.0
    
    def adapt_rates(self, population: List[Individual], generation: int):
        """Adapt mutation and crossover rates based on diversity and convergence"""
        diversity = self.calculate_population_diversity(population)
        
        # Check if fitness is improving
        # Edge case: Check for empty population
        if not population:
            print("Warning: Empty population in adapt_rates, skipping adaptation")
            return
        
        # Edge case: Validate base rates are reasonable
        if self.base_mutation_rate <= 0 or self.base_mutation_rate > 1:
            print(f"Warning: Invalid base_mutation_rate {self.base_mutation_rate}, resetting to 0.1")
            self.base_mutation_rate = 0.1
        
        if self.base_crossover_rate <= 0 or self.base_crossover_rate > 1:
            print(f"Warning: Invalid base_crossover_rate {self.base_crossover_rate}, resetting to 0.8")  
            self.base_crossover_rate = 0.8
        
        if self.best_fitness_history:
            # Edge case: Filter out individuals with invalid fitness values
            valid_fitness_population = []
            for ind in population:
                if hasattr(ind, 'fitness') and not (np.isnan(ind.fitness) or np.isinf(ind.fitness)):
                    valid_fitness_population.append(ind)
            
            if not valid_fitness_population:
                print("Warning: No individuals with valid fitness values found")
                return
            
            current_best = max(ind.fitness for ind in valid_fitness_population)
            
            # Edge case: Validate current_best is reasonable
            if np.isnan(current_best) or np.isinf(current_best):
                print(f"Warning: Invalid current_best fitness {current_best}, skipping improvement tracking")
            elif len(self.best_fitness_history) >= 3:
                # Edge case: Validate historical fitness values
                last_fitness = self.best_fitness_history[-3]
                if np.isnan(last_fitness) or np.isinf(last_fitness):
                    print(f"Warning: Invalid historical fitness {last_fitness}, skipping improvement calculation")
                else:
                    recent_improvement = current_best - last_fitness
                    
                    # Edge case: Handle very small differences that could be numerical noise
                    improvement_threshold = max(0.001, abs(current_best) * 1e-6)
                    if recent_improvement < improvement_threshold:  # No significant improvement
                        self.generations_without_improvement += 1
                    else:
                        self.generations_without_improvement = 0
        
        # Edge case: Ensure generations_without_improvement doesn't grow unbounded
        self.generations_without_improvement = min(self.generations_without_improvement, 50)
        
        # Adjust mutation rate based on diversity and stagnation
        if diversity < 0.1:  # Low diversity
            self.mutation_rate = min(0.5, self.base_mutation_rate * 2.0)
        elif self.generations_without_improvement > 5:  # Stagnation
            self.mutation_rate = min(0.4, self.base_mutation_rate * 1.5)
        else:
            self.mutation_rate = self.base_mutation_rate
        
        # Edge case: Ensure mutation rate stays within valid bounds
        self.mutation_rate = max(0.001, min(1.0, self.mutation_rate))
        
        # Adjust crossover rate inversely to mutation
        if self.mutation_rate > self.base_mutation_rate:
            self.crossover_rate = max(0.5, self.base_crossover_rate * 0.8)
        else:
            self.crossover_rate = self.base_crossover_rate
        
        # Edge case: Ensure crossover rate stays within valid bounds
        self.crossover_rate = max(0.001, min(1.0, self.crossover_rate))
        
        # Adaptive info removed for cleaner output
    
    def age_based_replacement(self, population: List[Individual], generation: int) -> List[Individual]:
        """Replace old elites to prevent premature convergence"""
        # Update ages
        for ind in population:
            ind_id = id(ind)
            if ind_id in self.individual_ages:
                self.individual_ages[ind_id] += 1
            else:
                self.individual_ages[ind_id] = 0
        
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Check elite ages and replace if too old
        new_population = []
        elites_replaced = 0
        
        for i, ind in enumerate(sorted_pop):
            ind_id = id(ind)
            age = self.individual_ages.get(ind_id, 0)
            
            # Replace old elites (except the very best)
            if i > 0 and i < self.elite_size and age > self.max_elite_age:
                # Create a new random individual to replace the old elite
                new_ind = self.create_individual()
                new_population.append(new_ind)
                elites_replaced += 1
                # Remove age tracking for replaced individual
                if ind_id in self.individual_ages:
                    del self.individual_ages[ind_id]
            else:
                new_population.append(ind)
        
        # Elite replacement info removed for cleaner output
        
        # Clean up age tracking for individuals not in population (memory leak prevention)
        current_ids = {id(ind) for ind in new_population}
        self.individual_ages = {k: v for k, v in self.individual_ages.items() if k in current_ids}
        
        # Edge case: Prevent unbounded growth of individual_ages dictionary
        if len(self.individual_ages) > self.max_individual_ages:
            print(f"Warning: individual_ages dictionary too large ({len(self.individual_ages)}), trimming oldest entries")
            # Keep only the most recent entries by sorting by age and keeping youngest
            sorted_ages = sorted(self.individual_ages.items(), key=lambda x: x[1])
            self.individual_ages = dict(sorted_ages[:self.max_individual_ages//2])  # Keep half
        
        return new_population
    
    def augment_data(self, df: pd.DataFrame, augmentation_type: str = "noise") -> pd.DataFrame:
        """Apply data augmentation for robustness
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original price data
        augmentation_type : str
            Type of augmentation: 'noise', 'resample', 'shift'
        """
        if augmentation_type == "noise":
            # Add small noise to OHLCV data (1% of price)
            noise_factor = 0.01
            augmented = df.copy()
            
            for col in ['open', 'high', 'low', 'close']:
                if col in augmented.columns:
                    noise = np.random.normal(0, augmented[col].std() * noise_factor, len(augmented))
                    augmented[col] = augmented[col] + noise
            
            # Ensure high >= low and OHLC consistency
            augmented['high'] = augmented[['open', 'high', 'close']].max(axis=1)
            augmented['low'] = augmented[['open', 'low', 'close']].min(axis=1)
            
        elif augmentation_type == "resample":
            # Randomly resample with replacement (bootstrap)
            n_samples = int(len(df) * 0.95)  # Use 95% of data
            indices = np.random.choice(len(df), n_samples, replace=True)
            augmented = df.iloc[sorted(indices)].copy()
            
        elif augmentation_type == "shift":
            # Time shift returns slightly
            augmented = df.copy()
            returns = augmented['close'].pct_change()
            shift_factor = np.random.uniform(-0.001, 0.001)  # Small shift
            shifted_returns = returns + shift_factor
            
            # Reconstruct prices
            augmented['close'] = augmented['close'].iloc[0] * (1 + shifted_returns).cumprod()
            for col in ['open', 'high', 'low']:
                if col in augmented.columns:
                    # Adjust other prices proportionally
                    ratio = augmented[col] / df['close']
                    augmented[col] = ratio * augmented['close']
        else:
            augmented = df.copy()
        
        return augmented
    
    def adjust_population_size(self, population: List[Individual], diversity: float) -> int:
        """Dynamically adjust population size based on diversity"""
        current_size = len(population)
        
        if diversity < 0.15:  # Very low diversity
            # Increase population to encourage exploration
            new_size = min(current_size + 10, self.max_population_size)
            # if new_size > current_size:
            #     print(f"  {cyan(f'↑ Low div ({diversity:.2f}), pop: {current_size} → {new_size}')}")
        elif diversity > 0.45 and current_size > self.min_population_size:  # High diversity
            # Can reduce population for efficiency
            new_size = max(current_size - 5, self.min_population_size)
            # if new_size < current_size:
            #     print(f"  {cyan(f'↓ High div ({diversity:.2f}), pop: {current_size} → {new_size}')}")
        else:
            new_size = current_size
        
        return new_size
    
    def create_ensemble_parameters(self, top_individuals: List[Individual], n_ensemble: int = 3) -> Dict[str, float]:
        """Create ensemble parameters from top N individuals
        
        Uses weighted average based on fitness scores
        """
        # Get top N individuals
        sorted_individuals = sorted(top_individuals, key=lambda x: x.fitness, reverse=True)
        ensemble_members = sorted_individuals[:n_ensemble]
        
        if not ensemble_members:
            return {}
        
        # Calculate weights based on fitness (normalized)
        min_fitness = min(ind.fitness for ind in ensemble_members)
        fitness_range = max(ind.fitness for ind in ensemble_members) - min_fitness
        
        if fitness_range > 0:
            weights = [(ind.fitness - min_fitness) / fitness_range for ind in ensemble_members]
        else:
            weights = [1.0 / len(ensemble_members)] * len(ensemble_members)
        
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Create ensemble parameters
        ensemble_params = {}
        for param in self.parameter_ranges.keys():
            # Check if all individuals have this parameter
            if all(param in ind.genes for ind in ensemble_members):
                if param.endswith('_int'):
                    # For integer parameters, use weighted average then round
                    weighted_sum = sum(ind.genes[param] * w for ind, w in zip(ensemble_members, weights))
                    ensemble_params[param] = int(round(weighted_sum))
                else:
                    # For continuous parameters, use weighted average
                    ensemble_params[param] = sum(ind.genes[param] * w for ind, w in zip(ensemble_members, weights))
            else:
                # If not all individuals have this parameter, use default value from parameter range
                param_range = self.parameter_ranges[param]
                if hasattr(param_range, 'default_value'):
                    ensemble_params[param] = param_range.default_value
                elif hasattr(param_range, 'min_value') and hasattr(param_range, 'max_value'):
                    # Use midpoint as default
                    ensemble_params[param] = (param_range.min_value + param_range.max_value) / 2.0
                else:
                    # Fallback to a reasonable default based on parameter name
                    if 'weight' in param:
                        ensemble_params[param] = 0.33
                    elif 'threshold' in param:
                        ensemble_params[param] = 0.0
                    elif 'lookback' in param:
                        ensemble_params[param] = 20
                    else:
                        ensemble_params[param] = 1.0
        
        # Silently create ensemble - details logged elsewhere if needed
        
        return ensemble_params
    
    def evaluate_fitness(self, individual: Individual, 
                        train_data: pd.DataFrame, 
                        volatility_scale: Optional[float] = None) -> float:
        """
        Evaluate fitness of an individual with volatility-aware adjustments
        
        Parameters:
        -----------
        individual : Individual
            Individual to evaluate
        train_data : pd.DataFrame
            Training data with OHLCV
        volatility_scale : Optional[float]
            Pre-calculated volatility scale, or None to auto-detect
        
        Returns:
        --------
        float
            Fitness score
        """
        genes = individual.genes
        
        # Auto-detect volatility scale if not provided
        if volatility_scale is None:
            volatility_scale = estimate_asset_volatility_scale(train_data, sample_size=min(500, len(train_data)))
        
        # Extract parameters
        weights = {
            'trend': genes.get('weight_trend', 0.33),
            'volatility': genes.get('weight_volatility', 0.33),
            'exhaustion': genes.get('weight_exhaustion', 0.34)
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        lookback = int(genes.get('lookback_int', 20))
        
        try:
            # Calculate indicators
            factors_df, regimes = calculate_mss(train_data, lookback, weights)
            
            # Add MACD and RSI
            macd_data = calculate_macd(train_data)
            factors_df['macd_hist'] = macd_data['histogram']
            factors_df['rsi'] = calculate_rsi(train_data)
            
            # Merge with main dataframe
            combined_df = pd.concat([train_data, factors_df], axis=1)
            
            # Run backtest with enhanced strategy if parameters are available
            initial_capital = 100000
            
            # Use enhanced strategy
            strategy = EnhancedTradingStrategy(
                initial_capital=initial_capital,
                max_position_fraction=genes.get('max_position_pct', 1.0),
                entry_step_size=genes.get('entry_step_size', 0.2),
                stop_loss_multiplier_strong=genes.get('stop_loss_multiplier_strong', 2.0),
                stop_loss_multiplier_weak=genes.get('stop_loss_multiplier_weak', 1.0),
                strong_bull_threshold=genes.get('strong_bull_threshold', 50.0),
                weak_bull_threshold=genes.get('weak_bull_threshold', 20.0),
                neutral_upper=genes.get('neutral_threshold_upper', 20.0),
                neutral_lower=genes.get('neutral_threshold_lower', -20.0),
                weak_bear_threshold=genes.get('weak_bear_threshold', -20.0),
                strong_bear_threshold=genes.get('strong_bear_threshold', -50.0),
                allow_shorts=self.allow_shorts,
            )
            
            # Enhanced error handling for malformed strategy results
            try:
                results = strategy.run_backtest(combined_df, combined_df)
                
                # Validate strategy results structure
                if results is None:
                    print("Warning: Strategy returned None results")
                    return -1000
                    
                if not isinstance(results, pd.DataFrame):
                    print(f"Warning: Strategy results not a DataFrame: {type(results)}")
                    return -1000
                    
                if len(results) < 2:
                    print("Warning: Strategy results too short (< 2 rows)")
                    return -1000
                    
                # Check for empty or all-NaN results
                if results.empty or results.isna().all().all():
                    print("Warning: Strategy results empty or all NaN")
                    return -1000
                    
                # Validate critical columns exist and have reasonable data
                required_cols = ['portfolio_value']
                for col in required_cols:
                    if col not in results.columns:
                        print(f"Warning: Missing required column '{col}' in strategy results")
                        return -1000
                    
                    # Check for all-NaN column
                    if results[col].isna().all():
                        print(f"Warning: Column '{col}' contains only NaN values")
                        return -1000
                        
                    # Check for non-numeric data
                    try:
                        numeric_col = pd.to_numeric(results[col], errors='coerce')
                        if numeric_col.isna().all():
                            print(f"Warning: Column '{col}' contains no valid numeric data")
                            return -1000
                    except Exception as e:
                        print(f"Warning: Error validating column '{col}': {e}")
                        return -1000
                
                # Validate index integrity
                if not isinstance(results.index, (pd.DatetimeIndex, pd.RangeIndex)):
                    try:
                        # Try to convert index to datetime if possible
                        results.index = pd.to_datetime(results.index)
                    except Exception:
                        print("Warning: Invalid index format in strategy results")
                        return -1000
                
                # Check for duplicate index values which can cause calculation issues
                if results.index.duplicated().any():
                    print("Warning: Duplicate index values in strategy results")
                    # Keep first occurrence of duplicates
                    results = results.loc[~results.index.duplicated(keep='first')]
                    if len(results) < 2:
                        return -1000
                
            except Exception as e:
                print(f"Warning: Strategy execution failed: {e}")
                return -1000
            
            # Get portfolio values (already validated above)
            portfolio_values = results['portfolio_value']
            
            # Calculate returns  
            returns = portfolio_values.pct_change().dropna()
            
            # Initialize common variables for debug output
            total_return = (portfolio_values.iloc[-1] / initial_capital) - 1
            n_periods = len(returns)
            
            # Calculate actual time span using timestamps
            try:
                # Get time span safely - portfolio_values should have DatetimeIndex
                # Type checker can't infer this, but it's safe within try block
                time_diff = portfolio_values.index[-1] - portfolio_values.index[0]  # type: ignore
                if hasattr(time_diff, 'days'):
                    time_span_days = time_diff.days
                else:
                    # For pandas Timedelta
                    time_span_days = time_diff.total_seconds() / 86400
                
                years_elapsed = max(time_span_days / 365.25, 1/365.25)  # At least 1 day
                annualized_return = (1 + total_return) ** (1 / years_elapsed) - 1
                periods_per_year = n_periods / years_elapsed  # Actual frequency
            except (AttributeError, TypeError):
                # Fallback if timestamp calculation fails
                periods_per_year = min(8760, n_periods * 4)  # Conservative estimate
                annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1 if n_periods > 1 else 0
                time_span_days = n_periods  # Fallback for debug output
            
            # Calculate base fitness metric using IMPROVED APPROACH
            # Always use ensemble fitness (legacy sortino calculation for base_fitness component)
            if n_periods < 2:
                return -1000  # Not enough data
            
            # Calculate downside deviation
            mar = 0  # Risk-free rate
            downside_returns = returns[returns < mar]
            
            # Volatility-aware fitness adjustments
            # Low-vol assets need different expectations
            if volatility_scale < 0.3:  # Very stable assets (blue-chips)
                fitness_multiplier = 1.5  # Reward consistency more
                sharpe_weight = 0.1  # Focus on Sortino for trend following
            elif volatility_scale < 0.5:  # Traditional stocks
                fitness_multiplier = 1.2  # Slight boost for stocks
                sharpe_weight = 0.05  # Minimal Sharpe weight - Sortino is better for our strategy
            else:  # High volatility (crypto, etc)
                fitness_multiplier = 1.0  # Baseline
                sharpe_weight = 0.0  # Pure Sortino for high volatility assets
            
            if len(downside_returns) == 0 or total_return <= 0:
                # No downside or negative total return
                base_fitness = max(-10.0, total_return * 10 * fitness_multiplier)
            else:
                downside_deviation = downside_returns.std()
                if downside_deviation == 0 or np.isnan(downside_deviation):
                    # Perfect strategy - high but finite score
                    base_fitness = min(10.0, annualized_return * 20 * fitness_multiplier)
                else:
                    # Standard Sortino calculation using actual time-based annualization
                    annualized_downside_dev = downside_deviation * np.sqrt(periods_per_year)
                    sortino = annualized_return / annualized_downside_dev
                    
                    # Also calculate Sharpe for stable assets
                    returns_vol = returns.std() * np.sqrt(periods_per_year)
                    if returns_vol > 0:
                        sharpe = annualized_return / returns_vol
                    else:
                        sharpe = 0.0
                    
                    # Cap Sharpe contribution to avoid negative drag
                    # If Sharpe is negative but Sortino is positive, limit the damage
                    if sharpe < 0 and sortino > 0:
                        sharpe_contribution = max(sharpe * sharpe_weight, -0.5)  # Cap negative contribution
                    else:
                        sharpe_contribution = sharpe * sharpe_weight
                    
                    # Blend Sortino and Sharpe based on asset type
                    blended_ratio = sortino * (1 - sharpe_weight) + sharpe_contribution
                    
                    # Apply volatility adjustment for more realistic fitness
                    base_fitness = max(-10.0, min(10.0, blended_ratio * fitness_multiplier))
            
            # Handle edge cases
            if np.isnan(base_fitness) or np.isinf(base_fitness):
                base_fitness = -10.0
            
            # Calculate maximum drawdown PERCENTAGE (OLD CODE APPROACH)
            equity_curve = portfolio_values
            peak = equity_curve.expanding().max()
            drawdown_dollars = equity_curve - peak
            max_drawdown_dollars = drawdown_dollars.min()
            
            if max_drawdown_dollars < 0:
                # Find the index where max drawdown occurred
                idx_max_drawdown = drawdown_dollars.idxmin()
                # Find the peak before this drawdown
                peak_before_drawdown = equity_curve.loc[:idx_max_drawdown].max()
                if peak_before_drawdown > 0:
                    max_drawdown_pct = abs(max_drawdown_dollars) / peak_before_drawdown * 100
                else:
                    max_drawdown_pct = 100.0
            else:
                max_drawdown_pct = 0.0
            
            # Ensure valid percentage
            if np.isnan(max_drawdown_pct) or np.isinf(max_drawdown_pct):
                max_drawdown_pct = 100.0
            
            # Store base_fitness for use in multi-objective calculation
            # (base_fitness contains Sortino/Sharpe blend from above)
            
            # Debug output (remove later)
            if False:  # Set to True for debugging
                print(f"  Base fitness (Sortino): {base_fitness:.4f}")
                print(f"  Max drawdown %: {max_drawdown_pct:.2f}%")
                print(f"  Combined fitness: {fitness:.4f}")
                print(f"  Total return: {total_return:.4f}")
                print(f"  Annualized return: {annualized_return:.4f}")
                print(f"  Time span: {time_span_days} days, Periods: {n_periods}, Freq: {periods_per_year:.1f}/year")
            
            # Calculate additional metrics for multi-objective optimization
            
            # Win Rate and Profit Factor (combined for efficiency)
            win_rate = 0.0
            profit_factor = 1.0
            if 'returns' in results.columns:
                trade_returns = pd.to_numeric(results['returns'].diff(), errors='coerce')

                # Win Rate
                winning_periods = (trade_returns > 0).sum()
                total_periods = (trade_returns != 0).sum()
                if total_periods > 0:
                    win_rate = winning_periods / total_periods

                # Profit Factor
                gross_profits = trade_returns[trade_returns > 0].sum()
                gross_losses = abs(trade_returns[trade_returns < 0].sum())
                if gross_losses > 0:
                    profit_factor = gross_profits / gross_losses
                elif gross_profits > 0:
                    profit_factor = 10.0  # Cap at 10 if no losses
            
            # 3. Trade Quality Metrics
            position_changes = 0
            if 'position' in results.columns:
                positions = results['position']
                position_changes = (positions.diff() != 0).sum()
            
            # Calculate trades per period for quality assessment
            trades_per_period = position_changes / max(1, n_periods)
            
            # 4. Additional Advanced Metrics for Trading GA Robustness
            
            # Omega Ratio (probability-weighted ratio of gains vs losses)
            omega_ratio = 1.0
            if len(returns) > 0:
                threshold = 0.0  # Use 0% as threshold for gains/losses
                gains = returns[returns > threshold] - threshold
                losses = threshold - returns[returns < threshold]
                if losses.sum() > 0:
                    omega_ratio = gains.sum() / losses.sum()
                elif gains.sum() > 0:
                    omega_ratio = 10.0  # Cap at 10 if no losses
            
            # Sterling Ratio (return/max drawdown, similar to Calmar but different calculation)
            sterling_ratio = 0.0
            if max_drawdown_pct > 0:
                # Use average drawdown instead of max for Sterling
                equity_curve = portfolio_values
                peak = equity_curve.expanding().max()
                drawdowns = (equity_curve - peak) / peak
                avg_drawdown_pct = abs(drawdowns[drawdowns < 0].mean()) * 100 if len(drawdowns[drawdowns < 0]) > 0 else 0.1
                if avg_drawdown_pct > 0:
                    sterling_ratio = annualized_return / avg_drawdown_pct
            
            # Ulcer Index (downside volatility measure)
            ulcer_index = 0.0
            if len(portfolio_values) > 1:
                equity_curve = portfolio_values
                peak = equity_curve.expanding().max()
                drawdowns = (equity_curve - peak) / peak * 100  # Percentage drawdowns
                ulcer_index = np.sqrt((drawdowns ** 2).mean())
            
            # Risk-adjusted return consistency (lower is better)
            return_consistency = returns.std() if len(returns) > 1 else 1.0
            
            # Multi-Objective Ensemble Fitness (always used)
            # Enhanced ensemble weights including advanced trading metrics
            if volatility_scale < 0.5:  # Stocks
                sortino_weight = 0.25      # Core risk-adjusted return
                sharpe_weight = 0.10       # Small weight for consistency
                calmar_weight = 0.15       # Drawdown-adjusted return
                sterling_weight = 0.10     # Average drawdown adjusted
                omega_weight = 0.15        # Probability-weighted gains/losses
                win_rate_weight = 0.10     # Trade success rate
                profit_factor_weight = 0.10  # Gross profit ratio
                ulcer_weight = 0.05        # Downside volatility penalty
            else:  # Crypto/High volatility
                sortino_weight = 0.30      # Higher weight for downside protection
                sharpe_weight = 0.05       # Minimal but non-zero
                calmar_weight = 0.20       # Important for volatile assets
                sterling_weight = 0.15     # Average drawdown matters more
                omega_weight = 0.15        # Tail risk management
                win_rate_weight = 0.05     # Less important for crypto
                profit_factor_weight = 0.05  # Less important for crypto
                ulcer_weight = 0.05        # Downside volatility penalty
            
            # Calculate Calmar ratio
            calmar_ratio = 0.0
            if max_drawdown_pct > 0:
                calmar_ratio = annualized_return / (max_drawdown_pct / 100)
            
            # Normalize metrics to similar scales [-1, 1]
            normalized_sortino = np.clip(base_fitness / 10, -1, 1)  # base_fitness is already Sortino-based
            normalized_sharpe = np.clip(sharpe / 3, -1, 1) if 'sharpe' in locals() else 0
            normalized_calmar = np.clip(calmar_ratio / 3, -1, 1)
            normalized_sterling = np.clip(sterling_ratio / 3, -1, 1)
            normalized_omega = np.clip((omega_ratio - 1) / 4, -1, 1)  # Omega of 5 = max score
            normalized_win_rate = (win_rate - 0.5) * 2  # Map [0,1] to [-1,1]
            normalized_profit_factor = np.clip((profit_factor - 1) / 4, -1, 1)  # PF of 5 = max score
            # Ulcer index penalty (lower is better, so negate)
            normalized_ulcer = -np.clip(ulcer_index / 20, 0, 1)  # UI of 20% = max penalty
            
            # Enhanced ensemble fitness calculation
            ensemble_fitness = (
                sortino_weight * normalized_sortino +
                sharpe_weight * normalized_sharpe +
                calmar_weight * normalized_calmar +
                sterling_weight * normalized_sterling +
                omega_weight * normalized_omega +
                win_rate_weight * normalized_win_rate +
                profit_factor_weight * normalized_profit_factor +
                ulcer_weight * normalized_ulcer
            ) * 10  # Scale back up
            
            # Trade Quality Penalties
            # Penalize both over-trading and under-trading
            
            # Define optimal trading frequency based on volatility
            if volatility_scale < 0.3:  # Blue-chip stocks
                optimal_trades_per_period = 1 / 300  # 1 trade per 300 periods
                min_trades_per_period = 1 / 500
                max_trades_per_period = 1 / 100
            elif volatility_scale < 0.5:  # Traditional stocks
                optimal_trades_per_period = 1 / 250
                min_trades_per_period = 1 / 400
                max_trades_per_period = 1 / 80
            elif volatility_scale < 1.0:  # Volatile stocks/crypto
                optimal_trades_per_period = 1 / 200
                min_trades_per_period = 1 / 350
                max_trades_per_period = 1 / 60
            else:  # Very high volatility
                optimal_trades_per_period = 1 / 150
                min_trades_per_period = 1 / 250
                max_trades_per_period = 1 / 40
            
            # Apply trading frequency penalty with adaptive scaling
            raw_trade_penalty = 0.0
            
            if trades_per_period < min_trades_per_period:
                # Under-trading penalty (granular based on severity)
                ratio = trades_per_period / min_trades_per_period
                # More granular penalty with parameter-based variation
                deficit = 1 - ratio
                # Add parameter-based variation to avoid constant penalties
                # Use defensive access for weights and parameters
                trend_variation = abs(weights.get('trend', 0.33) - 0.5) if 'weight_trend' in genes else 0.0
                lookback_variation = abs(lookback - 50) / 100 if 'lookback_int' in genes else 0.0
                param_variation = trend_variation + lookback_variation
                raw_trade_penalty = 1.0 + (deficit ** 1.2) * 1.5 + param_variation * 0.2  # Range: 1.0 to 2.9
            elif trades_per_period > max_trades_per_period:
                # Over-trading penalty (potential overfitting) 
                ratio = trades_per_period / max_trades_per_period
                # Exponential penalty for severe overtrading
                excess = ratio - 1
                raw_trade_penalty = 1.5 + (excess ** 1.1) * 2.5  # Range: 1.5 to 4.0+
            else:
                # Small penalty for deviation from optimal
                deviation = abs(trades_per_period - optimal_trades_per_period) / (optimal_trades_per_period + 1e-6)
                raw_trade_penalty = 0.2 + deviation * 0.8  # Range: 0.2 to 1.0
            
            # Apply adaptive scaling to trade quality penalty
            adaptive_trade_penalty = self._calculate_adaptive_penalty_scale('trade_quality', raw_trade_penalty)
            
            # Apply trade quality penalty
            fitness = ensemble_fitness - adaptive_trade_penalty
            
            # Additional penalties for extreme parameters with adaptive scaling
            raw_param_penalty = 0.0
            
            # Granular trend weight penalty (gets worse as it approaches extremes)
            # Only apply penalty if trend weight parameter exists
            if 'weight_trend' in genes:
                trend_weight = weights['trend']
                if trend_weight < 0.05:
                    raw_param_penalty += (0.05 - trend_weight) / 0.05 * 2.0  # 0-2.0 based on severity
                elif trend_weight > 0.95:
                    raw_param_penalty += (trend_weight - 0.95) / 0.05 * 2.0  # 0-2.0 based on severity
            
            # Granular lookback penalty (exponential for extreme values)
            # Only apply penalty if lookback parameter exists
            if 'lookback_int' in genes:
                if lookback < 3:
                    raw_param_penalty += ((3 - lookback) / 3) ** 1.5 * 1.5
                elif lookback > 100:
                    raw_param_penalty += ((lookback - 100) / 100) ** 1.2 * 1.5
                
            # Apply adaptive scaling to parameter penalty
            if raw_param_penalty > 0:
                adaptive_param_penalty = self._calculate_adaptive_penalty_scale('parameter_extreme', raw_param_penalty)
                fitness -= adaptive_param_penalty
            
            # Penalty for inverted thresholds (critical for proper strategy operation)
            # Bull thresholds: strong should be > weak
            strong_bull = genes.get('strong_bull_threshold', 50.0)
            weak_bull = genes.get('weak_bull_threshold', 20.0)
            if strong_bull <= weak_bull:
                fitness -= 5.0  # Heavy penalty
                if False:  # Debug
                    print(f"  Inverted bull thresholds: strong={strong_bull:.1f} <= weak={weak_bull:.1f}")
            
            # Bear thresholds: strong should be < weak (more negative)
            strong_bear = genes.get('strong_bear_threshold', -50.0)
            weak_bear = genes.get('weak_bear_threshold', -20.0)
            if strong_bear >= weak_bear:
                fitness -= 5.0  # Heavy penalty
                if False:  # Debug
                    print(f"  Inverted bear thresholds: strong={strong_bear:.1f} >= weak={weak_bear:.1f}")
            
            # Neutral thresholds should be between weak bull and weak bear
            neutral_upper = genes.get('neutral_threshold_upper', 20.0)
            neutral_lower = genes.get('neutral_threshold_lower', -20.0)
            if neutral_upper >= weak_bull or neutral_lower <= weak_bear:
                fitness -= 2.0
            if neutral_upper <= neutral_lower:
                fitness -= 3.0
            
            # Add adaptive drawdown penalty based on max drawdown percentage
            raw_drawdown_penalty = 0.0
            if max_drawdown_pct > 5:  # Start penalizing drawdowns above 5% (more realistic threshold)
                # Exponential penalty for severe drawdowns
                excess_drawdown = max_drawdown_pct - 5
                # Add small parameter-based variation to ensure some penalty variation
                # Use defensive access for weights and parameters
                volatility_variation = abs(weights.get('volatility', 0.33) - 0.33) if 'weight_volatility' in genes else 0.0
                rebalance_variation = abs(genes.get('rebalance_frequency', 1) - 5) / 10
                param_variation = volatility_variation + rebalance_variation
                raw_drawdown_penalty = 0.5 + (excess_drawdown / 100) ** 1.2 * 2.5 + param_variation * 0.1  # Range: 0.5 to 3.1+
            
            if raw_drawdown_penalty > 0:
                adaptive_drawdown_penalty = self._calculate_adaptive_penalty_scale('drawdown', raw_drawdown_penalty)
                fitness -= adaptive_drawdown_penalty
            
            # Debug output for multi-objective fitness
            if False:  # Set to True for debugging
                print(f"  Enhanced multi-objective fitness breakdown:")
                print(f"    Sortino component: {normalized_sortino:.3f} (weight: {sortino_weight})")
                print(f"    Sharpe component: {normalized_sharpe:.3f} (weight: {sharpe_weight})")
                print(f"    Calmar component: {normalized_calmar:.3f} (weight: {calmar_weight})")
                print(f"    Sterling component: {normalized_sterling:.3f} (weight: {sterling_weight})")
                print(f"    Omega component: {normalized_omega:.3f} (weight: {omega_weight})")
                print(f"    Win rate component: {normalized_win_rate:.3f} (weight: {win_rate_weight})")
                print(f"    Profit factor component: {normalized_profit_factor:.3f} (weight: {profit_factor_weight})")
                print(f"    Ulcer penalty: {normalized_ulcer:.3f} (weight: {ulcer_weight})")
                print(f"    Ensemble fitness: {ensemble_fitness:.3f}")
                print(f"    Trade quality penalty: {adaptive_trade_penalty:.3f}")
                print(f"    Trades per period: {trades_per_period:.6f}")
                print(f"    Position changes: {position_changes}")
                print(f"    Final fitness: {fitness:.3f}")
                print(f"    Raw metrics - Omega: {omega_ratio:.2f}, Sterling: {sterling_ratio:.2f}, Ulcer: {ulcer_index:.2f}%")
                print(f"    Adaptive penalties - Trade: {adaptive_trade_penalty:.3f}, Drawdown: {adaptive_drawdown_penalty if 'adaptive_drawdown_penalty' in locals() else 0:.3f}")
            
            # Record fitness score for adaptive penalty system
            # Use lock to prevent race conditions during shared state modification
            with self._state_lock:
                self.fitness_history.append(fitness)
                
                # Periodically save fitness history (every N evaluations)
                if len(self.fitness_history) % 100 == 0:
                    self._save_fitness_history()
            
            return fitness
            
        except Exception as e:
            # Invalid parameters
            return -1000
    
    def tournament_selection(self, population: List[Individual], 
                           tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection"""
        # Ensure tournament size doesn't exceed population size
        actual_tournament_size = min(tournament_size, len(population))
        tournament = random.sample(population, actual_tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents with threshold constraint fixing
        
        After standard single-point crossover, both children are checked for 
        threshold constraint violations. Any violations are fixed by swapping 
        the values to maintain proper ordering.
        """
        # Edge case: Validate crossover rate
        if self.crossover_rate <= 0 or self.crossover_rate > 1:
            print(f"Warning: Invalid crossover_rate {self.crossover_rate}, skipping crossover")
            return parent1, parent2
            
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Edge case: Validate parents have genes
        if not hasattr(parent1, 'genes') or not hasattr(parent2, 'genes'):
            print("Warning: Parent missing genes attribute, skipping crossover")
            return parent1, parent2
            
        if not parent1.genes or not parent2.genes:
            print("Warning: Parent has empty genes, skipping crossover")
            return parent1, parent2
        
        # Single-point crossover with bounds checking
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        # Edge case: Ensure both parents have same parameter set
        keys1 = set(genes1.keys())
        keys2 = set(genes2.keys())
        
        if keys1 != keys2:
            print(f"Warning: Parents have mismatched parameter sets: {keys1.symmetric_difference(keys2)}")
            # Use intersection of keys to ensure compatibility
            common_keys = keys1.intersection(keys2)
            if not common_keys:
                print("Warning: No common parameters between parents, skipping crossover")
                return parent1, parent2
            
            # Keep only common parameters
            genes1 = {k: v for k, v in genes1.items() if k in common_keys}
            genes2 = {k: v for k, v in genes2.items() if k in common_keys}
        
        param_names = list(genes1.keys())
        
        # Edge case: Check if enough parameters for crossover
        if len(param_names) < 2:
            print("Warning: Not enough parameters for crossover")
            return parent1, parent2
            
        # Edge case: Bounds check for crossover point calculation
        try:
            crossover_point = random.randint(1, len(param_names) - 1)
        except ValueError as e:
            print(f"Warning: Invalid range for crossover point: {e}")
            return parent1, parent2
        
        # Perform crossover with parameter validation
        for i in range(crossover_point, len(param_names)):
            param = param_names[i]
            
            # Edge case: Validate parameter values before swapping
            val1 = genes2.get(param)
            val2 = genes1.get(param)
            
            if val1 is None or val2 is None:
                print(f"Warning: Missing parameter {param} in one parent, skipping swap")
                continue
                
            if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                print(f"Warning: Non-numeric parameter values for {param}, skipping swap")
                continue
                
            if np.isnan(val1) or np.isinf(val1) or np.isnan(val2) or np.isinf(val2):
                print(f"Warning: Invalid parameter values for {param}, skipping swap")
                continue
            
            # Perform the swap
            genes1[param], genes2[param] = val1, val2
        
        # Edge case: Validate parameter values are within valid ranges after crossover
        def validate_parameter_ranges(genes):
            """Ensure all parameters are within their defined ranges"""
            for param, value in genes.items():
                if param in self.parameter_ranges:
                    param_range = self.parameter_ranges[param]
                    try:
                        # Check if value is within bounds by attempting to clip/validate
                        if hasattr(param_range, 'clip'):
                            # Use range's clip method if available
                            clipped_value = param_range.clip(value)
                            if clipped_value != value:
                                # print(yellow(f"Warning: Parameter {param} value {value:.6f} clipped to {clipped_value:.6f}"))
                                genes[param] = clipped_value
                        elif hasattr(param_range, 'sample'):
                            # For other range types, sample a new value if out of bounds
                            # Check using range properties if available
                            if hasattr(param_range, 'min_val') and hasattr(param_range, 'max_val'):
                                if value < param_range.min_val or value > param_range.max_val:
                                    new_value = param_range.sample()
                                    # print(yellow(f"Warning: Parameter {param} value {value:.6f} out of bounds, resampled to {new_value:.6f}"))
                                    genes[param] = new_value
                    except Exception as e:
                        print(yellow(f"Warning: Could not validate parameter {param}: {e}"))
                        # Re-sample the parameter if validation fails
                        try:
                            new_value = param_range.sample()
                            # print(yellow(f"Warning: Resampling parameter {param} to {new_value:.6f}"))
                            genes[param] = new_value
                        except Exception as e2:
                            print(red(f"Error: Failed to resample parameter {param}: {e2}"))
        
        validate_parameter_ranges(genes1)
        validate_parameter_ranges(genes2)
        
        # Check if both children still have valid threshold constraints
        def fix_thresholds(genes):
            """Fix threshold ordering if needed"""
            strong_bull = genes.get('strong_bull_threshold', 50.0)
            weak_bull = genes.get('weak_bull_threshold', 20.0)
            neutral_upper = genes.get('neutral_threshold_upper', 10.0)
            neutral_lower = genes.get('neutral_threshold_lower', -10.0)
            weak_bear = genes.get('weak_bear_threshold', -20.0)
            strong_bear = genes.get('strong_bear_threshold', -50.0)
            
            # If constraints are violated, swap values to fix them
            if strong_bull <= weak_bull:
                genes['strong_bull_threshold'], genes['weak_bull_threshold'] = max(strong_bull, weak_bull), min(strong_bull, weak_bull)
            if weak_bull <= neutral_upper:
                genes['weak_bull_threshold'], genes['neutral_threshold_upper'] = max(weak_bull, neutral_upper), min(weak_bull, neutral_upper)
            if neutral_upper <= neutral_lower:
                genes['neutral_threshold_upper'], genes['neutral_threshold_lower'] = max(neutral_upper, neutral_lower), min(neutral_upper, neutral_lower)
            if neutral_lower <= weak_bear:
                genes['neutral_threshold_lower'], genes['weak_bear_threshold'] = max(neutral_lower, weak_bear), min(neutral_lower, weak_bear)
            if weak_bear <= strong_bear:
                genes['weak_bear_threshold'], genes['strong_bear_threshold'] = max(weak_bear, strong_bear), min(weak_bear, strong_bear)
        
        fix_thresholds(genes1)
        fix_thresholds(genes2)
        
        return Individual(genes=genes1), Individual(genes=genes2)
    
    def mutate(self, individual: Individual, mutation_rate: Optional[float] = None) -> Individual:
        """Mutate an individual while maintaining threshold constraints
        
        Mutation strategy:
        - Non-threshold parameters mutate normally based on mutation rate
        - Threshold parameters only accept mutations that preserve valid ordering
        - Tries up to 10 times to find a valid mutation for each threshold
        - Reverts to original value if no valid mutation is found
        """
        genes = individual.genes.copy()
        
        # Use provided mutation rate or default
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        
        # Mutate non-threshold parameters normally
        for param, param_range in self.parameter_ranges.items():
            if 'threshold' not in param and random.random() < mutation_rate:
                if param.endswith('_int'):
                    # Integer mutation - mutate and convert to int
                    mutated_value = param_range.mutate(genes[param])
                    genes[param] = int(mutated_value)
                else:
                    # All other parameters use the parameter range's mutate method
                    genes[param] = param_range.mutate(genes[param])
        
        # Mutate threshold parameters with constraint checking
        threshold_params = [p for p in self.parameter_ranges.keys() if 'threshold' in p]
        for param in threshold_params:
            if random.random() < mutation_rate:
                param_range = self.parameter_ranges[param]
                # Try multiple times to get valid mutation
                for _ in range(10):
                    new_value = param_range.mutate(genes[param])
                    
                    # Temporarily set the new value
                    old_value = genes[param]
                    genes[param] = new_value
                    
                    # Check if constraints are still satisfied
                    strong_bull = genes.get('strong_bull_threshold', 50.0)
                    weak_bull = genes.get('weak_bull_threshold', 20.0)
                    neutral_upper = genes.get('neutral_threshold_upper', 10.0)
                    neutral_lower = genes.get('neutral_threshold_lower', -10.0)
                    weak_bear = genes.get('weak_bear_threshold', -20.0)
                    strong_bear = genes.get('strong_bear_threshold', -50.0)
                    
                    if (strong_bull > weak_bull and 
                        weak_bull > neutral_upper and
                        neutral_upper > neutral_lower and
                        neutral_lower > weak_bear and
                        weak_bear > strong_bear):
                        # Constraints satisfied, keep the mutation
                        break
                    else:
                        # Revert the mutation
                        genes[param] = old_value
        
        return Individual(genes=genes)
    
    def evolve_population(self, population: List[Individual]) -> List[Individual]:
        """Evolve population for one generation with diversity injection"""
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep elite - ensure we don't try to select more elites than population size
        actual_elite_size = min(self.elite_size, len(population))
        if actual_elite_size < self.elite_size:
            print(f"Warning: Population size ({len(population)}) smaller than elite size ({self.elite_size}), using {actual_elite_size} elites")
        new_population = population[:actual_elite_size]
        
        # Inject diversity: Add 10% completely random individuals each generation
        diversity_count = max(1, int(len(population) * 0.1))
        for _ in range(diversity_count):
            new_population.append(self.create_individual())
        
        # Generate rest of population
        while len(new_population) < len(population):
            # Occasionally (5% chance) create a completely new random individual
            if random.random() < 0.05:
                new_population.append(self.create_individual())
                continue
                
            # Selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation with adaptive rate from adapt_rates method
            child1 = self.mutate(child1, mutation_rate=self.mutation_rate)
            child2 = self.mutate(child2, mutation_rate=self.mutation_rate)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        return new_population[:len(population)]
    
    def optimize(self, train_data: pd.DataFrame, 
                verbose: bool = True,
                use_data_augmentation: bool = True,
                use_ensemble: bool = True) -> Tuple[Individual, List[float], Optional[Dict]]:
        """
        Run genetic algorithm optimization with advanced features
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        verbose : bool
            Print progress
        use_data_augmentation : bool
            Apply data augmentation for robustness
        use_ensemble : bool
            Return ensemble parameters from top individuals
        
        Returns:
        --------
        Tuple[Individual, List[float], Optional[Dict]]
            Best individual, fitness history, and ensemble parameters (if enabled)
        """
        # Create initial population
        population = self.create_population()
        current_pop_size = len(population)
        
        # Pre-calculate volatility scale once for efficiency
        volatility_scale = estimate_asset_volatility_scale(train_data, sample_size=min(500, len(train_data)), verbose=False)
        
        # Log volatility regime detection and adaptive penalty ranges only on first run
        ##if verbose:
        ##    self._log_volatility_regime_detection(volatility_scale, verbose=True)
        ##    if not self.penalty_table_shown:
        ##        self._log_adaptive_penalty_ranges(verbose=True)
        ##        self.penalty_table_shown = True
        
        # Prepare augmented datasets if enabled
        augmented_datasets = []
        if use_data_augmentation:
            augmented_datasets = [
                self.augment_data(train_data, "noise"),
                self.augment_data(train_data, "shift")
            ]
            # Silent - augmentation enabled
        
        # Evaluate initial population
        for individual in population:
            fitness_value = self.evaluate_fitness(individual, train_data, volatility_scale)
            # Validate fitness assignment
            if np.isnan(fitness_value) or np.isinf(fitness_value):
                print(f"Warning: Invalid fitness value {fitness_value} for individual, setting to -1000")
                individual.fitness = -1000.0
            else:
                individual.fitness = float(fitness_value)
            
            # Additional evaluation on augmented data
            if use_data_augmentation and augmented_datasets:
                aug_fitness_scores = []
                for aug_data in augmented_datasets:
                    aug_fitness = self.evaluate_fitness(individual, aug_data, volatility_scale)
                    # Validate augmented fitness values
                    if not (np.isnan(aug_fitness) or np.isinf(aug_fitness)):
                        aug_fitness_scores.append(aug_fitness)
                
                if aug_fitness_scores:  # Only compute average if we have valid scores
                    # Use weighted average: 70% original, 30% augmented
                    combined_fitness = 0.7 * individual.fitness + 0.3 * np.mean(aug_fitness_scores)
                    # Validate combined fitness
                    if np.isnan(combined_fitness) or np.isinf(combined_fitness):
                        print(f"Warning: Invalid combined fitness {combined_fitness}, keeping original")
                    else:
                        individual.fitness = float(combined_fitness)
        
        self.best_fitness_history = []
        ensemble_params = None
        
        # Add separator before generation progress
        if verbose:
            print()  # Empty line for separation
        
        # Evolution loop
        for generation in range(self.generations):
            # Adaptive rate adjustment
            self.adapt_rates(population, generation)
            
            # Age-based elite replacement
            population = self.age_based_replacement(population, generation)
            
            # Dynamic population size adjustment
            diversity = self.calculate_population_diversity(population)
            new_pop_size = self.adjust_population_size(population, diversity)
            
            # Adjust population size if needed
            if new_pop_size > current_pop_size:
                # Add new random individuals
                for _ in range(new_pop_size - current_pop_size):
                    population.append(self.create_individual())
                current_pop_size = new_pop_size
            elif new_pop_size < current_pop_size:
                # Remove worst individuals
                population.sort(key=lambda x: x.fitness, reverse=True)
                population = population[:new_pop_size]
                current_pop_size = new_pop_size
            
            # Evolve population
            population = self.evolve_population(population)
            
            # Evaluate new individuals
            for individual in population:
                if individual.fitness == 0:  # Not evaluated yet
                    fitness_value = self.evaluate_fitness(individual, train_data, volatility_scale)
                    # Validate fitness assignment
                    if np.isnan(fitness_value) or np.isinf(fitness_value):
                        print(f"Warning: Invalid fitness value {fitness_value} for individual, setting to -1000")
                        individual.fitness = -1000.0
                    else:
                        individual.fitness = float(fitness_value)
                    
                    # Additional evaluation on augmented data
                    if use_data_augmentation and augmented_datasets:
                        aug_fitness_scores = []
                        for aug_data in augmented_datasets:
                            aug_fitness = self.evaluate_fitness(individual, aug_data, volatility_scale)
                            # Validate augmented fitness values
                            if not (np.isnan(aug_fitness) or np.isinf(aug_fitness)):
                                aug_fitness_scores.append(aug_fitness)
                        
                        if aug_fitness_scores:  # Only compute average if we have valid scores
                            # Use weighted average
                            combined_fitness = 0.7 * individual.fitness + 0.3 * np.mean(aug_fitness_scores)
                            # Validate combined fitness
                            if np.isnan(combined_fitness) or np.isinf(combined_fitness):
                                print(f"Warning: Invalid combined fitness {combined_fitness}, keeping original")
                            else:
                                individual.fitness = float(combined_fitness)
            
            # Track best fitness with error handling for generation statistics
            try:
                if not population:
                    print("Warning: Empty population in generation statistics")
                    continue
                
                # Filter out individuals with invalid fitness values for statistics
                valid_fitness_population = []
                for ind in population:
                    if hasattr(ind, 'fitness') and not (np.isnan(ind.fitness) or np.isinf(ind.fitness)):
                        valid_fitness_population.append(ind)
                
                if not valid_fitness_population:
                    print(f"Warning: No valid fitness values in generation {generation+1}")
                    continue
                
                best_individual = max(valid_fitness_population, key=lambda x: x.fitness)
                self.best_fitness_history.append(best_individual.fitness)
                
                # Edge case: Prevent unbounded growth of best_fitness_history (memory leak prevention)
                if len(self.best_fitness_history) > self.max_fitness_history:
                    self.best_fitness_history = self.best_fitness_history[-self.max_fitness_history:]
                
                if verbose:
                    # Calculate average fitness with validation
                    fitness_values = [ind.fitness for ind in valid_fitness_population]
                    if fitness_values:
                        avg_fitness = np.mean(fitness_values)
                        # Validate average fitness calculation
                        if np.isnan(avg_fitness) or np.isinf(avg_fitness):
                            print(f"Warning: Invalid average fitness in generation {generation+1}")
                            avg_fitness = 0.0
                    else:
                        avg_fitness = 0.0
                    
                    # Compact output - no graph
                    # Clear line and print progress (avoid stray characters)
                    progress_line = (f"Gen {bold(f'{generation+1:>2}/{self.generations}')} | "
                                   f"Best: {cyan(f'{best_individual.fitness:.3f}')} | "
                                   f"Avg: {black(f'{avg_fitness:.3f}')} | "
                                   f"Pop: {current_pop_size:>3}")
                    print(f"\r{progress_line:<80}", end='', flush=True)
                    
            except Exception as e:
                print(f"Warning: Error in generation {generation+1} statistics: {e}")
                # Continue with next generation instead of crashing
        
        # Get final best individual with validation
        try:
            if not population:
                raise ValueError("Empty population at end of optimization")
            
            # Filter population for valid fitness values
            valid_final_population = []
            for ind in population:
                if hasattr(ind, 'fitness') and not (np.isnan(ind.fitness) or np.isinf(ind.fitness)):
                    valid_final_population.append(ind)
            
            if not valid_final_population:
                raise ValueError("No individuals with valid fitness values found")
                
            best_individual = max(valid_final_population, key=lambda x: x.fitness)
            
        except Exception as e:
            print(yellow(f"Warning: Error selecting final best individual: {e}"))
            # Fallback: return first individual with valid fitness, or create default
            for ind in population:
                if hasattr(ind, 'fitness') and not (np.isnan(ind.fitness) or np.isinf(ind.fitness)):
                    best_individual = ind
                    break
            else:
                # Create a default individual as last resort
                best_individual = self.create_individual()
                best_individual.fitness = -1000.0
        
        if verbose:
            print()  # New line after progress bar
        
        # Create ensemble parameters if requested
        if use_ensemble:
            ensemble_params = self.create_ensemble_parameters(population, n_ensemble=3)
        
        # Update candidate archive with top performers
        self.update_archive(population)
        
        # Save final fitness history for adaptive penalty system
        self._save_fitness_history()
        
        return best_individual, self.best_fitness_history, ensemble_params
    
    def update_archive(self, population: List[Individual]):
        """Update the candidate archive with top performers from current population"""
        # Get top performers from current population
        top_performers = sorted(population, key=lambda x: x.fitness, reverse=True)[:3]
        
        # Add their parameters to archive (memory leak prevention)
        for performer in top_performers:
            # Edge case: Validate performer has genes before adding to archive
            if hasattr(performer, 'genes') and performer.genes:
                # Create a lightweight copy with only essential parameters
                essential_genes = {}
                for key, value in performer.genes.items():
                    if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                        essential_genes[key] = value
                
                if essential_genes:  # Only add if we have valid genes
                    self.candidate_archive.append(essential_genes)
        
        # Keep only the best candidates in archive (memory leak prevention)
        if len(self.candidate_archive) > self.archive_size:
            # Keep only the most recent ones (FIFO with size limit)
            self.candidate_archive = self.candidate_archive[-self.archive_size:]
        
        # Edge case: Prevent archive from growing too large due to accumulated overhead
        if len(self.candidate_archive) > self.archive_size * 2:
            print(f"Warning: Candidate archive too large ({len(self.candidate_archive)}), trimming to size")
            self.candidate_archive = self.candidate_archive[-self.archive_size:]
        
        # Silent archive update - no print needed

def estimate_asset_volatility_scale(data: pd.DataFrame, sample_size: int = 500, verbose: bool = True) -> float:
    """
    Estimate volatility scale factor for the asset based on recent price movements
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data
    sample_size : int
        Number of recent bars to analyze
        
    Returns:
    --------
    float
        Volatility scale factor (0.1 to 3.0)
        ~0.3-0.5 for traditional stocks
        ~0.7-1.5 for volatile stocks/ETFs
        ~1.0-3.0 for crypto
    """
    # Use most recent data
    sample_data = data.tail(sample_size)
    
    # Calculate various volatility metrics
    returns = sample_data['close'].pct_change().dropna()
    
    # Daily volatility (standard deviation of returns)
    daily_vol = returns.std()
    
    # Annualized volatility
    annualized_vol = daily_vol * np.sqrt(252)  # Assuming daily data
    
    
    if verbose:
        print(f"\n{black('Asset volatility analysis:')}")
        print(f"  {black('Annualized volatility:'.ljust(22))} {annualized_vol*100:.1f}%")
    
    # Enhanced volatility categories for better granularity
    # More categories = better regime threshold tuning
    
    if annualized_vol < 0.15:  # < 15% annual vol
        scale = 0.2
        asset_type = "Ultra-low volatility (bonds/utilities)"
    elif annualized_vol < 0.20:  # 15-20% annual vol
        scale = 0.25
        asset_type = "Blue-chip stocks (mega-caps)"
    elif annualized_vol < 0.25:  # 20-25% annual vol
        scale = 0.3
        asset_type = "Large-cap stocks"
    elif annualized_vol < 0.30:  # 25-30% annual vol
        scale = 0.4
        asset_type = "Traditional stocks (S&P 500)"
    elif annualized_vol < 0.40:  # 30-40% annual vol
        scale = 0.5
        asset_type = "Growth stocks"
    elif annualized_vol < 0.50:  # 40-50% annual vol
        scale = 0.7
        asset_type = "Volatile stocks/sector ETFs"
    elif annualized_vol < 0.60:  # 50-60% annual vol
        scale = 0.9
        asset_type = "Small-cap stocks"
    elif annualized_vol < 0.80:  # 60-80% annual vol
        scale = 1.2
        asset_type = "High volatility (emerging markets/crypto)"
    elif annualized_vol < 1.0:  # 80-100% annual vol
        scale = 1.5
        asset_type = "Very high volatility (altcoins)"
    else:  # > 100% annual vol
        scale = 2.0
        asset_type = "Extreme volatility (meme coins/penny stocks)"
    
    if verbose:
        print(f"  {black('Detected asset type:'.ljust(22))} {asset_type}")
        print(f"  {black('Volatility scale:'.ljust(22))} {scale}")
    
    return scale
