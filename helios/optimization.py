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

from factors import calculate_mss, calculate_macd, calculate_rsi
from strategy_enhanced import EnhancedTradingStrategy
from performance import calculate_sortino_ratio, calculate_calmar_ratio
from data_processing import prepare_data
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
                 population_size: int = 50,
                 generations: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1,
                 fitness_metric: str = 'sortino',
                 allow_shorts: bool = False):
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
        fitness_metric : str
            Metric to optimize ('sortino' or 'calmar')
        allow_shorts : bool
            Whether to allow short positions
        """
        # Convert parameter config to uniform ParameterRange objects
        self.parameter_ranges = {}
        for param, config in parameter_config.items():
            if isinstance(config, ParameterRange):
                self.parameter_ranges[param] = config
            elif isinstance(config, tuple) and len(config) == 2:
                # Legacy tuple format - convert to MinMaxRange
                self.parameter_ranges[param] = MinMaxRange(config[0], config[1])
            else:
                raise ValueError(f"Invalid parameter config for {param}: {config}")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.fitness_metric = fitness_metric
        self.allow_shorts = allow_shorts
        self.candidate_archive: List[Dict] = []  # Archive of recent top candidates
        self.archive_size = 5  # Keep top 5 candidates from recent runs
        
        # Calculate elite size
        self.elite_size = max(1, int(population_size * elitism_rate))
        
        # Adaptive parameters
        self.base_mutation_rate = mutation_rate
        self.base_crossover_rate = crossover_rate
        self.generations_without_improvement = 0
        self.best_fitness_history = []
        
        # Age-based elitism
        self.individual_ages = {}  # Track age of individuals
        self.max_elite_age = 5  # Maximum generations an elite can survive
        
        # Dynamic population
        self.min_population_size = max(20, population_size // 2)
        self.max_population_size = min(200, population_size * 2)
        self.target_diversity = 0.3  # Target diversity level
    
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
            values = [ind.genes[param] for ind in population]
            if len(set(values)) > 1:  # More than one unique value
                param_range = self.parameter_ranges[param]
                normalized_values = [(v - min(values)) / (max(values) - min(values) + 1e-8) 
                                   for v in values]
                variance = np.var(normalized_values)
                variances.append(variance)
        
        return float(np.mean(variances)) if variances else 0.0
    
    def adapt_rates(self, population: List[Individual], generation: int):
        """Adapt mutation and crossover rates based on diversity and convergence"""
        diversity = self.calculate_population_diversity(population)
        
        # Check if fitness is improving
        if self.best_fitness_history:
            current_best = max(ind.fitness for ind in population)
            if len(self.best_fitness_history) >= 3:
                recent_improvement = current_best - self.best_fitness_history[-3]
                if recent_improvement < 0.001:  # No significant improvement
                    self.generations_without_improvement += 1
                else:
                    self.generations_without_improvement = 0
        
        # Adjust mutation rate based on diversity and stagnation
        if diversity < 0.1:  # Low diversity
            self.mutation_rate = min(0.5, self.base_mutation_rate * 2.0)
        elif self.generations_without_improvement > 5:  # Stagnation
            self.mutation_rate = min(0.4, self.base_mutation_rate * 1.5)
        else:
            self.mutation_rate = self.base_mutation_rate
        
        # Adjust crossover rate inversely to mutation
        if self.mutation_rate > self.base_mutation_rate:
            self.crossover_rate = max(0.5, self.base_crossover_rate * 0.8)
        else:
            self.crossover_rate = self.base_crossover_rate
        
        # Only print adaptive info every 5 generations or on significant changes
        if generation % 5 == 0 or self.generations_without_improvement == 1:
            from chalk import black
            print(f"  {black(f'Adaptive: Div={diversity:.2f}, Mut={self.mutation_rate:.2f}, Stag={self.generations_without_improvement}')}")
    
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
        
        # Only print if elites were replaced
        if elites_replaced > 0:
            from chalk import yellow
            print(f"  {yellow(f'↻ Replaced {elites_replaced} old elites')}")
        
        # Clean up age tracking for individuals not in population
        current_ids = {id(ind) for ind in new_population}
        self.individual_ages = {k: v for k, v in self.individual_ages.items() if k in current_ids}
        
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
            if new_size > current_size:
                from chalk import cyan
                print(f"  {cyan(f'↑ Low diversity ({diversity:.2f}), population: {current_size} → {new_size}')}")
        elif diversity > 0.45 and current_size > self.min_population_size:  # High diversity
            # Can reduce population for efficiency
            new_size = max(current_size - 5, self.min_population_size)
            if new_size < current_size:
                from chalk import cyan
                print(f"  {cyan(f'↓ High diversity ({diversity:.2f}), population: {current_size} → {new_size}')}")
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
            if param.endswith('_int'):
                # For integer parameters, use weighted average then round
                weighted_sum = sum(ind.genes[param] * w for ind, w in zip(ensemble_members, weights))
                ensemble_params[param] = int(round(weighted_sum))
            else:
                # For continuous parameters, use weighted average
                ensemble_params[param] = sum(ind.genes[param] * w for ind, w in zip(ensemble_members, weights))
        
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
            
            # Check if we have enhanced parameters
            has_enhanced_params = any(param in genes for param in [
                'entry_step_size', 'stop_loss_multiplier_strong', 'stop_loss_multiplier_weak',
                'strong_bull_threshold', 'weak_bull_threshold', 'strong_bear_threshold'
            ])
            
            if has_enhanced_params:
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
            else:
                # Use enhanced strategy with default parameters
                strategy = EnhancedTradingStrategy(
                    initial_capital=initial_capital,
                    max_position_fraction=genes.get('max_position_pct', 0.95),
                    allow_shorts=self.allow_shorts
                )
            
            results = strategy.run_backtest(combined_df, combined_df)
            
            # Calculate fitness metric
            if len(results) < 2:
                return -1000  # Invalid strategy
            
            # Get portfolio values
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
            if self.fitness_metric == 'sortino':
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
                
            elif self.fitness_metric == 'calmar':
                # Use already calculated annualized_return (time-based)
                
                # Max drawdown calculation
                equity_curve = portfolio_values
                peak = equity_curve.expanding().max()
                drawdown_dollars = equity_curve - peak
                max_drawdown_dollars = drawdown_dollars.min()
                
                if max_drawdown_dollars >= 0:
                    base_fitness = 10.0  # No drawdown
                else:
                    base_fitness = annualized_return / abs(max_drawdown_dollars / initial_capital)
            else:
                # Default to total return
                base_fitness = (portfolio_values.iloc[-1] / initial_capital - 1)
            
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
            
            # Combined fitness with drawdown penalty (simplified approach)
            # Use the base fitness (Sortino/Sharpe blend) directly for stocks
            # Apply minimal drawdown penalty since Sortino already accounts for downside risk
            
            if volatility_scale < 0.5:  # Stocks
                # For stocks, use mostly the risk-adjusted return metrics
                # Small penalty for extreme drawdowns only
                if max_drawdown_pct > 20:  # Only penalize large drawdowns
                    drawdown_penalty = (max_drawdown_pct - 20) / 100 * 0.5
                else:
                    drawdown_penalty = 0
                fitness = base_fitness - drawdown_penalty
            else:  # Crypto and high volatility
                # For crypto, drawdown matters more
                weight_primary = 0.8
                weight_drawdown = 0.2
                drawdown_ratio = max_drawdown_pct / 100.0
                fitness = (base_fitness * weight_primary) - (drawdown_ratio * 5 * weight_drawdown)
            
            # Debug output (remove later)
            if False:  # Set to True for debugging
                print(f"  Base fitness (Sortino): {base_fitness:.4f}")
                print(f"  Max drawdown %: {max_drawdown_pct:.2f}%")
                print(f"  Combined fitness: {fitness:.4f}")
                print(f"  Total return: {total_return:.4f}")
                print(f"  Annualized return: {annualized_return:.4f}")
                print(f"  Time span: {time_span_days} days, Periods: {n_periods}, Freq: {periods_per_year:.1f}/year")
            
            # Additional penalties for extreme parameters
            if weights['trend'] < 0.05 or weights['trend'] > 0.95:
                fitness -= 1.0
            if lookback < 3 or lookback > 100:
                fitness -= 1.0
            
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
            
            # Volatility-aware penalty for too few trades
            # Low-vol assets naturally trade less frequently
            position_changes = 0
            if 'position' in results.columns:
                positions = results['position']
                position_changes = (positions.diff() != 0).sum()
            
            # Adjust minimum trade expectations based on volatility
            if volatility_scale < 0.3:  # Blue-chip stocks
                trades_per_period = 300  # 1 trade per 300 periods (less frequent)
                penalty_multiplier = 0.5  # Lower penalty
            elif volatility_scale < 0.5:  # Traditional stocks
                trades_per_period = 250  # 1 trade per 250 periods
                penalty_multiplier = 1.0
            elif volatility_scale < 1.0:  # Volatile stocks/crypto
                trades_per_period = 200  # 1 trade per 200 periods
                penalty_multiplier = 1.5
            else:  # Very high volatility
                trades_per_period = 150  # 1 trade per 150 periods (more frequent)
                penalty_multiplier = 2.0
            
            min_trades = max(5, n_periods / trades_per_period)
            if position_changes < min_trades:
                trade_penalty = penalty_multiplier * (1 - position_changes / min_trades)
                fitness -= trade_penalty
                if False:  # Debug
                    print(f"  Trade penalty: {trade_penalty:.2f} (only {position_changes} position changes, vol_scale={volatility_scale:.2f})")
            
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
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Single-point crossover
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        crossover_point = random.randint(1, len(genes1) - 1)
        param_names = list(genes1.keys())
        
        for i in range(crossover_point, len(param_names)):
            param = param_names[i]
            genes1[param], genes2[param] = genes2[param], genes1[param]
        
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
        
        # Keep elite
        new_population = population[:self.elite_size]
        
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
        volatility_scale = estimate_asset_volatility_scale(train_data, sample_size=min(500, len(train_data)))
        
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
            individual.fitness = self.evaluate_fitness(individual, train_data, volatility_scale)
            
            # Additional evaluation on augmented data
            if use_data_augmentation and augmented_datasets:
                aug_fitness_scores = []
                for aug_data in augmented_datasets:
                    aug_fitness = self.evaluate_fitness(individual, aug_data, volatility_scale)
                    aug_fitness_scores.append(aug_fitness)
                # Use weighted average: 70% original, 30% augmented
                individual.fitness = float(0.7 * individual.fitness + 0.3 * np.mean(aug_fitness_scores))
        
        self.best_fitness_history = []
        ensemble_params = None
        
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
                    individual.fitness = self.evaluate_fitness(individual, train_data, volatility_scale)
                    
                    # Additional evaluation on augmented data
                    if use_data_augmentation and augmented_datasets:
                        aug_fitness_scores = []
                        for aug_data in augmented_datasets:
                            aug_fitness = self.evaluate_fitness(individual, aug_data, volatility_scale)
                            aug_fitness_scores.append(aug_fitness)
                        # Use weighted average
                        individual.fitness = float(0.7 * individual.fitness + 0.3 * np.mean(aug_fitness_scores))
            
            # Track best fitness
            best_individual = max(population, key=lambda x: x.fitness)
            self.best_fitness_history.append(best_individual.fitness)
            
            if verbose:
                from chalk import green, cyan, bold, black
                avg_fitness = np.mean([ind.fitness for ind in population])
                
                # Progress bar
                progress = (generation + 1) / self.generations
                bar_length = 20
                filled = int(bar_length * progress)
                bar = green("█" * filled) + black("░" * (bar_length - filled))
                
                # Compact output
                print(f"\r{bar} Gen {bold(f'{generation+1}/{self.generations}')} | "
                      f"Best: {cyan(f'{best_individual.fitness:.3f}')} | "
                      f"Avg: {black(f'{avg_fitness:.3f}')} | "
                      f"Pop: {current_pop_size}", end='', flush=True)
        
        # Get final best individual
        best_individual = max(population, key=lambda x: x.fitness)
        
        if verbose:
            print()  # New line after progress bar
        
        # Create ensemble parameters if requested
        if use_ensemble:
            ensemble_params = self.create_ensemble_parameters(population, n_ensemble=3)
        
        # Update candidate archive with top performers
        self.update_archive(population)
        
        return best_individual, self.best_fitness_history, ensemble_params
    
    def update_archive(self, population: List[Individual]):
        """Update the candidate archive with top performers from current population"""
        # Get top performers from current population
        top_performers = sorted(population, key=lambda x: x.fitness, reverse=True)[:3]
        
        # Add their parameters to archive
        for performer in top_performers:
            self.candidate_archive.append(performer.genes.copy())
        
        # Keep only the best candidates in archive (remove duplicates and sort by fitness)
        # Note: This is simplified - in production you'd want to track fitness with params
        if len(self.candidate_archive) > self.archive_size:
            # Keep only the most recent ones (FIFO with size limit)
            self.candidate_archive = self.candidate_archive[-self.archive_size:]
        
        # Silent archive update - no print needed


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
                'mutation_rate': ga.mutation_rate,
                'fitness_metric': ga.fitness_metric
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




def estimate_asset_volatility_scale(data: pd.DataFrame, sample_size: int = 500) -> float:
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
    
    
    print(f"\nAsset volatility analysis:")
    print(f"  Annualized volatility: {annualized_vol*100:.1f}%")
    
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
    
    print(f"  Detected asset type: {asset_type}")
    print(f"  Volatility scale factor: {scale}")
    
    return scale


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
    
    print(f"Auto-threshold analysis:")
    print(f"  Average price: ${avg_price:.2f}")
    print(f"  Average volume: {avg_volume:,.0f}")
    print(f"  Median dollar volume: ${median_dollar_vol:,.0f}")
    print(f"  Mean dollar volume: ${mean_dollar_vol:,.0f}")
    
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
    
    print(f"  Target: ~{target_bars_per_day} bars per day")
    
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