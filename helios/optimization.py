"""
Genetic Algorithm optimization for Helios Trader
Implements walk-forward optimization of strategy parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import random
from datetime import datetime, timedelta
import json
from pathlib import Path

from factors import calculate_mss, calculate_macd, calculate_rsi
from strategy import TradingStrategy
from strategy_enhanced import EnhancedTradingStrategy
from performance import calculate_sortino_ratio, calculate_calmar_ratio
from data_processing import prepare_data


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
                 parameter_ranges: Dict[str, Tuple[float, float]],
                 population_size: int = 50,
                 generations: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1,
                 fitness_metric: str = 'sortino'):
        """
        Initialize genetic algorithm
        
        Parameters:
        -----------
        parameter_ranges : Dict[str, Tuple[float, float]]
            Min and max values for each parameter
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
        """
        self.parameter_ranges = parameter_ranges
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.fitness_metric = fitness_metric
        
        # Calculate elite size
        self.elite_size = max(1, int(population_size * elitism_rate))
    
    def create_individual(self) -> Individual:
        """Create a random individual"""
        genes = {}
        for param, (min_val, max_val) in self.parameter_ranges.items():
            if param.endswith('_int'):
                # Integer parameters
                genes[param] = random.randint(int(min_val), int(max_val))
            else:
                # Float parameters
                genes[param] = random.uniform(min_val, max_val)
        
        return Individual(genes=genes)
    
    def create_population(self) -> List[Individual]:
        """Create initial population"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def evaluate_fitness(self, individual: Individual, 
                        train_data: pd.DataFrame) -> float:
        """
        Evaluate fitness of an individual
        
        Parameters:
        -----------
        individual : Individual
            Individual to evaluate
        train_data : pd.DataFrame
            Training data with OHLCV
        
        Returns:
        --------
        float
            Fitness score
        """
        genes = individual.genes
        
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
                )
            else:
                # Use basic strategy
                strategy = TradingStrategy(
                    initial_capital=initial_capital,
                    max_position_pct=genes.get('max_position_pct', 0.95),
                    min_position_pct=genes.get('min_position_pct', 0.1)
                )
            
            results = strategy.run_backtest(combined_df, combined_df)
            
            # Calculate fitness metric
            if len(results) < 2:
                return -1000  # Invalid strategy
            
            # Get portfolio values
            portfolio_values = results['portfolio_value']
            
            # Calculate returns  
            returns = portfolio_values.pct_change().dropna()
            
            # Calculate base fitness metric using OLD CODE APPROACH
            if self.fitness_metric == 'sortino':
                # OLD CODE: MAR = 0, no risk-free rate
                mar = 0
                downside_returns = returns[returns < mar]
                
                if len(downside_returns) == 0:
                    # No downside returns
                    sortino = float('inf')
                else:
                    downside_deviation = downside_returns.std()
                    if downside_deviation == 0 or np.isnan(downside_deviation):
                        sortino = float('inf')
                    else:
                        # Annualized return and Sortino
                        periods_per_year = 365  # Dollar bars, not 252
                        n_periods = len(returns)
                        total_return = (portfolio_values.iloc[-1] / initial_capital) - 1
                        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
                        sortino = (annualized_return - mar) / (downside_deviation * np.sqrt(periods_per_year))
                
                # Handle special cases
                if np.isnan(sortino):
                    return -1000  # Invalid strategy
                elif np.isinf(sortino):
                    sortino = 10.0  # Cap infinite Sortino at 10
                base_fitness = sortino
                
            elif self.fitness_metric == 'calmar':
                # Calculate using old approach
                periods_per_year = 365
                n_periods = len(returns)
                total_return = (portfolio_values.iloc[-1] / initial_capital) - 1
                annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
                
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
            
            # Combined fitness with drawdown penalty (from old implementation)
            weight_primary = 0.7
            weight_drawdown = 0.3
            fitness = (base_fitness * weight_primary) - (max_drawdown_pct * weight_drawdown)
            
            # Debug output (remove later)
            if False:  # Set to True for debugging
                print(f"  Base fitness (Sortino): {base_fitness:.4f}")
                print(f"  Max drawdown %: {max_drawdown_pct:.2f}%")
                print(f"  Combined fitness: {fitness:.4f}")
            
            # Additional penalties for extreme parameters
            if weights['trend'] < 0.1 or weights['trend'] > 0.7:
                fitness -= 0.5
            if lookback < 5 or lookback > 100:
                fitness -= 0.5
            
            return fitness
            
        except Exception as e:
            # Invalid parameters
            return -1000
    
    def tournament_selection(self, population: List[Individual], 
                           tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents"""
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
        
        return Individual(genes=genes1), Individual(genes=genes2)
    
    def mutate(self, individual: Individual) -> Individual:
        """Mutate an individual"""
        genes = individual.genes.copy()
        
        for param, (min_val, max_val) in self.parameter_ranges.items():
            if random.random() < self.mutation_rate:
                if param.endswith('_int'):
                    # Integer mutation
                    genes[param] = random.randint(int(min_val), int(max_val))
                else:
                    # Float mutation with Gaussian noise
                    current = genes[param]
                    noise = random.gauss(0, (max_val - min_val) * 0.1)
                    genes[param] = max(min_val, min(max_val, current + noise))
        
        return Individual(genes=genes)
    
    def evolve_population(self, population: List[Individual]) -> List[Individual]:
        """Evolve population for one generation"""
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep elite
        new_population = population[:self.elite_size]
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        return new_population[:self.population_size]
    
    def optimize(self, train_data: pd.DataFrame, 
                verbose: bool = True) -> Tuple[Individual, List[float]]:
        """
        Run genetic algorithm optimization
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        verbose : bool
            Print progress
        
        Returns:
        --------
        Tuple[Individual, List[float]]
            Best individual and fitness history
        """
        # Create initial population
        population = self.create_population()
        
        # Evaluate initial population
        for individual in population:
            individual.fitness = self.evaluate_fitness(individual, train_data)
        
        best_fitness_history = []
        
        # Evolution loop
        for generation in range(self.generations):
            # Evolve population
            population = self.evolve_population(population)
            
            # Evaluate new individuals
            for individual in population:
                if individual.fitness == 0:  # Not evaluated yet
                    individual.fitness = self.evaluate_fitness(individual, train_data)
            
            # Track best fitness
            best_individual = max(population, key=lambda x: x.fitness)
            best_fitness_history.append(best_individual.fitness)
            
            if verbose:
                avg_fitness = np.mean([ind.fitness for ind in population])
                print(f"Generation {generation+1}/{self.generations}: "
                      f"Best Fitness = {best_individual.fitness:.4f}, "
                      f"Avg Fitness = {avg_fitness:.4f}")
        
        # Return best individual
        best_individual = max(population, key=lambda x: x.fitness)
        return best_individual, best_fitness_history


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
                save_results: bool = True) -> Dict:
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
            best_individual, fitness_history = ga.optimize(train_data, verbose=False)
            
            # Evaluate on test data
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
            results_dir = Path('./optimization_results')
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = results_dir / f'walk_forward_{timestamp}.json'
            
            with open(results_file, 'w') as f:
                json.dump(optimization_results, f, indent=2)
            
            print(f"\nResults saved to: {results_file}")
        
        return optimization_results


def create_default_parameter_ranges() -> Dict[str, Tuple[float, float]]:
    """Create default parameter ranges for optimization"""
    return {
        'weight_trend': (0.1, 0.6),
        'weight_volatility': (0.1, 0.5),
        'weight_exhaustion': (0.1, 0.5),
        'lookback_int': (10, 50),
        'max_position_pct': (0.5, 1.0),
        'min_position_pct': (0.05, 0.2),
        'strong_bull_threshold_int': (50, 80),
        'weak_bull_threshold_int': (10, 40),
        'weak_bear_threshold_int': (-40, -10),
        'strong_bear_threshold_int': (-80, -50),
        'stop_loss_multiplier_strong': (1.5, 3.0),
        'stop_loss_multiplier_weak': (0.5, 1.5)
    }


def create_enhanced_parameter_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Create enhanced parameter ranges matching the old notebook implementation
    Includes gradual entry parameters and all thresholds
    """
    return {
        # Factor weights
        'weight_trend': (0.0, 1.0),
        'weight_volatility': (0.0, 1.0),
        'weight_exhaustion': (0.0, 1.0),
        
        # Lookback periods
        'lookback_int': (10, 50),
        
        # Regime thresholds (as floats, not ints)
        'strong_bull_threshold': (30.0, 80.0),
        'weak_bull_threshold': (10.0, 40.0),
        'neutral_threshold_upper': (10.0, 40.0),
        'neutral_threshold_lower': (-40.0, -10.0),
        'weak_bear_threshold': (-40.0, -10.0),
        'strong_bear_threshold': (-80.0, -30.0),
        
        # Stop-loss multipliers
        'stop_loss_multiplier_strong': (1.0, 5.0),
        'stop_loss_multiplier_weak': (0.5, 3.0),
        
        # Gradual entry parameters
        'entry_step_size': (0.1, 1.0),
        'max_position_pct': (0.5, 1.0),
    }