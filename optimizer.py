"""
Strategy optimizer combining Walk-Forward Optimization and Genetic Algorithm.

This module provides an Optimizer class that performs walk-forward optimization
using genetic algorithms to find optimal strategy parameters in each window.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Type, TypeVar, Generic
from pydantic import BaseModel, Field
from walk_forward_optimizer import WalkForwardOptimizer
from genetic_algorithm import GeneticAlgorithm, OptimizationResult
from strategy_parameters import BaseStrategyParameters
import chalk

T = TypeVar('T', bound=BaseStrategyParameters)


class WalkForwardResult(BaseModel, Generic[T]):
    """Result from a single walk-forward window."""

    split_id: int = Field(description="Split identifier")
    train_indices: Tuple[int, int] = Field(description="Training indices (start, end)")
    test_indices: Tuple[int, int] = Field(description="Test indices (start, end)")
    best_params: T = Field(description="Best parameters from training")
    train_fitness: float = Field(description="Best fitness on training data")
    test_performance: float = Field(description="Performance on test data")
    optimization_result: OptimizationResult[T] = Field(description="Full GA optimization result")

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class OptimizerConfig(BaseModel):
    """Configuration for the optimizer."""
    
    # Walk-forward parameters
    n_splits: int = Field(default=5, gt=0, description="Number of walk-forward splits")
    # test_ratio: Optional[float] = Field(default=None, ge=0.1, le=0.9, description="Test ratio as fraction (0.3 = 30% test, 70% train) or None for expanding window")
    gap: int = Field(default=0, ge=0, description="Gap between train and test")
    
    # Genetic algorithm parameters
    param_class: Type[BaseStrategyParameters] = Field(description="Strategy parameter class")
    population_size: int = Field(default=50, gt=0, description="GA population size")
    generations: int = Field(default=100, gt=0, description="Number of GA generations")
    crossover_prob: float = Field(default=0.8, ge=0, le=1, description="Crossover probability")
    mutation_prob: float = Field(default=0.1, ge=0, le=1, description="Mutation probability")
    tournament_size: int = Field(default=3, gt=0, description="Tournament selection size")
    
    # General parameters
    verbose: bool = Field(default=True, description="Print progress information")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class Optimizer(Generic[T]):
    """
    Combines Walk-Forward Optimization with Genetic Algorithm for robust parameter optimization.
    
    This class implements the complete optimization workflow:
    1. Split data using walk-forward windows
    2. Optimize parameters on each training window using GA
    3. Evaluate on corresponding test window
    4. Aggregate results across all windows
    """

    def __init__(self, config: OptimizerConfig) -> None:
        """
        Initialize the optimizer.
        
        Args:
            config: Optimizer configuration
        """
        self.config = config
        self.wfo = WalkForwardOptimizer(
            n_splits=config.n_splits,
            test_ratio=0.3,  # Always use 70/30 split
            gap=config.gap
        )
        self.results: List[WalkForwardResult[T]] = []

    def optimize(
        self,
        data: pd.DataFrame,
        fitness_function: Callable[[T, pd.DataFrame], float],
        evaluation_function: Optional[Callable[[T, pd.DataFrame], float]] = None,
        seed_params: Optional[List[T]] = None
    ) -> List[WalkForwardResult[T]]:
        """
        Run walk-forward optimization with genetic algorithm.
        
        Args:
            data: DataFrame with market data
            fitness_function: Function to optimize (maximize) during training
            evaluation_function: Function to evaluate on test data (if None, uses fitness_function)
            seed_params: Optional list of parameter sets to seed GA populations
            
        Returns:
            List of WalkForwardResult objects for each split
        """
        if evaluation_function is None:
            evaluation_function = fitness_function

        # Get walk-forward splits
        splits = self.wfo.get_splits(data)

        if self.config.verbose:
            print(chalk.cyan + "\n" + "━"*60 + chalk.RESET)
            print(chalk.cyan + chalk.bold + "🚀 Starting Walk-Forward Optimization with Genetic Algorithm" + chalk.RESET)
            print(chalk.cyan + "━"*60 + chalk.RESET)
            self.wfo.print_splits_summary(splits)

        self.results = []

        # Process each split
        for split in splits:
            if self.config.verbose:
                print(chalk.blue + f"\n{'─'*60}" + chalk.RESET)
                print(chalk.blue + chalk.bold + f"📊 Processing Split {split.split_id}" + chalk.RESET)
                print(chalk.blue + f"{'─'*60}" + chalk.RESET)

            # Get train and test data
            train_data, test_data = self.wfo.get_split_data(data, split)

            # 1. Optimize on training data using GA
            ga = GeneticAlgorithm(
                param_class=self.config.param_class,
                population_size=self.config.population_size,
                generations=self.config.generations,
                crossover_prob=self.config.crossover_prob,
                mutation_prob=self.config.mutation_prob,
                tournament_size=self.config.tournament_size,
                seed=self.config.seed
            )

            if self.config.verbose:
                print(chalk.yellow + f"\n🔧 Optimizing on {len(train_data):,} training bars..." + chalk.RESET)

            optimization_result = ga.run(
                fitness_function=fitness_function,
                data=train_data,
                verbose=False,  # Suppress generation-by-generation output
                seed_params=seed_params  # Pass seed parameters if available
            )

            # 2. Evaluate on test data
            if self.config.verbose:
                print(chalk.magenta + f"\n📈 Evaluating on {len(test_data):,} test bars..." + chalk.RESET)

            test_performance = evaluation_function(optimization_result.best_params, test_data)

            # 3. Store results
            result = WalkForwardResult[T](
                split_id=split.split_id,
                train_indices=(split.train_start_idx, split.train_end_idx),
                test_indices=(split.test_start_idx, split.test_end_idx),
                best_params=optimization_result.best_params,
                train_fitness=optimization_result.best_fitness,
                test_performance=test_performance,
                optimization_result=optimization_result
            )

            self.results.append(result)

            if self.config.verbose:
                print(chalk.green + f"\n✅ Split {split.split_id} Results:" + chalk.RESET)
                if result.train_fitness >= 0:
                    print(f"  {chalk.green}▸{chalk.RESET} Training Fitness: {chalk.bold}{result.train_fitness:.4f}{chalk.RESET}")
                else:
                    print(f"  {chalk.red}▸{chalk.RESET} Training Fitness: {chalk.red}{result.train_fitness:.4f}{chalk.RESET}")
                
                if result.test_performance >= 0:
                    print(f"  {chalk.green}▸{chalk.RESET} Test Performance: {chalk.bold}{result.test_performance:.4f}{chalk.RESET}")
                else:
                    print(f"  {chalk.red}▸{chalk.RESET} Test Performance: {chalk.red}{result.test_performance:.4f}{chalk.RESET}")
                #  print(f"  Best Parameters: {result.best_params.to_dict()}")

        if self.config.verbose:
            self._print_summary()

        return self.results

    def _print_summary(self) -> None:
        """Print summary of optimization results."""
        print(chalk.cyan + "\n" + "━"*60 + chalk.RESET)
        print(chalk.cyan + chalk.bold + "   🏆 Walk-Forward Optimization Summary" + chalk.RESET)
        print(chalk.cyan + "━"*60 + chalk.RESET)

        # Calculate aggregate statistics
        train_scores = [r.train_fitness for r in self.results]
        test_scores = [r.test_performance for r in self.results]

        print(f"\nNumber of windows: {len(self.results)}")
        print(f"\nTraining Performance:")
        print(f"  Mean: {np.mean(train_scores):.4f}")
        print(f"  Std:  {np.std(train_scores):.4f}")
        print(f"  Min:  {np.min(train_scores):.4f}")
        print(f"  Max:  {np.max(train_scores):.4f}")

        print(f"\nTest Performance:")
        print(f"  Mean: {np.mean(test_scores):.4f}")
        print(f"  Std:  {np.std(test_scores):.4f}")
        print(f"  Min:  {np.min(test_scores):.4f}")
        print(f"  Max:  {np.max(test_scores):.4f}")

        # Check for overfitting
        overfitting_ratio = np.mean(train_scores) / np.mean(test_scores) if np.mean(test_scores) > 0 else float('inf')
        print(f"\n   {chalk.white}Overfitting Ratio (train/test):{chalk.RESET} {chalk.bold}{overfitting_ratio:.2f}{chalk.RESET}")

        if overfitting_ratio > 1.5:
            print(chalk.red + "   ⚠️  Warning: Significant overfitting detected!" + chalk.RESET)
        elif overfitting_ratio > 1.2:
            print(chalk.yellow + "   ⚠️  Warning: Moderate overfitting detected" + chalk.RESET)
        else:
            print(chalk.green + "   ✓ Good generalization performance" + chalk.RESET)
        
        print(chalk.cyan + "━"*60 + chalk.RESET)

    def get_consensus_parameters(self) -> Dict[str, float]:
        """
        Calculate consensus parameters across all windows.
        
        Can use mean, median, or weighted average based on test performance.
        
        Returns:
            Dictionary of consensus parameter values
        """
        if not self.results:
            raise ValueError("No optimization results available")

        # Collect all parameters
        all_params = {}
        param_names = list(self.config.param_class.get_param_ranges().keys())
        for param in param_names:
            all_params[param] = []

        for result in self.results:
            param_dict = result.best_params.to_dict()
            for param, value in param_dict.items():
                all_params[param].append(value)

        # Calculate median (more robust than mean)
        consensus_params = {}
        for param, values in all_params.items():
            consensus_params[param] = float(np.median(values))

        return consensus_params

    def get_performance_weighted_parameters(self) -> Dict[str, float]:
        """
        Calculate parameters weighted by test performance.
        
        Better performing windows get more weight in the final parameters.
        
        Returns:
            Dictionary of weighted parameter values
        """
        if not self.results:
            raise ValueError("No optimization results available")

        # Get test performances (ensure positive)
        performances = np.array([r.test_performance for r in self.results])
        min_perf = performances.min()
        if min_perf < 0:
            performances = performances - min_perf + 1  # Shift to positive

        # Calculate weights
        weights = performances / performances.sum()

        # Calculate weighted parameters
        weighted_params = {}
        param_names = list(self.config.param_class.get_param_ranges().keys())
        for param in param_names:
            weighted_sum = 0
            for i, result in enumerate(self.results):
                param_dict = result.best_params.to_dict()
                weighted_sum += param_dict[param] * weights[i]
            weighted_params[param] = float(weighted_sum)

        return weighted_params

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a DataFrame for analysis.
        
        Returns:
            DataFrame with optimization results
        """
        if not self.results:
            return pd.DataFrame()

        records = []
        for result in self.results:
            record = {
                'split_id': result.split_id,
                'train_start_idx': result.train_indices[0],
                'train_end_idx': result.train_indices[1],
                'test_start_idx': result.test_indices[0],
                'test_end_idx': result.test_indices[1],
                'train_fitness': result.train_fitness,
                'test_performance': result.test_performance,
                'overfit_ratio': result.train_fitness / result.test_performance if result.test_performance > 0 else float('inf')
            }

            # Add parameter values
            param_dict = result.best_params.to_dict()
            for param, value in param_dict.items():
                record[f'param_{param}'] = value

            records.append(record)

        return pd.DataFrame(records)
