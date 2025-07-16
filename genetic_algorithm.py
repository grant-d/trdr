"""
Genetic Algorithm for parameter optimization using DEAP.

This module provides a GeneticAlgorithm class for optimizing strategy parameters
using the DEAP (Distributed Evolutionary Algorithms in Python) framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Any, Type, TypeVar, Generic
from pydantic import BaseModel, Field
import random
from deap import base, creator, tools, algorithms
from strategy_parameters import BaseStrategyParameters, BaseRange
from typing import cast

T = TypeVar('T', bound=BaseStrategyParameters)


class OptimizationResult(BaseModel, Generic[T]):
    """Result from genetic algorithm optimization."""

    best_params: T = Field(description="Best parameter values found")
    best_fitness: float = Field(description="Best fitness score achieved")
    generation_stats: List[Dict[str, float]] = Field(description="Statistics per generation")

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class GeneticAlgorithm:
    """
    Implements a genetic algorithm for parameter optimization using DEAP.
    
    The GA evolves a population of parameter sets to find optimal values
    for trading strategy parameters.
    """
    
    def __init__(
        self,
        param_class: Type[T],
        population_size: int = 50,
        generations: int = 100,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        mutation_indpb: float = 0.2,
        tournament_size: int = 3,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the Genetic Algorithm.
        
        Args:
            param_class: Strategy parameter class (subclass of BaseStrategyParameters)
            population_size: Number of individuals in each generation
            generations: Number of generations to evolve
            crossover_prob: Probability of crossover between parents
            mutation_prob: Probability of mutation for an individual
            mutation_indpb: Probability of mutating each gene
            tournament_size: Number of individuals in tournament selection
            seed: Random seed for reproducibility
        """
        self.param_class = param_class
        self.param_ranges = param_class.get_param_ranges()
        self.param_names = list(self.param_ranges.keys())
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_indpb = mutation_indpb
        self.tournament_size = tournament_size
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self._setup_deap()
        
    def _setup_deap(self) -> None:
        """Set up DEAP framework components."""
        # Create fitness class (maximize)
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Register attribute generators for each parameter
        for i, (param_name, range_obj) in enumerate(self.param_ranges.items()):
            # Create a sampling function for this parameter
            self.toolbox.register(f"attr_{i}", range_obj.sample)
        
        # Register individual and population generators
        self.toolbox.register("individual", self._create_valid_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._dummy_evaluate)  # Will be replaced
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", self._mutate_bounded)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
    def _create_valid_individual(self) -> List[float]:
        """
        Create an individual with valid parameter constraints.
        
        Returns:
            List of parameter values that satisfy all constraints
        """
        # For regime strategy parameters, use smart generation
        param_indices = {name: i for i, name in enumerate(self.param_names)}
        
        if 'strong_bull_threshold' in param_indices:
            # Generate regime thresholds with proper ordering
            max_attempts = 100
            
            for _ in range(max_attempts):
                values = [0.0] * len(self.param_names)
                
                # Generate non-threshold parameters first
                for param_name, i in param_indices.items():
                    if 'threshold' not in param_name and 'neutral' not in param_name:
                        range_obj = self.param_ranges[param_name]
                        values[i] = range_obj.sample()
                
                # Generate thresholds in order to ensure constraints
                # Pick random values but ensure proper spacing
                strong_bear = random.uniform(-70, -50)
                weak_bear = random.uniform(strong_bear + 5, -25)
                neutral_lower = random.uniform(weak_bear, -5)
                neutral_upper = random.uniform(max(neutral_lower + 5, 5), 20)
                weak_bull = random.uniform(max(neutral_upper, 25), 45)
                strong_bull = random.uniform(max(weak_bull + 5, 50), 70)
                
                # Assign the values
                values[param_indices['strong_bear_threshold']] = strong_bear
                values[param_indices['weak_bear_threshold']] = weak_bear
                values[param_indices['neutral_lower']] = neutral_lower
                values[param_indices['neutral_upper']] = neutral_upper
                values[param_indices['weak_bull_threshold']] = weak_bull
                values[param_indices['strong_bull_threshold']] = strong_bull
                
                # Create individual and verify
                individual = creator.Individual(values)
                params = self._individual_to_params(individual)
                if params.validate_constraints():
                    return individual
            
            # Fallback to guaranteed valid values
            return self._create_fallback_valid_individual()
        else:
            # For non-regime strategies, use the original approach
            max_attempts = 100
            
            for _ in range(max_attempts):
                values = []
                for i in range(len(self.param_names)):
                    param_name = self.param_names[i]
                    range_obj = self.param_ranges[param_name]
                    values.append(range_obj.sample())
                
                individual = creator.Individual(values)
                params = self._individual_to_params(individual)
                if params.validate_constraints():
                    return individual
            
            return self._create_fallback_valid_individual()
    
    def _create_fallback_valid_individual(self) -> List[float]:
        """
        Create a valid individual with guaranteed valid constraints.
        Used as fallback when random generation fails.
        """
        # Start with middle values for each range
        values = []
        for i in range(len(self.param_names)):
            param_name = self.param_names[i]
            range_obj = self.param_ranges[param_name]
            min_val, max_val = range_obj.get_bounds()
            values.append((min_val + max_val) / 2)
        
        # For regime parameters, ensure proper ordering
        param_indices = {name: i for i, name in enumerate(self.param_names)}
        
        # If this looks like regime strategy parameters, fix the ordering
        if 'strong_bull_threshold' in param_indices:
            # Set thresholds with proper spacing to guarantee ordering:
            # strong_bear < weak_bear <= neutral_lower < neutral_upper <= weak_bull < strong_bull
            values[param_indices['strong_bear_threshold']] = -60.0
            values[param_indices['weak_bear_threshold']] = -35.0
            values[param_indices['neutral_lower']] = -12.0
            values[param_indices['neutral_upper']] = 12.0
            values[param_indices['weak_bull_threshold']] = 35.0
            values[param_indices['strong_bull_threshold']] = 60.0
        
        return creator.Individual(values)
    
    def _mutate_bounded(self, individual: List[float]) -> Tuple[List[float]]:
        """
        Mutate an individual with bounds checking using range types.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual (tuple for DEAP compatibility)
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_indpb:
                param_name = self.param_names[i]
                range_obj = self.param_ranges[param_name]
                
                # Get bounds
                min_val, max_val = range_obj.get_bounds()
                
                # Gaussian mutation with 10% of range as sigma
                sigma = (max_val - min_val) * 0.1
                mutated_value = individual[i] + random.gauss(0, sigma)
                
                # Use range's clip method to ensure valid value
                individual[i] = range_obj.clip(mutated_value)
        
        # After mutation, ensure constraints are still satisfied
        # If not, try to fix them
        params = self._individual_to_params(individual)
        if not params.validate_constraints():
            # Try smaller mutations until valid
            for attempt in range(10):
                valid = True
                temp_individual = individual[:]
                
                for i in range(len(temp_individual)):
                    if random.random() < self.mutation_indpb:
                        param_name = self.param_names[i]
                        range_obj = self.param_ranges[param_name]
                        min_val, max_val = range_obj.get_bounds()
                        
                        # Use progressively smaller sigma
                        sigma = (max_val - min_val) * 0.1 * (0.5 ** attempt)
                        mutated_value = temp_individual[i] + random.gauss(0, sigma)
                        temp_individual[i] = range_obj.clip(mutated_value)
                
                params = self._individual_to_params(temp_individual)
                if params.validate_constraints():
                    individual[:] = temp_individual
                    break
                
        return (individual,)
        
    def _dummy_evaluate(self, individual: List[float]) -> Tuple[float]:
        """Dummy evaluation function (replaced during run)."""
        return (0.0,)
        
    def _individual_to_params(self, individual: List[float]) -> BaseStrategyParameters:
        """Convert individual representation to parameter object."""
        param_dict = {param: individual[i] for i, param in enumerate(self.param_names)}
        return self.param_class.from_dict(param_dict)
        
    def _params_to_individual(self, params: BaseStrategyParameters) -> List[float]:
        """Convert parameter object to individual representation."""
        param_dict = params.to_dict()
        return [param_dict[name] for name in self.param_names]
        
    def run(
        self,
        fitness_function: Callable[[T, pd.DataFrame], float],
        data: pd.DataFrame,
        verbose: bool = True,
        seed_params: Optional[List[T]] = None
    ) -> OptimizationResult[T]:
        """
        Run the genetic algorithm optimization.
        
        Args:
            fitness_function: Function that takes (parameters, data) and returns fitness
            data: Data to optimize on
            verbose: Whether to print progress
            seed_params: Optional list of parameter sets to seed the initial population
            
        Returns:
            OptimizationResult with best parameters and statistics
        """
        # Create evaluation wrapper
        def evaluate_wrapper(individual):
            params = self._individual_to_params(individual)
            
            # Constraints should already be satisfied, but double-check
            if not params.validate_constraints():
                # This should rarely happen with our new approach
                return (-1000.0,)
            
            fitness = fitness_function(cast(T, params), data)
            return (fitness,)
        
        # Register the actual evaluation function
        self.toolbox.register("evaluate", evaluate_wrapper)
        
        # Create initial population
        if seed_params:
            # Seed with provided parameters
            seeded_individuals = []
            for params in seed_params[:self.population_size // 2]:  # Use up to half the population
                individual = creator.Individual(self._params_to_individual(params))
                seeded_individuals.append(individual)
            
            # Fill the rest with random individuals
            remaining = self.population_size - len(seeded_individuals)
            random_individuals = self.toolbox.population(n=remaining)
            
            population = seeded_individuals + random_individuals
            
            if verbose:
                print(f"Seeded population with {len(seeded_individuals)} hall of fame members")
        else:
            # Create fully random population
            population = self.toolbox.population(n=self.population_size)
        
        # Set up statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Set up hall of fame (tracks best individuals)
        hof = tools.HallOfFame(1)
        
        # Run the genetic algorithm
        if verbose:
            print(f"Starting GA optimization with {self.population_size} individuals for {self.generations} generations")
            
        population, logbook = algorithms.eaSimple(
            population, 
            self.toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=verbose
        )
        
        # Extract results
        best_individual = hof[0]
        best_params = self._individual_to_params(best_individual)
        best_fitness = best_individual.fitness.values[0]
        
        # Convert logbook to generation stats
        generation_stats = []
        for record in logbook:
            generation_stats.append({
                'generation': record['gen'],
                'best_fitness': record['max'],
                'avg_fitness': record['avg'],
                'std_fitness': record['std']
            })
            
        if verbose:
            print(f"\nOptimization complete. Best fitness: {best_fitness:.4f}")
            print(f"Best parameters: {best_params.to_dict()}")
            
        return OptimizationResult[T](
            best_params=best_params,
            best_fitness=best_fitness,
            generation_stats=generation_stats
        )