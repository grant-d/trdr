#!/usr/bin/env python3
"""
Genetic Algorithm optimizer using DEAP for trading strategy parameters.
"""
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from typing import Dict, List, Tuple, Callable, Any
import random
from dataclasses import dataclass


@dataclass
class ParamRange:
    """Define parameter range for optimization."""

    name: str
    min_val: float
    max_val: float
    type: type = float  # float or int


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for trading strategy parameters using DEAP.
    """

    def __init__(
        self,
        param_ranges: List[ParamRange],
        population_size: int = 50,
        generations: int = 20,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        tournament_size: int = 3,
    ):
        self.param_ranges = param_ranges
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size

        # Setup DEAP
        self._setup_deap()

    def _setup_deap(self):
        """Initialize DEAP framework."""
        # Create fitness and individual classes
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create toolbox
        self.toolbox = base.Toolbox()

        # Register attribute generators for each parameter
        for i, param in enumerate(self.param_ranges):
            if param.type == int:
                self.toolbox.register(
                    f"attr_{i}", random.randint, param.min_val, param.max_val
                )
            else:
                self.toolbox.register(
                    f"attr_{i}", random.uniform, param.min_val, param.max_val
                )

        # Register individual and population
        attr_list = [
            getattr(self.toolbox, f"attr_{i}") for i in range(len(self.param_ranges))
        ]
        self.toolbox.register(
            "individual", tools.initCycle, creator.Individual, attr_list, n=1
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Register genetic operators
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self._mutate_bounded)
        self.toolbox.register(
            "select", tools.selTournament, tournsize=self.tournament_size
        )

    def _mutate_bounded(self, individual):
        """Custom mutation that respects parameter bounds."""
        for i, param in enumerate(self.param_ranges):
            if random.random() < 0.3:  # 30% chance to mutate each gene
                if param.type == int:
                    individual[i] = random.randint(param.min_val, param.max_val)
                else:
                    # Gaussian mutation
                    sigma = (param.max_val - param.min_val) * 0.1
                    individual[i] += random.gauss(0, sigma)
                    individual[i] = max(
                        param.min_val, min(param.max_val, individual[i])
                    )
        return (individual,)

    def _decode_individual(self, individual: List[float]) -> Dict[str, Any]:
        """Convert individual to parameter dictionary."""
        params = {}
        for i, (gene, param_range) in enumerate(zip(individual, self.param_ranges)):
            if param_range.type == int:
                params[param_range.name] = int(gene)
            else:
                params[param_range.name] = float(gene)
        return params

    def optimize(
        self, fitness_func: Callable[[Dict[str, Any]], float], verbose: bool = True
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run genetic optimization.

        Args:
            fitness_func: Function that takes params dict and returns fitness score
            verbose: Print progress

        Returns:
            Tuple of (best_params, best_fitness)
        """

        # Register fitness function
        def evaluate(individual):
            params = self._decode_individual(individual)
            try:
                fitness = fitness_func(params)
                return (fitness,) if not np.isnan(fitness) else (-1000,)
            except Exception as e:
                if verbose:
                    print(f"Error evaluating individual: {e}")
                return (-1000,)

        self.toolbox.register("evaluate", evaluate)

        # Create initial population
        population = self.toolbox.population(n=self.population_size)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)

        # Hall of fame to track best individuals
        hof = tools.HallOfFame(1)

        # Run evolution
        population, logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=verbose,
        )

        # Get best individual
        best_individual = hof[0]
        best_params = self._decode_individual(best_individual)
        best_fitness = best_individual.fitness.values[0]

        if verbose:
            print(f"\nBest parameters found:")
            for name, value in best_params.items():
                print(f"  {name}: {value}")
            print(f"Best fitness: {best_fitness:.4f}")

        return best_params, best_fitness


def example_matrix_profile_optimization():
    """Example: Optimize Matrix Profile strategy parameters."""
    from matrix_profile_strategy import MatrixProfileStrategy
    from strategy_evaluator import StrategyEvaluator

    # Define parameter ranges
    param_ranges = [
        ParamRange("window_size", 20, 100, int),
        ParamRange("buy_threshold", 0.1, 0.5, float),
        ParamRange("sell_threshold", 0.1, 0.5, float),
        ParamRange("lookback_periods", 100, 500, int),
        ParamRange("stop_loss_pct", 0.005, 0.02, float),
        ParamRange("take_profit_pct", 0.01, 0.04, float),
    ]

    # Create optimizer
    optimizer = GeneticOptimizer(
        param_ranges=param_ranges, population_size=30, generations=10
    )

    # Load data for fitness evaluation
    df = pd.read_csv("data/sol_usd_1m.transform.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Create strategy and evaluator instances
    strategy = MatrixProfileStrategy()
    evaluator = StrategyEvaluator(
        initial_capital=100000, position_size_pct=0.02, commission_pct=0.001
    )

    # Define fitness function
    def fitness_func(params):
        """Calculate Sharpe ratio for given parameters."""
        try:
            # Update strategy parameters
            strategy.set_parameters(params)

            # Generate signals
            signals = strategy.generate_signals(df)

            # Run backtest
            results = evaluator.backtest(
                df,
                signals,
                stop_loss_pct=params["stop_loss_pct"],
                take_profit_pct=params["take_profit_pct"],
            )

            # Use Sharpe ratio as fitness
            sharpe = results["metrics"].get("sharpe", -1000)

            # Penalize if no trades
            if results["metrics"].get("total_trades", 0) < 5:
                sharpe *= 0.1

            return max(sharpe, -1000)

        except Exception as e:
            print(f"Error in fitness evaluation: {e}")
            return -1000

    # Run optimization
    best_params, best_fitness = optimizer.optimize(fitness_func)

    return best_params, best_fitness


if __name__ == "__main__":
    print("Testing Genetic Optimizer with Matrix Profile strategy...")
    best_params, best_fitness = example_matrix_profile_optimization()
    print(f"\nOptimization complete!")
    print(f"Best Sharpe ratio: {best_fitness:.4f}")
