"""
Test script to compare old vs new fitness function approaches
"""

import pandas as pd
import numpy as np
from optimization import GeneticAlgorithm
from data_processing import create_dollar_bars, prepare_data

# Load test data
print("Loading data...")
df = pd.read_csv('/Users/grantdickinson/repos/trdr/data/BTCUSD-feed.csv', parse_dates=True, index_col=0)
df.columns = df.columns.str.lower()
df = prepare_data(df)

# Create dollar bars
print("Creating dollar bars...")
df = create_dollar_bars(df, 1000000)

# Create a simple parameter set to test
test_params = {
    'weight_trend': 0.4,
    'weight_volatility': 0.3,
    'weight_exhaustion': 0.3,
    'lookback_int': 20,
    'max_position_pct': 0.95,
    'min_position_pct': 0.1
}

# Create GA instance
ga = GeneticAlgorithm(
    parameter_ranges={
        'weight_trend': (0.1, 0.7),
        'weight_volatility': (0.1, 0.5),
        'weight_exhaustion': (0.1, 0.5),
        'lookback_int': (10, 50),
        'max_position_pct': (0.8, 1.0),
        'min_position_pct': (0.05, 0.2)
    },
    population_size=10,
    generations=5,
    fitness_metric='sortino'
)

# Create individual
from optimization import Individual
individual = Individual(genes=test_params)

# Evaluate fitness with debugging
print("\nEvaluating fitness with updated function...")

# Temporarily modify evaluate_fitness to print details
original_evaluate_fitness = ga.evaluate_fitness

def debug_evaluate_fitness(individual, train_data):
    fitness = original_evaluate_fitness(individual, train_data)
    print(f"  Raw fitness score: {fitness:.4f}")
    return fitness

ga.evaluate_fitness = debug_evaluate_fitness
fitness = ga.evaluate_fitness(individual, df)
print(f"Final fitness score: {fitness:.4f}")

# Restore original
ga.evaluate_fitness = original_evaluate_fitness

# Now let's run a quick optimization
print("\nRunning quick GA optimization (10 pop, 5 gen)...")
best_individual, fitness_history = ga.optimize(df, verbose=True)

print(f"\nBest fitness achieved: {best_individual.fitness:.4f}")
print("Best parameters:")
for param, value in best_individual.genes.items():
    print(f"  {param}: {value:.4f}")