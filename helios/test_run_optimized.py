#!/usr/bin/env python3
"""Test that run-optimized works without data parameter"""

import json
import os
from pathlib import Path

# Create a test optimized parameters file with absolute data path
test_params = {
    "parameters": {
        "weight_trend": 0.5,
        "weight_volatility": 0.3,
        "weight_exhaustion": 0.2,
        "lookback_int": 20.0,
        "strong_bull_threshold": 25.0,
        "weak_bull_threshold": 10.0,
        "neutral_threshold_upper": 5.0,
        "neutral_threshold_lower": -5.0,
        "weak_bear_threshold": -10.0,
        "strong_bear_threshold": -25.0,
        "stop_loss_multiplier_strong": 2.0,
        "stop_loss_multiplier_weak": 1.5,
        "entry_step_size": 0.3,
        "max_position_pct": 0.9
    },
    "fitness": 18.5,
    "fitness_metric": "sortino",
    "allow_shorts": False,
    "data_file": str(Path("../data/MSFT.csv").resolve()),  # Absolute path
    "dollar_threshold": 100000000,
    "population_size": 10,
    "generations": 5
}

# Save test parameters
test_file = "test_params.json"
with open(test_file, "w") as f:
    json.dump(test_params, f, indent=2)

print(f"Created test parameters file: {test_file}")
print(f"Data file path saved: {test_params['data_file']}")
print("\nNow you can run:")
print(f"  python main.py run-optimized --params {test_file}")
print("\nWithout needing to specify --data!")