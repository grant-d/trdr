#!/usr/bin/env python3
"""Multi-objective optimization for Ensemble strategy."""

from trdr.core import Duration, Symbol, Timeframe
from trdr.optimize import ParamBounds
from trdr.optimize.objectives import OBJECTIVES_FULL
from trdr.strategy.moo_runner import run_moo_benchmark

if __name__ == "__main__":
    run_moo_benchmark(
        strategy_module="trdr.strategy.ensemble.strategy",
        config_class="EnsembleConfig",
        strategy_class="EnsembleStrategy",
        symbol=Symbol.parse("crypto:ETH/USD"),
        timeframe=Timeframe.parse("15m"),
        lookback=Duration.parse("1M"),
        param_bounds=[
            ParamBounds("lookback_bars", lower=1000, upper=3000, dtype="int"),
            ParamBounds("retrain_every", lower=100, upper=500, dtype="int"),
            ParamBounds("n_estimators", lower=50, upper=200, dtype="int"),
            ParamBounds("max_tree_depth", lower=5, upper=20, dtype="int"),
            ParamBounds("min_samples_split", lower=20, upper=100, dtype="int"),
        ],
        objectives=OBJECTIVES_FULL,
    )
