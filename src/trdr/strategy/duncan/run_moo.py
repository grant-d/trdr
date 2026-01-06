#!/usr/bin/env python3
"""Multi-objective optimization for Duncan Trailer v2 strategy."""

from trdr.core import Duration, Symbol, Timeframe
from trdr.optimize import ParamBounds
from trdr.optimize.objectives import OBJECTIVES_FULL
from trdr.strategy.moo_runner import run_moo_benchmark

if __name__ == "__main__":
    run_moo_benchmark(
        strategy_module="trdr.strategy.duncan.strategy",
        config_class="DuncanConfig",
        strategy_class="DuncanStrategy",
        symbol=Symbol.parse("crypto:ETH/USD"),
        timeframe=Timeframe.parse("15m"),
        lookback=Duration.parse("6M"),
        param_bounds=[
            ParamBounds("atr_period", lower=10, upper=30, dtype="int"),
            ParamBounds("multiplier", lower=10.0, upper=30.0),
            ParamBounds("rsi_weight", lower=0.3, upper=0.9),
            ParamBounds("min_trail_factor", lower=0.2, upper=0.7),
            ParamBounds("max_trail_factor", lower=0.7, upper=1.0),
            ParamBounds("grid_count", lower=1.0, upper=5.0),
        ],
        objectives=OBJECTIVES_FULL,
    )
