#!/usr/bin/env python3
"""Multi-objective optimization for Trailing Grid strategy."""

from trdr.core import Duration, Symbol, Timeframe
from trdr.optimize import ParamBounds
from trdr.optimize.objectives import OBJECTIVES_FULL
from trdr.strategy.moo_runner import run_moo_benchmark

if __name__ == "__main__":
    run_moo_benchmark(
        strategy_module="trdr.strategy.trailing_grid.strategy",
        config_class="TrailingGridConfig",
        strategy_class="TrailingGridStrategy",
        symbol=Symbol.parse("crypto:ETH/USD"),
        timeframe=Timeframe.parse("15m"),
        lookback=Duration.parse("750h"),
        param_bounds=[
            ParamBounds("grid_width_pct", lower=0.02, upper=0.10),
            ParamBounds("trail_pct", lower=0.01, upper=0.05),
            ParamBounds("max_dca", lower=1, upper=5, dtype="int"),
            ParamBounds("downtrend_bars", lower=1, upper=5, dtype="int"),
            ParamBounds("stop_loss_multiplier", lower=1.0, upper=4.0),
            ParamBounds("sell_target_multiplier", lower=0.5, upper=2.0),
        ],
        objectives=OBJECTIVES_FULL,  # cagr, calmar, sortino, pf, wr, trades, dd, alpha
    )
