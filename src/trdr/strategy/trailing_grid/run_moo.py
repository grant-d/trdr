#!/usr/bin/env python3
"""Multi-objective optimization for Trailing Grid strategy."""

from trdr.core import Duration, Symbol, Timeframe
from trdr.optimize import ParamBounds
from trdr.optimize.objectives import OBJECTIVES_CORE

# Add total_trades to objectives for frequency optimization
OBJECTIVES_WITH_TRADES = ["sharpe", "max_drawdown", "profit_factor", "total_trades"]
from trdr.strategy.moo_runner import run_moo_benchmark

if __name__ == "__main__":
    run_moo_benchmark(
        strategy_module="trdr.strategy.trailing_grid.strategy",
        config_class="TrailingGridConfig",
        strategy_class="TrailingGridStrategy",
        symbol=Symbol.parse("crypto:ETH/USD"),
        timeframe=Timeframe.parse("5m"),
        lookback=Duration.parse("250h"),  # ~3000 bars @ 5m
        param_bounds=[
            # Centered around SICA-optimized values: 0.018, 0.014, 2, 1.35, 1.25
            ParamBounds("grid_width_pct", lower=0.012, upper=0.025),
            ParamBounds("trail_pct", lower=0.008, upper=0.020),
            ParamBounds("max_dca", lower=1, upper=3, dtype="int"),
            ParamBounds("downtrend_bars", lower=1, upper=4, dtype="int"),
            ParamBounds("stop_loss_multiplier", lower=1.0, upper=2.0),
            ParamBounds("sell_target_multiplier", lower=0.8, upper=1.5),
        ],
        objectives=OBJECTIVES_WITH_TRADES,  # sharpe, max_drawdown, profit_factor, total_trades
        min_trades=10,  # Require at least 10 trades
    )
