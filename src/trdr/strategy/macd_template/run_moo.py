#!/usr/bin/env python3
"""Multi-objective optimization for MACD template strategy."""

from trdr.core import Duration, Symbol, Timeframe
from trdr.optimize import ParamBounds
from trdr.optimize.objectives import OBJECTIVES_FULL
from trdr.strategy.moo_runner import run_moo_benchmark

if __name__ == "__main__":
    run_moo_benchmark(
        strategy_module="trdr.strategy.macd_template.strategy",
        config_class="MACDConfig",
        strategy_class="MACDStrategy",
        symbol=Symbol.parse("crypto:ETH/USD"),
        timeframe=Timeframe.parse("4h"),
        lookback=Duration.parse("6M"),
        param_bounds=[
            ParamBounds("fast_period", lower=8, upper=20, dtype="int"),
            ParamBounds("slow_period", lower=20, upper=40, dtype="int"),
            ParamBounds("signal_period", lower=5, upper=15, dtype="int"),
            ParamBounds("stop_loss_pct", lower=0.01, upper=0.05),
        ],
        objectives=OBJECTIVES_FULL,
    )
