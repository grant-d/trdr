#!/usr/bin/env python3
"""Multi-objective optimization for VolumeAreaBreakout strategy."""

from trdr.core import Duration, Symbol, Timeframe
from trdr.optimize import ParamBounds
from trdr.optimize.objectives import OBJECTIVES_FULL
from trdr.strategy.moo_runner import run_moo_benchmark

if __name__ == "__main__":
    run_moo_benchmark(
        strategy_module="trdr.strategy.volume_area_breakout.strategy",
        config_class="VolumeAreaBreakoutConfig",
        strategy_class="VolumeAreaBreakoutStrategy",
        symbol=Symbol.parse("crypto:ETH/USD"),
        timeframe=Timeframe.parse("15m"),
        lookback=Duration.parse("750h"),
        param_bounds=[
            ParamBounds("atr_threshold", lower=1.0, upper=4.0),
            ParamBounds("stop_loss_multiplier", lower=1.0, upper=3.0),
        ],
        objectives=OBJECTIVES_FULL,
    )
