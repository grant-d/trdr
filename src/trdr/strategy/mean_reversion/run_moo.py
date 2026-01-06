#!/usr/bin/env python3
"""Multi-objective optimization for MeanReversion strategy."""

from trdr.core import Duration, Symbol, Timeframe
from trdr.optimize import ParamBounds
from trdr.optimize.objectives import OBJECTIVES_FULL
from trdr.strategy.moo_runner import run_moo_benchmark

if __name__ == "__main__":
    run_moo_benchmark(
        strategy_module="trdr.strategy.mean_reversion.strategy",
        config_class="MeanReversionConfig",
        strategy_class="MeanReversionStrategy",
        symbol=Symbol.parse("crypto:BTC/USD"),
        timeframe=Timeframe.parse("1d"),
        lookback=Duration.parse("3y"),
        position_pct=0.5,
        param_bounds=[
            ParamBounds("breakout_period", lower=5, upper=20, dtype="int"),
            ParamBounds("volume_multiplier", lower=1.0, upper=2.0),
            ParamBounds("stop_loss_atr_mult", lower=1.0, upper=4.0),
            ParamBounds("trailing_stop_atr_mult", lower=1.5, upper=5.0),
            ParamBounds("max_holding_days", lower=20, upper=100, dtype="int"),
        ],
        objectives=OBJECTIVES_FULL,
    )
