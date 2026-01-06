#!/usr/bin/env python3
"""SICA benchmark for Ensemble random forest strategy."""

from trdr.core import Duration, Symbol, Timeframe
from trdr.strategy.sica_runner import run_sica_benchmark

if __name__ == "__main__":
    run_sica_benchmark(
        strategy_module="trdr.strategy.ensemble.strategy",
        config_class="EnsembleConfig",
        strategy_class="EnsembleStrategy",
        symbol=Symbol.parse("crypto:ETH/USD"),
        timeframe=Timeframe.parse("15m"),
        lookback=Duration.parse("1M"),
        position_pct=1.0,
    )
