#!/usr/bin/env python3
"""SICA benchmark for Duncan Trailer v2 strategy."""

from trdr.core import Duration, Symbol, Timeframe
from trdr.strategy.sica_runner import run_sica_benchmark

if __name__ == "__main__":
    run_sica_benchmark(
        strategy_module="trdr.strategy.duncan.strategy",
        config_class="DuncanConfig",
        strategy_class="DuncanStrategy",
        symbol=Symbol.parse("crypto:ETH/USD"),
        timeframe=Timeframe.parse("15m"),
        lookback=Duration.parse("6M"),
        position_pct=1.0,
    )
