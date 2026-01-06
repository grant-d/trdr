#!/usr/bin/env python3
"""SICA benchmark for Trailing Grid strategy."""

from trdr.core import Duration, Symbol, Timeframe
from trdr.strategy.sica_runner import run_sica_benchmark

if __name__ == "__main__":
    run_sica_benchmark(
        strategy_module="trdr.strategy.trailing_grid.strategy",
        config_class="TrailingGridConfig",
        strategy_class="TrailingGridStrategy",
        symbol=Symbol.parse("crypto:ETH/USD"),
        timeframe=Timeframe.parse("5m"),
        lookback=Duration.parse("250h"),  # ~3000 bars @ 5m
        position_pct=0.5,  # 50% per entry (DCA=2)
    )
