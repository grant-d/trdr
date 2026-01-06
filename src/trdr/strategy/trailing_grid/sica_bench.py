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
        timeframe=Timeframe.parse("15m"),
        lookback=Duration.parse("750h"),  # 3000 bars @ 15m
        position_pct=0.5,  # 50% per entry (max 100% with 2 DCA levels)
    )
