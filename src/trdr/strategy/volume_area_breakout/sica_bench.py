#!/usr/bin/env python3
"""SICA benchmark for VolumeAreaBreakout strategy."""

from trdr.core import Duration, Symbol, Timeframe
from trdr.strategy.sica_runner import run_sica_benchmark

if __name__ == "__main__":
    run_sica_benchmark(
        strategy_module="trdr.strategy.volume_area_breakout.strategy",
        config_class="VolumeAreaBreakoutConfig",
        strategy_class="VolumeAreaBreakoutStrategy",
        symbol=Symbol.parse("crypto:ETH/USD"),
        timeframe=Timeframe.parse("15m"),
        lookback=Duration.parse("750h"),  # 3000 bars @ 15m (45000min = 750h)
        position_pct=1.0,
    )
