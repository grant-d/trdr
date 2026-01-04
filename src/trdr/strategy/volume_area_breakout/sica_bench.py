#!/usr/bin/env python3
"""SICA benchmark for VolumeAreaBreakout strategy."""

from trdr.strategy.sica_runner import run_sica_benchmark

if __name__ == "__main__":
    run_sica_benchmark(
        strategy_module="trdr.strategy.volume_area_breakout.strategy",
        config_class="VolumeAreaBreakoutConfig",
        strategy_class="VolumeAreaBreakoutStrategy",
        default_symbol="stock:AAPL",
        default_position_pct=1.0,
    )
