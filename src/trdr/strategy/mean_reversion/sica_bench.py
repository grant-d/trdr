#!/usr/bin/env python3
"""SICA benchmark for MeanReversion strategy."""

from trdr.strategy.sica_runner import run_sica_benchmark

if __name__ == "__main__":
    run_sica_benchmark(
        strategy_module="trdr.strategy.mean_reversion.strategy",
        config_class="MeanReversionConfig",
        strategy_class="MeanReversionStrategy",
        default_symbol="crypto:BTC/USD",
        default_position_pct=0.5,
    )
