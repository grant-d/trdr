#!/usr/bin/env python3
"""SICA benchmark for MACD template strategy."""

from trdr.strategy.sica_runner import run_sica_benchmark

if __name__ == "__main__":
    run_sica_benchmark(
        strategy_module="trdr.strategy.macd_template.strategy",
        config_class="MACDConfig",
        strategy_class="MACDStrategy",
        default_symbol="stock:AAPL",
        default_position_pct=1.0,
    )
