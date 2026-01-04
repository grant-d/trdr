#!/usr/bin/env python3
"""SICA benchmark for MACD template strategy."""

from trdr.strategy.sica_runner import run_sica_benchmark

if __name__ == "__main__":
    run_sica_benchmark(
        strategy_module="trdr.strategy.macd_template.strategy",
        config_class="MACDConfig",
        strategy_class="MACDStrategy",
        symbol="crypto:ETH/USD",
        timeframe="4h",
        lookback=1000,  # ~167 days crypto (24/7), ~2.6 years stock
        position_pct=1.0,
    )
