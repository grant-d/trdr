#!/usr/bin/env python3
"""SICA benchmark for MACD template strategy."""

from trdr.core import Duration, Timeframe
from trdr.strategy.sica_runner import run_sica_benchmark

if __name__ == "__main__":
    run_sica_benchmark(
        strategy_module="trdr.strategy.macd_template.strategy",
        config_class="MACDConfig",
        strategy_class="MACDStrategy",
        symbol="crypto:ETH/USD",
        timeframe=Timeframe.parse("4h"),
        lookback=Duration.parse("6M"),
        position_pct=1.0,
    )
