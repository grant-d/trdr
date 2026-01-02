#!/usr/bin/env python3
"""SICA benchmark script for VolumeAreaBreakout strategy.

DO NOT MODIFY THE PASS/FAIL CRITERIA IN THIS FILE.
SICA uses this to measure strategy performance. Modify the strategy, not this.

Runs backtest and exits with code based on pass/fail.
Exit 0 = all tests pass, Exit 1 = any test fails.

Usage:
  BACKTEST_SYMBOL=stock:AAPL BACKTEST_TIMEFRAME=1d python sica_bench.py
"""

import asyncio
import importlib
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def reload_strategy():
    """Force reimport of strategy module."""
    mod_name = "trdr.strategy.volume_area_breakout.strategy"
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

    from trdr.strategy.volume_area_breakout.strategy import (
        VolumeAreaBreakoutConfig,
        VolumeAreaBreakoutStrategy,
    )
    return VolumeAreaBreakoutConfig, VolumeAreaBreakoutStrategy


async def get_bars(symbol: str, timeframe: str, lookback: int = 10000):
    """Fetch bars from market data client."""
    from trdr.core import load_config
    from trdr.data import MarketDataClient
    from trdr.data.market import Symbol
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    config = load_config()
    client = MarketDataClient(config.alpaca, Path(project_root / "data/cache"))
    sym = Symbol.parse(symbol)

    # Parse timeframe
    tf_map = {
        "1d": TimeFrame(1, TimeFrameUnit.Day),
        "d": TimeFrame(1, TimeFrameUnit.Day),
        "day": TimeFrame(1, TimeFrameUnit.Day),
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
        "4h": TimeFrame(4, TimeFrameUnit.Hour),
    }
    tf = tf_map.get(timeframe.lower(), TimeFrame(1, TimeFrameUnit.Day))

    return await client.get_bars(sym, lookback, tf)


def run_backtest(symbol: str, timeframe: str):
    """Run backtest and return results."""
    # Force reimport to pick up code changes
    Config, Strategy = reload_strategy()

    from trdr.backtest.backtest_engine import BacktestConfig, BacktestEngine

    # Get bars
    bars = asyncio.run(get_bars(symbol, timeframe))

    # Create strategy
    config = Config(symbol=symbol, timeframe=timeframe)
    strategy = Strategy(config)

    # Run backtest
    bt_config = BacktestConfig(
        symbol=symbol,
        initial_capital=10000,
        position_size=1.0,
    )
    engine = BacktestEngine(bt_config, strategy)
    return engine.run(bars)


def main():
    symbol = os.environ.get("BACKTEST_SYMBOL", "stock:AAPL")
    timeframe = os.environ.get("BACKTEST_TIMEFRAME", "1d")

    result = run_backtest(symbol, timeframe)

    # Print summary
    print("=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total P&L: ${result.total_pnl:.2f}")
    sortino = result.sortino_ratio  # Can be None, inf, or number
    sortino_str = f"{sortino:.2f}" if sortino and sortino != float("inf") else str(sortino)
    print(f"Sortino: {sortino_str}")
    print("=" * 50)

    # Check pass/fail criteria (DO NOT LOWER THESE - improve the strategy)
    failures = []
    if result.profit_factor <= 1.5:
        failures.append(f"FAIL: Profit factor {result.profit_factor:.2f} <= 1.5")
    if sortino is not None and sortino != float("inf") and sortino <= 1.0:
        failures.append(f"FAIL: Sortino {sortino:.2f} <= 1.0")
    if result.total_pnl <= 1000:
        failures.append(f"FAIL: P&L ${result.total_pnl:.2f} <= $1000 (10% min over test period)")
    if result.win_rate < 0.40:
        failures.append(f"FAIL: Win rate {result.win_rate:.1%} < 40%")
    if result.max_drawdown > 0.20:
        failures.append(f"FAIL: Max drawdown {result.max_drawdown:.1%} > 20%")

    if failures:
        print("\nFailed tests:")
        for f in failures:
            print(f"  {f}")
        # MUST output pytest-style summary for SICA parser
        print(f"\n{len(failures)} failed, 0 passed")
        sys.exit(1)
    else:
        print("\n1 passed, 0 failed")
        sys.exit(0)


if __name__ == "__main__":
    main()
