#!/usr/bin/env python3
"""SICA benchmark script for MACD strategy.

DO NOT MODIFY THE TARGETS IN THIS FILE.
SICA uses this to measure strategy performance. Modify the strategy, not this.

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

from trdr.strategy.score import compute_composite_score


def reload_strategy():
    """Force reimport of strategy module."""
    mod_name = "trdr.strategy.macd_template.strategy"
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

    from trdr.strategy.macd_template.strategy import (
        MACDConfig,
        MACDStrategy,
    )
    return MACDConfig, MACDStrategy


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
        position_size_pct=1.0,  # 100% of capital per trade
    )
    engine = BacktestEngine(bt_config, strategy)
    return engine.run(bars)


def main():
    symbol = os.environ.get("BACKTEST_SYMBOL", "stock:AAPL")
    timeframe = os.environ.get("BACKTEST_TIMEFRAME", "1d")

    result = run_backtest(symbol, timeframe)

    # Get metrics
    sortino = result.sortino_ratio

    # Compute composite score
    score, details = compute_composite_score(
        profit_factor=result.profit_factor,
        sortino=sortino,
        pnl=result.total_pnl,
        win_rate=result.win_rate,
        max_drawdown=result.max_drawdown,
        total_trades=result.total_trades,
    )

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
    sortino_str = f"{sortino:.2f}" if sortino and sortino != float("inf") else str(sortino)
    print(f"Sortino: {sortino_str}")
    print(f"Max Drawdown: {result.max_drawdown:.1%}")
    print("=" * 50)

    # Print scoring breakdown
    print("\nScoring breakdown:")
    for d in details:
        print(f"  {d}")

    # COMPOSITE SCORING: "N passed, M failed" represents score percentage, NOT test counts
    # Example: score=0.034 (3.4%) → "3 passed, 97 failed"
    # This format satisfies SICA parser which expects pytest-style output
    passed = int(score * 100)
    failed = 100 - passed
    print(f"\n{passed} passed, {failed} failed")

    # Exit 0 if score >= 0.95 (near-perfect), else 1
    sys.exit(0 if score >= 0.95 else 1)


if __name__ == "__main__":
    main()
