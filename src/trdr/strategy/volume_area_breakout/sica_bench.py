#!/usr/bin/env python3
"""SICA benchmark script for VolumeAreaBreakout strategy.

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

from trdr.strategy.targets import score_result


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


async def get_bars(symbol: str, timeframe: str, lookback: int):
    """Fetch bars from market data client."""
    from trdr.backtest import parse_timeframe
    from trdr.core import load_config
    from trdr.data import MarketDataClient
    from trdr.data.market import Symbol

    config = load_config()
    client = MarketDataClient(config.alpaca, Path(project_root / "data/cache"))
    sym = Symbol.parse(symbol)

    # Parse timeframe
    tf = parse_timeframe(timeframe)

    return await client.get_bars(sym, lookback, tf)


def run_backtest(symbol: str, timeframe: str, lookback: int):
    """Run backtest and return results plus buy-hold info."""
    # Force reimport to pick up code changes
    Config, Strategy = reload_strategy()

    from trdr.backtest import PaperExchange, PaperExchangeConfig

    # Get bars
    bars = asyncio.run(get_bars(symbol, timeframe, lookback))

    # Calculate buy-hold return
    initial_capital = 10000
    if bars and len(bars) >= 2:
        buyhold_return = (bars[-1].close / bars[0].close) - 1
    else:
        buyhold_return = 0.0

    # Create strategy
    config = Config(symbol=symbol, timeframe=timeframe)
    strategy = Strategy(config)

    # Run backtest with PaperExchange
    bt_config = PaperExchangeConfig(
        symbol=symbol,
        initial_capital=initial_capital,
        default_position_pct=1.0,  # 100% of capital per trade
    )
    engine = PaperExchange(bt_config, strategy)
    result = engine.run(bars)

    return result, initial_capital, buyhold_return, bars


def main():
    symbol = os.environ.get("BACKTEST_SYMBOL", "stock:AAPL")
    timeframe = os.environ.get("BACKTEST_TIMEFRAME", "1d")
    lookback = int(os.environ.get("BACKTEST_LOOKBACK", "1000"))

    result, initial_capital, buyhold_return, bars = run_backtest(symbol, timeframe, lookback)

    # Compute composite score (uses metrics from result directly)
    score, details = score_result(result, buyhold_return)

    # Print summary
    print("=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)
    print(f"Symbol: {symbol}")
    print(f"Trades: {result.total_trades} ({result.trades_per_year:.0f}/yr)")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total P&L: ${result.total_pnl:.2f}")
    print(f"CAGR: {result.cagr:.1%}" if result.cagr else "CAGR: N/A")
    print(f"Buy-Hold P&L: ${initial_capital * buyhold_return:,.2f}")
    print(f"Sharpe: {result.sharpe_ratio:.2f}" if result.sharpe_ratio else "Sharpe: N/A")
    print(f"Sortino: {result.sortino_ratio:.2f}" if result.sortino_ratio else "Sortino: N/A")
    print(f"Calmar: {result.calmar_ratio:.2f}" if result.calmar_ratio else "Calmar: N/A")
    print(f"Max Drawdown: {result.max_drawdown:.1%}")
    print("=" * 50)

    # Print scoring breakdown
    print("\nScoring breakdown:")
    for d in details:
        print(f"  {d}")

    # COMPOSITE SCORING: "N passed, M failed" represents score (3 decimal precision)
    # Example: score=0.849 → "849 passed, 151 failed" → hook calculates 849/1000 = 0.849
    # This format satisfies SICA parser which expects pytest-style output
    passed = int(score * 1000)
    failed = 1000 - passed
    print(f"\n{passed} passed, {failed} failed")

    # Exit 0 if score >= 0.95 (near-perfect), else 1
    sys.exit(0 if score >= 0.95 else 1)


if __name__ == "__main__":
    main()
