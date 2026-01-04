"""Base SICA benchmark runner for strategies.

Concrete sica_bench.py files specify strategy module and defaults. All common logic here.

Usage in concrete sica_bench.py:
    from trdr.strategy.sica_runner import run_sica_benchmark
    run_sica_benchmark(
        strategy_module="trdr.strategy.volume_area_breakout.strategy",
        config_class="VolumeAreaBreakoutConfig",
        strategy_class="VolumeAreaBreakoutStrategy",
        default_symbol="stock:AAPL",
        default_position_pct=1.0,
    )
"""

import asyncio
import importlib
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from trdr.strategy.targets import score_result


async def _get_bars(symbol: str, timeframe: str, lookback: int):
    """Fetch bars from market data client."""
    from trdr.backtest import parse_timeframe
    from trdr.core import load_config
    from trdr.data import MarketDataClient
    from trdr.data.market import Symbol

    config = load_config()
    client = MarketDataClient(config.alpaca, Path(project_root / "data/cache"))
    sym = Symbol.parse(symbol)
    tf = parse_timeframe(timeframe)

    return await client.get_bars(sym, lookback, tf)


def _reload_strategy(module_name: str, config_class: str, strategy_class: str):
    """Force reimport of strategy module.

    Uses dynamic import to avoid hardcoding strategy imports in base class.
    Example: _reload_strategy("trdr.strategy.foo.strategy", "FooConfig", "FooStrategy")
    is equivalent to:
        from trdr.strategy.foo.strategy import FooConfig, FooStrategy
        return FooConfig, FooStrategy
    """
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

    mod = importlib.import_module(module_name)
    return getattr(mod, config_class), getattr(mod, strategy_class)


def run_sica_benchmark(
    strategy_module: str,
    config_class: str,
    strategy_class: str,
    default_symbol: str = "stock:AAPL",
    default_timeframe: str = "1d",
    default_lookback: int = 1000,
    default_position_pct: float = 1.0,
) -> None:
    """Run SICA benchmark for a strategy.

    Args:
        strategy_module: Full module path (e.g., "trdr.strategy.volume_area_breakout.strategy")
        config_class: Config class name (e.g., "VolumeAreaBreakoutConfig")
        strategy_class: Strategy class name (e.g., "VolumeAreaBreakoutStrategy")
        default_symbol: Default trading symbol
        default_timeframe: Default timeframe
        default_lookback: Default lookback bars
        default_position_pct: Default position size as fraction of capital
    """
    from trdr.backtest import PaperExchange, PaperExchangeConfig

    # Get params from environment or defaults
    symbol = os.environ.get("BACKTEST_SYMBOL", default_symbol)
    timeframe = os.environ.get("BACKTEST_TIMEFRAME", default_timeframe)
    lookback = int(os.environ.get("BACKTEST_LOOKBACK", str(default_lookback)))

    # Load strategy
    Config, Strategy = _reload_strategy(strategy_module, config_class, strategy_class)

    # Get bars
    bars = asyncio.run(_get_bars(symbol, timeframe, lookback))

    # Calculate buy-hold return
    initial_capital = 10000
    if bars and len(bars) >= 2:
        buyhold_return = (bars[-1].close / bars[0].close) - 1
    else:
        buyhold_return = 0.0

    # Create and run strategy
    config = Config(symbol=symbol, timeframe=timeframe)
    strategy = Strategy(config)

    bt_config = PaperExchangeConfig(
        symbol=symbol,
        initial_capital=initial_capital,
        default_position_pct=default_position_pct,
    )
    engine = PaperExchange(bt_config, strategy)
    result = engine.run(bars)

    # Compute score
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

    # Scoring breakdown
    print("\nScoring breakdown:")
    for d in details:
        print(f"  {d}")

    # SICA parser output (3 decimal precision)
    passed = round(score * 1000)
    failed = 1000 - passed
    print(f"\n{passed} passed, {failed} failed")

    sys.exit(0 if score >= 0.95 else 1)
