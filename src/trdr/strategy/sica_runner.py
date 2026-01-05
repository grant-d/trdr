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
from typing import TYPE_CHECKING

from .types import DataRequirement

if TYPE_CHECKING:
    from ..core import Symbol

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from trdr.strategy.targets import score_result


def get_primary_requirement(requirements: list[DataRequirement]) -> DataRequirement:
    """Find the primary requirement. Raises if not exactly one."""
    primaries = [r for r in requirements if r.role == "primary"]
    if len(primaries) != 1:
        raise ValueError(f"Expected exactly 1 primary requirement, got {len(primaries)}")
    return primaries[0]


async def _get_bars(strategy) -> tuple[dict[str, list], DataRequirement]:
    """Fetch bars based on strategy requirements.

    Args:
        strategy: Strategy instance with get_data_requirements()

    Returns:
        Tuple of (bars dict keyed by "symbol:tf", primary requirement)
    """
    from trdr.backtest import align_feeds
    from trdr.core import load_config, Feed
    from trdr.data import AlpacaDataClient

    config = load_config()
    client = AlpacaDataClient(config.alpaca, Path(project_root / "data/cache"))

    # Get requirements from strategy
    requirements = strategy.get_data_requirements()
    primary = get_primary_requirement(requirements)

    # Apply BACKTEST_TIMEFRAME override if specified
    timeframe_override = os.environ.get("BACKTEST_TIMEFRAME")
    if timeframe_override:
        # Replace primary's timeframe
        requirements = [
            (
                DataRequirement(r.symbol, timeframe_override, r.lookback, r.role)
                if r.role == "primary"
                else r
            )
            for r in requirements
        ]
        primary = get_primary_requirement(requirements)

    # Fetch all feeds
    bars_dict = await client.get_bars_multi(requirements)
    primary_bars = bars_dict[primary.key]

    # Align informative feeds to primary
    aligned = {primary.key: primary_bars}
    for req in requirements:
        if req.role != "primary":
            aligned[req.key] = align_feeds(primary_bars, bars_dict[req.key])

    return aligned, primary


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
    symbol: "Symbol",
    timeframe: "Timeframe",
    lookback: "Duration",
    position_pct: float,
) -> None:
    """Run SICA benchmark for a strategy.

    Args:
        strategy_module: Full module path (e.g., "trdr.strategy.volume_area_breakout.strategy")
        config_class: Config class name (e.g., "VolumeAreaBreakoutConfig")
        strategy_class: Strategy class name (e.g., "VolumeAreaBreakoutStrategy")
        symbol: Symbol object (can be overridden by BACKTEST_SYMBOL env var)
        timeframe: Timeframe (can be overridden by BACKTEST_TIMEFRAME env var)
        lookback: Duration
        position_pct: Position size as fraction of capital (1.0 = 100%)
    """
    from trdr.backtest import PaperExchange, PaperExchangeConfig
    from trdr.core import Symbol, Timeframe, Feed

    # Env vars can override specific code-driven values
    env_symbol = os.environ.get("BACKTEST_SYMBOL")
    if env_symbol:
        symbol = Symbol.parse(env_symbol)
    tf_override = os.environ.get("BACKTEST_TIMEFRAME")
    if tf_override:
        timeframe = Timeframe.parse(tf_override)

    # Resolve Duration to bar count
    lookback_bars = lookback.to_bars(timeframe, symbol)

    # Load strategy
    Config, Strategy = _reload_strategy(strategy_module, config_class, strategy_class)

    # Create strategy instance (needed to get data requirements)
    config = Config(symbol=symbol, timeframe=timeframe, lookback=lookback)
    strategy = Strategy(config)

    # Get bars using strategy's data requirements
    bars, primary = asyncio.run(_get_bars(strategy))

    # Calculate buy-hold return using primary bars
    initial_capital = 10000
    primary_bars = bars[primary.key]
    if primary_bars and len(primary_bars) >= 2:
        buyhold_return = (primary_bars[-1].close / primary_bars[0].close) - 1
    else:
        buyhold_return = 0.0

    bt_config = PaperExchangeConfig(
        symbol=primary.symbol,
        initial_capital=initial_capital,
        default_position_pct=position_pct,
        primary_feed=primary.key,
    )
    engine = PaperExchange(bt_config, strategy)
    result = engine.run(bars)

    # Compute score
    score, details = score_result(result, buyhold_return)

    # Print summary
    print("=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)
    print(f"Symbol: {primary.symbol}")
    print(f"Timeframe: {primary.timeframe}")
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
