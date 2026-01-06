"""Base MOO runner for strategies.

Concrete run_moo.py files specify strategy module and param bounds. All common logic here.

Usage in concrete run_moo.py:
    from trdr.strategy.moo_runner import run_moo_benchmark
    run_moo_benchmark(
        strategy_module="trdr.strategy.volume_area_breakout.strategy",
        config_class="VolumeAreaBreakoutConfig",
        strategy_class="VolumeAreaBreakoutStrategy",
        symbol=Symbol.parse("crypto:ETH/USD"),
        timeframe=Timeframe.parse("15m"),
        lookback=Duration.parse("750h"),
        param_bounds=[
            ParamBounds("atr_threshold", 1.0, 4.0),
            ParamBounds("stop_loss_multiplier", 1.0, 3.0),
        ],
    )
"""

import asyncio
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from trdr.backtest import PaperExchangeConfig
from trdr.core import Feed, load_config
from trdr.data import AlpacaDataClient
from trdr.optimize import MooConfig, ParamBounds, run_moo, select_from_pareto
from trdr.optimize.pareto import display_pareto_front
from trdr.optimize.walk_forward_moo import WalkForwardMooResult, run_walk_forward_moo

from .sica_runner import get_primary_requirement

if TYPE_CHECKING:
    from trdr.core import Duration, Symbol, Timeframe

project_root = Path(__file__).parent.parent.parent.parent


async def _get_bars(strategy) -> tuple[dict[str, list], "DataRequirement"]:
    """Fetch bars based on strategy requirements."""
    from trdr.backtest import align_feeds

    config = load_config()
    client = AlpacaDataClient(config.alpaca, Path(project_root / "data/cache"))

    requirements = strategy.get_data_requirements()
    primary = get_primary_requirement(requirements)

    bars_dict = await client.get_bars_multi(requirements)
    primary_bars = bars_dict[primary.key]

    aligned = {primary.key: primary_bars}
    for req in requirements:
        if req.role != "primary":
            aligned[req.key] = align_feeds(primary_bars, bars_dict[req.key])

    return aligned, primary


def _reload_strategy(module_name: str, config_class: str, strategy_class: str):
    """Force reimport of strategy module."""
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

    mod = importlib.import_module(module_name)
    return getattr(mod, config_class), getattr(mod, strategy_class)


def run_moo_benchmark(
    strategy_module: str,
    config_class: str,
    strategy_class: str,
    symbol: "Symbol",
    timeframe: "Timeframe",
    lookback: "Duration",
    param_bounds: list[ParamBounds],
    objectives: list[str] | None = None,
    population_size: int = 50,
    generations: int = 100,
    min_trades: int = 10,
    position_pct: float = 1.0,
    walk_forward: bool = False,
    n_folds: int = 5,
    seed: int | None = 42,
) -> None:
    """Run MOO benchmark for a strategy.

    Args:
        strategy_module: Full module path (e.g., "trdr.strategy.volume_area_breakout.strategy")
        config_class: Config class name (e.g., "VolumeAreaBreakoutConfig")
        strategy_class: Strategy class name (e.g., "VolumeAreaBreakoutStrategy")
        symbol: Symbol object
        timeframe: Timeframe
        lookback: Duration
        param_bounds: List of ParamBounds for optimization
        objectives: List of objective names (default: sharpe, max_drawdown, profit_factor)
        population_size: GA population size
        generations: Number of generations
        min_trades: Minimum trades for valid solution
        position_pct: Position size as fraction of capital
        walk_forward: If True, run walk-forward MOO
        n_folds: Number of folds for walk-forward
        seed: Random seed for reproducibility
    """
    if objectives is None:
        objectives = ["sharpe", "max_drawdown", "profit_factor"]

    # Load strategy classes
    config_cls, strategy_cls = _reload_strategy(strategy_module, config_class, strategy_class)

    # Create temp strategy for data requirements
    temp_config = config_cls(symbol=symbol, timeframe=timeframe, lookback=lookback)
    temp_strategy = strategy_cls(temp_config)

    # Load bars
    print("Loading bar data...")
    bars, primary = asyncio.run(_get_bars(temp_strategy))
    print(f"Loaded {len(bars[primary.key])} bars")

    # Extract param names for factory
    param_names = [p.name for p in param_bounds]

    def create_strategy(params: dict):
        """Factory function to create strategy with given parameters."""
        # Build config kwargs from params
        config_kwargs = {
            "symbol": symbol,
            "timeframe": timeframe,
            "lookback": lookback,
        }
        for name in param_names:
            # Find the bound to check dtype
            bound = next(p for p in param_bounds if p.name == name)
            if bound.dtype == "int":
                config_kwargs[name] = int(round(params[name]))
            else:
                config_kwargs[name] = params[name]

        config = config_cls(**config_kwargs)
        return strategy_cls(config)

    # MOO config
    moo_config = MooConfig(
        param_bounds=param_bounds,
        objectives=objectives,
        population_size=population_size,
        generations=generations,
        min_trades=min_trades,
        seed=seed,
    )

    # Exchange config
    primary_feed = Feed(symbol=primary.symbol, timeframe=primary.timeframe)
    exchange_config = PaperExchangeConfig(
        primary_feed=primary_feed,
        initial_capital=10000,
        default_position_pct=position_pct,
    )

    print(f"\nRunning {'walk-forward ' if walk_forward else ''}multi-objective optimization...")
    print(f"  Population: {population_size}")
    print(f"  Generations: {generations}")
    print(f"  Objectives: {objectives}")
    print(f"  Parameters: {[p.name for p in param_bounds]}")
    print()

    if walk_forward:
        from trdr.backtest.walk_forward import WalkForwardConfig

        wf_config = WalkForwardConfig(n_folds=n_folds, train_pct=0.7)
        wf_result = run_walk_forward_moo(
            bars=bars,
            strategy_factory=create_strategy,
            exchange_config=exchange_config,
            moo_config=moo_config,
            wf_config=wf_config,
            verbose=True,
        )

        print(f"\nWalk-forward MOO complete!")
        print(f"  Folds: {wf_result.n_folds}")
        oos_summary = wf_result.get_oos_summary()
        if oos_summary:
            print(f"  Avg OOS Sharpe: {oos_summary['avg_oos_sharpe']:.2f}")
            print(f"  Avg OOS Drawdown: {oos_summary['avg_oos_drawdown']:.1%}")
            print(f"  Avg OOS Win Rate: {oos_summary['avg_oos_win_rate']:.1%}")

        # Get robust params
        print("\n" + "=" * 50)
        print("ROBUST PARAMETERS (median across folds)")
        print("=" * 50)
        robust = wf_result.get_robust_params(method="median")
        if robust:
            for i, (name, value) in enumerate(robust.items()):
                # Use dtype from param_bounds
                if param_bounds[i].dtype == "int":
                    print(f"  {name}: {int(round(value))}")
                else:
                    print(f"  {name}: {value:.4f}")
        print("=" * 50)

    else:
        result = run_moo(
            strategy_factory=create_strategy,
            bars=bars,
            exchange_config=exchange_config,
            moo_config=moo_config,
            verbose=True,
        )

        print(f"\nOptimization complete!")
        print(f"  Pareto solutions: {result.n_solutions}")
        print(f"  Generations: {result.n_generations}")
        print(f"  Evaluations: {result.n_evaluations}")

        # Display and select
        display_pareto_front(result)

        # Interactive selection
        selected = select_from_pareto(result, interactive=True)
        if selected:
            print("\n" + "=" * 50)
            print("SELECTED PARAMETERS")
            print("=" * 50)
            for name, value in selected.items():
                if isinstance(value, int):
                    print(f"  {name}: {value}")
                else:
                    print(f"  {name}: {value:.4f}")
            print("=" * 50)
