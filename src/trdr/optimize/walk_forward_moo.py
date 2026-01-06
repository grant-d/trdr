"""Walk-forward multi-objective optimization.

Integrates MOO with walk-forward validation for robust parameter selection.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..backtest.walk_forward import Fold, WalkForwardConfig, generate_folds
from .multi_objective import MooConfig, MooResult, run_moo
from .objectives import calculate_objectives

if TYPE_CHECKING:
    from ..backtest.types import PaperExchangeConfig, PaperExchangeResult
    from ..data import Bar
    from ..strategy import BaseStrategy


@dataclass
class FoldMooResult:
    """MOO result for a single fold.

    Args:
        fold: Fold info (train/test indices)
        train_moo: MOO result from training data
        test_results: Backtest results for Pareto solutions on test data
    """

    fold: Fold
    train_moo: MooResult
    test_results: list["PaperExchangeResult"]


@dataclass
class WalkForwardMooResult:
    """Aggregated walk-forward MOO results.

    Args:
        fold_results: Results for each fold
        wf_config: Walk-forward configuration
        moo_config: MOO configuration
        exchange_config: Paper exchange configuration
    """

    fold_results: list[FoldMooResult]
    wf_config: WalkForwardConfig
    moo_config: MooConfig
    exchange_config: "PaperExchangeConfig"

    @property
    def n_folds(self) -> int:
        """Number of folds."""
        return len(self.fold_results)

    def get_robust_params(self, method: str = "median") -> dict[str, float] | None:
        """Get robust parameter set across all folds.

        Args:
            method: Aggregation method ("median", "mean", "best_oos")

        Returns:
            Parameter dictionary or None if no solutions
        """
        if not self.fold_results:
            return None

        if method == "best_oos":
            return self._best_oos_params()

        # Collect all Pareto solutions across folds
        all_params = []
        for fold_result in self.fold_results:
            moo = fold_result.train_moo
            for i in range(moo.n_solutions):
                all_params.append(moo.get_params_dict(i))

        if not all_params:
            return None

        # Aggregate parameters
        param_names = list(all_params[0].keys())
        result = {}
        for name in param_names:
            values = [p[name] for p in all_params]
            if method == "median":
                result[name] = float(np.median(values))
            else:  # mean
                result[name] = float(np.mean(values))

        return result

    def _best_oos_params(self) -> dict[str, float] | None:
        """Select parameters with best out-of-sample Sharpe."""
        best_params = None
        best_sharpe = float("-inf")

        for fold_result in self.fold_results:
            moo = fold_result.train_moo
            for i, test_result in enumerate(fold_result.test_results):
                if i >= moo.n_solutions:
                    continue
                obj = calculate_objectives(test_result)
                if obj.sharpe > best_sharpe:
                    best_sharpe = obj.sharpe
                    best_params = moo.get_params_dict(i)

        return best_params

    def get_oos_summary(self) -> dict:
        """Summarize out-of-sample performance across folds.

        Returns:
            Dict with aggregated OOS metrics
        """
        all_sharpes = []
        all_drawdowns = []
        all_win_rates = []

        for fold_result in self.fold_results:
            for test_result in fold_result.test_results:
                obj = calculate_objectives(test_result)
                all_sharpes.append(obj.sharpe)
                all_drawdowns.append(obj.max_drawdown)
                all_win_rates.append(obj.win_rate)

        if not all_sharpes:
            return {}

        return {
            "avg_oos_sharpe": float(np.mean(all_sharpes)),
            "std_oos_sharpe": float(np.std(all_sharpes)),
            "avg_oos_drawdown": float(np.mean(all_drawdowns)),
            "max_oos_drawdown": float(np.max(all_drawdowns)),
            "avg_oos_win_rate": float(np.mean(all_win_rates)),
        }


def run_walk_forward_moo(
    bars: dict[str, list["Bar"]],
    strategy_factory: Callable[[dict], "BaseStrategy"],
    exchange_config: "PaperExchangeConfig",
    moo_config: MooConfig,
    wf_config: WalkForwardConfig | None = None,
    validate_oos: bool = True,
    verbose: bool = True,
) -> WalkForwardMooResult:
    """Run walk-forward multi-objective optimization.

    For each fold:
    1. Optimize on training data using NSGA-II/III
    2. Optionally validate Pareto solutions on test data

    Args:
        bars: Bar data keyed by "symbol:timeframe" (aligned to primary feed)
        strategy_factory: Callable that creates strategy from param dict
        exchange_config: Paper exchange configuration
        moo_config: MOO configuration
        wf_config: Walk-forward configuration (default 5 folds, 70% train)
        validate_oos: If True, run Pareto solutions on test data
        verbose: Print progress

    Returns:
        WalkForwardMooResult with all fold results
    """
    from ..backtest import PaperExchange

    if wf_config is None:
        wf_config = WalkForwardConfig()

    primary_key = str(exchange_config.primary_feed)
    primary_bars = bars.get(primary_key)
    if primary_bars is None:
        raise ValueError(f"Primary feed '{primary_key}' missing from bars")

    # Generate folds
    folds = generate_folds(
        total_bars=len(primary_bars),
        wf_config=wf_config,
        warmup_bars=exchange_config.warmup_bars,
    )

    if not folds:
        return WalkForwardMooResult(
            fold_results=[],
            wf_config=wf_config,
            moo_config=moo_config,
            exchange_config=exchange_config,
        )

    fold_results = []
    for fold in folds:
        if verbose:
            print(f"\n{'='*50}")
            print(f"FOLD {fold.fold_num}/{len(folds)}")
            print(f"Train: {fold.train_size} bars, Test: {fold.test_size} bars")
            print("=" * 50)

        # Extract training bars for all feeds
        train_bars_dict = {
            key: feed_bars[fold.train_start : fold.train_end]
            for key, feed_bars in bars.items()
        }

        # Run MOO on training data
        moo_result = run_moo(
            strategy_factory=strategy_factory,
            bars=train_bars_dict,
            exchange_config=exchange_config,
            moo_config=moo_config,
            verbose=verbose,
        )

        if verbose:
            print(f"Found {moo_result.n_solutions} Pareto-optimal solutions")

        # Validate on test data
        test_results = []
        if validate_oos and moo_result.n_solutions > 0:
            test_bars_dict = {
                key: feed_bars[fold.test_start : fold.test_end]
                for key, feed_bars in bars.items()
            }

            for i in range(moo_result.n_solutions):
                params = moo_result.get_params_dict(i)
                strategy = strategy_factory(params)
                strategy.reset()
                engine = PaperExchange(exchange_config, strategy)
                result = engine.run(test_bars_dict)
                test_results.append(result)

            if verbose:
                # Print OOS summary for this fold
                sharpes = [calculate_objectives(r).sharpe for r in test_results]
                print(
                    f"OOS Sharpe: min={min(sharpes):.2f}, max={max(sharpes):.2f}, "
                    f"avg={np.mean(sharpes):.2f}"
                )

        fold_results.append(
            FoldMooResult(
                fold=fold,
                train_moo=moo_result,
                test_results=test_results,
            )
        )

    return WalkForwardMooResult(
        fold_results=fold_results,
        wf_config=wf_config,
        moo_config=moo_config,
        exchange_config=exchange_config,
    )
