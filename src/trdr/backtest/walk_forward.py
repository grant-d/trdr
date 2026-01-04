"""Walk-forward K-fold validation for backtesting."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..data import Bar
from .paper_exchange import PaperExchange, PaperExchangeConfig, PaperExchangeResult, Trade

if TYPE_CHECKING:
    from ..strategy import BaseStrategy


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Currently runs out-of-sample backtests on rolling test windows.
    Training windows are reserved for future parameter optimization.

    Args:
        n_folds: Number of rolling windows (default 5)
        train_pct: Fraction reserved for training/optimization (default 0.70)
        min_test_bars: Minimum bars in test set (default 100)
    """

    n_folds: int = 5
    train_pct: float = 0.70
    min_test_bars: int = 100


@dataclass
class Fold:
    """Single train/test fold.

    Args:
        fold_num: Fold number (1-indexed)
        train_start: Training start index
        train_end: Training end index (exclusive)
        test_start: Test start index
        test_end: Test end index (exclusive)
    """

    fold_num: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        """Number of training bars."""
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        """Number of test bars."""
        return self.test_end - self.test_start


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results.

    Args:
        folds: Individual fold results
        config: Walk-forward config used
        exchange_config: PaperExchange config used
    """

    folds: list[PaperExchangeResult]
    fold_info: list[Fold]
    config: WalkForwardConfig
    exchange_config: PaperExchangeConfig

    @property
    def all_trades(self) -> list[Trade]:
        """All trades across all folds."""
        trades = []
        for result in self.folds:
            trades.extend(result.trades)
        return trades

    @property
    def total_trades(self) -> int:
        """Total trades across all folds."""
        return len(self.all_trades)

    @property
    def winning_trades(self) -> int:
        """Total winning trades across all folds."""
        return sum(1 for t in self.all_trades if t.is_winner)

    @property
    def win_rate(self) -> float:
        """Overall win rate across all folds."""
        if not self.all_trades:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_pnl(self) -> float:
        """Total net P&L across all folds."""
        return sum(t.net_pnl for t in self.all_trades)

    @property
    def avg_sortino(self) -> float | None:
        """Average Sortino ratio across folds (excluding None)."""
        sortinos = [f.sortino_ratio for f in self.folds if f.sortino_ratio is not None]
        if not sortinos:
            return None
        return sum(sortinos) / len(sortinos)

    @property
    def avg_profit_factor(self) -> float:
        """Average profit factor across folds."""
        pfs = [f.profit_factor for f in self.folds if f.profit_factor != float("inf")]
        if not pfs:
            return 0.0
        return sum(pfs) / len(pfs)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "walk_forward_config": {
                "n_folds": self.config.n_folds,
                "train_pct": self.config.train_pct,
                "min_test_bars": self.config.min_test_bars,
            },
            "exchange_config": {
                "symbol": self.exchange_config.symbol,
                "warmup_bars": self.exchange_config.warmup_bars,
                "transaction_cost_pct": self.exchange_config.transaction_cost_pct,
                "slippage_pct": self.exchange_config.slippage_pct,
                "default_position_pct": self.exchange_config.default_position_pct,
                "initial_capital": self.exchange_config.initial_capital,
            },
            "aggregate_metrics": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "win_rate": round(self.win_rate, 4),
                "total_pnl": round(self.total_pnl, 2),
                "avg_sortino_ratio": round(self.avg_sortino, 4) if self.avg_sortino else None,
                "avg_profit_factor": round(self.avg_profit_factor, 4),
            },
            "folds": [
                {
                    "fold_num": fold_info.fold_num,
                    "train_bars": fold_info.train_size,
                    "test_bars": fold_info.test_size,
                    "period": {
                        "start": result.start_time,
                        "end": result.end_time,
                    },
                    "metrics": {
                        "total_trades": result.total_trades,
                        "win_rate": round(result.win_rate, 4),
                        "total_pnl": round(result.total_pnl, 2),
                        "sortino_ratio": (
                            round(result.sortino_ratio, 4) if result.sortino_ratio else None
                        ),
                        "max_drawdown": round(result.max_drawdown, 4),
                    },
                }
                for fold_info, result in zip(self.fold_info, self.folds)
            ],
        }


def generate_folds(
    total_bars: int,
    wf_config: WalkForwardConfig,
    warmup_bars: int,
) -> list[Fold]:
    """Generate train/test fold indices for walk-forward validation.

    Uses non-overlapping rolling windows. Each fold divides its chunk into:
    - Training portion (train_pct): Reserved for future parameter optimization
    - Test portion (1 - train_pct): First warmup_bars for indicator init,
      remainder for actual OOS trading. No overlap with training.

    Args:
        total_bars: Total number of bars available
        wf_config: Walk-forward configuration
        warmup_bars: Bars needed for strategy warmup

    Returns:
        List of Fold objects with train/test indices
    """
    # Need enough bars for warmup + at least one test window
    min_required = warmup_bars * 2 + wf_config.min_test_bars
    if total_bars < min_required:
        return []

    # Calculate usable bars after initial warmup
    usable_bars = total_bars - warmup_bars
    fold_size = usable_bars // wf_config.n_folds

    # Each fold needs: warmup + train + warmup + test
    min_fold_size = warmup_bars + wf_config.min_test_bars
    if fold_size < min_fold_size:
        return []

    folds = []
    for i in range(wf_config.n_folds):
        # Rolling windows - each fold is independent (no cumulative training)
        fold_start = warmup_bars + (i * fold_size)
        fold_end = warmup_bars + ((i + 1) * fold_size)

        if i == wf_config.n_folds - 1:
            fold_end = total_bars

        this_fold_size = fold_end - fold_start
        train_size = int(this_fold_size * wf_config.train_pct)
        test_size = this_fold_size - train_size
        # Test window must include warmup_bars (for indicator init) PLUS min_test_bars
        # (for actual trading). Without this, a test window of exactly min_test_bars
        # would have zero trading bars after warmup.
        required_test_bars = warmup_bars + wf_config.min_test_bars

        if test_size < required_test_bars:
            continue

        # Training window (for future optimization)
        train_start = fold_start
        train_end = fold_start + train_size

        # Test window: first warmup_bars are used by engine for indicator warmup,
        # remaining bars are actual out-of-sample trading period. No overlap with train.
        test_start = train_end
        test_end = fold_end

        folds.append(
            Fold(
                fold_num=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    return folds


def run_walk_forward(
    bars: list[Bar],
    strategy: "BaseStrategy",
    exchange_config: PaperExchangeConfig,
    wf_config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Run walk-forward validation on bar data.

    Args:
        bars: Historical bars (oldest first)
        strategy: Strategy instance to test
        exchange_config: PaperExchange configuration
        wf_config: Walk-forward configuration (default 5 folds, 70% train)

    Returns:
        WalkForwardResult with aggregated metrics
    """
    if wf_config is None:
        wf_config = WalkForwardConfig()

    folds = generate_folds(
        total_bars=len(bars),
        wf_config=wf_config,
        warmup_bars=exchange_config.warmup_bars,
    )

    if not folds:
        return WalkForwardResult(
            folds=[],
            fold_info=[],
            config=wf_config,
            exchange_config=exchange_config,
        )

    results = []
    for fold in folds:
        # Run backtest on test set only (train set would be for optimization)
        test_bars = bars[fold.test_start : fold.test_end]
        engine = PaperExchange(exchange_config, strategy)
        strategy.reset()  # Reset strategy state between folds
        result = engine.run(test_bars)
        results.append(result)

    return WalkForwardResult(
        folds=results,
        fold_info=folds,
        config=wf_config,
        exchange_config=exchange_config,
    )
