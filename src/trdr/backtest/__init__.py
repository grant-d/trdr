"""Backtesting framework for strategy validation."""

from .engine import BacktestConfig, BacktestEngine, BacktestResult, Trade
from .walk_forward import (
    Fold,
    WalkForwardConfig,
    WalkForwardResult,
    generate_folds,
    run_walk_forward,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "Trade",
    "Fold",
    "WalkForwardConfig",
    "WalkForwardResult",
    "generate_folds",
    "run_walk_forward",
]
